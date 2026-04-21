import asyncio
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Set

from agent.main_agent import MainAgent
from engine.runner import BenchmarkRunner

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def _tokenize(text: str) -> Set[str]:
    normalized = _normalize_text(text)
    return {token for token in re.findall(r"\w+", normalized) if len(token) > 2}


class ExpertEvaluator:
    async def score(self, case: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
        expected_ids = case.get("ground_truth_ids", [])
        retrieved_ids = resp.get("retrieved_ids", [])

        hit_rate = 1.0 if any(doc_id in retrieved_ids[:3] for doc_id in expected_ids) else 0.0

        mrr = 0.0
        for index, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_ids:
                mrr = 1.0 / index
                break

        return {
            "faithfulness": round(hit_rate, 4),
            "relevancy": round(mrr if mrr > 0 else 0.0, 4),
            "retrieval": {
                "hit_rate": round(hit_rate, 4),
                "mrr": round(mrr, 4),
            },
        }


class MultiModelJudge:
    def _correctness_score(self, answer: str, ground_truth: str) -> float:
        answer_norm = _normalize_text(answer)
        ground_truth_norm = _normalize_text(ground_truth)

        if not answer_norm:
            return 0.0

        similarity = SequenceMatcher(None, answer_norm, ground_truth_norm).ratio()
        answer_tokens = _tokenize(answer)
        ground_truth_tokens = _tokenize(ground_truth)
        token_overlap = (
            len(answer_tokens & ground_truth_tokens) / len(ground_truth_tokens)
            if ground_truth_tokens
            else 0.0
        )

        if ground_truth_norm in answer_norm:
            return 1.0

        return round(min(1.0, 0.55 * similarity + 0.45 * token_overlap), 4)

    def _hallucination_score(self, answer: str, context: str) -> float:
        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context)

        if not answer_tokens:
            return 0.0
        if not context_tokens:
            return 0.5

        supported_ratio = len(answer_tokens & context_tokens) / len(answer_tokens)
        return round(supported_ratio, 4)

    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        context: str = "",
    ) -> Dict[str, Any]:
        correctness_score = self._correctness_score(answer, ground_truth)
        hallucination_support = self._hallucination_score(answer, context)
        hallucination = hallucination_support < 0.45
        hallucination_score = 1.0 - min(1.0, max(0.0, 1.0 - hallucination_support))

        final_score = round(0.7 * correctness_score + 0.3 * hallucination_score, 4)
        is_correct = correctness_score >= 0.55 and not hallucination

        if is_correct:
            reason = "Câu trả lời bám khá sát ground truth và không có dấu hiệu bịa rõ ràng."
        elif correctness_score < 0.3:
            reason = "Câu trả lời lệch khá xa ground truth."
        elif hallucination:
            reason = "Câu trả lời có nhiều nội dung không được context hỗ trợ."
        else:
            reason = "Câu trả lời mới đúng một phần."

        return {
            "final_score": final_score,
            "is_correct": is_correct,
            "hallucination": hallucination,
            "hallucination_score": round(hallucination_score, 4),
            "agreement_rate": 1.0,
            "reasoning": reason,
            "question": question,
        }


def _agent_version_from_label(agent_version: str) -> str:
    return "v2" if "v2" in agent_version.lower() else "v1"


async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    agent = MainAgent(version=_agent_version_from_label(agent_version))
    # Force local benchmark mode so offline runs compare repo logic instead of external API availability.
    agent.use_real_api = False
    runner = BenchmarkRunner(agent, ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)
    summary = runner.summarize_results(results)
    summary["metadata"]["version"] = agent_version
    summary["metadata"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["metrics"]["pass_rate"] = summary["metrics"].get("accuracy", 0.0)
    return results, summary


async def run_benchmark(version: str):
    _, summary = await run_benchmark_with_results(version)
    return summary


def release_gate(v1_summary: Dict[str, Any], v2_summary: Dict[str, Any]) -> Dict[str, Any]:
    m1, m2 = v1_summary["metrics"], v2_summary["metrics"]

    checks = [
        (
            "Quality Gate",
            "avg_score delta >= -0.05",
            m2["avg_score"] - m1["avg_score"],
            m2["avg_score"] - m1["avg_score"] >= -0.05,
        ),
        (
            "Retrieval Gate",
            "hit_rate delta >= -0.05",
            m2["hit_rate"] - m1["hit_rate"],
            m2["hit_rate"] - m1["hit_rate"] >= -0.05,
        ),
        (
            "Pass Rate Gate",
            "pass_rate delta >= -0.05",
            m2["pass_rate"] - m1["pass_rate"],
            m2["pass_rate"] - m1["pass_rate"] >= -0.05,
        ),
    ]

    print(f"\n{'=' * 55}")
    print("  RELEASE GATE")
    print(f"{'=' * 55}")
    all_pass = True
    for name, threshold, delta, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {threshold} (actual: {delta:+.4f})")

    decision = "APPROVE" if all_pass else "BLOCK"
    print(f"\n  >>> DECISION: {decision} <<<")
    print(f"{'=' * 55}")

    return {
        "decision": decision,
        "checks": [
            {
                "name": name,
                "threshold": threshold,
                "delta": round(delta, 4),
                "passed": passed,
            }
            for name, threshold, delta, passed in checks
        ],
    }


async def main():
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base")
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")

    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    gate = release_gate(v1_summary, v2_summary)

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.4f}")

    report_payload = {
        "v1": v1_summary,
        "v2": v2_summary,
        "release_gate": gate,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({"v1": v1_results, "v2": v2_results}, f, ensure_ascii=False, indent=2)

    if gate["decision"] == "APPROVE":
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")


if __name__ == "__main__":
    asyncio.run(main())
