import asyncio
import json
import os
import time
from typing import Dict, Any
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent


async def run_benchmark_with_results(agent_version: str):
    print(f"Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    version = "v2" if "V2" in agent_version else "v1"
    agent = MainAgent(version=version)
    judge = LLMJudge()

    runner = BenchmarkRunner(agent, judge)
    results = await runner.run_all(dataset)

    total = len(results)
    pass_count = sum(1 for r in results if r["status"] == "pass")

    # Aggregate token usage across all test cases
    agg_tokens: Dict[str, Any] = {"total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0, "by_model": {}}
    for r in results:
        tu = r.get("token_usage", {})
        agg_tokens["total_input_tokens"] += tu.get("total_input_tokens", 0)
        agg_tokens["total_output_tokens"] += tu.get("total_output_tokens", 0)
        agg_tokens["total_tokens"] += tu.get("total_tokens", 0)
        agg_tokens["total_cost_usd"] += tu.get("total_cost_usd", 0.0)
        for model, usage in tu.get("by_model", {}).items():
            if model not in agg_tokens["by_model"]:
                agg_tokens["by_model"][model] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
            agg_tokens["by_model"][model]["input_tokens"] += usage.get("input_tokens", 0)
            agg_tokens["by_model"][model]["output_tokens"] += usage.get("output_tokens", 0)
            agg_tokens["by_model"][model]["total_tokens"] += usage.get("total_tokens", 0)
            agg_tokens["by_model"][model]["cost_usd"] += usage.get("cost_usd", 0.0)
    agg_tokens["total_cost_usd"] = round(agg_tokens["total_cost_usd"], 6)

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "mrr": sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total,
            "agreement_rate": sum(r["judge"].get("agreement_rate", 0) for r in results) / total,
            "pass_rate": pass_count / total,
        },
        "token_usage": agg_tokens,
    }
    return results, summary


async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary


def release_gate(v1_summary, v2_summary):
    """Release Gate tự động — quyết định APPROVE/BLOCK."""
    m1, m2 = v1_summary["metrics"], v2_summary["metrics"]

    checks = [
        ("Quality Gate",   "avg_score delta >= -0.5",  m2["avg_score"] - m1["avg_score"],   m2["avg_score"] - m1["avg_score"] >= -0.5),
        ("Retrieval Gate", "hit_rate delta >= -0.05",   m2["hit_rate"] - m1["hit_rate"],     m2["hit_rate"] - m1["hit_rate"] >= -0.05),
        ("Pass Rate Gate", "pass_rate delta >= -0.05",  m2["pass_rate"] - m1["pass_rate"],   m2["pass_rate"] - m1["pass_rate"] >= -0.05),
    ]

    print(f"\n{'='*55}")
    print("  RELEASE GATE")
    print(f"{'='*55}")
    all_pass = True
    for name, threshold, delta, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {threshold} (actual: {delta:+.4f})")

    decision = "APPROVE" if all_pass else "BLOCK"
    print(f"\n  >>> DECISION: {decision} <<<")
    print(f"{'='*55}")

    return {
        "decision": decision,
        "checks": [
            {"name": c[0], "threshold": c[1], "delta": round(c[2], 4), "passed": c[3]}
            for c in checks
        ],
    }


async def main():
    (v1_results, v1_summary), (v2_results, v2_summary) = await asyncio.gather(
        run_benchmark_with_results("Agent_V1_Base"),
        run_benchmark_with_results("Agent_V2_Optimized"),
    )

    if not v1_summary or not v2_summary:
        print("Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n--- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    for metric in ("avg_score", "hit_rate", "mrr", "agreement_rate", "pass_rate"):
        v1_val = v1_summary["metrics"][metric]
        v2_val = v2_summary["metrics"][metric]
        delta = v2_val - v1_val
        print(f"{metric:>15}: V1={v1_val:.4f}  V2={v2_val:.4f}  delta={delta:+.4f}")

    gate = release_gate(v1_summary, v2_summary)

    os.makedirs("reports", exist_ok=True)
    with open("reports/v1_summary.json", "w", encoding="utf-8") as f:
        json.dump(v1_summary, f, ensure_ascii=False, indent=2)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/v1_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v1_results, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)
    with open("reports/release_gate.json", "w", encoding="utf-8") as f:
        json.dump(gate, f, ensure_ascii=False, indent=2)

    print(f"\nReports saved to reports/")


if __name__ == "__main__":
    asyncio.run(main())
