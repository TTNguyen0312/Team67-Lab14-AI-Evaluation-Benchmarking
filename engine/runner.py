import asyncio
import time
from typing import List, Dict, Any


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # 1. Gọi Agent
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        # Kỳ vọng response có dạng:
        # {
        #   "answer": "...",
        #   "retrieved_chunk_ids": [...],   # optional nhưng nên có
        #   "retrieved_doc_ids": [...],     # optional
        #   ...
        # }

        agent_answer = response.get("answer", "")
        retrieved_chunk_ids = response.get("retrieved_chunk_ids", [])
        retrieved_doc_ids = response.get("retrieved_doc_ids", [])

        # 2. Chạy retrieval / RAG metrics
        ragas_scores = await self.evaluator.score(test_case, response)

        # 3. Chạy Multi-Judge (GPT + Claude)
        judge_result = await self.judge.evaluate_multi_judge(
            question=test_case["question"],
            answer=agent_answer,
            ground_truth=test_case["expected_answer"],
            context=test_case.get("context", "")
        )

        # 4. Quyết định pass/fail
        status = "pass" if judge_result["is_correct"] and judge_result["final_score"] >= 0.7 else "fail"

        return {
            "test_case": test_case["question"],
            "expected_answer": test_case.get("expected_answer", ""),
            "agent_response": agent_answer,
            "ground_truth_ids": test_case.get("ground_truth_ids", []),
            "metadata": test_case.get("metadata", {}),
            "context": test_case.get("context", ""),

            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_doc_ids": retrieved_doc_ids,

            "latency": round(latency, 4),
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": status
        }

    async def run_all(self, dataset: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Chạy theo batch để tránh rate limit.
        """
        results = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            results.extend(batch_results)

        return results

    def summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        if total == 0:
            return {
                "metadata": {
                    "total": 0
                },
                "metrics": {}
            }

        pass_count = sum(1 for r in results if r["status"] == "pass")
        fail_count = total - pass_count

        avg_score = sum(r["judge"].get("final_score", 0.0) for r in results) / total
        avg_agreement = sum(r["judge"].get("agreement_rate", 0.0) for r in results) / total
        hallucination_rate = sum(1 for r in results if r["judge"].get("hallucination", False)) / total
        avg_latency = sum(r.get("latency", 0.0) for r in results) / total

        # retrieval metrics
        hit_rates = []
        mrrs = []

        for r in results:
            ragas = r.get("ragas", {})
            retrieval = ragas.get("retrieval", {})
            hit_rates.append(retrieval.get("hit_rate", 0.0))
            mrrs.append(retrieval.get("mrr", 0.0))

        avg_hit_rate = sum(hit_rates) / total if hit_rates else 0.0
        avg_mrr = sum(mrrs) / total if mrrs else 0.0

        return {
            "metadata": {
                "total": total
            },
            "metrics": {
                "accuracy": round(pass_count / total, 4),
                "pass_count": pass_count,
                "fail_count": fail_count,
                "avg_score": round(avg_score, 4),
                "hallucination_rate": round(hallucination_rate, 4),
                "hit_rate": round(avg_hit_rate, 4),
                "mrr": round(avg_mrr, 4),
                "agreement_rate": round(avg_agreement, 4),
                "avg_latency": round(avg_latency, 4)
            }
        }