import asyncio
import time
from typing import List, Dict
from engine.retrieval_eval import RetrievalEvaluator


class BenchmarkRunner:
    def __init__(self, agent, judge):
        self.agent = agent
        self.judge = judge
        self.retrieval_eval = RetrievalEvaluator()

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        # 1. Call Agent
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        # 2. Retrieval + RAG quality metrics
        ragas = self.retrieval_eval.evaluate(
            question=test_case["question"],
            answer=response["answer"],
            contexts=response.get("contexts", []),
            expected_ids=test_case.get("ground_truth_ids", []),
            retrieved_ids=response.get("retrieved_ids", []),
        )

        # 3. Multi-Judge evaluation
        judge_result = await self.judge.evaluate_multi_judge(
            question=test_case["question"],
            answer=response["answer"],
            ground_truth=test_case.get("expected_answer", ""),
            context=test_case.get("context", ""),
        )

        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
            "ragas": ragas,
            "judge": judge_result,
            "token_usage": judge_result.get("token_usage", {}),
            "status": "pass" if judge_result["final_score"] >= 3 else "fail",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Run tests in parallel batches to avoid rate limits."""
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.run_single_test(case) for case in batch]
            )
            results.extend(batch_results)
        return results
