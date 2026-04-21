import re
from typing import List, Dict


def _tokens(text: str) -> set:
    """Lowercase word tokens, stripping punctuation, min length 2."""
    return {w for w in re.findall(r"\b\w{2,}\b", text.lower())}


class RetrievalEvaluator:
    def __init__(self):
        pass

    # Retrieval metrics
    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """1.0 if at least one expected chunk appears in the top-k retrieved chunks."""
        top_retrieved = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """1 / rank of the first relevant chunk; 0 if none found."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    # RAG quality metrics

    def calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Proportion of answer tokens that appear in the retrieved context.
        Score 0-1: 1.0 means every answer word is grounded in context.
        """
        if not answer or not contexts:
            return 0.0
        answer_tokens = _tokens(answer)
        if not answer_tokens:
            return 0.0
        context_tokens = _tokens(" ".join(contexts))
        grounded = answer_tokens & context_tokens
        return round(len(grounded) / len(answer_tokens), 4)

    def calculate_relevancy(self, question: str, answer: str) -> float:
        """
        Proportion of question tokens covered by the answer.
        Score 0-1: 1.0 means the answer addresses every keyword in the question.
        """
        if not question or not answer:
            return 0.0
        question_tokens = _tokens(question)
        if not question_tokens:
            return 0.0
        answer_tokens = _tokens(answer)
        covered = question_tokens & answer_tokens
        return round(len(covered) / len(question_tokens), 4)

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int = 3,
    ) -> Dict:
        """Run all metrics for a single test case and return the ragas dict."""
        return {
            "faithfulness": self.calculate_faithfulness(answer, contexts),
            "relevancy": self.calculate_relevancy(question, answer),
            "retrieval": {
                "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids, top_k),
                "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
            },
        }

    async def evaluate_batch(self, dataset: List[Dict], top_k: int = 3) -> Dict:
        """Aggregate metrics over a dataset of result dicts."""
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "avg_faithfulness": 0.0, "avg_relevancy": 0.0, "total": 0}

        hit_rates, mrr_scores, faithfulness_scores, relevancy_scores = [], [], [], []

        for item in dataset:
            expected = item.get("ground_truth_ids") or item.get("expected_retrieval_ids", [])
            retrieved = item.get("retrieved_ids", [])
            answer = item.get("answer", "")
            question = item.get("question", "")
            contexts = item.get("contexts", [])

            hit_rates.append(self.calculate_hit_rate(expected, retrieved, top_k))
            mrr_scores.append(self.calculate_mrr(expected, retrieved))
            faithfulness_scores.append(self.calculate_faithfulness(answer, contexts))
            relevancy_scores.append(self.calculate_relevancy(question, answer))

        n = len(dataset)
        return {
            "avg_hit_rate": round(sum(hit_rates) / n, 4),
            "avg_mrr": round(sum(mrr_scores) / n, 4),
            "avg_faithfulness": round(sum(faithfulness_scores) / n, 4),
            "avg_relevancy": round(sum(relevancy_scores) / n, 4),
            "total": n,
        }
