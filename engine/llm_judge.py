import asyncio
import json
import os
from typing import Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
import anthropic


class LLMJudge:
    """
    Multi-judge evaluator using:
    - GPT (OpenAI)
    - Claude (Anthropic)

    Focus:
    - correctness
    - hallucination
    - agreement rate
    """

    def __init__(
        self,
        openai_model: str = "gpt-4o",
        claude_model: str = "claude-3-haiku-20240307"
    ):
        load_dotenv()

        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        if not openai_api_key:
            raise ValueError("Thiếu OPENAI_API_KEY trong .env")
        if not anthropic_api_key:
            raise ValueError("Thiếu ANTHROPIC_API_KEY trong .env")

        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

        self.openai_model = openai_model
        self.claude_model = claude_model

    def _build_prompt(
        self,
        question: str,
        ground_truth: str,
        answer: str,
        context: str = ""
    ) -> str:
        return f"""
Bạn là AI Judge dùng để benchmark hệ thống hỏi đáp nội bộ.

Chỉ đánh giá 2 tiêu chí:
1. correctness: câu trả lời của hệ thống có đúng với ground truth không
2. hallucination: câu trả lời có bịa thêm / suy diễn sai / không được hỗ trợ bởi context không

Dữ liệu đầu vào:

[Câu hỏi]
{question}

[Ground Truth]
{ground_truth}

[Câu trả lời của hệ thống]
{answer}

[Context tham chiếu]
{context}

Quy tắc chấm:

- correctness_score:
  - 1.0 = đúng hoàn toàn hoặc gần như hoàn toàn
  - 0.5 = đúng một phần
  - 0.0 = sai hoặc lệch ý chính

- is_correct:
  - true nếu câu trả lời chấp nhận được về mặt nội dung
  - false nếu sai hoặc không trả lời đúng trọng tâm

- hallucination:
  - true nếu có thông tin bịa, sai, hoặc không được support bởi context
  - false nếu không có hallucination rõ ràng

- hallucination_score:
  - 1.0 = không hallucination
  - 0.5 = có dấu hiệu nhẹ / không chắc chắn
  - 0.0 = hallucination rõ ràng

Hãy trả về DUY NHẤT JSON hợp lệ theo đúng schema này, không thêm markdown, không giải thích ngoài JSON:

{{
  "correctness_score": 0.0,
  "is_correct": false,
  "hallucination": false,
  "hallucination_score": 0.0,
  "reason": "giải thích ngắn gọn"
}}
""".strip()

    async def _call_openai(
        self,
        prompt: str
    ) -> Dict[str, Any]:
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là AI Judge. Luôn trả về JSON hợp lệ, không markdown."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=400
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("OpenAI trả về content rỗng")

            parsed = json.loads(content)
            return {
                "model": self.openai_model,
                "correctness_score": float(parsed.get("correctness_score", 0.0)),
                "is_correct": bool(parsed.get("is_correct", False)),
                "hallucination": bool(parsed.get("hallucination", False)),
                "hallucination_score": float(parsed.get("hallucination_score", 0.0)),
                "reason": parsed.get("reason", "")
            }

        except Exception as e:
            return {
                "model": self.openai_model,
                "correctness_score": 0.0,
                "is_correct": False,
                "hallucination": True,
                "hallucination_score": 0.0,
                "reason": f"OpenAI judge error: {str(e)}",
                "error": True
            }

    async def _call_claude(
        self,
        prompt: str
    ) -> Dict[str, Any]:
        try:
            response = self.anthropic_client.messages.create(
                model=self.claude_model,
                max_tokens=400,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            if not response.content or not response.content[0].text:
                raise ValueError("Claude trả về content rỗng")

            content = response.content[0].text
            parsed = json.loads(content)

            return {
                "model": self.claude_model,
                "correctness_score": float(parsed.get("correctness_score", 0.0)),
                "is_correct": bool(parsed.get("is_correct", False)),
                "hallucination": bool(parsed.get("hallucination", False)),
                "hallucination_score": float(parsed.get("hallucination_score", 0.0)),
                "reason": parsed.get("reason", "")
            }

        except Exception as e:
            return {
                "model": self.claude_model,
                "correctness_score": 0.0,
                "is_correct": False,
                "hallucination": True,
                "hallucination_score": 0.0,
                "reason": f"Claude judge error: {str(e)}",
                "error": True
            }

    async def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Run GPT + Claude in parallel and compute consensus.
        """

        prompt = self._build_prompt(
            question=question,
            ground_truth=ground_truth,
            answer=answer,
            context=context
        )

        gpt_result, claude_result = await asyncio.gather(
            self._call_openai(prompt),
            self._call_claude(prompt)
        )

        results = {
            "gpt": gpt_result,
            "claude": claude_result
        }

        valid_results = [r for r in results.values() if not r.get("error")]

        if not valid_results:
            return {
                "final_score": 0.0,
                "is_correct": False,
                "hallucination": True,
                "agreement_rate": 0.0,
                "consensus": "failed",
                "details": results,
                "reasoning": "Cả GPT và Claude đều lỗi"
            }

        # Score từng judge: correctness quan trọng hơn hallucination
        per_judge_scores = [
            0.7 * r["correctness_score"] + 0.3 * r["hallucination_score"]
            for r in valid_results
        ]
        final_score = sum(per_judge_scores) / len(per_judge_scores)

        # Majority vote for correctness
        correct_votes = sum(1 for r in valid_results if r["is_correct"])
        is_correct = correct_votes >= ((len(valid_results) // 2) + 1)

        # Majority vote for hallucination
        hallucination_votes = sum(1 for r in valid_results if r["hallucination"])
        hallucination = hallucination_votes >= ((len(valid_results) // 2) + 1)

        # Agreement rate based on correctness verdict
        ratio = correct_votes / len(valid_results)
        agreement_rate = max(ratio, 1 - ratio)

        if agreement_rate >= 0.9:
            consensus = "high"
        elif agreement_rate >= 0.7:
            consensus = "medium"
        else:
            consensus = "low"

        return {
            "final_score": round(final_score, 4),      # scale 0-1
            "is_correct": is_correct,
            "hallucination": hallucination,
            "agreement_rate": round(agreement_rate, 4),
            "consensus": consensus,
            "details": results,
            "reasoning": "Consensus from GPT and Claude based on correctness and hallucination"
        }