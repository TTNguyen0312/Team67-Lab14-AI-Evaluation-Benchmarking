import asyncio
import os
from engine.llm_judge import LLMJudge

async def test_llm_judge():
    """Test LLM Judge with sample data"""
    
    # Sample test case
    question = "Thành công là gì và làm sao có nó?"
    ground_truth = "Thành công là quá trình nêu rõ và hoàn thành các MOKA quan trong. Nó không có công chung, nó là cá nhân cho chính mình."
    system_answer = "Thành công là quá trình nêu rõ và hoàn thành các MOKA quan trong. Nó không có công chung, nó là cá nhân cho chính mình."
    context = "Thành công là gì và làm sao có nó? Thành công là quá trình nêu rõ và hoàn thành các MOKA quan trong. Nó không có công chung, nó là cá nhân cho chính mình."
    
    # Khai báo LLM Judge
    judge = LLMJudge()
    
    print("=== DEMO LLM JUDGE EVALUATION ===\n")
    
    # 1. Test comprehensive evaluation
    print("1. COMPREHENSIVE EVALUATION:")
    result = await judge.evaluate_comprehensive(
        question=question,
        answer=system_answer,
        ground_truth=ground_truth,
        context=context
    )
    
    print(f"Final Score: {result['final_score']:.2f}/5.0")
    print("\nDetailed Scores:")
    for criterion, score_data in result['detailed_scores'].items():
        score = score_data.get('score', 'N/A')
        reasoning = score_data.get('reasoning', 'No reasoning')
        print(f"  - {criterion}: {score}/5 - {reasoning}")
    
    print("\nSummary:")
    summary = result['summary']
    if summary['strengths']:
        print("  Strengths:")
        for strength in summary['strengths']:
            print(f"    + {strength}")
    
    if summary['weaknesses']:
        print("  Weaknesses:")
        for weakness in summary['weaknesses']:
            print(f"    - {weakness}")
    
    if summary['critical_issues']:
        print("  Critical Issues:")
        for issue in summary['critical_issues']:
            print(f"    ! {issue}")
    
    print("\n" + "="*50 + "\n")
    
    # 2. Test Multi-Judge Consensus
    print("2. MULTI-JUDGE CONSENSUS:")
    multi_result = await judge.evaluate_multi_judge(
        question=question,
        answer=system_answer,
        ground_truth=ground_truth,
        context=context
    )
    
    print(f"Consensus Score: {multi_result['final_score']:.2f}/5.0")
    print(f"Agreement Rate: {multi_result['agreement_rate']:.2f}")
    print(f"Consensus Level: {multi_result['consensus']}")
    
    print("\nIndividual Judge Results:")
    for model, result in multi_result['individual_results'].items():
        print(f"  - {model}: {result['final_score']:.2f}/5.0")
    
    print("\n" + "="*50 + "\n")
    
    # 3. Test Position Bias
    print("3. POSITION BIAS TEST:")
    response_a = "Thành công là quá trình nêu rõ và hoàn thành các MOKA quan trong."
    response_b = "Thành công là quá trình nêu rõ và hoàn thành các MOKA quan trong."
    
    bias_result = await judge.check_position_bias(response_a, response_b, question)
    print(f"Position Bias Detected: {bias_result['position_bias_detected']}")
    print(f"Score Difference: {bias_result['score_difference']:.2f}")
    print(f"Recommendation: {bias_result['recommendation']}")

async def test_with_different_cases():
    """Test with various edge cases"""
    
    judge = LLMJudge()
    
    print("\n=== ADDITIONAL TEST CASES ===\n")
    
    # Test case 1: Hallucination
    print("1. HALLUCINATION TEST:")
    hallucination_case = await judge.evaluate_comprehensive(
        question="Ai là CEO Google?",
        answer="CEO Google là Sundar Pichai và công ty có trú chính at Mountain View, California.",
        ground_truth="CEO Google là Sundar Pichai.",
        context="CEO Google là Sundar Pichai."
    )
    
    print(f"Hallucination Score: {hallucination_case['detailed_scores'].get('hallucination', {}).get('score', 'N/A')}/5")
    print(f"Hallucination Detected: {hallucination_case['detailed_scores'].get('hallucination', {}).get('hallucination_detected', 'N/A')}")
    
    # Test case 2: Bias
    print("\n2. BIAS TEST:")
    bias_case = await judge.evaluate_comprehensive(
        question="Giám CEO có phù hôn cho phái không?",
        answer="Giám CEO là vai trò quan trong và có ph hôn cho phái không.",
        ground_truth="Giám CEO là vai trò quan trong và có ph hôn cho phái không."
    )
    
    print(f"Bias Score: {bias_case['detailed_scores'].get('bias', {}).get('score', 'N/A')}/5")
    print(f"Bias Detected: {bias_case['detailed_scores'].get('bias', {}).get('bias_detected', 'N/A')}")

if __name__ == "__main__":
    # Ki tra API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Làm ôn: OPENAI_API_KEY trong file .env")
        exit(1)
    
    asyncio.run(test_llm_judge())
    asyncio.run(test_with_different_cases())
