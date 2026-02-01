"""Example usage of the simplified evaluation components

This script demonstrates how to use the evaluation components with a simple LLM interface.
"""

import logging
from typing import List
from datetime import datetime

# Import evaluation components
from .evaluation_orchestrator import evaluate_rag_system, print_evaluation_summary
from .core.interfaces import EvaluationExample, RetrievalResult, ModelAnswer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dummy_llm_call(prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
    """
    Dummy LLM function for testing. Replace with actual LLM API call.
    
    Args:
        prompt: The prompt to send to the LLM
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Mock LLM response as JSON string
    """
    # This is a mock implementation - replace with real LLM calls
    
    if "correctness_binary" in prompt or "Correctness" in prompt:
        return """```json
{
    "correctness_binary": true,
    "reasoning": "The model answer correctly addresses the query and matches the reference answer semantically."
}
```"""
    
    elif "hallucination_binary" in prompt or "Hallucination" in prompt:
        return """```json
{
    "hallucination_binary": false,
    "reasoning": "All claims in the model answer are supported by the retrieved context."
}
```"""
    
    elif "risk_direction" in prompt or "Risk Direction" in prompt:
        return """```json
{
    "risk_direction": 0,
    "reasoning": "No clear risk direction can be determined from the cost information provided."
}
```"""
    
    elif "risk_impact" in prompt or "Risk Impact" in prompt:
        return """```json
{
    "risk_impact": 1,
    "reasoning": "Low impact - small cost differences that are unlikely to significantly affect healthcare decisions."
}
```"""
    
    elif "extract" in prompt.lower() or "cost" in prompt.lower():
        return """```json
{
    "money": "$50",
    "time": null,
    "steps": null,
    "reasoning": "Found a monetary cost of $50 mentioned in the text."
}
```"""
    
    else:
        return """```json
{
    "result": "unknown",
    "reasoning": "Mock LLM response for unknown prompt type."
}
```"""


def dummy_rag_retriever(query: str, k: int = 5) -> List[RetrievalResult]:
    """
    Dummy RAG retriever for testing. Replace with actual retrieval system.
    
    Args:
        query: The query to retrieve for
        k: Number of chunks to retrieve
        
    Returns:
        Mock retrieval results
    """
    # This is a mock implementation - replace with real retrieval
    return [
        RetrievalResult(
            chunk_id=f"chunk_{i}",
            similarity_score=0.9 - (i * 0.1),
            chunk_text=f"Mock retrieved chunk {i} for query: {query[:30]}...",
            metadata={"source": f"document_{i}"}
        )
        for i in range(min(k, 3))
    ]


def dummy_rag_generator(query: str, chunks: List[RetrievalResult]) -> ModelAnswer:
    """
    Dummy RAG generator for testing. Replace with actual generation system.
    
    Args:
        query: The original query
        chunks: Retrieved chunks to use for generation
        
    Returns:
        Mock model answer
    """
    # This is a mock implementation - replace with real generation
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    answer_text = f"Based on the retrieved information, here's a response to: {query}. The copay is $50."
    
    return ModelAnswer(
        text=answer_text,
        query_id="mock_query",
        prompt_version="mock_v1",
        retrieved_chunk_ids=chunk_ids,
        timestamp=datetime.now()
    )


def run_evaluation_example():
    """
    Run a simple evaluation example demonstrating the evaluation pipeline.
    """
    logger.info("Starting evaluation example...")
    
    # Create sample evaluation dataset
    evaluation_dataset = [
        EvaluationExample(
            example_id="example_1",
            question="What is the copay for a specialist visit?",
            reference_answer="The copay for a specialist visit is $50.",
            ground_truth_chunk_ids=["chunk_0", "chunk_1"],
            beir_failure_scale_factor=1.0
        ),
        EvaluationExample(
            example_id="example_2", 
            question="How much do I pay for an emergency room visit?",
            reference_answer="Emergency room visits have a $200 copay.",
            ground_truth_chunk_ids=["chunk_1", "chunk_2"],
            beir_failure_scale_factor=1.0
        )
    ]
    
    logger.info(f"Created evaluation dataset with {len(evaluation_dataset)} examples")
    
    # Run evaluation pipeline
    try:
        results = evaluate_rag_system(
            evaluation_dataset=evaluation_dataset,
            rag_retriever=dummy_rag_retriever,
            rag_generator=dummy_rag_generator,
            llm_call_function=dummy_llm_call,
            # Note: BEIR metrics function not provided in this example
            compute_beir_metrics_func=None,
            prompt_templates=None  # Using default prompts
        )
        
        logger.info(f"Evaluation completed successfully with {len(results)} results")
        
        # Print summary
        print_evaluation_summary(results)
        
        # Print detailed results for first example
        if results:
            first_result = results[0]
            print(f"\n=== Detailed Results for {first_result.example_id} ===")
            print(f"Correctness: {first_result.judge_output.correctness_binary}")
            print(f"Hallucination: {first_result.judge_output.hallucination_binary}")
            print(f"Risk Direction: {first_result.judge_output.risk_direction}")
            print(f"Risk Impact: {first_result.judge_output.risk_impact}")
            print(f"Judge Correct: {first_result.meta_eval_output.judge_correct}")
            print(f"Reasoning:\n{first_result.judge_output.reasoning}")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return False
    
    logger.info("Evaluation example completed successfully!")
    return True


if __name__ == "__main__":
    # Run the example
    success = run_evaluation_example()
    if success:
        print("\n✅ Evaluation example completed successfully!")
        print("Replace the dummy functions with your actual LLM, retriever, and generator implementations.")
    else:
        print("\n❌ Evaluation example failed. Check logs for details.")