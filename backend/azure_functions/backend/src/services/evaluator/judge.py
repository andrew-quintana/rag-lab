"""LLM-as-Judge orchestrator for evaluation"""

from typing import Optional, List
from dataclasses import dataclass

from src.core.config import Config
from src.core.exceptions import AzureServiceError
from src.core.interfaces import RetrievalResult, JudgeEvaluationResult
from src.services.evaluator.correctness import CorrectnessEvaluator
from src.services.evaluator.hallucination import HallucinationEvaluator
from src.services.evaluator.risk_direction import RiskDirectionEvaluator
from src.services.evaluator.cost_extraction import CostExtractionEvaluator
from src.services.evaluator.risk_impact import RiskImpactEvaluator
from src.db.queries import QueryExecutor


def evaluate_answer_with_judge(
    query: str,
    retrieved_context: List[RetrievalResult],
    model_answer: str,
    reference_answer: str,
    config: Optional[Config] = None,
    query_executor: Optional[QueryExecutor] = None,
    prompt_version: Optional[str] = None,
    live: bool = True
) -> JudgeEvaluationResult:
    """
    Evaluate model answer using LLM-as-Judge with deterministic orchestration.
    
    Orchestrates evaluation process:
    1. Correctness classification (direct comparison to reference answer)
    2. Hallucination binary classification (grounding analysis against retrieved context)
    3. Cost type classification (if correctness is True)
    4. Impact magnitude calculation (if correctness is True)
    
    Args:
        query: Original user query
        retrieved_context: List of retrieved chunks with similarity scores
        model_answer: Generated answer from RAG system
        reference_answer: Gold reference answer for comparison
        config: Application configuration (optional, defaults to Config.from_env())
        query_executor: QueryExecutor instance for database operations (optional, for database prompt loading)
        prompt_version: Prompt version name (e.g., "v1"). Defaults to None (uses live version)
        live: If True and prompt_version is None, loads the live version. Defaults to True
        
    Returns:
        JudgeEvaluationResult object containing:
        - correctness_binary: bool (true if answer is correct)
        - hallucination_binary: bool (true if hallucination detected)
        - risk_direction: Optional[int] (-1, 0, or 1: care avoidance risk, no clear direction, or unexpected cost risk, None if no deviation)
        - risk_impact: Optional[int] (0, 1, 2, or 3: discrete impact magnitude, None if no deviation)
        - reasoning: str (reasoning trace from all LLM node outputs)
        - failure_mode: Optional[str] (e.g., "cost misstatement", "omitted deductible")
        
    Raises:
        AzureServiceError: If Azure Foundry LLM calls fail
        ValueError: If inputs are invalid or empty
    """
    # Validate inputs
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if not retrieved_context:
        raise ValueError("Retrieved context cannot be empty")
    
    if not model_answer or not model_answer.strip():
        raise ValueError("Model answer cannot be empty")
    
    if not reference_answer or not reference_answer.strip():
        raise ValueError("Reference answer cannot be empty")
    
    # Use provided config or default
    if config is None:
        config = Config.from_env()
    
    # Step 1: Call correctness LLM-node (always)
    correctness_evaluator = CorrectnessEvaluator(
        config=config,
        query_executor=query_executor,
        prompt_version=prompt_version,
        live=live
    )
    correctness_result, correctness_reasoning = _call_evaluator_with_reasoning(
        correctness_evaluator,
        lambda: correctness_evaluator._construct_prompt(query, model_answer, reference_answer),
        "correctness_binary"
    )
    
    # Step 2: Call hallucination LLM-node (always)
    hallucination_evaluator = HallucinationEvaluator(
        config=config,
        query_executor=query_executor,
        prompt_version=prompt_version,
        live=live
    )
    hallucination_result, hallucination_reasoning = _call_evaluator_with_reasoning(
        hallucination_evaluator,
        lambda: hallucination_evaluator._construct_prompt(retrieved_context, model_answer),
        "hallucination_binary"
    )
    
    # Step 3 & 4: Conditional - if correctness is True, call cost classification and impact nodes
    risk_direction = None
    risk_impact = None
    cost_reasoning = None
    impact_reasoning = None
    
    if correctness_result:
        # Step 3: Call cost classification node
        risk_direction_evaluator = RiskDirectionEvaluator(
            config=config,
            query_executor=query_executor,
            prompt_version=prompt_version,
            live=live
        )
        risk_direction, cost_reasoning = _call_evaluator_with_reasoning(
            risk_direction_evaluator,
            lambda: risk_direction_evaluator._construct_prompt(model_answer, retrieved_context),
            "risk_direction"
        )
        
        # Step 4: Extract costs and calculate impact
        cost_extractor = CostExtractionEvaluator(
            config=config,
            query_executor=query_executor,
            prompt_version=prompt_version,
            live=live
        )
        
        # Extract costs from model answer
        model_answer_cost = cost_extractor.extract_costs(model_answer)
        
        # Extract costs from retrieved chunks
        chunks_text = " ".join([chunk.chunk_text for chunk in retrieved_context])
        actual_cost = cost_extractor.extract_costs(chunks_text)
        
        # Calculate impact
        impact_evaluator = RiskImpactEvaluator(
            config=config,
            query_executor=query_executor,
            prompt_version=prompt_version,
            live=live
        )
        risk_impact, impact_reasoning = _call_evaluator_with_reasoning(
            impact_evaluator,
            lambda: impact_evaluator._construct_prompt(model_answer_cost, actual_cost),
            "risk_impact"
        )
    
    # Step 5: Construct reasoning trace from all LLM node outputs
    reasoning = _construct_reasoning_trace(
        correctness_result=correctness_result,
        hallucination_result=hallucination_result,
        risk_direction=risk_direction,
        risk_impact=risk_impact,
        correctness_reasoning=correctness_reasoning,
        hallucination_reasoning=hallucination_reasoning,
        cost_reasoning=cost_reasoning,
        impact_reasoning=impact_reasoning
    )
    
    # Step 6: Extract failure mode from reasoning (if present)
    failure_mode = _extract_failure_mode(reasoning)
    
    # Assemble and return JudgeEvaluationResult
    return JudgeEvaluationResult(
        correctness_binary=correctness_result,
        hallucination_binary=hallucination_result,
        risk_direction=risk_direction,
        risk_impact=risk_impact,
        reasoning=reasoning,
        failure_mode=failure_mode
    )


def _call_evaluator_with_reasoning(
    evaluator,
    construct_prompt_fn,
    result_field_name: str
):
    """
    Helper function to call an evaluator and get both classification result and reasoning.
    
    This function constructs the prompt, calls the LLM, parses the response,
    and extracts both the result field and reasoning.
    
    Args:
        evaluator: Evaluator instance
        construct_prompt_fn: Function that constructs the prompt
        result_field_name: Name of the result field in the JSON response (e.g., "correctness_binary")
        
    Returns:
        Tuple of (result, reasoning)
        
    Raises:
        AzureServiceError: If LLM call fails
        ValueError: If response cannot be parsed or is invalid
    """
    # Construct prompt
    prompt = construct_prompt_fn()
    
    # Call LLM
    response_text = evaluator._call_llm(
        prompt=prompt,
        temperature=0.1,
        max_tokens=500
    )
    
    # Parse JSON response
    classification = evaluator._parse_json_response(response_text)
    
    # Extract result field
    if result_field_name not in classification:
        raise ValueError(
            f"Classification missing '{result_field_name}' field. "
            f"Response: {classification}"
        )
    
    result = classification[result_field_name]
    
    # Extract reasoning
    reasoning = classification.get("reasoning", "No reasoning provided")
    
    return result, reasoning


def _construct_reasoning_trace(
    correctness_result: bool,
    hallucination_result: bool,
    risk_direction: Optional[int],
    risk_impact: Optional[int],
    correctness_reasoning: str,
    hallucination_reasoning: str,
    cost_reasoning: Optional[str],
    impact_reasoning: Optional[str]
) -> str:
    """
    Construct combined reasoning trace from all LLM node outputs.
    
    Args:
        correctness_result: Correctness classification result
        hallucination_result: Hallucination classification result
        risk_direction: Risk direction classification result (if available)
        risk_impact: Risk impact calculation result (if available)
        correctness_reasoning: Reasoning from correctness node
        hallucination_reasoning: Reasoning from hallucination node
        cost_reasoning: Reasoning from cost classification node (if available)
        impact_reasoning: Reasoning from impact node (if available)
        
    Returns:
        Combined reasoning trace string
    """
    trace_parts = []
    
    # Always include correctness reasoning
    trace_parts.append(f"Correctness Analysis: {correctness_reasoning}")
    trace_parts.append(f"Correctness Result: {correctness_result}")
    
    # Always include hallucination reasoning
    trace_parts.append(f"Hallucination Analysis: {hallucination_reasoning}")
    trace_parts.append(f"Hallucination Result: {hallucination_result}")
    
    # Conditionally include cost and impact reasoning
    if risk_direction is not None:
        if risk_direction == -1:
            risk_type = "care avoidance risk (-1)"
        elif risk_direction == 0:
            risk_type = "no clear direction (0)"
        else:
            risk_type = "unexpected cost risk (+1)"
        trace_parts.append(f"Risk Direction Analysis: {cost_reasoning}")
        trace_parts.append(f"Risk Direction Result: {risk_type}")
    
    if risk_impact is not None:
        trace_parts.append(f"Risk Impact Analysis: {impact_reasoning}")
        trace_parts.append(f"Risk Impact Result: {risk_impact}")
    
    return "\n\n".join(trace_parts)


def _extract_failure_mode(reasoning: str) -> Optional[str]:
    """
    Extract failure mode from reasoning trace.
    
    This is a simple heuristic that looks for common failure mode patterns
    in the reasoning text. Can be enhanced in the future.
    
    Args:
        reasoning: Combined reasoning trace
        
    Returns:
        Failure mode string if detected, None otherwise
    """
    reasoning_lower = reasoning.lower()
    
    # Common failure mode patterns
    failure_modes = [
        "cost misstatement",
        "omitted deductible",
        "incorrect copay",
        "wrong coverage",
        "missing information",
        "hallucination",
        "incorrect cost"
    ]
    
    for mode in failure_modes:
        if mode in reasoning_lower:
            return mode.replace("_", " ").title()
    
    return None

