"""LLM-as-Judge orchestrator for evaluation - Simplified for local use"""

import logging
from pathlib import Path
from typing import Optional, List, Callable, Dict

from ..core.interfaces import RetrievalResult, JudgeEvaluationResult
from .correctness_evaluator import CorrectnessEvaluator
from .hallucination_evaluator import HallucinationEvaluator
from .risk_direction_evaluator import RiskDirectionEvaluator
from .risk_impact_evaluator import RiskImpactEvaluator
from .cost_extraction_evaluator import CostExtractionEvaluator

logger = logging.getLogger(__name__)


def load_prompts_from_disk(prompts_dir: str) -> Dict[str, str]:
    """
    Load evaluation prompt templates from a directory.
    Expects filenames: correctness.md, hallucination.md, risk_direction.md,
    risk_impact.md, cost_extraction_prompt.md (or cost_extraction.md).
    Keys in returned dict: correctness, hallucination, risk_direction, risk_impact, cost_extraction.
    """
    base = Path(prompts_dir)
    if not base.exists():
        logger.warning(f"Prompts directory not found: {prompts_dir}, using in-code defaults")
        return {}
    out = {}
    for key, candidates in [
        ("correctness", ["correctness.md"]),
        ("hallucination", ["hallucination.md"]),
        ("risk_direction", ["risk_direction.md"]),
        ("risk_impact", ["risk_impact.md"]),
        ("cost_extraction", ["cost_extraction_prompt.md", "cost_extraction.md"]),
    ]:
        for name in candidates:
            p = base / name
            if p.exists():
                try:
                    out[key] = p.read_text(encoding="utf-8")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {p}: {e}")
    return out


class JudgeEvaluator:
    """Thin wrapper around evaluate_answer_with_judge for use in the pipeline."""

    def __init__(
        self,
        llm_function: Callable[[str, float, int], str],
        prompt_templates: Optional[Dict[str, str]] = None,
    ):
        self.llm_function = llm_function
        self.prompt_templates = prompt_templates or {}

    def evaluate(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
        retrieved_chunks: List[RetrievalResult],
    ) -> JudgeEvaluationResult:
        return evaluate_answer_with_judge(
            query=question,
            retrieved_context=retrieved_chunks,
            model_answer=generated_answer,
            reference_answer=reference_answer,
            llm_call_function=self.llm_function,
            prompt_templates=self.prompt_templates,
        )


def evaluate_answer_with_judge(
    query: str,
    retrieved_context: List[RetrievalResult],
    model_answer: str,
    reference_answer: str,
    llm_call_function: Callable[[str, float, int], str],
    prompt_templates: Optional[dict] = None
) -> JudgeEvaluationResult:
    """
    Evaluate model answer using LLM-as-Judge with deterministic orchestration.
    
    Orchestrates evaluation process:
    1. Correctness classification (direct comparison to reference answer)
    2. Hallucination binary classification (grounding analysis against retrieved context)
    3. Risk direction classification (if correctness is True)
    4. Impact magnitude calculation (if correctness is True)
    
    Args:
        query: Original user query
        retrieved_context: List of retrieved chunks with similarity scores
        model_answer: Generated answer from RAG system
        reference_answer: Gold reference answer for comparison
        llm_call_function: Function to call LLM (prompt, temperature, max_tokens) -> response
        prompt_templates: Optional dict of custom prompt templates for each evaluator
        
    Returns:
        JudgeEvaluationResult object containing:
        - correctness_binary: bool (true if answer is correct)
        - hallucination_binary: bool (true if hallucination detected)
        - risk_direction: Optional[int] (-1, 0, or 1, None if no deviation)
        - risk_impact: Optional[int] (0, 1, 2, or 3, None if no deviation)
        - reasoning: str (reasoning trace from all LLM node outputs)
        - failure_mode: Optional[str] (e.g., "cost misstatement", "omitted deductible")
        
    Raises:
        Exception: If LLM calls fail
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
    
    # Initialize prompt templates if provided
    prompt_templates = prompt_templates or {}
    
    logger.info(f"Starting judge evaluation for query: '{query[:50]}...'")
    
    # Step 1: Call correctness LLM-node (always)
    logger.debug("Step 1: Evaluating correctness")
    correctness_evaluator = CorrectnessEvaluator(
        llm_call_function=llm_call_function,
        prompt_template=prompt_templates.get("correctness")
    )
    correctness_result, correctness_reasoning = _call_evaluator_with_reasoning(
        correctness_evaluator,
        lambda: correctness_evaluator._construct_prompt(query, model_answer, reference_answer),
        "correctness_binary"
    )
    
    # Step 2: Call hallucination LLM-node (always)
    logger.debug("Step 2: Evaluating hallucination")
    hallucination_evaluator = HallucinationEvaluator(
        llm_call_function=llm_call_function,
        prompt_template=prompt_templates.get("hallucination")
    )
    hallucination_result, hallucination_reasoning = _call_evaluator_with_reasoning(
        hallucination_evaluator,
        lambda: hallucination_evaluator._construct_prompt(retrieved_context, model_answer),
        "hallucination_binary"
    )
    
    # Step 3 & 4: Conditional - if correctness is True, call risk classification and impact nodes
    risk_direction = None
    risk_impact = None
    cost_reasoning = None
    impact_reasoning = None
    
    if correctness_result:
        logger.debug("Step 3: Evaluating risk direction (correctness=True)")
        # Step 3: Call risk direction classification node
        risk_direction_evaluator = RiskDirectionEvaluator(
            llm_call_function=llm_call_function,
            prompt_template=prompt_templates.get("risk_direction")
        )
        risk_direction, cost_reasoning = _call_evaluator_with_reasoning(
            risk_direction_evaluator,
            lambda: risk_direction_evaluator._construct_prompt(model_answer, retrieved_context),
            "risk_direction"
        )
        
        logger.debug("Step 4: Extracting costs and calculating impact")
        # Step 4: Extract costs and calculate impact
        cost_extractor = CostExtractionEvaluator(
            llm_call_function=llm_call_function,
            prompt_template=prompt_templates.get("cost_extraction")
        )
        
        # Extract costs from model answer
        try:
            model_answer_cost = cost_extractor.extract_costs(model_answer)
        except Exception as e:
            logger.warning(f"Failed to extract costs from model answer: {e}")
            model_answer_cost = {"reasoning": f"Cost extraction failed: {str(e)}"}
        
        # Extract costs from retrieved chunks
        chunks_text = " ".join([chunk.chunk_text for chunk in retrieved_context])
        try:
            actual_cost = cost_extractor.extract_costs(chunks_text)
        except Exception as e:
            logger.warning(f"Failed to extract costs from retrieved chunks: {e}")
            actual_cost = {"reasoning": f"Cost extraction failed: {str(e)}"}
        
        # Calculate impact only if we successfully extracted costs
        if "money" in model_answer_cost or "time" in model_answer_cost or "steps" in model_answer_cost:
            if "money" in actual_cost or "time" in actual_cost or "steps" in actual_cost:
                impact_evaluator = RiskImpactEvaluator(
                    llm_call_function=llm_call_function,
                    prompt_template=prompt_templates.get("risk_impact")
                )
                risk_impact, impact_reasoning = _call_evaluator_with_reasoning(
                    impact_evaluator,
                    lambda: impact_evaluator._construct_prompt(model_answer_cost, actual_cost),
                    "risk_impact"
                )
            else:
                logger.debug("No costs found in actual context, skipping impact calculation")
                impact_reasoning = "No costs found in retrieved context"
        else:
            logger.debug("No costs found in model answer, skipping impact calculation")
            impact_reasoning = "No costs found in model answer"
    else:
        logger.debug("Skipping risk evaluation (correctness=False)")
    
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
    result = JudgeEvaluationResult(
        correctness_binary=correctness_result,
        hallucination_binary=hallucination_result,
        risk_direction=risk_direction,
        risk_impact=risk_impact,
        reasoning=reasoning,
        failure_mode=failure_mode
    )
    
    logger.info(f"Judge evaluation completed: correctness={correctness_result}, hallucination={hallucination_result}, risk_direction={risk_direction}, risk_impact={risk_impact}")
    return result


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
        result_field_name: Name of the result field in the JSON response
        
    Returns:
        Tuple of (result, reasoning)
        
    Raises:
        Exception: If LLM call fails
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
    in the reasoning text.
    
    Args:
        reasoning: Combined reasoning trace
        
    Returns:
        Failure mode string if detected, None otherwise
    """
    reasoning_lower = reasoning.lower()
    
    # Common failure mode patterns for insurance contexts
    failure_modes = [
        "cost misstatement",
        "omitted deductible",
        "incorrect copay", 
        "wrong coverage",
        "missing information",
        "hallucination",
        "incorrect cost",
        "premium error",
        "coinsurance mistake",
        "out-of-pocket error"
    ]
    
    for mode in failure_modes:
        if mode in reasoning_lower:
            return mode.replace("_", " ").title()
    
    return None


class JudgeEvaluator:
    """Thin wrapper for eval.py: loads prompts from disk and calls evaluate_answer_with_judge."""

    def __init__(
        self,
        llm_function: Callable[[str, float, int], str],
        prompts_dir: Optional[str] = None,
        prompt_templates: Optional[dict] = None,
    ):
        """
        Initialize judge evaluator.

        Args:
            llm_function: Function (prompt, temperature, max_tokens) -> response text.
            prompts_dir: Optional path to prompts directory (e.g. "prompts/evaluation").
                        If set, prompts are loaded from {prompts_dir}/correctness.md, etc.
            prompt_templates: Optional dict of name -> template string. Overrides prompts_dir if provided.
        """
        self.llm_function = llm_function
        self.prompts_dir = prompts_dir
        self._prompt_templates = prompt_templates

    def _load_prompts(self) -> dict:
        """Load prompt templates from prompts_dir or return cached dict."""
        if self._prompt_templates is not None:
            return self._prompt_templates
        if not self.prompts_dir:
            return {}
        from pathlib import Path
        d = Path(self.prompts_dir)
        if not d.exists():
            logger.warning(f"Prompts dir not found: {d}, using no custom prompts")
            return {}
        out = {}
        for name, filename in [
            ("correctness", "correctness_prompt.md"),
            ("hallucination", "hallucination_prompt.md"),
            ("risk_direction", "risk_direction_prompt.md"),
            ("risk_impact", "risk_impact_prompt.md"),
            ("cost_extraction", "cost_extraction_prompt.md"),
        ]:
            p = d / filename
            if p.exists():
                try:
                    out[name] = p.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Failed to load {p}: {e}")
        return out

    def evaluate(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
        context_chunks: List[str],
    ) -> JudgeEvaluationResult:
        """
        Evaluate one example. context_chunks are plain text strings; they are converted to RetrievalResult for the judge.
        """
        # Build minimal RetrievalResult list from context text list
        retrieved_context = [
            RetrievalResult(chunk_id=f"ctx_{i}", similarity_score=0.0, chunk_text=t)
            for i, t in enumerate(context_chunks)
        ]
        if not retrieved_context:
            retrieved_context = [
                RetrievalResult(chunk_id="ctx_0", similarity_score=0.0, chunk_text="[No context]")
            ]
        prompt_templates = self._load_prompts()
        return evaluate_answer_with_judge(
            query=question,
            retrieved_context=retrieved_context,
            model_answer=generated_answer,
            reference_answer=reference_answer,
            llm_call_function=self.llm_function,
            prompt_templates=prompt_templates or None,
        )