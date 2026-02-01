"""Meta-Evaluator: Deterministic validation of judge verdicts - Simplified for local use

This module includes bias correction for LLM-as-a-Judge evaluations, adapted from the original
rag_evaluator implementation but simplified for local interfaces.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from math import sqrt

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    # Fallback to approximation if scipy is not available
    HAS_SCIPY = False
    def norm_ppf(p):
        """Simple normal distribution inverse CDF approximation"""
        if p <= 0 or p >= 1:
            raise ValueError("p must be between 0 and 1")
        if p == 0.5:
            return 0.0
        # Simple approximation for common alpha values
        if abs(p - 0.975) < 0.001:  # 95% CI
            return 1.96
        if abs(p - 0.995) < 0.001:  # 99% CI
            return 2.576
        # Linear approximation (rough)
        return (p - 0.5) * 4

from ..core.interfaces import (
    JudgeEvaluationResult,
    MetaEvaluationResult,
    RetrievalResult,
    JudgeMetricScores,
    JudgePerformanceMetrics
)


def meta_evaluate_judge(
    judge_output: JudgeEvaluationResult,
    retrieved_context: List[RetrievalResult],
    model_answer: str,
    reference_answer: str,
    extracted_costs: Optional[Dict[str, Any]] = None,
    actual_costs: Optional[Dict[str, Any]] = None
) -> MetaEvaluationResult:
    """
    Deterministically evaluate whether LLM-as-Judge's verdicts are correct.
    
    Uses rule-based validation logic to compare judge output against ground truth.
    No LLM calls - pure deterministic function.
    
    Validation checks:
    1. correctness_binary: Compare model answer to reference answer (exact or semantic similarity)
    2. hallucination_binary: Check if model answer claims are supported by retrieved chunks
    3. risk_direction: Validate risk direction against extracted costs (if available)
    4. risk_impact: Validate impact magnitude against cost differences (if available)
    
    Args:
        judge_output: JudgeEvaluationResult from LLM-as-Judge
        retrieved_context: Original retrieved chunks (ground truth for hallucination detection)
        model_answer: Original model answer evaluated by judge
        reference_answer: Original reference answer (ground truth for correctness)
        extracted_costs: Optional cost information extracted from model answer
        actual_costs: Optional cost information extracted from retrieved chunks
        
    Returns:
        MetaEvaluationResult object containing:
        - judge_correct: bool (True if all judge verdicts are correct, False otherwise)
        - explanation: Optional[str] (Deterministic explanation of validation results)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if not model_answer or not model_answer.strip():
        raise ValueError("Model answer cannot be empty")
    
    if not reference_answer or not reference_answer.strip():
        raise ValueError("Reference answer cannot be empty")
    
    if not retrieved_context:
        raise ValueError("Retrieved context cannot be empty")
    
    # Perform validations
    validation_results = {}
    
    # Validation 1: correctness_binary
    correctness_valid = _validate_correctness(
        judge_output.correctness_binary,
        model_answer,
        reference_answer
    )
    validation_results["correctness"] = correctness_valid
    
    # Validation 2: hallucination_binary
    hallucination_valid = _validate_hallucination(
        judge_output.hallucination_binary,
        model_answer,
        retrieved_context
    )
    validation_results["hallucination"] = hallucination_valid
    
    # Validation 3 & 4: risk_direction and risk_impact (only if correctness is True and costs available)
    risk_direction_valid = True
    risk_impact_valid = True
    
    if judge_output.correctness_binary and extracted_costs and actual_costs:
        # Only validate risk metrics if correctness is True and we have cost data
        risk_direction_valid = _validate_cost_classification(
            judge_output.risk_direction,
            extracted_costs,
            actual_costs
        )
        validation_results["risk_direction"] = risk_direction_valid
        
        risk_impact_valid = _validate_impact_magnitude(
            judge_output.risk_impact,
            extracted_costs,
            actual_costs
        )
        validation_results["risk_impact"] = risk_impact_valid
    else:
        # If conditions not met, mark as valid (not applicable)
        validation_results["risk_direction"] = True
        validation_results["risk_impact"] = True
    
    # Determine overall judge correctness (all validations must pass)
    judge_correct = all(validation_results.values())
    
    # Generate explanation
    explanation = _generate_explanation(validation_results, judge_output)
    
    # Compute ground truth values for metrics calculation
    ground_truth_correctness = _compute_ground_truth_correctness(model_answer, reference_answer)
    ground_truth_hallucination = _compute_ground_truth_hallucination(model_answer, retrieved_context)
    ground_truth_risk_direction = _compute_ground_truth_risk_direction(extracted_costs, actual_costs) if (judge_output.correctness_binary and extracted_costs and actual_costs) else None
    ground_truth_risk_impact = _compute_ground_truth_risk_impact(extracted_costs, actual_costs) if (judge_output.correctness_binary and extracted_costs and actual_costs) else None
    
    return MetaEvaluationResult(
        judge_correct=judge_correct,
        explanation=explanation,
        ground_truth_correctness=ground_truth_correctness,
        ground_truth_hallucination=ground_truth_hallucination,
        ground_truth_risk_direction=ground_truth_risk_direction,
        ground_truth_risk_impact=ground_truth_risk_impact
    )


def _validate_correctness(
    judge_correctness: bool,
    model_answer: str,
    reference_answer: str
) -> bool:
    """
    Validate correctness_binary verdict by comparing model answer to reference answer.
    
    Uses normalized string comparison (case-insensitive, whitespace-normalized)
    and basic semantic similarity (keyword overlap).
    
    Args:
        judge_correctness: Judge's correctness verdict (True/False)
        model_answer: Model-generated answer
        reference_answer: Gold reference answer
        
    Returns:
        bool: True if judge verdict matches ground truth, False otherwise
    """
    # Normalize answers for comparison
    model_normalized = _normalize_text(model_answer)
    reference_normalized = _normalize_text(reference_answer)
    
    # Check exact match (after normalization)
    exact_match = model_normalized == reference_normalized
    
    # Check semantic similarity (keyword overlap)
    model_keywords = set(_extract_keywords(model_normalized))
    reference_keywords = set(_extract_keywords(reference_normalized))
    
    if not model_keywords or not reference_keywords:
        # If no keywords, fall back to exact match
        semantic_similar = exact_match
    else:
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(model_keywords & reference_keywords)
        union = len(model_keywords | reference_keywords)
        similarity = intersection / union if union > 0 else 0.0
        # Consider similar if > 70% keyword overlap
        semantic_similar = similarity >= 0.7
    
    # Ground truth: answer is correct if exact match OR semantic similar
    ground_truth_correct = exact_match or semantic_similar
    
    # Judge verdict matches ground truth
    return judge_correctness == ground_truth_correct


def _validate_hallucination(
    judge_hallucination: bool,
    model_answer: str,
    retrieved_context: List[RetrievalResult]
) -> bool:
    """
    Validate hallucination_binary verdict by checking if model answer claims are supported.
    
    Checks if key claims in model answer are present in retrieved chunks.
    Uses keyword and phrase matching to determine grounding.
    
    Args:
        judge_hallucination: Judge's hallucination verdict (True/False)
        model_answer: Model-generated answer
        retrieved_context: List of retrieved chunks (ground truth)
        
    Returns:
        bool: True if judge verdict matches ground truth, False otherwise
    """
    if not retrieved_context:
        # No context means any answer is ungrounded (hallucination)
        return judge_hallucination == True
    
    # Combine all retrieved chunk texts
    context_text = " ".join([chunk.chunk_text for chunk in retrieved_context])
    context_normalized = _normalize_text(context_text)
    
    # Extract key claims from model answer (numbers, specific terms, key phrases)
    model_claims = _extract_claims(model_answer)
    
    if not model_claims:
        # No specific claims to validate - assume grounded if answer is generic
        # If answer is very short or generic, likely no hallucination
        is_grounded = len(model_answer.strip()) < 50 or _is_generic_answer(model_answer)
        return judge_hallucination == (not is_grounded)
    
    # Check if claims are supported by context
    unsupported_claims = []
    for claim in model_claims:
        claim_normalized = _normalize_text(claim)
        # Check if claim appears in context (exact or substring match)
        if claim_normalized not in context_normalized:
            # Try partial match (claim keywords in context)
            claim_keywords = _extract_keywords(claim_normalized)
            context_keywords = set(_extract_keywords(context_normalized))
            claim_keywords_set = set(claim_keywords)
            
            # If less than 50% of claim keywords are in context, consider unsupported
            if claim_keywords_set:
                overlap = len(claim_keywords_set & context_keywords) / len(claim_keywords_set)
                if overlap < 0.5:
                    unsupported_claims.append(claim)
            else:
                unsupported_claims.append(claim)
    
    # Ground truth: hallucination exists if there are unsupported claims
    ground_truth_hallucination = len(unsupported_claims) > 0
    
    # Judge verdict matches ground truth
    return judge_hallucination == ground_truth_hallucination


def _validate_cost_classification(
    judge_cost: Optional[int],
    extracted_costs: Dict[str, Any],
    actual_costs: Dict[str, Any]
) -> bool:
    """
    Validate risk_direction verdict by comparing extracted costs vs actual costs.
    
    Determines expected cost direction:
    - -1 (opportunity cost): Model overestimated cost (extracted > actual)
    - +1 (resource cost): Model underestimated cost (extracted < actual)
    - 0 (no clear direction): Costs are similar or ambiguous
    
    Args:
        judge_cost: Judge's risk_direction verdict (-1, 0, or 1, or None)
        extracted_costs: Cost extracted from model answer
        actual_costs: Cost extracted from retrieved chunks
        
    Returns:
        bool: True if judge verdict matches expected direction, False otherwise
    """
    if judge_cost is None:
        # If judge didn't provide a verdict, consider it valid (not applicable)
        return True
    
    # Extract numeric values for comparison
    extracted_money = _extract_numeric_cost(extracted_costs.get("money"))
    actual_money = _extract_numeric_cost(actual_costs.get("money"))
    
    extracted_time = _extract_numeric_cost(extracted_costs.get("time"))
    actual_time = _extract_numeric_cost(actual_costs.get("time"))
    
    extracted_steps = _extract_numeric_cost(extracted_costs.get("steps"))
    actual_steps = _extract_numeric_cost(actual_costs.get("steps"))
    
    # Determine expected direction based on cost differences
    # Priority: money > time > steps
    expected_direction = 0  # Default to no clear direction
    
    if extracted_money is not None and actual_money is not None:
        if extracted_money > actual_money * 1.1:  # 10% threshold
            expected_direction = -1  # Overestimated (opportunity cost)
        elif extracted_money < actual_money * 0.9:  # 10% threshold
            expected_direction = 1  # Underestimated (resource cost)
    elif extracted_time is not None and actual_time is not None:
        if extracted_time > actual_time * 1.1:
            expected_direction = -1
        elif extracted_time < actual_time * 0.9:
            expected_direction = 1
    elif extracted_steps is not None and actual_steps is not None:
        if extracted_steps > actual_steps:
            expected_direction = -1
        elif extracted_steps < actual_steps:
            expected_direction = 1
    
    # Judge verdict matches expected direction
    return judge_cost == expected_direction


def _validate_impact_magnitude(
    judge_impact: Optional[int],
    extracted_costs: Dict[str, Any],
    actual_costs: Dict[str, Any]
) -> bool:
    """
    Validate risk_impact verdict by calculating expected impact from cost differences.
    
    Calculates expected impact magnitude (0-3) based on relative cost differences.
    
    Args:
        judge_impact: Judge's risk_impact verdict (0, 1, 2, or 3, or None)
        extracted_costs: Cost extracted from model answer
        actual_costs: Cost extracted from retrieved chunks
        
    Returns:
        bool: True if judge verdict is within reasonable range of expected impact, False otherwise
    """
    if judge_impact is None:
        # If judge didn't provide a verdict, consider it valid (not applicable)
        return True
    
    # Calculate cost differences
    extracted_money = _extract_numeric_cost(extracted_costs.get("money"))
    actual_money = _extract_numeric_cost(actual_costs.get("money"))
    
    extracted_time = _extract_numeric_cost(extracted_costs.get("time"))
    actual_time = _extract_numeric_cost(actual_costs.get("time"))
    
    # Calculate relative differences
    max_diff = 0.0
    
    if extracted_money is not None and actual_money is not None and actual_money > 0:
        diff = abs(extracted_money - actual_money) / actual_money
        max_diff = max(max_diff, diff)
    
    if extracted_time is not None and actual_time is not None and actual_time > 0:
        diff = abs(extracted_time - actual_time) / actual_time
        max_diff = max(max_diff, diff)
    
    # Map difference to expected impact scale
    # 0: < 5% difference
    # 1: 5-20% difference
    # 2: 20-50% difference
    # 3: > 50% difference
    if max_diff < 0.05:
        expected_impact = 0
    elif max_diff < 0.20:
        expected_impact = 1
    elif max_diff < 0.50:
        expected_impact = 2
    else:
        expected_impact = 3
    
    # Judge verdict is valid if within 1 level of expected impact (tolerance for judgment)
    return abs(judge_impact - expected_impact) <= 1


def _generate_explanation(
    validation_results: Dict[str, bool],
    judge_output: JudgeEvaluationResult
) -> str:
    """
    Generate deterministic explanation of validation results.
    
    Args:
        validation_results: Dictionary mapping validation type to result (True/False)
        judge_output: Original judge output for context
        
    Returns:
        str: Explanation of validation results
    """
    parts = []
    
    if validation_results.get("correctness", True):
        parts.append("Correctness verdict validated")
    else:
        parts.append("Correctness verdict mismatch")
    
    if validation_results.get("hallucination", True):
        parts.append("Hallucination verdict validated")
    else:
        parts.append("Hallucination verdict mismatch")
    
    if validation_results.get("risk_direction", True):
        parts.append("Risk direction validated")
    else:
        parts.append("Risk direction mismatch")
    
    if validation_results.get("risk_impact", True):
        parts.append("Risk impact validated")
    else:
        parts.append("Risk impact mismatch")
    
    overall = "All validations passed" if all(validation_results.values()) else "Some validations failed"
    
    return f"{overall}. {'; '.join(parts)}."


# Helper functions for text processing

def _normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove extra whitespace)."""
    return " ".join(text.lower().split())


def _extract_keywords(text: str) -> List[str]:
    """Extract keywords from text (remove common stop words)."""
    # Simple stop words list
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "should",
        "could", "may", "might", "can", "this", "that", "these", "those"
    }
    
    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b\w+\b', text.lower())
    # Filter out stop words and short words
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords


def _extract_claims(text: str) -> List[str]:
    """Extract key claims from text (numbers, specific terms, key phrases)."""
    claims = []
    
    # Extract numbers with context (e.g., "$50", "2 hours", "step 3")
    number_patterns = [
        r'\$\d+(?:\.\d+)?',  # Money amounts
        r'\d+\s*(?:hours?|minutes?|days?|weeks?)',  # Time amounts
        r'(?:step|step\s+)\d+',  # Step numbers
        r'\d+(?:\.\d+)?\s*(?:percent|%)',  # Percentages
    ]
    
    for pattern in number_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        claims.extend(matches)
    
    # Extract key phrases (capitalized terms, specific terminology)
    # This is a simple heuristic - could be improved
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and len(sentence) < 100:
            # Extract phrases with numbers or specific terms
            if re.search(r'\d+', sentence) or any(term in sentence.lower() for term in 
                ["copay", "deductible", "coinsurance", "out-of-pocket", "maximum", "minimum"]):
                claims.append(sentence.strip())
    
    return claims[:5]  # Limit to top 5 claims


def _is_generic_answer(text: str) -> bool:
    """Check if answer is generic (lacks specific information)."""
    generic_phrases = [
        "i don't know", "i'm not sure", "cannot determine", "unable to",
        "please consult", "contact your", "refer to", "check with"
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in generic_phrases)


def _extract_numeric_cost(cost_value: Any) -> Optional[float]:
    """Extract numeric value from cost field (handles strings, numbers, etc.)."""
    if cost_value is None:
        return None
    
    if isinstance(cost_value, (int, float)):
        return float(cost_value)
    
    if isinstance(cost_value, str):
        # Try to extract number from string (e.g., "$50" -> 50.0, "2 hours" -> 2.0)
        # Remove currency symbols and extract first number
        number_match = re.search(r'(\d+(?:\.\d+)?)', cost_value.replace(',', ''))
        if number_match:
            return float(number_match.group(1))
    
    return None


# Ground truth computation functions for metrics calculation

def _compute_ground_truth_correctness(
    model_answer: str,
    reference_answer: str
) -> bool:
    """
    Compute ground truth correctness value.
    
    Args:
        model_answer: Model-generated answer
        reference_answer: Gold reference answer
        
    Returns:
        bool: True if model answer is correct, False otherwise
    """
    # Use same logic as _validate_correctness but return ground truth
    model_normalized = _normalize_text(model_answer)
    reference_normalized = _normalize_text(reference_answer)
    
    exact_match = model_normalized == reference_normalized
    
    model_keywords = set(_extract_keywords(model_normalized))
    reference_keywords = set(_extract_keywords(reference_normalized))
    
    if not model_keywords or not reference_keywords:
        semantic_similar = exact_match
    else:
        intersection = len(model_keywords & reference_keywords)
        union = len(model_keywords | reference_keywords)
        similarity = intersection / union if union > 0 else 0.0
        semantic_similar = similarity >= 0.7
    
    return exact_match or semantic_similar


def _compute_ground_truth_hallucination(
    model_answer: str,
    retrieved_context: List[RetrievalResult]
) -> bool:
    """
    Compute ground truth hallucination value.
    
    Args:
        model_answer: Model-generated answer
        retrieved_context: List of retrieved chunks
        
    Returns:
        bool: True if hallucination exists, False otherwise
    """
    if not retrieved_context:
        return True  # No context means hallucination
    
    context_text = " ".join([chunk.chunk_text for chunk in retrieved_context])
    context_normalized = _normalize_text(context_text)
    
    model_claims = _extract_claims(model_answer)
    
    if not model_claims:
        is_grounded = len(model_answer.strip()) < 50 or _is_generic_answer(model_answer)
        return not is_grounded
    
    unsupported_claims = []
    for claim in model_claims:
        claim_normalized = _normalize_text(claim)
        if claim_normalized not in context_normalized:
            claim_keywords = _extract_keywords(claim_normalized)
            context_keywords = set(_extract_keywords(context_normalized))
            claim_keywords_set = set(claim_keywords)
            
            if claim_keywords_set:
                overlap = len(claim_keywords_set & context_keywords) / len(claim_keywords_set)
                if overlap < 0.5:
                    unsupported_claims.append(claim)
            else:
                unsupported_claims.append(claim)
    
    return len(unsupported_claims) > 0


def _compute_ground_truth_risk_direction(
    extracted_costs: Dict[str, Any],
    actual_costs: Dict[str, Any]
) -> Optional[int]:
    """
    Compute ground truth risk direction value.
    
    Args:
        extracted_costs: Cost extracted from model answer
        actual_costs: Cost extracted from retrieved chunks
        
    Returns:
        Optional[int]: -1, 0, or 1, or None if cannot determine
    """
    extracted_money = _extract_numeric_cost(extracted_costs.get("money"))
    actual_money = _extract_numeric_cost(actual_costs.get("money"))
    
    extracted_time = _extract_numeric_cost(extracted_costs.get("time"))
    actual_time = _extract_numeric_cost(actual_costs.get("time"))
    
    extracted_steps = _extract_numeric_cost(extracted_costs.get("steps"))
    actual_steps = _extract_numeric_cost(actual_costs.get("steps"))
    
    expected_direction = 0
    
    if extracted_money is not None and actual_money is not None:
        if extracted_money > actual_money * 1.1:
            expected_direction = -1
        elif extracted_money < actual_money * 0.9:
            expected_direction = 1
    elif extracted_time is not None and actual_time is not None:
        if extracted_time > actual_time * 1.1:
            expected_direction = -1
        elif extracted_time < actual_time * 0.9:
            expected_direction = 1
    elif extracted_steps is not None and actual_steps is not None:
        if extracted_steps > actual_steps:
            expected_direction = -1
        elif extracted_steps < actual_steps:
            expected_direction = 1
    else:
        return None
    
    return expected_direction


def _compute_ground_truth_risk_impact(
    extracted_costs: Dict[str, Any],
    actual_costs: Dict[str, Any]
) -> Optional[int]:
    """
    Compute ground truth risk impact value.
    
    Args:
        extracted_costs: Cost extracted from model answer
        actual_costs: Cost extracted from retrieved chunks
        
    Returns:
        Optional[int]: 0, 1, 2, or 3, or None if cannot determine
    """
    extracted_money = _extract_numeric_cost(extracted_costs.get("money"))
    actual_money = _extract_numeric_cost(actual_costs.get("money"))
    
    extracted_time = _extract_numeric_cost(extracted_costs.get("time"))
    actual_time = _extract_numeric_cost(actual_costs.get("time"))
    
    max_diff = 0.0
    
    if extracted_money is not None and actual_money is not None and actual_money > 0:
        diff = abs(extracted_money - actual_money) / actual_money
        max_diff = max(max_diff, diff)
    
    if extracted_time is not None and actual_time is not None and actual_time > 0:
        diff = abs(extracted_time - actual_time) / actual_time
        max_diff = max(max_diff, diff)
    
    if max_diff < 0.05:
        return 0
    elif max_diff < 0.20:
        return 1
    elif max_diff < 0.50:
        return 2
    else:
        return 3


# Metrics calculation functions

def calculate_judge_metrics(
    evaluation_results: List[Tuple[JudgeEvaluationResult, MetaEvaluationResult]]
) -> JudgePerformanceMetrics:
    """
    Calculate recall, precision, and F1 scores for LLM-as-Judge across all output metrics.
    
    Computes performance metrics by comparing judge verdicts against ground truth values
    from meta-evaluation results.
    
    Args:
        evaluation_results: List of tuples containing (judge_output, meta_eval_result) pairs
        
    Returns:
        JudgePerformanceMetrics object containing metrics for all judge output types
        
    Raises:
        ValueError: If evaluation_results is empty
    """
    if not evaluation_results:
        raise ValueError("evaluation_results cannot be empty")
    
    # Extract pairs for each metric
    correctness_pairs = []
    hallucination_pairs = []
    risk_direction_pairs = []
    risk_impact_pairs = []
    
    for judge_output, meta_eval in evaluation_results:
        # Correctness
        if meta_eval.ground_truth_correctness is not None:
            correctness_pairs.append((
                judge_output.correctness_binary,
                meta_eval.ground_truth_correctness
            ))
        
        # Hallucination
        if meta_eval.ground_truth_hallucination is not None:
            hallucination_pairs.append((
                judge_output.hallucination_binary,
                meta_eval.ground_truth_hallucination
            ))
        
        # Risk direction (only if available)
        if (judge_output.risk_direction is not None and 
            meta_eval.ground_truth_risk_direction is not None):
            risk_direction_pairs.append((
                judge_output.risk_direction,
                meta_eval.ground_truth_risk_direction
            ))
        
        # Risk impact (only if available)
        if (judge_output.risk_impact is not None and 
            meta_eval.ground_truth_risk_impact is not None):
            risk_impact_pairs.append((
                judge_output.risk_impact,
                meta_eval.ground_truth_risk_impact
            ))
    
    # Calculate metrics for each type
    correctness_metrics = _calculate_binary_metrics(correctness_pairs)
    hallucination_metrics = _calculate_binary_metrics(hallucination_pairs)
    
    risk_direction_metrics = None
    if risk_direction_pairs:
        risk_direction_metrics = _calculate_multiclass_metrics(risk_direction_pairs, classes=[-1, 0, 1])
    
    risk_impact_metrics = None
    if risk_impact_pairs:
        risk_impact_metrics = _calculate_multiclass_metrics(risk_impact_pairs, classes=[0, 1, 2, 3])
    
    return JudgePerformanceMetrics(
        correctness=correctness_metrics,
        hallucination=hallucination_metrics,
        risk_direction=risk_direction_metrics,
        risk_impact=risk_impact_metrics
    )


def _calculate_binary_metrics(
    pairs: List[Tuple[bool, bool]]
) -> JudgeMetricScores:
    """
    Calculate precision, recall, and F1 for binary classification.
    
    Args:
        pairs: List of (predicted, actual) boolean pairs
        
    Returns:
        JudgeMetricScores object with calculated metrics
    """
    if not pairs:
        return JudgeMetricScores(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            true_positives=0,
            true_negatives=0,
            false_positives=0,
            false_negatives=0,
            total_samples=0
        )
    
    tp = sum(1 for pred, actual in pairs if pred and actual)
    tn = sum(1 for pred, actual in pairs if not pred and not actual)
    fp = sum(1 for pred, actual in pairs if pred and not actual)
    fn = sum(1 for pred, actual in pairs if not pred and actual)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return JudgeMetricScores(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        total_samples=len(pairs)
    )


def _calculate_multiclass_metrics(
    pairs: List[Tuple[Any, Any]],
    classes: List[Any]
) -> JudgeMetricScores:
    """
    Calculate precision, recall, and F1 for multiclass classification.
    
    Uses macro-averaging across all classes.
    
    Args:
        pairs: List of (predicted, actual) pairs
        classes: List of possible class values
        
    Returns:
        JudgeMetricScores object with macro-averaged metrics
    """
    if not pairs:
        return JudgeMetricScores(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            true_positives=0,
            true_negatives=0,
            false_positives=0,
            false_negatives=0,
            total_samples=0
        )
    
    # Calculate metrics for each class
    class_metrics = []
    
    for class_val in classes:
        # Treat current class as positive, others as negative
        tp = sum(1 for pred, actual in pairs if pred == class_val and actual == class_val)
        tn = sum(1 for pred, actual in pairs if pred != class_val and actual != class_val)
        fp = sum(1 for pred, actual in pairs if pred == class_val and actual != class_val)
        fn = sum(1 for pred, actual in pairs if pred != class_val and actual == class_val)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics.append((precision, recall, f1, tp, tn, fp, fn))
    
    # Macro-average across classes
    avg_precision = sum(m[0] for m in class_metrics) / len(class_metrics) if class_metrics else 0.0
    avg_recall = sum(m[1] for m in class_metrics) / len(class_metrics) if class_metrics else 0.0
    avg_f1 = sum(m[2] for m in class_metrics) / len(class_metrics) if class_metrics else 0.0
    
    # Sum confusion matrix elements
    total_tp = sum(m[3] for m in class_metrics)
    total_tn = sum(m[4] for m in class_metrics)
    total_fp = sum(m[5] for m in class_metrics)
    total_fn = sum(m[6] for m in class_metrics)
    
    return JudgeMetricScores(
        precision=avg_precision,
        recall=avg_recall,
        f1_score=avg_f1,
        true_positives=total_tp,
        true_negatives=total_tn,
        false_positives=total_fp,
        false_negatives=total_fn,
        total_samples=len(pairs)
    )


# Bias correction functions for LLM-as-a-Judge evaluations
# Simplified version based on the original implementation

def clip(x: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clip value to [low, high] range."""
    return max(low, min(high, x))


def compute_calibration_parameters(
    evaluation_results: List[Tuple[JudgeEvaluationResult, MetaEvaluationResult]]
) -> Tuple[float, float, int, int]:
    """
    Compute judge calibration parameters (q0, q1) from meta-evaluation results.
    
    q0 (specificity): Probability judge correctly identifies incorrect answers
    q1 (sensitivity): Probability judge correctly identifies correct answers
    
    Args:
        evaluation_results: List of (judge_output, meta_eval_result) pairs
        
    Returns:
        Tuple of (q0, q1, m0, m1)
    """
    if not evaluation_results:
        raise ValueError("evaluation_results cannot be empty")
    
    # Count true negatives and false positives (for q0)
    tn_count = 0  # True negatives: judge says False, ground truth is False
    fp_count = 0  # False positives: judge says True, ground truth is False
    
    # Count true positives and false negatives (for q1)
    tp_count = 0  # True positives: judge says True, ground truth is True
    fn_count = 0  # False negatives: judge says False, ground truth is True
    
    for judge_output, meta_eval in evaluation_results:
        if meta_eval.ground_truth_correctness is None:
            continue
        
        judge_verdict = judge_output.correctness_binary
        ground_truth = meta_eval.ground_truth_correctness
        
        if not ground_truth:  # Ground truth is False (incorrect answer)
            if not judge_verdict:  # Judge correctly says False
                tn_count += 1
            else:  # Judge incorrectly says True
                fp_count += 1
        else:  # Ground truth is True (correct answer)
            if judge_verdict:  # Judge correctly says True
                tp_count += 1
            else:  # Judge incorrectly says False
                fn_count += 1
    
    m0 = tn_count + fp_count  # Total examples where ground truth is False
    m1 = tp_count + fn_count  # Total examples where ground truth is True
    
    # Estimate q0 (specificity): P(judge says False | ground truth is False)
    # Use Laplace smoothing to avoid division by zero
    q0 = (tn_count + 1) / (m0 + 2) if m0 > 0 else 0.5
    
    # Estimate q1 (sensitivity): P(judge says True | ground truth is True)
    # Use Laplace smoothing to avoid division by zero
    q1 = (tp_count + 1) / (m1 + 2) if m1 > 0 else 0.5
    
    return (q0, q1, m0, m1)


def compute_bias_corrected_accuracy(
    raw_accuracy: float,
    q0: float,
    q1: float
) -> float:
    """
    Compute bias-corrected accuracy estimate using Rogan-Gladen adjustment.
    
    Formula: θ = (p + q0 - 1) / (q0 + q1 - 1)
    where p is the raw accuracy estimate.
    
    Args:
        raw_accuracy: Raw proportion of "correct" judgments (p)
        q0: Judge specificity
        q1: Judge sensitivity
        
    Returns:
        Bias-corrected accuracy estimate, clipped to [0, 1]
    """
    if q0 + q1 - 1 == 0:
        # Degenerate case: judge is completely unreliable
        return clip(raw_accuracy)
    
    # Rogan-Gladen adjustment
    corrected = (raw_accuracy + q0 - 1) / (q0 + q1 - 1)
    return clip(corrected)


def compute_bias_corrected_accuracy_from_results(
    evaluation_results: List[Tuple[JudgeEvaluationResult, MetaEvaluationResult]],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compute bias-corrected accuracy estimate from evaluation results.
    
    Args:
        evaluation_results: List of (judge_output, meta_eval_result) pairs
        alpha: Significance level for confidence interval (default 0.05 for 95% CI)
        
    Returns:
        Dictionary containing bias-corrected accuracy and related statistics
    """
    if not evaluation_results:
        raise ValueError("evaluation_results cannot be empty")
    
    # Compute raw accuracy
    n = len(evaluation_results)
    correct_count = sum(1 for judge_output, _ in evaluation_results 
                       if judge_output.correctness_binary)
    raw_accuracy = correct_count / n if n > 0 else 0.0
    
    # Compute calibration parameters
    q0, q1, m0, m1 = compute_calibration_parameters(evaluation_results)
    
    # Compute bias-corrected accuracy
    corrected_accuracy = compute_bias_corrected_accuracy(raw_accuracy, q0, q1)
    
    # Simple confidence interval (without scipy)
    if HAS_SCIPY:
        z = norm.ppf(1 - alpha / 2)
    else:
        z = norm_ppf(1 - alpha / 2)
    
    # Simplified confidence interval calculation
    se = sqrt(raw_accuracy * (1 - raw_accuracy) / n) if n > 0 else 0.0
    confidence_interval = (
        clip(corrected_accuracy - z * se),
        clip(corrected_accuracy + z * se)
    )
    
    return {
        "raw_accuracy": raw_accuracy,
        "corrected_accuracy": corrected_accuracy,
        "confidence_interval": confidence_interval,
        "q0": q0,
        "q1": q1,
        "n": n,
        "m0": m0,
        "m1": m1,
        "confidence_level": 1 - alpha
    }


class MetaEvaluator:
    """Thin wrapper for eval.py: validates judge output using meta_evaluate_judge."""

    def validate_judge_output(
        self,
        judge_output: JudgeEvaluationResult,
        question: str,
        reference_answer: str,
        generated_answer: str,
        retrieved_context: Optional[List[RetrievalResult]] = None,
        extracted_costs: Optional[Dict[str, Any]] = None,
        actual_costs: Optional[Dict[str, Any]] = None,
    ) -> MetaEvaluationResult:
        """
        Validate judge verdicts against ground truth. If retrieved_context is not provided,
        only correctness can be validated; hallucination/risk checks use empty context.
        """
        ctx = retrieved_context if retrieved_context is not None else []
        return meta_evaluate_judge(
            judge_output=judge_output,
            retrieved_context=ctx,
            model_answer=generated_answer,
            reference_answer=reference_answer,
            extracted_costs=extracted_costs,
            actual_costs=actual_costs,
        )