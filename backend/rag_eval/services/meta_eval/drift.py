"""Drift detection and judge stability analysis"""

from typing import List
from rag_eval.core.interfaces import EvaluationScore
from rag_eval.core.logging import get_logger
import statistics

logger = get_logger("services.meta_eval.drift")


def compute_judge_consistency(scores: List[EvaluationScore]) -> float:
    """
    Compute judge consistency (variance) across multiple evaluations.
    
    Lower variance indicates more consistent judging.
    
    Args:
        scores: List of evaluation scores
        
    Returns:
        Consistency score (inverse of variance, normalized to 0-1)
    """
    if len(scores) < 2:
        return 1.0
    
    logger.info(f"Computing judge consistency for {len(scores)} scores")
    
    # Compute variance across all metrics
    groundings = [s.grounding for s in scores]
    relevances = [s.relevance for s in scores]
    hallucination_risks = [s.hallucination_risk for s in scores]
    
    variances = [
        statistics.variance(groundings) if len(groundings) > 1 else 0.0,
        statistics.variance(relevances) if len(relevances) > 1 else 0.0,
        statistics.variance(hallucination_risks) if len(hallucination_risks) > 1 else 0.0,
    ]
    
    avg_variance = sum(variances) / len(variances)
    # Convert variance to consistency (inverse, normalized)
    consistency = 1.0 / (1.0 + avg_variance)
    
    return consistency


def detect_drift(
    baseline_scores: List[EvaluationScore],
    current_scores: List[EvaluationScore],
    threshold: float = 0.1
) -> bool:
    """
    Detect if there is significant drift between baseline and current scores.
    
    Args:
        baseline_scores: Baseline evaluation scores
        current_scores: Current evaluation scores
        threshold: Threshold for considering drift significant
        
    Returns:
        True if drift is detected
    """
    if len(baseline_scores) != len(current_scores):
        raise ValueError("Score lists must have the same length")
    
    logger.info(f"Detecting drift between baseline and current scores (threshold: {threshold})")
    
    # Compute average deltas
    avg_grounding_delta = abs(
        sum(s2.grounding - s1.grounding for s1, s2 in zip(baseline_scores, current_scores)) / len(baseline_scores)
    )
    avg_relevance_delta = abs(
        sum(s2.relevance - s1.relevance for s1, s2 in zip(baseline_scores, current_scores)) / len(baseline_scores)
    )
    avg_hallucination_delta = abs(
        sum(s2.hallucination_risk - s1.hallucination_risk for s1, s2 in zip(baseline_scores, current_scores)) / len(baseline_scores)
    )
    
    max_delta = max(avg_grounding_delta, avg_relevance_delta, avg_hallucination_delta)
    
    drift_detected = max_delta > threshold
    logger.info(f"Drift detection result: {drift_detected} (max delta: {max_delta:.3f})")
    
    return drift_detected

