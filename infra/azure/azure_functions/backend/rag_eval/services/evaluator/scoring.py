"""Scoring utilities for evaluation"""

from typing import Dict, Any
from rag_eval.core.interfaces import EvaluationScore
from rag_eval.core.logging import get_logger

logger = get_logger("services.evaluator.scoring")


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize a score to the range [min_val, max_val].
    
    Args:
        score: Raw score
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Normalized score
    """
    return max(min_val, min(max_val, score))


def aggregate_scores(scores: list[EvaluationScore]) -> Dict[str, float]:
    """
    Aggregate multiple evaluation scores.
    
    Args:
        scores: List of evaluation scores
        
    Returns:
        Dictionary with aggregated metrics
    """
    if not scores:
        return {
            "avg_grounding": 0.0,
            "avg_relevance": 0.0,
            "avg_hallucination_risk": 0.0,
        }
    
    return {
        "avg_grounding": sum(s.grounding for s in scores) / len(scores),
        "avg_relevance": sum(s.relevance for s in scores) / len(scores),
        "avg_hallucination_risk": sum(s.hallucination_risk for s in scores) / len(scores),
    }

