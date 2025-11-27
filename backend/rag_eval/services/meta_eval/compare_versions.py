"""Version-to-version performance comparison"""

from typing import List, Dict, Any
from rag_eval.core.interfaces import EvaluationScore
from rag_eval.core.logging import get_logger

logger = get_logger("services.meta_eval.compare_versions")


def compare_versions(
    scores_v1: List[EvaluationScore],
    scores_v2: List[EvaluationScore]
) -> Dict[str, float]:
    """
    Compare evaluation scores between two prompt versions.
    
    Args:
        scores_v1: Evaluation scores for version 1
        scores_v2: Evaluation scores for version 2
        
    Returns:
        Dictionary with delta metrics
    """
    logger.info(f"Comparing {len(scores_v1)} v1 scores with {len(scores_v2)} v2 scores")
    
    if len(scores_v1) != len(scores_v2):
        raise ValueError("Score lists must have the same length")
    
    deltas = {
        "delta_grounding": sum(s2.grounding - s1.grounding for s1, s2 in zip(scores_v1, scores_v2)) / len(scores_v1),
        "delta_relevance": sum(s2.relevance - s1.relevance for s1, s2 in zip(scores_v1, scores_v2)) / len(scores_v1),
        "delta_hallucination_risk": sum(s2.hallucination_risk - s1.hallucination_risk for s1, s2 in zip(scores_v1, scores_v2)) / len(scores_v1),
    }
    
    return deltas

