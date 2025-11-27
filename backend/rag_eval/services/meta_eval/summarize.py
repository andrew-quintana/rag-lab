"""Meta-evaluation summary generation"""

from typing import Dict, Any, List
from rag_eval.core.interfaces import EvaluationScore
from rag_eval.core.logging import get_logger

logger = get_logger("services.meta_eval.summarize")


def generate_summary(
    version_1: str,
    version_2: str,
    deltas: Dict[str, float],
    judge_consistency: float
) -> Dict[str, Any]:
    """
    Generate a summary of meta-evaluation results.
    
    Args:
        version_1: Name of first prompt version
        version_2: Name of second prompt version
        deltas: Delta metrics from comparison
        judge_consistency: Judge consistency score
        
    Returns:
        Summary dictionary
    """
    logger.info(f"Generating meta-eval summary for {version_1} vs {version_2}")
    
    return {
        "version_1": version_1,
        "version_2": version_2,
        "deltas": deltas,
        "judge_consistency": judge_consistency,
        "summary": f"Comparison between {version_1} and {version_2} shows "
                   f"grounding delta: {deltas['delta_grounding']:.3f}, "
                   f"relevance delta: {deltas['delta_relevance']:.3f}, "
                   f"hallucination risk delta: {deltas['delta_hallucination_risk']:.3f}"
    }

