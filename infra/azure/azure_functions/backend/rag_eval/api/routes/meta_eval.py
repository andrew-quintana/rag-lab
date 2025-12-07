"""POST /meta_eval endpoint

NOTE: This endpoint is not currently implemented. The meta-evaluation functionality
for validating judge verdicts is available via rag_eval.services.evaluator.meta_eval.meta_evaluate_judge().
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from rag_eval.core.logging import get_logger

logger = get_logger("api.routes.meta_eval")
router = APIRouter()


class MetaEvalRequest(BaseModel):
    """Request model for meta-eval endpoint"""
    version_1: str
    version_2: str
    query_ids: list[str]  # List of query IDs to compare


class MetaEvalResponse(BaseModel):
    """Response model for meta-eval endpoint"""
    summary: Dict[str, Any]
    deltas: Dict[str, float]
    judge_consistency: float


@router.post("/meta_eval", response_model=MetaEvalResponse)
async def handle_meta_eval(request: MetaEvalRequest):
    """
    Trigger version-to-version comparison and meta-evaluation.
    
    NOTE: This endpoint is not currently implemented.
    For meta-evaluation of judge verdicts, use rag_eval.services.evaluator.meta_eval.meta_evaluate_judge()
    
    Args:
        request: Meta-eval request with versions and query IDs
        
    Returns:
        MetaEvalResponse with comparison results
    """
    logger.info(f"Meta-evaluation endpoint called: {request.version_1} vs {request.version_2}")
    raise HTTPException(
        status_code=501, 
        detail="Version comparison meta-evaluation endpoint not yet implemented. Use rag_eval.services.evaluator.meta_eval.meta_evaluate_judge() for judge verdict validation."
    )

