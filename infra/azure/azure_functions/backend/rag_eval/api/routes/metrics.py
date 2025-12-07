"""GET /metrics endpoint"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from rag_eval.db.connection import DatabaseConnection
from rag_eval.db.queries import QueryExecutor
from rag_eval.core.config import Config
from rag_eval.core.logging import get_logger

logger = get_logger("api.routes.metrics")
router = APIRouter()

config = Config.from_env()
db_conn = DatabaseConnection(config)
query_executor = QueryExecutor(db_conn)


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get aggregated metrics from Supabase Postgres.
    
    Returns:
        Dictionary with metrics for dashboard visualization
    """
    logger.info("Fetching metrics")
    
    try:
        # TODO: Implement metrics aggregation from database
        # This should query:
        # - Model quality by prompt version
        # - Retrieval performance
        # - Prompt stability trends
        
        return {
            "model_quality_by_version": {},
            "retrieval_performance": {},
            "prompt_stability": {},
            "message": "Metrics endpoint - implementation pending"
        }
    except Exception as e:
        logger.error(f"Metrics fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

