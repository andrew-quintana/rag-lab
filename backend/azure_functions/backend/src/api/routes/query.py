"""POST /query endpoint"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.core.interfaces import Query, ModelAnswer
from src.services.rag.pipeline import run_rag
from src.core.config import Config
from src.core.logging import get_logger

logger = get_logger("api.routes.query")
router = APIRouter()

config = Config.from_env()


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    text: str
    prompt_version: str = "v1"


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    query_id: str
    prompt_version: str


@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Process a query through the RAG pipeline.
    
    Args:
        request: Query request with text and prompt version
        
    Returns:
        QueryResponse with answer and metadata
    """
    logger.info(f"Received query: {request.text}")
    
    try:
        query = Query(text=request.text)
        answer = run_rag(query, prompt_version=request.prompt_version, config=config)
        
        return QueryResponse(
            answer=answer.text,
            query_id=answer.query_id,
            prompt_version=answer.prompt_version
        )
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="RAG pipeline not yet implemented")
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

