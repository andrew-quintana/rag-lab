"""RAG pipeline logging to Supabase"""

from typing import List
from src.core.interfaces import Query, ModelAnswer, RetrievalResult
from src.db.queries import QueryExecutor
from src.core.logging import get_logger

logger = get_logger("services.rag.logging")


def log_query(query: Query, query_executor: QueryExecutor) -> str:
    """
    Log a query to the database.
    
    Args:
        query: Query object
        query_executor: Database query executor
        
    Returns:
        Query ID
    """
    logger.info(f"Logging query: {query.text}")
    # TODO: Implement query logging
    raise NotImplementedError("Query logging not yet implemented")


def log_retrieval(
    query_id: str,
    retrieval_results: List[RetrievalResult],
    query_executor: QueryExecutor
) -> None:
    """
    Log retrieval results to the database.
    
    Args:
        query_id: ID of the query
        retrieval_results: List of retrieval results
        query_executor: Database query executor
    """
    logger.info(f"Logging {len(retrieval_results)} retrieval results for query: {query_id}")
    # TODO: Implement retrieval logging
    raise NotImplementedError("Retrieval logging not yet implemented")


def log_model_answer(answer: ModelAnswer, query_executor: QueryExecutor) -> str:
    """
    Log a model answer to the database.
    
    Args:
        answer: ModelAnswer object
        query_executor: Database query executor
        
    Returns:
        Answer ID
    """
    logger.info(f"Logging model answer for query: {answer.query_id}")
    # TODO: Implement model answer logging
    raise NotImplementedError("Model answer logging not yet implemented")

