"""Supabase logging for RAG pipeline operations"""

import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from rag_eval.core.interfaces import Query, ModelAnswer, RetrievalResult
from rag_eval.db.queries import QueryExecutor
from rag_eval.core.logging import get_logger
from rag_eval.core.exceptions import DatabaseError
from rag_eval.utils.ids import generate_id

logger = get_logger("services.rag.logging")


def log_query(query: Query, query_executor: QueryExecutor) -> str:
    """
    Log a query to the Supabase `queries` table.
    
    This function inserts a query record into the database, generating a query ID
    if one is not already present in the Query object. Logging failures are handled
    gracefully and do not raise exceptions, ensuring the pipeline continues even
    if logging fails.
    
    **Database Schema:**
    - `queries` table: query_id (PK), query_text, timestamp, metadata (JSONB)
    
    **Error Handling:**
    - Logging failures are caught and logged as warnings
    - Returns the query_id even if logging fails (ensures pipeline continuity)
    - Does not raise exceptions to avoid breaking the pipeline
    
    Args:
        query: Query object with text and optional query_id, timestamp, metadata
        query_executor: Database query executor for executing SQL queries
        
    Returns:
        str: Query ID (generated if missing, or from query object)
        
    Example:
        >>> from rag_eval.core.interfaces import Query
        >>> from rag_eval.db.queries import QueryExecutor
        >>> 
        >>> query = Query(text="What is the coverage limit?")
        >>> query_id = log_query(query, query_executor)
        >>> print(query_id)
        "query_abc123..."
    """
    # Generate query ID if missing
    query_id = query.query_id
    if not query_id:
        query_id = generate_id("query")
        logger.debug(f"Generated query_id: {query_id} for query logging")
    
    # Use current timestamp if not provided
    query_timestamp = query.timestamp
    if not query_timestamp:
        query_timestamp = datetime.now(timezone.utc)
    
    try:
        # Prepare metadata as JSON (empty dict if None)
        metadata = query.__dict__.get('metadata', {}) if hasattr(query, 'metadata') else {}
        if metadata is None:
            metadata = {}
        
        # Convert metadata dict to JSON string for JSONB column
        metadata_json = json.dumps(metadata) if metadata else '{}'
        
        # Insert query into database
        insert_query = """
            INSERT INTO queries (query_id, query_text, timestamp, metadata)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (query_id) DO NOTHING
        """
        params = (query_id, query.text, query_timestamp, metadata_json)
        
        query_executor.execute_insert(insert_query, params)
        logger.info(f"Successfully logged query '{query_id}' to database")
        
    except DatabaseError as e:
        # Log error but don't raise - logging failures shouldn't break pipeline
        logger.warning(
            f"Failed to log query '{query_id}' to database (non-fatal): {e}",
            exc_info=True
        )
    except Exception as e:
        # Catch any unexpected errors
        logger.warning(
            f"Unexpected error logging query '{query_id}' (non-fatal): {e}",
            exc_info=True
        )
    
    # Always return query_id, even if logging failed
    return query_id


def log_retrieval(
    query_id: str,
    retrieval_results: List[RetrievalResult],
    query_executor: QueryExecutor
) -> None:
    """
    Log retrieval results to the Supabase `retrieval_logs` table using batch insertion.
    
    This function performs a batch insert of all retrieval results for efficiency.
    If the retrieval_results list is empty, the function returns early without
    attempting database operations. Logging failures are handled gracefully
    and do not raise exceptions.
    
    **Database Schema:**
    - `retrieval_logs` table: log_id (PK), query_id (FK), chunk_id, similarity_score, timestamp
    
    **Batch Insertion Strategy:**
    - Uses a single SQL INSERT statement with multiple VALUES clauses
    - Generates log_id for each retrieval result
    - All logs are inserted in a single transaction for atomicity
    
    **Error Handling:**
    - Logging failures are caught and logged as warnings
    - Does not raise exceptions to avoid breaking the pipeline
    - Empty retrieval results are handled gracefully (early return)
    
    Args:
        query_id: ID of the query that triggered the retrieval
        retrieval_results: List of RetrievalResult objects with chunk_id and similarity_score
        query_executor: Database query executor for executing SQL queries
        
    Example:
        >>> from rag_eval.core.interfaces import RetrievalResult
        >>> 
        >>> results = [
        ...     RetrievalResult(chunk_id="chunk_1", similarity_score=0.92, chunk_text="..."),
        ...     RetrievalResult(chunk_id="chunk_2", similarity_score=0.85, chunk_text="...")
        ... ]
        >>> log_retrieval("query_123", results, query_executor)
    """
    # Handle empty retrieval results
    if not retrieval_results:
        logger.debug(f"No retrieval results to log for query '{query_id}'")
        return
    
    try:
        # Prepare batch insert data
        timestamp = datetime.now(timezone.utc)
        values_placeholders = []
        params = []
        
        for result in retrieval_results:
            log_id = generate_id("log")
            values_placeholders.append("(%s, %s, %s, %s, %s)")
            params.extend([log_id, query_id, result.chunk_id, result.similarity_score, timestamp])
        
        # Construct batch insert query using psycopg2 parameter syntax
        insert_query = f"""
            INSERT INTO retrieval_logs (log_id, query_id, chunk_id, similarity_score, timestamp)
            VALUES {', '.join(values_placeholders)}
        """
        
        query_executor.execute_insert(insert_query, tuple(params))
        logger.info(
            f"Successfully logged {len(retrieval_results)} retrieval results "
            f"for query '{query_id}'"
        )
        
    except DatabaseError as e:
        # Log error but don't raise - logging failures shouldn't break pipeline
        logger.warning(
            f"Failed to log retrieval results for query '{query_id}' (non-fatal): {e}",
            exc_info=True
        )
    except Exception as e:
        # Catch any unexpected errors
        logger.warning(
            f"Unexpected error logging retrieval results for query '{query_id}' (non-fatal): {e}",
            exc_info=True
        )


def log_model_answer(answer: ModelAnswer, query_executor: QueryExecutor) -> str:
    """
    Log a model answer to the Supabase `model_answers` table.
    
    This function inserts a model answer record into the database, generating an
    answer ID if one is not already present. The retrieved_chunk_ids are stored
    as a PostgreSQL array. Logging failures are handled gracefully and do not
    raise exceptions.
    
    **Database Schema:**
    - `model_answers` table: answer_id (PK), query_id (FK), answer_text, 
      prompt_version (FK), retrieved_chunk_ids (TEXT[]), timestamp
    
    **Error Handling:**
    - Logging failures are caught and logged as warnings
    - Returns the answer_id even if logging fails (ensures pipeline continuity)
    - Does not raise exceptions to avoid breaking the pipeline
    
    Args:
        answer: ModelAnswer object with text, query_id, prompt_version, retrieved_chunk_ids
        query_executor: Database query executor for executing SQL queries
        
    Returns:
        str: Answer ID (generated if missing, or from answer object)
        
    Example:
        >>> from rag_eval.core.interfaces import ModelAnswer
        >>> 
        >>> answer = ModelAnswer(
        ...     text="The coverage limit is $500,000...",
        ...     query_id="query_123",
        ...     prompt_version="v1",
        ...     retrieved_chunk_ids=["chunk_1", "chunk_2"]
        ... )
        >>> answer_id = log_model_answer(answer, query_executor)
        >>> print(answer_id)
        "answer_abc123..."
    """
    # Generate answer ID if missing
    answer_id = getattr(answer, 'answer_id', None)
    if not answer_id:
        answer_id = generate_id("answer")
        logger.debug(f"Generated answer_id: {answer_id} for model answer logging")
    
    # Use current timestamp if not provided
    answer_timestamp = answer.timestamp
    if not answer_timestamp:
        answer_timestamp = datetime.now(timezone.utc)
    
    try:
        # Ensure retrieved_chunk_ids is a list (handle None case)
        retrieved_chunk_ids = answer.retrieved_chunk_ids or []
        
        # Insert model answer into database
        # PostgreSQL array syntax: ARRAY['value1', 'value2']
        insert_query = """
            INSERT INTO model_answers (
                answer_id, query_id, answer_text, prompt_version, 
                retrieved_chunk_ids, timestamp
            )
            VALUES (%s, %s, %s, %s, %s::text[], %s)
            ON CONFLICT (answer_id) DO NOTHING
        """
        params = (
            answer_id,
            answer.query_id,
            answer.text,
            answer.prompt_version,
            retrieved_chunk_ids,  # PostgreSQL will convert list to array
            answer_timestamp
        )
        
        query_executor.execute_insert(insert_query, params)
        logger.info(
            f"Successfully logged model answer '{answer_id}' for query '{answer.query_id}' "
            f"(prompt_version: {answer.prompt_version})"
        )
        
    except DatabaseError as e:
        # Log error but don't raise - logging failures shouldn't break pipeline
        logger.warning(
            f"Failed to log model answer '{answer_id}' to database (non-fatal): {e}",
            exc_info=True
        )
    except Exception as e:
        # Catch any unexpected errors
        logger.warning(
            f"Unexpected error logging model answer '{answer_id}' (non-fatal): {e}",
            exc_info=True
        )
    
    # Always return answer_id, even if logging failed
    return answer_id

