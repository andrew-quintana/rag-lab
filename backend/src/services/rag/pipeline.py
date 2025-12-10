"""Main RAG pipeline orchestration"""

import time
from typing import Optional
from datetime import datetime, timezone
from src.core.interfaces import Query, ModelAnswer
from src.core.config import Config
from src.core.logging import get_logger
from src.core.exceptions import AzureServiceError, DatabaseError, ValidationError
from src.utils.ids import generate_id
from src.utils.timing import timer
from src.services.rag.embeddings import generate_query_embedding
from src.services.rag.search import retrieve_chunks
from src.services.rag.generation import generate_answer
from src.services.rag.logging import log_query, log_retrieval, log_model_answer
from src.db.connection import DatabaseConnection
from src.db.queries import QueryExecutor

logger = get_logger("services.rag.pipeline")


def run_rag(query: Query, prompt_version: str = "v1", config: Optional[Config] = None) -> ModelAnswer:
    """
    Run the complete RAG pipeline: query embedding → retrieval → generation → logging.
    
    This function orchestrates all RAG pipeline components in a deterministic order:
    1. Generate query ID and timestamp if missing
    2. Generate query embedding (Phase 3)
    3. Retrieve top-k chunks from Azure AI Search (Phase 4)
    4. Load prompt template and construct prompt (Phase 5)
    5. Generate answer using LLM (Phase 6)
    6. Assemble ModelAnswer with metadata
    7. Log to Supabase (Phase 8 - stubbed initially)
    
    **Latency Measurement**: The function measures and logs total pipeline latency,
    as well as individual step latencies for observability.
    
    **Error Handling**: Errors from individual steps are propagated with context,
    allowing callers to identify which step failed. All errors are logged before
    being raised.
    
    **Deterministic Execution**: The pipeline executes steps in a fixed order,
    ensuring reproducible results for testing and validation.
    
    Args:
        query: Query object with text and optional metadata (query_id, timestamp)
        prompt_version: Version of the prompt template to use (default: "v1")
        config: Application configuration (optional, defaults to Config.from_env())
        
    Returns:
        ModelAnswer object containing:
        - text: Generated answer
        - query_id: Unique query identifier (generated if missing)
        - prompt_version: Prompt version used
        - retrieved_chunk_ids: List of chunk IDs used in generation
        - timestamp: Generation timestamp
        
    Raises:
        AzureServiceError: If Azure service calls fail (embedding, search, generation)
        ValidationError: If query is invalid or prompt version not found
        ValueError: If query text is empty or config is missing required fields
        DatabaseError: If database operations fail (prompt loading, logging)
        
    Example:
        >>> from src.core.config import Config
        >>> from src.core.interfaces import Query
        >>> 
        >>> config = Config.from_env()
        >>> query = Query(text="What is the coverage limit?")
        >>> answer = run_rag(query, prompt_version="v1", config=config)
        >>> print(answer.text)
        "The coverage limit is $500,000..."
    """
    # Start total pipeline timer
    pipeline_start_time = time.time()
    
    # Use default config if not provided
    if config is None:
        config = Config.from_env()
    
    # Validate query
    if not query.text or not query.text.strip():
        error_msg = "Query text cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Generate query ID and timestamp if missing
    query_id = query.query_id
    if not query_id:
        query_id = generate_id("query")
        logger.debug(f"Generated query_id: {query_id} for query: {query.text[:50]}...")
    
    query_timestamp = query.timestamp
    if not query_timestamp:
        query_timestamp = datetime.now(timezone.utc)
        logger.debug(f"Generated timestamp for query_id: {query_id}")
    
    logger.info(
        f"Starting RAG pipeline for query_id '{query_id}': {query.text[:100]}... "
        f"(prompt_version: {prompt_version})"
    )
    
    try:
        # Step 1: Generate query embedding
        logger.info(f"[Step 1/5] Generating query embedding for query_id '{query_id}'")
        embedding_start_time = time.time()
        try:
            query_embedding = generate_query_embedding(query, config)
            embedding_latency = time.time() - embedding_start_time
            logger.info(
                f"[Step 1/5] Query embedding generated in {embedding_latency:.2f}s "
                f"(dimension: {len(query_embedding)})"
            )
        except Exception as e:
            logger.error(f"[Step 1/5] Query embedding generation failed: {e}", exc_info=True)
            raise AzureServiceError(f"Query embedding generation failed: {str(e)}") from e
        
        # Step 2: Retrieve top-k chunks
        logger.info(f"[Step 2/5] Retrieving top-k chunks for query_id '{query_id}'")
        retrieval_start_time = time.time()
        try:
            retrieval_results = retrieve_chunks(query, top_k=5, config=config)
            retrieval_latency = time.time() - retrieval_start_time
            logger.info(
                f"[Step 2/5] Retrieved {len(retrieval_results)} chunks in {retrieval_latency:.2f}s "
                f"for query_id '{query_id}'"
            )
        except Exception as e:
            logger.error(f"[Step 2/5] Chunk retrieval failed: {e}", exc_info=True)
            raise AzureServiceError(f"Chunk retrieval failed: {str(e)}") from e
        
        # Step 3 & 4: Generate answer (includes prompt construction and LLM generation)
        logger.info(f"[Step 3-4/5] Generating answer for query_id '{query_id}' using prompt version '{prompt_version}'")
        generation_start_time = time.time()
        db_conn = None
        try:
            # Get QueryExecutor for prompt loading
            db_conn = DatabaseConnection(config)
            db_conn.connect()
            query_executor = QueryExecutor(db_conn)
            
            # Generate answer (this internally handles prompt construction)
            answer = generate_answer(
                query=query,
                retrieved_chunks=retrieval_results,
                prompt_version=prompt_version,
                config=config,
                query_executor=query_executor
            )
            
            # Ensure query_id matches (generate_answer may have generated one)
            answer.query_id = query_id
            
            generation_latency = time.time() - generation_start_time
            logger.info(
                f"[Step 3-4/5] Answer generated in {generation_latency:.2f}s "
                f"for query_id '{query_id}' ({len(answer.text)} characters)"
            )
            
        except ValidationError:
            # Re-raise ValidationError as-is
            raise
        except DatabaseError:
            # Re-raise DatabaseError as-is
            raise
        except Exception as e:
            logger.error(f"[Step 3-4/5] Answer generation failed: {e}", exc_info=True)
            raise AzureServiceError(f"Answer generation failed: {str(e)}") from e
        finally:
            # Always close database connection, even on error
            if db_conn is not None:
                try:
                    db_conn.close()
                except Exception as close_error:
                    logger.warning(f"Error closing database connection: {close_error}")
        
        # Step 5: Log to Supabase
        logger.info(f"[Step 5/5] Logging pipeline results for query_id '{query_id}'")
        logging_start_time = time.time()
        try:
            # Ensure we have a QueryExecutor for logging
            # Reuse existing connection if available, otherwise create new one
            if db_conn is None:
                db_conn = DatabaseConnection(config)
                db_conn.connect()
                query_executor = QueryExecutor(db_conn)
                created_db_conn = True
            else:
                created_db_conn = False
            
            # Log query (generates query_id if missing)
            logged_query_id = log_query(query, query_executor)
            if logged_query_id != query_id:
                logger.warning(
                    f"Query ID mismatch: expected '{query_id}', got '{logged_query_id}'. "
                    f"Using logged query_id."
                )
                query_id = logged_query_id
                answer.query_id = query_id
            
            # Log retrieval results
            log_retrieval(query_id, retrieval_results, query_executor)
            
            # Log model answer
            log_model_answer(answer, query_executor)
            
            # Close database connection if we created it
            if created_db_conn and db_conn is not None:
                try:
                    db_conn.close()
                except Exception as close_error:
                    logger.warning(f"Error closing database connection: {close_error}")
            
            logging_latency = time.time() - logging_start_time
            logger.info(
                f"[Step 5/5] Logging completed in {logging_latency:.2f}s "
                f"(query_id: '{query_id}')"
            )
        except Exception as e:
            # Logging failures should not break the pipeline
            logger.warning(f"[Step 5/5] Logging failed (non-fatal): {e}", exc_info=True)
            # Continue execution even if logging fails
        
        # Calculate total pipeline latency
        total_latency = time.time() - pipeline_start_time
        
        logger.info(
            f"RAG pipeline completed successfully for query_id '{query_id}' in {total_latency:.2f}s. "
            f"Breakdown: embedding={embedding_latency:.2f}s, "
            f"retrieval={retrieval_latency:.2f}s, "
            f"generation={generation_latency:.2f}s, "
            f"logging={logging_latency:.2f}s"
        )
        
        return answer
        
    except (AzureServiceError, ValidationError, DatabaseError, ValueError):
        # Re-raise known exceptions as-is
        raise
    except Exception as e:
        # Catch any unexpected errors
        total_latency = time.time() - pipeline_start_time
        logger.error(
            f"Unexpected error in RAG pipeline for query_id '{query_id}' "
            f"(failed after {total_latency:.2f}s): {e}",
            exc_info=True
        )
        raise AzureServiceError(f"Unexpected error in RAG pipeline: {str(e)}") from e

