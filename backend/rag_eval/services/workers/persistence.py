"""Persistence layer for worker-queue architecture

This module provides load/persist functions for intermediate data (extracted text,
chunks, embeddings) between pipeline stages, enabling worker independence.
"""

from typing import List, Optional
import json
from rag_eval.core.interfaces import Chunk
from rag_eval.db.connection import DatabaseConnection
from rag_eval.db.queries import QueryExecutor
from rag_eval.core.exceptions import DatabaseError
from rag_eval.core.logging import get_logger

logger = get_logger("services.workers.persistence")


def load_extracted_text(document_id: str, config) -> str:
    """Load extracted text for a document from the database.
    
    Args:
        document_id: Document identifier (UUID string)
        config: Application configuration with database credentials
        
    Returns:
        Extracted text as string
        
    Raises:
        DatabaseError: If database query fails
        ValueError: If document_id is empty or extracted text not found
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    db_conn = DatabaseConnection(config)
    executor = QueryExecutor(db_conn)
    
    try:
        query = "SELECT extracted_text FROM documents WHERE id = %s"
        results = executor.execute_query(query, (document_id,))
        
        if not results:
            raise ValueError(f"Document {document_id} not found")
        
        extracted_text = results[0].get("extracted_text")
        if extracted_text is None:
            raise ValueError(f"Extracted text not found for document {document_id}")
        
        return extracted_text
    except Exception as e:
        logger.error(f"Failed to load extracted text for document {document_id}: {e}")
        if isinstance(e, (ValueError, DatabaseError)):
            raise
        raise DatabaseError(f"Failed to load extracted text: {e}") from e


def persist_extracted_text(document_id: str, text: str, config) -> None:
    """Persist extracted text for a document to the database.
    
    Args:
        document_id: Document identifier (UUID string)
        text: Extracted text to persist
        config: Application configuration with database credentials
        
    Raises:
        DatabaseError: If database insert/update fails
        ValueError: If document_id is empty
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    db_conn = DatabaseConnection(config)
    executor = QueryExecutor(db_conn)
    
    try:
        query = """
            UPDATE documents 
            SET extracted_text = %s 
            WHERE id = %s
        """
        executor.execute_insert(query, (text, document_id))
        logger.info(f"Persisted extracted text for document {document_id}")
    except Exception as e:
        logger.error(f"Failed to persist extracted text for document {document_id}: {e}")
        raise DatabaseError(f"Failed to persist extracted text: {e}") from e


def load_chunks(document_id: str, config) -> List[Chunk]:
    """Load all chunks for a document from the database.
    
    Args:
        document_id: Document identifier (UUID string)
        config: Application configuration with database credentials
        
    Returns:
        List of Chunk objects
        
    Raises:
        DatabaseError: If database query fails
        ValueError: If document_id is empty
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    db_conn = DatabaseConnection(config)
    executor = QueryExecutor(db_conn)
    
    try:
        query = """
            SELECT chunk_id, document_id, text, metadata
            FROM chunks
            WHERE document_id = %s
            ORDER BY created_at ASC
        """
        results = executor.execute_query(query, (document_id,))
        
        chunks = []
        for row in results:
            metadata = row.get("metadata")
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            elif metadata is None:
                metadata = {}
            
            chunk = Chunk(
                text=row["text"],
                chunk_id=row["chunk_id"],
                document_id=str(row["document_id"]),
                metadata=metadata
            )
            chunks.append(chunk)
        
        logger.info(f"Loaded {len(chunks)} chunks for document {document_id}")
        return chunks
    except Exception as e:
        logger.error(f"Failed to load chunks for document {document_id}: {e}")
        if isinstance(e, (ValueError, DatabaseError)):
            raise
        raise DatabaseError(f"Failed to load chunks: {e}") from e


def persist_chunks(document_id: str, chunks: List[Chunk], config) -> None:
    """Persist chunks for a document to the database.
    
    This function will overwrite existing chunks for the document if they exist.
    This enables idempotent chunking operations.
    
    Args:
        document_id: Document identifier (UUID string)
        chunks: List of Chunk objects to persist
        config: Application configuration with database credentials
        
    Raises:
        DatabaseError: If database insert fails
        ValueError: If document_id is empty or chunks list is invalid
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    if not chunks:
        logger.warning(f"No chunks to persist for document {document_id}")
        return
    
    db_conn = DatabaseConnection(config)
    executor = QueryExecutor(db_conn)
    
    try:
        # Delete existing chunks for this document (idempotency)
        delete_query = "DELETE FROM chunks WHERE document_id = %s"
        executor.execute_insert(delete_query, (document_id,))
        
        # Insert new chunks
        for chunk in chunks:
            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else None
            insert_query = """
                INSERT INTO chunks (chunk_id, document_id, text, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE
                SET document_id = EXCLUDED.document_id,
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata
            """
            executor.execute_insert(
                insert_query,
                (chunk.chunk_id, document_id, chunk.text, metadata_json)
            )
        
        logger.info(f"Persisted {len(chunks)} chunks for document {document_id}")
    except Exception as e:
        logger.error(f"Failed to persist chunks for document {document_id}: {e}")
        raise DatabaseError(f"Failed to persist chunks: {e}") from e


def load_embeddings(document_id: str, config) -> List[List[float]]:
    """Load embeddings for all chunks of a document from the database.
    
    Args:
        document_id: Document identifier (UUID string)
        config: Application configuration with database credentials
        
    Returns:
        List of embedding vectors (each vector is a list of floats)
        Returns embeddings in the same order as chunks (ordered by created_at)
        
    Raises:
        DatabaseError: If database query fails
        ValueError: If document_id is empty or embeddings not found
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    db_conn = DatabaseConnection(config)
    executor = QueryExecutor(db_conn)
    
    try:
        query = """
            SELECT embedding
            FROM chunks
            WHERE document_id = %s AND embedding IS NOT NULL
            ORDER BY created_at ASC
        """
        results = executor.execute_query(query, (document_id,))
        
        if not results:
            raise ValueError(f"Embeddings not found for document {document_id}")
        
        embeddings = []
        for row in results:
            embedding = row.get("embedding")
            if embedding is None:
                raise ValueError(f"Null embedding found for document {document_id}")
            
            # Handle JSONB - it might be a string or already a list
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            elif not isinstance(embedding, list):
                raise ValueError(f"Invalid embedding format for document {document_id}")
            
            embeddings.append(embedding)
        
        logger.info(f"Loaded {len(embeddings)} embeddings for document {document_id}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings for document {document_id}: {e}")
        if isinstance(e, (ValueError, DatabaseError)):
            raise
        raise DatabaseError(f"Failed to load embeddings: {e}") from e


def persist_embeddings(
    document_id: str, 
    chunks: List[Chunk], 
    embeddings: List[List[float]], 
    config
) -> None:
    """Persist embeddings alongside chunks for a document.
    
    This function updates the embedding column in the chunks table for each chunk.
    The chunks must already exist in the database (should be called after persist_chunks).
    
    Args:
        document_id: Document identifier (UUID string)
        chunks: List of Chunk objects (must match embeddings in order)
        embeddings: List of embedding vectors (each vector is a list of floats)
        config: Application configuration with database credentials
        
    Raises:
        DatabaseError: If database update fails
        ValueError: If document_id is empty, chunks/embeddings length mismatch, or chunks not found
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
        )
    
    if not chunks:
        logger.warning(f"No chunks/embeddings to persist for document {document_id}")
        return
    
    db_conn = DatabaseConnection(config)
    executor = QueryExecutor(db_conn)
    
    try:
        for chunk, embedding in zip(chunks, embeddings):
            embedding_json = json.dumps(embedding)
            update_query = """
                UPDATE chunks
                SET embedding = %s
                WHERE chunk_id = %s AND document_id = %s
            """
            executor.execute_insert(
                update_query,
                (embedding_json, chunk.chunk_id, document_id)
            )
        
        logger.info(f"Persisted {len(embeddings)} embeddings for document {document_id}")
    except Exception as e:
        logger.error(f"Failed to persist embeddings for document {document_id}: {e}")
        raise DatabaseError(f"Failed to persist embeddings: {e}") from e


def update_document_status(
    document_id: str, 
    status: str, 
    timestamp_field: Optional[str] = None, 
    config = None
) -> None:
    """Update document status and optionally set a timestamp field.
    
    Args:
        document_id: Document identifier (UUID string)
        status: New status value (e.g., 'parsed', 'chunked', 'embedded', 'indexed', 'failed_*')
        timestamp_field: Optional timestamp field to update ('parsed_at', 'chunked_at', 'embedded_at', 'indexed_at')
        config: Application configuration with database credentials
        
    Raises:
        DatabaseError: If database update fails
        ValueError: If document_id is empty or invalid timestamp_field
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    valid_timestamp_fields = ['parsed_at', 'chunked_at', 'embedded_at', 'indexed_at']
    if timestamp_field and timestamp_field not in valid_timestamp_fields:
        raise ValueError(f"Invalid timestamp_field: {timestamp_field}. Must be one of {valid_timestamp_fields}")
    
    db_conn = DatabaseConnection(config)
    executor = QueryExecutor(db_conn)
    
    try:
        if timestamp_field:
            query = f"""
                UPDATE documents 
                SET status = %s, {timestamp_field} = CURRENT_TIMESTAMP
                WHERE id = %s
            """
            executor.execute_insert(query, (status, document_id))
        else:
            query = "UPDATE documents SET status = %s WHERE id = %s"
            executor.execute_insert(query, (status, document_id))
        
        logger.info(f"Updated document {document_id} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update document status for {document_id}: {e}")
        raise DatabaseError(f"Failed to update document status: {e}") from e


def check_document_status(document_id: str, config) -> str:
    """Check the current status of a document.
    
    Args:
        document_id: Document identifier (UUID string)
        config: Application configuration with database credentials
        
    Returns:
        Current status string (e.g., 'uploaded', 'parsed', 'chunked', etc.)
        
    Raises:
        DatabaseError: If database query fails
        ValueError: If document_id is empty or document not found
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    db_conn = DatabaseConnection(config)
    executor = QueryExecutor(db_conn)
    
    try:
        query = "SELECT status FROM documents WHERE id = %s"
        results = executor.execute_query(query, (document_id,))
        
        if not results:
            raise ValueError(f"Document {document_id} not found")
        
        status = results[0].get("status", "uploaded")
        return status
    except Exception as e:
        logger.error(f"Failed to check document status for {document_id}: {e}")
        if isinstance(e, (ValueError, DatabaseError)):
            raise
        raise DatabaseError(f"Failed to check document status: {e}") from e


def should_process_document(document_id: str, target_status: str, config) -> bool:
    """Check if a document should be processed for a target status.
    
    This implements idempotency checks: if the document status is already at or
    beyond the target status, the document should not be processed.
    
    Status progression: uploaded -> parsed -> chunked -> embedded -> indexed
    
    Args:
        document_id: Document identifier (UUID string)
        target_status: Target status to check against ('parsed', 'chunked', 'embedded', 'indexed')
        config: Application configuration with database credentials
        
    Returns:
        True if document should be processed, False if already at or beyond target status
        
    Raises:
        DatabaseError: If database query fails
        ValueError: If document_id is empty or invalid target_status
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    valid_statuses = ['uploaded', 'parsed', 'chunked', 'embedded', 'indexed']
    if target_status not in valid_statuses:
        raise ValueError(f"Invalid target_status: {target_status}. Must be one of {valid_statuses}")
    
    current_status = check_document_status(document_id, config)
    
    # Status progression order
    status_order = {
        'uploaded': 0,
        'parsed': 1,
        'chunked': 2,
        'embedded': 3,
        'indexed': 4
    }
    
    current_order = status_order.get(current_status, -1)
    target_order = status_order[target_status]
    
    # If current status is at or beyond target, don't process
    if current_order >= target_order:
        logger.info(
            f"Document {document_id} already at status {current_status} "
            f"(target: {target_status}), skipping processing"
        )
        return False
    
    return True


def delete_chunks_by_document_id(document_id: str, config) -> int:
    """Delete all chunks (and embeddings) for a document from the chunks table.
    
    This function is used when deleting a document to ensure complete cleanup.
    
    Args:
        document_id: Document identifier (UUID string)
        config: Application configuration with database credentials
        
    Returns:
        Number of chunks deleted from chunks table
        
    Raises:
        DatabaseError: If deletion fails
        ValueError: If document_id is empty
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    db_conn = DatabaseConnection(config)
    executor = QueryExecutor(db_conn)
    
    try:
        # First, count chunks before deletion
        count_query = "SELECT COUNT(*) as count FROM chunks WHERE document_id = %s"
        count_results = executor.execute_query(count_query, (document_id,))
        chunk_count = count_results[0].get("count", 0) if count_results else 0
        
        # Delete chunks (CASCADE will handle any foreign key constraints)
        delete_query = "DELETE FROM chunks WHERE document_id = %s"
        executor.execute_insert(delete_query, (document_id,))
        
        logger.info(f"Deleted {chunk_count} chunks for document {document_id}")
        return chunk_count
    except Exception as e:
        logger.error(f"Failed to delete chunks for document {document_id}: {e}")
        raise DatabaseError(f"Failed to delete chunks: {e}") from e

