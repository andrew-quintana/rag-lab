"""Persistence layer for worker-queue architecture

This module provides load/persist functions for intermediate data (extracted text,
chunks, embeddings) between pipeline stages, enabling worker independence.

Uses Supabase REST API for all database operations (more reliable in serverless environments).
"""

from typing import List, Optional, Set
import json
from src.core.interfaces import Chunk
from src.core.exceptions import DatabaseError
from src.core.logging import get_logger
from src.db.supabase_db_service import SupabaseDatabaseService

logger = get_logger("services.workers.persistence")


def load_extracted_text(document_id: str, config) -> str:
    """Load extracted text for a document from the database using Supabase REST API.
    
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
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        doc = supabase_service.get_document(document_id)
        
        if not doc:
            raise ValueError(f"Document {document_id} not found")
        
        extracted_text = doc.get("extracted_text")
        if extracted_text is None:
            raise ValueError(f"Extracted text not found for document {document_id}")
        
        return extracted_text
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to load extracted text for document {document_id}: {e}")
        raise DatabaseError(f"Failed to load extracted text: {e}") from e


def persist_extracted_text(document_id: str, text: str, config) -> None:
    """Persist extracted text for a document to the database using Supabase REST API.
    
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
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        supabase_service.update_extracted_text(document_id, text)
        logger.info(f"Persisted extracted text for document {document_id}")
    except Exception as e:
        logger.error(f"Failed to persist extracted text for document {document_id}: {e}")
        raise DatabaseError(f"Failed to persist extracted text: {e}") from e


def load_chunks(document_id: str, config) -> List[Chunk]:
    """Load all chunks for a document from the database using Supabase REST API.
    
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
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        
        # Use Supabase client directly for chunks query
        result = supabase_service.client.table("chunks")\
            .select("chunk_id, document_id, text, metadata")\
            .eq("document_id", document_id)\
            .order("created_at")\
            .execute()
        
        chunks = []
        for row in result.data or []:
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
        raise DatabaseError(f"Failed to load chunks: {e}") from e


def persist_chunks(document_id: str, chunks: List[Chunk], config) -> None:
    """Persist chunks for a document to the database using Supabase REST API.
    
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
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        
        # Delete existing chunks for this document (idempotency)
        supabase_service.client.table("chunks")\
            .delete()\
            .eq("document_id", document_id)\
            .execute()
        
        # Prepare chunks data for bulk insert
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "chunk_id": chunk.chunk_id,
                "document_id": document_id,
                "text": chunk.text,
                "metadata": json.dumps(chunk.metadata) if chunk.metadata else None
            })
        
        # Bulk insert chunks
        supabase_service.client.table("chunks")\
            .upsert(chunks_data)\
            .execute()
        
        logger.info(f"Persisted {len(chunks)} chunks for document {document_id}")
    except Exception as e:
        logger.error(f"Failed to persist chunks for document {document_id}: {e}")
        raise DatabaseError(f"Failed to persist chunks: {e}") from e


def load_embeddings(document_id: str, config) -> List[List[float]]:
    """Load embeddings for all chunks of a document using Supabase REST API.
    
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
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        
        result = supabase_service.client.table("chunks")\
            .select("embedding")\
            .eq("document_id", document_id)\
            .not_.is_("embedding", "null")\
            .order("created_at")\
            .execute()
        
        if not result.data:
            raise ValueError(f"Embeddings not found for document {document_id}")
        
        embeddings = []
        for row in result.data:
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
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to load embeddings for document {document_id}: {e}")
        raise DatabaseError(f"Failed to load embeddings: {e}") from e


def persist_embeddings(
    document_id: str, 
    chunks: List[Chunk], 
    embeddings: List[List[float]], 
    config
) -> None:
    """Persist embeddings alongside chunks using Supabase REST API.
    
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
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        
        # Update embeddings for each chunk
        for chunk, embedding in zip(chunks, embeddings):
            supabase_service.client.table("chunks")\
                .update({"embedding": json.dumps(embedding)})\
                .eq("chunk_id", chunk.chunk_id)\
                .eq("document_id", document_id)\
                .execute()
        
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
    """Update document status and optionally set a timestamp field using Supabase REST API.
    
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
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        
        # Build update payload
        update_data = {"status": status}
        if timestamp_field:
            from datetime import datetime
            update_data[timestamp_field] = datetime.utcnow().isoformat()
        
        # Use Supabase client directly for timestamp updates
        supabase_service.client.table("documents")\
            .update(update_data)\
            .eq("id", document_id)\
            .execute()
        
        logger.info(f"Updated document {document_id} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update document status for {document_id}: {e}")
        raise DatabaseError(f"Failed to update document status: {e}") from e


def check_document_status(document_id: str, config) -> str:
    """Check the current status of a document using Supabase REST API.
    
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
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        status = supabase_service.get_document_status(document_id)
        if status:
            logger.debug(f"Got document status via REST API: {status}")
            return status
        else:
            raise ValueError(f"Document {document_id} not found")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to check document status for {document_id}: {e}")
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
    """Delete all chunks (and embeddings) using Supabase REST API.
    
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
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        
        # Count chunks before deletion
        count_result = supabase_service.client.table("chunks")\
            .select("chunk_id", count="exact")\
            .eq("document_id", document_id)\
            .execute()
        
        chunk_count = count_result.count if hasattr(count_result, 'count') else 0
        
        # Delete chunks
        supabase_service.client.table("chunks")\
            .delete()\
            .eq("document_id", document_id)\
            .execute()
        
        logger.info(f"Deleted {chunk_count} chunks for document {document_id}")
        return chunk_count
    except Exception as e:
        logger.error(f"Failed to delete chunks for document {document_id}: {e}")
        raise DatabaseError(f"Failed to delete chunks: {e}") from e


def persist_batch_result(
    document_id: str,
    batch_index: int,
    batch_text: str,
    start_page: int,
    end_page: int,
    config
) -> None:
    """Persist a batch extraction result using Supabase REST API.
    
    Stores batch results temporarily in the chunks table with chunk_id = "batch_XXX"
    for resumability and crash recovery. These are cleaned up after merging.
    
    Args:
        document_id: Document identifier (UUID string)
        batch_index: Batch index (0-based, used in chunk_id)
        batch_text: Extracted text from this batch
        start_page: First page in batch (1-indexed, inclusive)
        end_page: Last page in batch (1-indexed, exclusive)
        config: Application configuration with database credentials
        
    Raises:
        DatabaseError: If persistence fails
        ValueError: If document_id is empty or batch_index < 0
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    if batch_index < 0:
        raise ValueError(f"batch_index must be >= 0, got {batch_index}")
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        
        chunk_id = f"batch_{batch_index:03d}"  # e.g., "batch_000", "batch_001"
        batch_metadata = {
            "batch_index": batch_index,
            "start_page": start_page,
            "end_page": end_page,
            "type": "ingestion_batch"
        }
        
        # Use upsert for idempotency
        supabase_service.client.table("chunks")\
            .upsert({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": batch_text,
                "metadata": json.dumps(batch_metadata)
            })\
            .execute()
        
        logger.debug(
            f"Persisted batch {batch_index} for document {document_id} "
            f"(pages {start_page}-{end_page}, {len(batch_text)} chars)"
        )
    except Exception as e:
        logger.error(f"Failed to persist batch result for document {document_id}: {e}")
        raise DatabaseError(f"Failed to persist batch result: {e}") from e


def get_completed_batches(document_id: str, config) -> set[int]:
    """Get set of completed batch indices for a document using Supabase REST API.
    
    Queries the chunks table for all batch chunks (chunk_id LIKE 'batch_%')
    and extracts the batch_index from the chunk_id.
    
    Args:
        document_id: Document identifier (UUID string)
        config: Application configuration with database credentials
        
    Returns:
        Set of completed batch indices (0-based)
        
    Raises:
        DatabaseError: If query fails
        ValueError: If document_id is empty
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        
        # Query chunks with chunk_id starting with "batch_"
        result = supabase_service.client.table("chunks")\
            .select("chunk_id")\
            .eq("document_id", document_id)\
            .like("chunk_id", "batch_%")\
            .order("chunk_id")\
            .execute()
        
        completed_batches = set()
        for row in result.data or []:
            chunk_id = row.get("chunk_id", "")
            # Extract batch_index from "batch_XXX" format
            if chunk_id.startswith("batch_"):
                try:
                    # Handle both "batch_0" and "batch_000" formats
                    batch_str = chunk_id[6:]  # Skip "batch_" prefix
                    # Remove leading zeros for comparison
                    batch_index = int(batch_str)
                    completed_batches.add(batch_index)
                except ValueError:
                    logger.warning(f"Invalid batch chunk_id format: {chunk_id}")
        
        logger.debug(
            f"Found {len(completed_batches)} completed batches for document {document_id}"
        )
        return completed_batches
    except Exception as e:
        logger.error(f"Failed to get completed batches for document {document_id}: {e}")
        raise DatabaseError(f"Failed to get completed batches: {e}") from e


def load_batch_result(document_id: str, batch_index: int, config) -> Optional[str]:
    """Load a specific batch result using Supabase REST API.
    
    Args:
        document_id: Document identifier (UUID string)
        batch_index: Batch index (0-based)
        config: Application configuration with database credentials
        
    Returns:
        Batch text if found, None otherwise
        
    Raises:
        DatabaseError: If query fails
        ValueError: If document_id is empty
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    if batch_index < 0:
        raise ValueError(f"batch_index must be >= 0, got {batch_index}")
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        
        chunk_id = f"batch_{batch_index:03d}"
        result = supabase_service.client.table("chunks")\
            .select("text")\
            .eq("chunk_id", chunk_id)\
            .eq("document_id", document_id)\
            .execute()
        
        if not result.data or len(result.data) == 0:
            return None
        
        return result.data[0].get("text")
    except Exception as e:
        logger.error(f"Failed to load batch result for document {document_id}: {e}")
        raise DatabaseError(f"Failed to load batch result: {e}") from e


def delete_batch_chunk(document_id: str, batch_index: int, config) -> None:
    """Delete a specific batch chunk using Supabase REST API.
    
    Used to clean up batch chunks after they've been merged into the final text.
    
    Args:
        document_id: Document identifier (UUID string)
        batch_index: Batch index (0-based)
        config: Application configuration with database credentials
        
    Raises:
        DatabaseError: If deletion fails
        ValueError: If document_id is empty
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    if batch_index < 0:
        raise ValueError(f"batch_index must be >= 0, got {batch_index}")
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        
        chunk_id = f"batch_{batch_index:03d}"
        supabase_service.client.table("chunks")\
            .delete()\
            .eq("chunk_id", chunk_id)\
            .eq("document_id", document_id)\
            .execute()
        
        logger.debug(f"Deleted batch chunk {batch_index} for document {document_id}")
    except Exception as e:
        logger.error(f"Failed to delete batch chunk for document {document_id}: {e}")
        raise DatabaseError(f"Failed to delete batch chunk: {e}") from e


def update_ingestion_metadata(document_id: str, updates: dict, config) -> None:
    """Update ingestion metadata in documents.metadata->'ingestion' JSONB field.
    
    Merges the provided updates with existing ingestion metadata without overwriting
    the entire metadata structure.
    
    Args:
        document_id: Document identifier (UUID string)
        updates: Dictionary of updates to merge into metadata->ingestion
        config: Application configuration with database credentials
        
    Raises:
        DatabaseError: If update fails
        ValueError: If document_id is empty
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    if not updates:
        logger.warning(f"No updates provided for document {document_id}")
        return
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        supabase_service.update_ingestion_metadata(document_id, updates)
        logger.debug(f"Updated ingestion metadata for document {document_id}")
    except Exception as e:
        logger.error(f"Failed to update ingestion metadata for document {document_id}: {e}")
        raise DatabaseError(f"Failed to update ingestion metadata: {e}") from e


def get_ingestion_metadata(document_id: str, config) -> dict:
    """Get ingestion metadata from documents.metadata->'ingestion' JSONB field.
    
    Returns the ingestion metadata structure, or default structure if not exists.
    
    Args:
        document_id: Document identifier (UUID string)
        config: Application configuration with database credentials
        
    Returns:
        Dictionary with ingestion metadata, or default structure if not found
        
    Raises:
        DatabaseError: If query fails
        ValueError: If document_id is empty or document not found
    """
    if not document_id:
        raise ValueError("document_id cannot be empty")
    
    try:
        supabase_service = SupabaseDatabaseService(config)
        doc = supabase_service.get_document(document_id)
        
        if not doc:
            raise ValueError(f"Document {document_id} not found")
        
        metadata = doc.get("metadata")
        
        # Extract ingestion metadata
        if metadata and isinstance(metadata, dict):
            ingestion_meta = metadata.get("ingestion", {})
            if ingestion_meta:
                return ingestion_meta
        
        # Return default structure
        return {
            "num_pages": 0,
            "num_batches_total": 0,
            "last_successful_page": 0,
            "next_unparsed_batch_index": 0,
            "parsing_status": "pending",
            "batch_size": 2,
            "batches_completed": {},
            "parsing_started_at": None,
            "parsing_completed_at": None,
            "errors": []
        }
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to get ingestion metadata for document {document_id}: {e}")
        raise DatabaseError(f"Failed to get ingestion metadata: {e}") from e

