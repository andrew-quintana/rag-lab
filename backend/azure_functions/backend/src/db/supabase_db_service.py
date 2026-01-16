"""Supabase REST API database service

Alternative to direct PostgreSQL connections using Supabase REST API.
Uses service role key to bypass RLS and perform database operations via HTTP.

This is more reliable in serverless environments (Azure Functions) where
direct PostgreSQL connections can have issues with:
- Connection pooling
- Password authentication
- Network/firewall restrictions
- Connection string parsing

Usage:
    from src.db.supabase_db_service import SupabaseDatabaseService
    
    service = SupabaseDatabaseService(config)
    status = service.get_document_status(document_id)
    service.update_document_status(document_id, "processing")
"""

from typing import Optional, Dict, Any, List
import json
from datetime import datetime
from supabase import create_client, Client
from src.core.config import Config
from src.core.exceptions import DatabaseError
from src.core.logging import get_logger

logger = get_logger("db.supabase_db_service")


class SupabaseDatabaseService:
    """Database operations using Supabase REST API with service role key"""
    
    def __init__(self, config: Config):
        """Initialize Supabase client with service role key
        
        Args:
            config: Application configuration with Supabase credentials
        """
        if not config.supabase_url:
            raise DatabaseError("SUPABASE_URL is not configured")
        
        # Use service role key if available, otherwise fall back to anon key
        api_key = config.supabase_service_role_key if config.supabase_service_role_key else config.supabase_anon_key
        
        if not api_key:
            raise DatabaseError("SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY must be configured")
        
        try:
            self.client: Client = create_client(
                config.supabase_url,
                api_key
            )
            logger.info("Supabase REST API client initialized")
        except Exception as e:
            logger.error(f"Failed to create Supabase client: {e}")
            raise DatabaseError(f"Failed to create Supabase client: {e}") from e
    
    def get_document_status(self, document_id: str) -> Optional[str]:
        """Get document status using Supabase REST API
        
        Args:
            document_id: Document identifier (UUID string)
            
        Returns:
            Document status string, or None if not found
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            result = self.client.table("documents")\
                .select("status")\
                .eq("id", document_id)\
                .execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0].get("status")
            return None
        except Exception as e:
            logger.error(f"Failed to get document status for {document_id}: {e}")
            raise DatabaseError(f"Failed to get document status: {e}") from e
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get full document record using Supabase REST API
        
        Args:
            document_id: Document identifier (UUID string)
            
        Returns:
            Document record as dictionary, or None if not found
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            result = self.client.table("documents")\
                .select("*")\
                .eq("id", document_id)\
                .execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            raise DatabaseError(f"Failed to get document: {e}") from e
    
    def update_document_status(self, document_id: str, status: str) -> None:
        """Update document status using Supabase REST API
        
        Args:
            document_id: Document identifier (UUID string)
            status: New status value
            
        Raises:
            DatabaseError: If update fails
        """
        try:
            self.client.table("documents")\
                .update({"status": status})\
                .eq("id", document_id)\
                .execute()
            
            logger.debug(f"Updated document {document_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update document status: {e}")
            raise DatabaseError(f"Failed to update document status: {e}") from e
    
    def update_document_chunks(self, document_id: str, chunks_created: int) -> None:
        """Update document chunks count using Supabase REST API
        
        Args:
            document_id: Document identifier (UUID string)
            chunks_created: Number of chunks created
            
        Raises:
            DatabaseError: If update fails
        """
        try:
            self.client.table("documents")\
                .update({"chunks_created": chunks_created})\
                .eq("id", document_id)\
                .execute()
            
            logger.debug(f"Updated document {document_id} chunks to {chunks_created}")
        except Exception as e:
            logger.error(f"Failed to update document chunks: {e}")
            raise DatabaseError(f"Failed to update document chunks: {e}") from e
    
    def update_extracted_text(self, document_id: str, extracted_text: str) -> None:
        """Update document extracted text using Supabase REST API
        
        Args:
            document_id: Document identifier (UUID string)
            extracted_text: Extracted text content
            
        Raises:
            DatabaseError: If update fails
        """
        try:
            self.client.table("documents")\
                .update({"extracted_text": extracted_text})\
                .eq("id", document_id)\
                .execute()
            
            logger.debug(f"Updated extracted text for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to update extracted text: {e}")
            raise DatabaseError(f"Failed to update extracted text: {e}") from e
    
    def insert_batch_result(
        self,
        document_id: str,
        batch_index: int,
        batch_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert batch result using Supabase REST API
        
        Args:
            document_id: Document identifier (UUID string)
            batch_index: Batch index number
            batch_text: Extracted text for this batch
            metadata: Optional metadata dictionary
            
        Raises:
            DatabaseError: If insert fails
        """
        try:
            batch_data = {
                "document_id": document_id,
                "batch_index": batch_index,
                "batch_text": batch_text,
                "metadata": json.dumps(metadata) if metadata else None,
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.client.table("batch_results")\
                .insert(batch_data)\
                .execute()
            
            logger.debug(f"Inserted batch result for document {document_id}, batch {batch_index}")
        except Exception as e:
            logger.error(f"Failed to insert batch result: {e}")
            raise DatabaseError(f"Failed to insert batch result: {e}") from e
    
    def get_batch_results(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all batch results for a document using Supabase REST API
        
        Args:
            document_id: Document identifier (UUID string)
            
        Returns:
            List of batch result dictionaries
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            result = self.client.table("batch_results")\
                .select("*")\
                .eq("document_id", document_id)\
                .order("batch_index")\
                .execute()
            
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get batch results for {document_id}: {e}")
            raise DatabaseError(f"Failed to get batch results: {e}") from e
    
    def update_ingestion_metadata(self, document_id: str, updates: Dict[str, Any]) -> None:
        """Update ingestion metadata using Supabase REST API
        
        Note: This uses RPC call since JSONB updates are complex via REST
        
        Args:
            document_id: Document identifier (UUID string)
            updates: Dictionary of updates to merge into metadata->ingestion
            
        Raises:
            DatabaseError: If update fails
        """
        try:
            # For JSONB updates, we might need to use RPC or get-update pattern
            # Get current document
            doc = self.get_document(document_id)
            if not doc:
                raise ValueError(f"Document {document_id} not found")
            
            # Merge metadata
            current_metadata = doc.get("metadata", {}) or {}
            if isinstance(current_metadata, str):
                current_metadata = json.loads(current_metadata)
            
            ingestion_meta = current_metadata.get("ingestion", {})
            merged_meta = {**ingestion_meta, **updates}
            
            # Update with merged metadata
            new_metadata = {**current_metadata, "ingestion": merged_meta}
            
            self.client.table("documents")\
                .update({"metadata": new_metadata})\
                .eq("id", document_id)\
                .execute()
            
            logger.debug(f"Updated ingestion metadata for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to update ingestion metadata: {e}")
            raise DatabaseError(f"Failed to update ingestion metadata: {e}") from e
