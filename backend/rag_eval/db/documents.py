"""Document database operations"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import json
from rag_eval.db.connection import DatabaseConnection
from rag_eval.db.queries import QueryExecutor
from rag_eval.db.models import Document
from rag_eval.core.exceptions import DatabaseError
from rag_eval.core.logging import get_logger

logger = get_logger("db.documents")


class DocumentService:
    """Service for document database operations"""
    
    def __init__(self, db_conn: DatabaseConnection):
        self.db_conn = db_conn
        self.executor = QueryExecutor(db_conn)
    
    def insert_document(self, document: Document) -> None:
        """Insert a new document record"""
        query = """
            INSERT INTO documents (
                document_id, filename, file_size, mime_type, upload_timestamp,
                status, chunks_created, storage_path, preview_image_path, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            document.document_id,
            document.filename,
            document.file_size,
            document.mime_type,
            document.upload_timestamp or datetime.utcnow(),
            document.status,
            document.chunks_created,
            document.storage_path,
            document.preview_image_path,
            json.dumps(document.metadata) if document.metadata else None
        )
        
        try:
            self.executor.execute_insert(query, params)
            logger.info(f"Inserted document record: {document.document_id}")
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            raise DatabaseError(f"Failed to insert document: {e}") from e
    
    def update_document_status(self, document_id: str, status: str) -> None:
        """Update document status"""
        query = "UPDATE documents SET status = %s WHERE document_id = %s"
        params = (status, document_id)
        
        try:
            self.executor.execute_insert(query, params)
            logger.info(f"Updated document {document_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update document status: {e}")
            raise DatabaseError(f"Failed to update document status: {e}") from e
    
    def update_document_chunks(self, document_id: str, chunks_created: int) -> None:
        """Update document chunks count"""
        query = "UPDATE documents SET chunks_created = %s WHERE document_id = %s"
        params = (chunks_created, document_id)
        
        try:
            self.executor.execute_insert(query, params)
            logger.info(f"Updated document {document_id} chunks to {chunks_created}")
        except Exception as e:
            logger.error(f"Failed to update document chunks: {e}")
            raise DatabaseError(f"Failed to update document chunks: {e}") from e
    
    def update_preview_path(self, document_id: str, preview_path: str) -> None:
        """Update document preview image path"""
        query = "UPDATE documents SET preview_image_path = %s WHERE document_id = %s"
        params = (preview_path, document_id)
        
        try:
            self.executor.execute_insert(query, params)
            logger.info(f"Updated document {document_id} preview path")
        except Exception as e:
            logger.error(f"Failed to update preview path: {e}")
            raise DatabaseError(f"Failed to update preview path: {e}") from e
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID"""
        query = "SELECT * FROM documents WHERE document_id = %s"
        params = (document_id,)
        
        try:
            results = self.executor.execute_query(query, params)
            if not results:
                return None
            
            row = results[0]
            return self._row_to_document(row)
        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            raise DatabaseError(f"Failed to get document: {e}") from e
    
    def list_documents(
        self,
        search: Optional[str] = None,
        file_type: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        sort_by: str = "upload_timestamp",
        sort_order: str = "desc",
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """List documents with filtering and sorting"""
        # Build WHERE clause
        conditions = []
        params = []
        
        if search:
            conditions.append("filename ILIKE %s")
            params.append(f"%{search}%")
        
        if file_type:
            conditions.append("mime_type = %s")
            params.append(file_type)
        
        if status:
            conditions.append("status = %s")
            params.append(status)
        
        if date_from:
            conditions.append("upload_timestamp >= %s")
            params.append(date_from)
        
        if date_to:
            conditions.append("upload_timestamp <= %s")
            params.append(date_to)
        
        if size_min is not None:
            conditions.append("file_size >= %s")
            params.append(size_min)
        
        if size_max is not None:
            conditions.append("file_size <= %s")
            params.append(size_max)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Validate sort_by
        valid_sort_fields = ["upload_timestamp", "filename", "file_size", "mime_type", "status"]
        if sort_by not in valid_sort_fields:
            sort_by = "upload_timestamp"
        
        # Validate sort_order
        sort_order = sort_order.lower()
        if sort_order not in ["asc", "desc"]:
            sort_order = "desc"
        
        query = f"""
            SELECT * FROM documents
            WHERE {where_clause}
            ORDER BY {sort_by} {sort_order}
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])
        
        try:
            results = self.executor.execute_query(query, tuple(params))
            return [self._row_to_document(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise DatabaseError(f"Failed to list documents: {e}") from e
    
    def count_documents(
        self,
        search: Optional[str] = None,
        file_type: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None
    ) -> int:
        """Count documents matching filters"""
        # Build WHERE clause (same as list_documents)
        conditions = []
        params = []
        
        if search:
            conditions.append("filename ILIKE %s")
            params.append(f"%{search}%")
        
        if file_type:
            conditions.append("mime_type = %s")
            params.append(file_type)
        
        if status:
            conditions.append("status = %s")
            params.append(status)
        
        if date_from:
            conditions.append("upload_timestamp >= %s")
            params.append(date_from)
        
        if date_to:
            conditions.append("upload_timestamp <= %s")
            params.append(date_to)
        
        if size_min is not None:
            conditions.append("file_size >= %s")
            params.append(size_min)
        
        if size_max is not None:
            conditions.append("file_size <= %s")
            params.append(size_max)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"SELECT COUNT(*) as count FROM documents WHERE {where_clause}"
        
        try:
            results = self.executor.execute_query(query, tuple(params))
            return results[0]["count"] if results else 0
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            raise DatabaseError(f"Failed to count documents: {e}") from e
    
    def delete_document(self, document_id: str) -> None:
        """Delete a document record"""
        query = "DELETE FROM documents WHERE document_id = %s"
        params = (document_id,)
        
        try:
            self.executor.execute_insert(query, params)
            logger.info(f"Deleted document record: {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise DatabaseError(f"Failed to delete document: {e}") from e
    
    def _row_to_document(self, row: Dict[str, Any]) -> Document:
        """Convert database row to Document model"""
        metadata = None
        if row.get("metadata"):
            try:
                metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            except (json.JSONDecodeError, TypeError):
                metadata = None
        
        return Document(
            document_id=row["document_id"],
            filename=row["filename"],
            file_size=row["file_size"],
            mime_type=row.get("mime_type"),
            upload_timestamp=row.get("upload_timestamp"),
            status=row.get("status", "uploaded"),
            chunks_created=row.get("chunks_created"),
            storage_path=row.get("storage_path", ""),
            preview_image_path=row.get("preview_image_path"),
            metadata=metadata
        )

