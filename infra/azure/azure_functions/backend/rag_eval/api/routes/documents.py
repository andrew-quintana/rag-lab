"""Document management API endpoints"""

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from rag_eval.core.config import Config
from rag_eval.core.logging import get_logger
from rag_eval.db.connection import DatabaseConnection
from rag_eval.db.documents import DocumentService
from rag_eval.db.models import Document
from rag_eval.services.rag.supabase_storage import (
    download_document_from_storage,
    delete_document_from_storage,
    get_public_url
)
from rag_eval.services.rag.search import delete_chunks_by_document_id as delete_chunks_from_ai_search
from rag_eval.services.workers.persistence import delete_chunks_by_document_id as delete_chunks_from_db
from rag_eval.db.queries import QueryExecutor
import io

logger = get_logger("api.routes.documents")
router = APIRouter()

config = Config.from_env()
db_conn = DatabaseConnection(config)
db_conn.connect()
doc_service = DocumentService(db_conn)


class DocumentResponse(BaseModel):
    """Response model for document metadata"""
    document_id: str
    filename: str
    file_size: int
    mime_type: Optional[str] = None
    upload_timestamp: Optional[datetime] = None
    status: str
    chunks_created: Optional[int] = None
    storage_path: str
    preview_image_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentListResponse(BaseModel):
    """Response model for document list"""
    documents: List[DocumentResponse]
    total: int
    limit: int
    offset: int


class DocumentStatusResponse(BaseModel):
    """Response model for document status query"""
    document_id: str
    status: str
    parsed_at: Optional[datetime] = None
    chunked_at: Optional[datetime] = None
    embedded_at: Optional[datetime] = None
    indexed_at: Optional[datetime] = None
    error_details: Optional[str] = None


class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion"""
    message: str
    document_id: str
    chunks_deleted_db: int = 0
    chunks_deleted_ai_search: int = 0


def _document_to_response(doc: Document) -> DocumentResponse:
    """Convert Document model to response"""
    return DocumentResponse(
        document_id=doc.id,
        filename=doc.filename,
        file_size=doc.file_size,
        mime_type=doc.mime_type,
        upload_timestamp=doc.upload_timestamp,
        status=doc.status,
        chunks_created=doc.chunks_created,
        storage_path=doc.storage_path,
        preview_image_path=doc.preview_image_path,
        metadata=doc.metadata
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    search: Optional[str] = Query(None, description="Search in filename"),
    file_type: Optional[str] = Query(None, description="Filter by MIME type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    date_from: Optional[str] = Query(None, description="Filter from date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter to date (ISO format)"),
    size_min: Optional[int] = Query(None, description="Minimum file size in bytes"),
    size_max: Optional[int] = Query(None, description="Maximum file size in bytes"),
    sort_by: str = Query("upload_timestamp", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    limit: int = Query(100, ge=1, le=1000, description="Number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List documents with filtering, sorting, and pagination.
    
    Returns:
        DocumentListResponse with documents and pagination info
    """
    logger.info("Listing documents with filters")
    
    try:
        # Parse date strings
        date_from_dt = None
        date_to_dt = None
        if date_from:
            try:
                date_from_dt = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use ISO format.")
        if date_to:
            try:
                date_to_dt = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use ISO format.")
        
        # Get documents
        documents = doc_service.list_documents(
            search=search,
            file_type=file_type,
            status=status,
            date_from=date_from_dt,
            date_to=date_to_dt,
            size_min=size_min,
            size_max=size_max,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset
        )
        
        # Get total count
        total = doc_service.count_documents(
            search=search,
            file_type=file_type,
            status=status,
            date_from=date_from_dt,
            date_to=date_to_dt,
            size_min=size_min,
            size_max=size_max
        )
        
        return DocumentListResponse(
            documents=[_document_to_response(doc) for doc in documents],
            total=total,
            limit=limit,
            offset=offset
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """
    Get document metadata by ID.
    
    Args:
        document_id: Document identifier
        
    Returns:
        DocumentResponse with document metadata
    """
    logger.info(f"Getting document: {document_id}")
    
    try:
        document = doc_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return _document_to_response(document)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.get("/documents/{document_id}/download")
async def download_document(document_id: str):
    """
    Download a document file.
    
    Args:
        document_id: Document identifier
        
    Returns:
        File download response
    """
    logger.info(f"Downloading document: {document_id}")
    
    try:
        # Get document metadata
        document = doc_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Download from storage
        file_content = download_document_from_storage(document_id, config)
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type=document.mime_type or "application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{document.filename}"'
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download document: {str(e)}")


@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str):
    """
    Get document processing status.
    
    Returns current status, timestamps for each pipeline stage,
    and error details if the document processing failed.
    
    Args:
        document_id: Document identifier
        
    Returns:
        DocumentStatusResponse with status, timestamps, and error details
    """
    logger.info(f"Getting document status: {document_id}")
    
    try:
        # Get document metadata
        document = doc_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Query timestamps directly from database
        executor = QueryExecutor(db_conn)
        query = """
            SELECT status, parsed_at, chunked_at, embedded_at, indexed_at, metadata
            FROM documents
            WHERE id = %s
        """
        results = executor.execute_query(query, (document_id,))
        
        if not results:
            raise HTTPException(status_code=404, detail="Document not found")
        
        row = results[0]
        status = row.get("status", "uploaded")
        
        # Extract error details from metadata if status indicates failure
        error_details = None
        if status.startswith("failed_"):
            metadata = row.get("metadata")
            if metadata:
                if isinstance(metadata, str):
                    import json
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = None
                if metadata and isinstance(metadata, dict):
                    error_details = metadata.get("error_details") or metadata.get("error_message")
        
        return DocumentStatusResponse(
            document_id=document_id,
            status=status,
            parsed_at=row.get("parsed_at"),
            chunked_at=row.get("chunked_at"),
            embedded_at=row.get("embedded_at"),
            indexed_at=row.get("indexed_at"),
            error_details=error_details
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get document status: {str(e)}")


@router.get("/documents/{document_id}/preview")
async def get_preview(document_id: str):
    """
    Get preview image for a document.
    
    Args:
        document_id: Document identifier
        
    Returns:
        Preview image response
    """
    logger.info(f"Getting preview for document: {document_id}")
    
    try:
        # Get document metadata
        document = doc_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not document.preview_image_path:
            raise HTTPException(status_code=404, detail="Preview not available for this document")
        
        # Download preview from storage
        preview_content = download_document_from_storage(document.preview_image_path, config)
        
        # Return as image response
        return Response(
            content=preview_content,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f'inline; filename="{document_id}_preview.jpg"'
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get preview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get preview: {str(e)}")


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(document_id: str):
    """
    Delete a document and all associated data.
    
    This function performs a complete deletion:
    1. Deletes chunks from Azure AI Search
    2. Deletes chunks and embeddings from chunks table
    3. Deletes file from Supabase Storage (main file and preview if exists)
    4. Deletes document record from database
    
    All deletion operations are attempted even if one fails (graceful degradation).
    
    Args:
        document_id: Document identifier
        
    Returns:
        DeleteDocumentResponse with counts of deleted chunks from both systems
    """
    logger.info(f"Deleting document: {document_id}")
    
    try:
        # Get document metadata
        document = doc_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        chunks_deleted_ai_search = 0
        chunks_deleted_db = 0
        
        # Step 1: Delete chunks from Azure AI Search
        try:
            logger.info(f"Deleting chunks from Azure AI Search for document: {document_id}")
            chunks_deleted_ai_search = delete_chunks_from_ai_search(document_id, config)
            logger.info(f"Deleted {chunks_deleted_ai_search} chunks from Azure AI Search")
        except Exception as e:
            logger.warning(f"Failed to delete chunks from Azure AI Search (continuing with other deletions): {e}")
            # Continue with other deletions even if Azure AI Search deletion fails
        
        # Step 2: Delete chunks and embeddings from chunks table
        try:
            logger.info(f"Deleting chunks from chunks table for document: {document_id}")
            chunks_deleted_db = delete_chunks_from_db(document_id, config)
            logger.info(f"Deleted {chunks_deleted_db} chunks from chunks table")
        except Exception as e:
            logger.warning(f"Failed to delete chunks from chunks table (continuing with other deletions): {e}")
            # Continue with other deletions even if chunks table deletion fails
        
        # Step 3: Delete from storage (main file and preview if exists)
        try:
            delete_document_from_storage(document_id, config)
            if document.preview_image_path:
                delete_document_from_storage(document.preview_image_path, config)
        except Exception as e:
            logger.warning(f"Failed to delete file from storage (continuing with DB delete): {e}")
        
        # Step 4: Delete from database
        doc_service.delete_document(document_id)
        
        return DeleteDocumentResponse(
            message="Document deleted successfully",
            document_id=document_id,
            chunks_deleted_db=chunks_deleted_db,
            chunks_deleted_ai_search=chunks_deleted_ai_search
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

