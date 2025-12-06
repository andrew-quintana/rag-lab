"""POST /upload endpoint for document ingestion"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from rag_eval.core.config import Config
from rag_eval.core.logging import get_logger
from rag_eval.services.rag.supabase_storage import (
    upload_document_to_storage,
    generate_image_preview
)
from rag_eval.services.workers.queue_client import (
    enqueue_message,
    QueueMessage,
    SourceStorage,
    ProcessingStage
)
from rag_eval.db.connection import DatabaseConnection
from rag_eval.db.documents import DocumentService
from rag_eval.db.models import Document
from rag_eval.utils.ids import generate_id

logger = get_logger("api.routes.upload")
router = APIRouter()

config = Config.from_env()
db_conn = DatabaseConnection(config)
db_conn.connect()
doc_service = DocumentService(db_conn)


class UploadResponse(BaseModel):
    """Response model for upload endpoint"""
    document_id: str
    status: str = "uploaded"
    message: Optional[str] = None


@router.post("/upload", response_model=UploadResponse)
async def handle_upload(file: UploadFile = File(...)):
    """
    Upload and enqueue a document for asynchronous processing.
    
    Processing flow:
    1. Upload file to Supabase Storage
    2. Generate preview image (if applicable)
    3. Save document metadata to database with status='uploaded'
    4. Enqueue message to ingestion-uploads queue
    5. Return immediately with document_id and status
    
    Processing happens asynchronously via workers:
    - Ingestion worker extracts text
    - Chunking worker chunks text
    - Embedding worker generates embeddings
    - Indexing worker indexes chunks
    
    Use GET /documents/{document_id}/status to track processing progress.
    
    Args:
        file: Uploaded file (PDF, images, documents)
        
    Returns:
        UploadResponse with document ID and status='uploaded'
    """
    logger.info(f"Received upload request for file: {file.filename}")
    
    document_id = None
    try:
        # Generate document ID
        document_id = generate_id()
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        mime_type = file.content_type or "application/octet-stream"
        logger.info(f"Read {file_size} bytes from file: {file.filename}")
        
        # Step 1: Upload file to Supabase Storage
        logger.info("Step 1: Uploading file to Supabase Storage")
        storage_path = upload_document_to_storage(
            file_content,
            document_id,
            file.filename or document_id,
            config,
            content_type=mime_type
        )
        
        # Step 2: Generate preview image (if applicable)
        logger.info("Step 2: Generating preview image (if applicable)")
        preview_path = None
        try:
            preview_path = generate_image_preview(
                file_content,
                document_id,
                mime_type,
                config
            )
        except Exception as e:
            logger.warning(f"Preview generation failed (non-fatal): {e}")
        
        # Step 3: Save document metadata to database with status='uploaded'
        logger.info("Step 3: Saving document metadata to database")
        document = Document(
            id=document_id,
            filename=file.filename or document_id,
            file_size=file_size,
            mime_type=mime_type,
            upload_timestamp=datetime.utcnow(),
            status="uploaded",
            storage_path=storage_path,
            preview_image_path=preview_path,
            metadata={"original_filename": file.filename}
        )
        doc_service.insert_document(document)
        
        # Step 4: Enqueue message to ingestion-uploads queue
        logger.info("Step 4: Enqueuing message to ingestion-uploads queue")
        queue_message = QueueMessage(
            document_id=document_id,
            source_storage=SourceStorage.SUPABASE,
            filename=file.filename or document_id,
            attempt=1,
            stage=ProcessingStage.UPLOADED,
            metadata={
                "mime_type": mime_type,
                "file_size": file_size,
                "original_filename": file.filename
            }
        )
        enqueue_message("ingestion-uploads", queue_message, config)
        logger.info(f"Enqueued document {document_id} for processing")
        
        return UploadResponse(
            document_id=document_id,
            status="uploaded",
            message="Document uploaded and enqueued for processing"
        )
        
    except HTTPException:
        # Update status on error if document was created
        if document_id:
            try:
                doc_service.update_document_status(document_id, "uploaded")
            except Exception:
                pass  # Ignore errors when updating status on failure
        raise
    except Exception as e:
        logger.error(f"Upload processing failed: {e}", exc_info=True)
        if document_id:
            try:
                doc_service.update_document_status(document_id, "uploaded")
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")



