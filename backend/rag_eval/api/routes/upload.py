"""POST /upload endpoint for document ingestion"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from datetime import datetime
from rag_eval.core.config import Config
from rag_eval.core.logging import get_logger
from rag_eval.services.rag.ingestion import ingest_document
from rag_eval.services.rag.chunking import chunk_text
from rag_eval.services.rag.embeddings import generate_embeddings
from rag_eval.services.rag.search import index_chunks
from rag_eval.services.rag.supabase_storage import (
    upload_document_to_storage,
    generate_image_preview
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
    status: str
    message: str
    chunks_created: int


@router.post("/upload", response_model=UploadResponse)
async def handle_upload(file: UploadFile = File(...)):
    """
    Upload and process a document through the ingestion pipeline.
    
    Processing flow:
    1. Upload file to Supabase Storage
    2. Generate preview image (if applicable)
    3. Save document metadata to database
    4. Extract text using Azure Document Intelligence
    5. Chunk text
    6. Generate embeddings
    7. Index chunks into Azure AI Search
    8. Update document status
    
    Args:
        file: Uploaded file (PDF, images, documents)
        
    Returns:
        UploadResponse with document ID and processing status
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
        
        # Step 3: Save document metadata to database
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
        
        # Update status to processing
        doc_service.update_document_status(document_id, "processing")
        
        # Step 4: Extract text using Azure Document Intelligence
        logger.info("Step 4: Extracting text using Azure Document Intelligence")
        extracted_text = ingest_document(file_content, config)
        
        if not extracted_text or len(extracted_text.strip()) == 0:
            doc_service.update_document_status(document_id, "uploaded")
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the document"
            )
        
        logger.info(f"Extracted {len(extracted_text)} characters of text")
        
        # Step 5: Chunk text
        logger.info("Step 5: Chunking text")
        chunks = chunk_text(extracted_text, config, document_id=document_id)
        
        if not chunks:
            doc_service.update_document_status(document_id, "uploaded")
            raise HTTPException(
                status_code=500,
                detail="Failed to create chunks from extracted text"
            )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 6: Generate embeddings using Azure AI Foundry
        logger.info("Step 6: Generating embeddings using Azure AI Foundry")
        embeddings = generate_embeddings(chunks, config)
        
        if not embeddings or len(embeddings) != len(chunks):
            doc_service.update_document_status(document_id, "uploaded")
            raise HTTPException(
                status_code=500,
                detail=f"Embedding generation failed: expected {len(chunks)} embeddings, got {len(embeddings) if embeddings else 0}"
            )
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Step 7: Index chunks and embeddings into Azure AI Search
        logger.info("Step 7: Indexing chunks and embeddings into Azure AI Search")
        index_chunks(chunks, embeddings, config)
        
        logger.info(f"Successfully indexed {len(chunks)} chunks into Azure AI Search")
        
        # Step 8: Update document status and chunks count
        doc_service.update_document_status(document_id, "processed")
        doc_service.update_document_chunks(document_id, len(chunks))
        
        return UploadResponse(
            document_id=document_id,
            status="success",
            message=f"Document processed and indexed successfully",
            chunks_created=len(chunks)
        )
        
    except HTTPException:
        # Update status on error if document was created
        if document_id:
            try:
                doc_service.update_document_status(document_id, "uploaded")
            except Exception:
                pass  # Ignore errors when updating status on failure
        raise
    except NotImplementedError as e:
        logger.error(f"Upload processing failed - feature not implemented: {e}")
        if document_id:
            try:
                doc_service.update_document_status(document_id, "uploaded")
            except Exception:
                pass
        raise HTTPException(status_code=501, detail=f"Feature not yet implemented: {str(e)}")
    except Exception as e:
        logger.error(f"Upload processing failed: {e}", exc_info=True)
        if document_id:
            try:
                doc_service.update_document_status(document_id, "uploaded")
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")



