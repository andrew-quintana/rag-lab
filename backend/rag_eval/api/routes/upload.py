"""POST /upload endpoint for document ingestion"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from rag_eval.core.config import Config
from rag_eval.core.logging import get_logger
from rag_eval.services.rag.ingestion import ingest_document
from rag_eval.services.rag.chunking import chunk_text
from rag_eval.services.rag.embeddings import generate_embeddings
from rag_eval.services.rag.search import index_chunks
from rag_eval.utils.ids import generate_id

logger = get_logger("api.routes.upload")
router = APIRouter()

config = Config.from_env()


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
    
    Processing flow (entirely serial, in-memory):
    1. Azure Document Intelligence (OCR, table extraction, text segmentation)
    2. Chunking (fixed-size, deterministic)
    3. Azure AI Foundry embedding generation
    4. Azure AI Search indexing (chunks, embeddings, metadata)
    
    Note: Documents are processed in-memory without persistence to blob storage.
    The file content is read directly and processed through the pipeline.
    
    Args:
        file: Uploaded file (PDF, images, documents)
        
    Returns:
        UploadResponse with document ID and processing status
    """
    logger.info(f"Received upload request for file: {file.filename}")
    
    try:
        # Generate document ID
        document_id = generate_id()
        
        # Read file content
        file_content = await file.read()
        logger.info(f"Read {len(file_content)} bytes from file: {file.filename}")
        
        # Step 1: Extract text using Azure Document Intelligence
        logger.info("Step 1: Extracting text using Azure Document Intelligence")
        extracted_text = ingest_document(file_content, config)
        
        if not extracted_text or len(extracted_text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the document"
            )
        
        logger.info(f"Extracted {len(extracted_text)} characters of text")
        
        # Step 2: Chunk text using Azure AI Foundry (free tier OpenAI)
        logger.info("Step 2: Chunking text using Azure AI Foundry")
        chunks = chunk_text(extracted_text, config, document_id=document_id)
        
        if not chunks:
            raise HTTPException(
                status_code=500,
                detail="Failed to create chunks from extracted text"
            )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings using Azure AI Foundry
        logger.info("Step 3: Generating embeddings using Azure AI Foundry")
        embeddings = generate_embeddings(chunks, config)
        
        if not embeddings or len(embeddings) != len(chunks):
            raise HTTPException(
                status_code=500,
                detail=f"Embedding generation failed: expected {len(chunks)} embeddings, got {len(embeddings) if embeddings else 0}"
            )
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Step 4: Index chunks and embeddings into Azure AI Search
        logger.info("Step 4: Indexing chunks and embeddings into Azure AI Search")
        index_chunks(chunks, embeddings, config)
        
        logger.info(f"Successfully indexed {len(chunks)} chunks into Azure AI Search")
        
        return UploadResponse(
            document_id=document_id,
            status="success",
            message=f"Document processed and indexed successfully",
            chunks_created=len(chunks)
        )
        
    except HTTPException:
        raise
    except NotImplementedError as e:
        logger.error(f"Upload processing failed - feature not implemented: {e}")
        raise HTTPException(status_code=501, detail=f"Feature not yet implemented: {str(e)}")
    except Exception as e:
        logger.error(f"Upload processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")



