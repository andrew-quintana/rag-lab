"""RAG pipeline engine"""

from rag_eval.services.rag.ingestion import (
    ingest_document,
    extract_text_from_document,
)
from rag_eval.services.rag.chunking import (
    chunk_text,
    chunk_text_fixed_size,
    chunk_text_with_llm,
)
from rag_eval.services.rag.storage import (
    upload_document_to_blob,
    download_document_from_blob,
)

__all__ = [
    # Ingestion
    "ingest_document",
    "extract_text_from_document",
    # Chunking
    "chunk_text",
    "chunk_text_fixed_size",
    "chunk_text_with_llm",
    # Storage
    "upload_document_to_blob",
    "download_document_from_blob",
]
