"""RAG pipeline engine"""

from rag_eval.services.rag.ingestion import (
    ingest_document,
    extract_text_from_document,
)
from rag_eval.services.rag.chunking import (
    chunk_text,
    chunk_text_fixed_size,
)
from rag_eval.services.rag.storage import (
    upload_document_to_blob,
    download_document_from_blob,
)
from rag_eval.services.rag.embeddings import (
    generate_embeddings,
    generate_query_embedding,
)
from rag_eval.services.rag.generation import (
    load_prompt_template,
    construct_prompt,
)
from rag_eval.services.rag.logging import (
    log_query,
    log_retrieval,
    log_model_answer,
)

__all__ = [
    # Ingestion
    "ingest_document",
    "extract_text_from_document",
    # Chunking
    "chunk_text",
    "chunk_text_fixed_size",
    # Storage
    "upload_document_to_blob",
    "download_document_from_blob",
    # Embeddings
    "generate_embeddings",
    "generate_query_embedding",
    # Generation
    "load_prompt_template",
    "construct_prompt",
    # Logging
    "log_query",
    "log_retrieval",
    "log_model_answer",
]
