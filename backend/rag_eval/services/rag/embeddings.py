"""Embedding generation via Azure AI Foundry"""

from typing import List
from rag_eval.core.interfaces import Chunk
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.logging import get_logger

logger = get_logger("services.rag.embeddings")


def generate_embeddings(chunks: List[Chunk], config) -> List[List[float]]:
    """
    Generate embeddings for chunks using Azure AI Foundry.
    
    Args:
        chunks: List of chunks to embed
        config: Application configuration
        
    Returns:
        List of embedding vectors
        
    Raises:
        AzureServiceError: If embedding generation fails
    """
    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    # TODO: Implement Azure AI Foundry embedding generation
    raise NotImplementedError("Embedding generation not yet implemented")

