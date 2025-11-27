"""Answer generation via Azure AI Foundry"""

from typing import List
from rag_eval.core.interfaces import Query, RetrievalResult, ModelAnswer
from rag_eval.core.exceptions import AzureServiceError
from rag_eval.core.logging import get_logger

logger = get_logger("services.rag.generation")


def generate_answer(
    query: Query,
    retrieved_chunks: List[RetrievalResult],
    prompt_version: str,
    config
) -> ModelAnswer:
    """
    Generate an answer using Azure AI Foundry.
    
    Args:
        query: Query object
        retrieved_chunks: Retrieved context chunks
        prompt_version: Version of the prompt to use
        config: Application configuration
        
    Returns:
        ModelAnswer object
        
    Raises:
        AzureServiceError: If generation fails
    """
    logger.info(f"Generating answer for query: {query.text} using prompt version: {prompt_version}")
    # TODO: Implement Azure AI Foundry answer generation
    raise NotImplementedError("Answer generation not yet implemented")

