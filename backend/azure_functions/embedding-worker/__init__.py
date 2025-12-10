"""Azure Function for embedding worker

This function is triggered by messages in the ingestion-embeddings queue.
It processes documents through the embedding generation stage.
"""
import json
from src.services.workers.embedding_worker import embedding_worker
from src.core.logging import get_logger

logger = get_logger("azure_functions.embedding_worker")


def main(queueMessage: str) -> None:
    """
    Azure Function entry point for embedding worker.
    
    Args:
        queueMessage: JSON string from queue trigger
    """
    try:
        # Parse queue message
        message_dict = json.loads(queueMessage)
        
        logger.info(f"Processing embedding message for document: {message_dict.get('document_id')}")
        
        # Call worker function
        embedding_worker(message_dict, context=None)
        
        logger.info(f"Successfully processed embedding for document: {message_dict.get('document_id')}")
    
    except Exception as e:
        logger.error(f"Error processing embedding message: {e}", exc_info=True)
        raise

