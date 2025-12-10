"""Azure Function for indexing worker

This function is triggered by messages in the ingestion-indexing queue.
It processes documents through the indexing stage (Azure AI Search).
"""
import json
from src.services.workers.indexing_worker import indexing_worker
from src.core.logging import get_logger

logger = get_logger("azure_functions.indexing_worker")


def main(queueMessage: str) -> None:
    """
    Azure Function entry point for indexing worker.
    
    Args:
        queueMessage: JSON string from queue trigger
    """
    try:
        # Parse queue message
        message_dict = json.loads(queueMessage)
        
        logger.info(f"Processing indexing message for document: {message_dict.get('document_id')}")
        
        # Call worker function
        indexing_worker(message_dict, context=None)
        
        logger.info(f"Successfully processed indexing for document: {message_dict.get('document_id')}")
    
    except Exception as e:
        logger.error(f"Error processing indexing message: {e}", exc_info=True)
        raise

