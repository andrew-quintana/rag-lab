"""Azure Function for chunking worker

This function is triggered by messages in the ingestion-chunking queue.
It processes documents through the chunking stage.
"""

import logging
import sys
from pathlib import Path

# Add backend to Python path
# From infra/azure/azure_functions/chunking-worker/__init__.py
# Go up 4 levels to project root, then add backend
backend_dir = Path(__file__).parent.parent.parent.parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from rag_eval.services.workers.chunking_worker import chunking_worker
from rag_eval.core.logging import get_logger

logger = get_logger("azure_functions.chunking_worker")


def main(queueMessage: str) -> None:
    """
    Azure Function entry point for chunking worker.
    
    Args:
        queueMessage: JSON string from queue trigger
    """
    import json
    
    try:
        # Parse queue message
        message_dict = json.loads(queueMessage)
        
        logger.info(f"Processing chunking message for document: {message_dict.get('document_id')}")
        
        # Call worker function
        chunking_worker(message_dict, context=None)
        
        logger.info(f"Successfully processed chunking for document: {message_dict.get('document_id')}")
    
    except Exception as e:
        logger.error(f"Error processing chunking message: {e}", exc_info=True)
        raise

