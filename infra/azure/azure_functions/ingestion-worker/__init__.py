"""Azure Function for ingestion worker

This function is triggered by messages in the ingestion-uploads queue.
It processes documents through the ingestion stage (text extraction).
"""

import logging
import sys
from pathlib import Path

# Add backend to Python path
# From infra/azure/azure_functions/ingestion-worker/__init__.py
# Go up 4 levels to project root, then add backend
# Note: build.sh updates this path during Git deployment
backend_dir = Path(__file__).parent.parent.parent.parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from rag_eval.services.workers.ingestion_worker import ingestion_worker
from rag_eval.core.logging import get_logger

logger = get_logger("azure_functions.ingestion_worker")


def main(queueMessage: str) -> None:
    """
    Azure Function entry point for ingestion worker.
    
    Args:
        queueMessage: JSON string from queue trigger
    """
    import json
    
    try:
        # Parse queue message
        message_dict = json.loads(queueMessage)
        
        logger.info(f"Processing ingestion message for document: {message_dict.get('document_id')}")
        
        # Call worker function
        ingestion_worker(message_dict, context=None)
        
        logger.info(f"Successfully processed ingestion for document: {message_dict.get('document_id')}")
    
    except Exception as e:
        logger.error(f"Error processing ingestion message: {e}", exc_info=True)
        raise

