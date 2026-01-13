"""Azure Function for ingestion worker

This function is triggered by messages in the ingestion-uploads queue.
It processes documents through the ingestion stage (text extraction).
"""
import json
import sys
from pathlib import Path

# Add backend directory to Python path for imports
# Deployment structure: /home/site/wwwroot/<function-name>/__init__.py
# Backend code is in: /home/site/wwwroot/backend/src/
# So we need to add /home/site/wwwroot/backend to sys.path
backend_dir = Path(__file__).parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from src.services.workers.ingestion_worker import ingestion_worker
from src.core.logging import get_logger

logger = get_logger("azure_functions.ingestion_worker")


def main(queueMessage: str) -> None:
    """
    Azure Function entry point for ingestion worker.
    
    Args:
        queueMessage: JSON string from queue trigger
    """
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

