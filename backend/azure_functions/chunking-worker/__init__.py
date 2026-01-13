"""Azure Function for chunking worker

This function is triggered by messages in the ingestion-chunking queue.
It processes documents through the chunking stage.
"""
import json
import sys
from pathlib import Path

# Add backend directory to Python path for imports
# With Git deployment, functions are in backend/azure_functions/ and src is in backend/src/
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from src.services.workers.chunking_worker import chunking_worker
from src.core.logging import get_logger

logger = get_logger("azure_functions.chunking_worker")


def main(queueMessage: str) -> None:
    """
    Azure Function entry point for chunking worker.
    
    Args:
        queueMessage: JSON string from queue trigger
    """
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

