"""Azure Function for embedding worker

This function is triggered by messages in the ingestion-embeddings queue.
It processes documents through the embedding generation stage.
"""

import logging
import sys
from pathlib import Path

# Load .env.local from project root BEFORE importing backend code
# From infra/azure/azure_functions/embedding-worker/__init__.py
# Go up 5 levels to project root: embedding-worker -> azure_functions -> azure -> infra -> project_root
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.parent.parent.parent
    env_file = project_root / ".env.local"
    if env_file.exists():
        load_dotenv(env_file, override=True)
except ImportError:
    # dotenv not available, continue without loading
    pass

# Add backend directory to Python path for backend imports
# This allows importing directly from rag_eval.* (backend/rag_eval/)
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from rag_eval.services.workers.embedding_worker import embedding_worker
from rag_eval.core.logging import get_logger

logger = get_logger("azure_functions.embedding_worker")


def main(queueMessage: str) -> None:
    """
    Azure Function entry point for embedding worker.
    
    Args:
        queueMessage: JSON string from queue trigger
    """
    import json
    
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

