"""Setup script for Azure Storage Queues

This script creates all required queues in Azure Storage Account.
Queues are created idempotently (no error if they already exist).
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from azure.storage.queue import QueueServiceClient
from azure.core.exceptions import AzureError
from rag_eval.core.config import Config
from rag_eval.core.logging import get_logger

logger = get_logger("scripts.setup_azure_queues")

# Queue names as specified in RFC001.md
QUEUE_NAMES = [
    "ingestion-uploads",
    "ingestion-chunking",
    "ingestion-embeddings",
    "ingestion-indexing",
    "ingestion-dead-letter"
]


def setup_queues():
    """Create all required queues in Azure Storage Account"""
    
    # Load config
    try:
        config = Config.from_env()
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        print("Make sure .env.local has AZURE_BLOB_CONNECTION_STRING set")
        return False
    
    # Try to get connection string from config or environment
    import os
    connection_string = config.azure_blob_connection_string or os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING", "")
    
    if not connection_string:
        print("❌ Azure Storage connection string not found")
        print("Please set AZURE_BLOB_CONNECTION_STRING in config or AZURE_STORAGE_QUEUES_CONNECTION_STRING in .env.local")
        return False
    
    print("🔧 Setting up Azure Storage Queues...")
    print(f"   Connection: {connection_string.split('AccountName=')[1].split(';')[0] if 'AccountName=' in connection_string else 'Unknown'}")
    print()
    
    try:
        # Create queue service client
        queue_service_client = QueueServiceClient.from_connection_string(connection_string)
        
        created_count = 0
        existing_count = 0
        error_count = 0
        
        for queue_name in QUEUE_NAMES:
            try:
                queue_client = queue_service_client.get_queue_client(queue_name)
                queue_client.create_queue()
                print(f"   ✅ Created queue: {queue_name}")
                created_count += 1
            except AzureError as e:
                error_code = getattr(e, 'error_code', None) or str(e)
                if "QueueAlreadyExists" in str(e) or "QUEUE_ALREADY_EXISTS" in str(error_code) or "409" in str(e):
                    print(f"   ℹ️  Queue already exists: {queue_name}")
                    existing_count += 1
                else:
                    print(f"   ❌ Failed to create queue '{queue_name}': {e}")
                    error_count += 1
            except Exception as e:
                # Check if it's a QueueAlreadyExists error in the exception message
                if "QueueAlreadyExists" in str(e) or "QUEUE_ALREADY_EXISTS" in str(e) or "409" in str(e):
                    print(f"   ℹ️  Queue already exists: {queue_name}")
                    existing_count += 1
                else:
                    print(f"   ❌ Unexpected error creating queue '{queue_name}': {e}")
                    error_count += 1
        
        print()
        print("📊 Summary:")
        print(f"   Created: {created_count}")
        print(f"   Already existed: {existing_count}")
        print(f"   Errors: {error_count}")
        print()
        
        if error_count == 0:
            print("✅ All queues set up successfully!")
            return True
        else:
            print(f"⚠️  {error_count} queue(s) failed to create")
            return False
            
    except Exception as e:
        print(f"❌ Failed to connect to Azure Storage: {e}")
        print("   Check your connection string and network connectivity")
        return False


if __name__ == "__main__":
    success = setup_queues()
    sys.exit(0 if success else 1)

