#!/usr/bin/env python3
"""Send a test message to ingestion-uploads queue"""

import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from azure.storage.queue import QueueServiceClient

# Load environment (prefer .env.prod for cloud)
env_file = Path(__file__).parent.parent / ".env.prod"
if not env_file.exists():
    env_file = Path(__file__).parent.parent / ".env.local"
if env_file.exists():
    load_dotenv(env_file)

def get_connection_string():
    """Get Azure Storage connection string"""
    conn_str = (
        os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING") or
        os.getenv("AZURE_BLOB_CONNECTION_STRING") or
        os.getenv("AzureWebJobsStorage")
    )
    
    if not conn_str:
        print("❌ Error: No Azure Storage connection string found")
        sys.exit(1)
    
    return conn_str

def send_test_message():
    """Send a test message to ingestion-uploads queue"""
    print("\n" + "=" * 70)
    print("🧪 Sending Test Message to ingestion-uploads Queue")
    print("=" * 70)
    
    conn_str = get_connection_string()
    client = QueueServiceClient.from_connection_string(conn_str)
    
    # Create test message
    test_doc_id = str(uuid.uuid4())
    message = {
        "document_id": test_doc_id,
        "source_storage": "supabase",
        "filename": "test-database-connection.pdf",
        "attempt": 1,
        "stage": "uploaded",
        "metadata": {
            "test": True,
            "timestamp": datetime.utcnow().isoformat(),
            "purpose": "database_connection_test"
        }
    }
    
    try:
        queue = client.get_queue_client("ingestion-uploads")
        queue.send_message(json.dumps(message))
        print(f"✅ Test message sent successfully")
        print(f"   Document ID: {test_doc_id}")
        print(f"   Timestamp: {datetime.utcnow().isoformat()}")
        print(f"\n💡 Monitor logs with: make logs-ingestion TIME_WINDOW=5m")
        print(f"💡 Check queue: python3 scripts/monitor_queue_health.py")
        return test_doc_id
    except Exception as e:
        print(f"❌ Failed to send test message: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    doc_id = send_test_message()
    if doc_id:
        print(f"\n✅ Test message queued! Document ID: {doc_id}")
        sys.exit(0)
    else:
        print("\n❌ Failed to send test message")
        sys.exit(1)
