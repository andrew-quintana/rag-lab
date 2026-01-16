#!/usr/bin/env python3
"""Quick diagnostic script for Azure Functions

This script performs a rapid health check of Azure Functions:
1. Checks queue status (main + poison queues)
2. Sends a test message
3. Monitors for 60 seconds
4. Reports outcome with next steps
5. Prompts for manual log check if needed

Usage:
    python scripts/diagnose_functions.py
    python scripts/diagnose_functions.py --skip-test  # Only check queue status
"""

import os
import sys
import time
import json
import argparse
import uuid
from datetime import datetime
from pathlib import Path

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from azure.storage.queue import QueueServiceClient
from dotenv import load_dotenv


# Load environment
env_file = Path(__file__).parent.parent / ".env.local"
if env_file.exists():
    load_dotenv(env_file)


QUEUE_NAMES = [
    "ingestion-uploads",
    "ingestion-chunking",
    "ingestion-embeddings",
    "ingestion-indexing"
]

POISON_QUEUES = [f"{q}-poison" for q in QUEUE_NAMES]


def get_connection_string():
    """Get Azure Storage connection string from environment"""
    conn_str = (
        os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING") or
        os.getenv("AZURE_BLOB_CONNECTION_STRING") or
        os.getenv("AzureWebJobsStorage")
    )
    
    if not conn_str:
        print("❌ Error: No Azure Storage connection string found")
        print("   Set one of: AZURE_STORAGE_QUEUES_CONNECTION_STRING, AZURE_BLOB_CONNECTION_STRING, AzureWebJobsStorage")
        sys.exit(1)
    
    return conn_str


def check_queue_health():
    """Check health of all queues"""
    print("=" * 60)
    print("📊 Queue Health Status")
    print("=" * 60)
    
    conn_str = get_connection_string()
    client = QueueServiceClient.from_connection_string(conn_str)
    
    queue_status = {}
    total_messages = 0
    poison_messages = 0
    
    # Check main queues
    print("\n🔵 Main Queues:")
    for queue_name in QUEUE_NAMES:
        try:
            queue = client.get_queue_client(queue_name)
            props = queue.get_queue_properties()
            count = props.approximate_message_count
            queue_status[queue_name] = count
            total_messages += count
            
            status = "✅" if count == 0 else f"📝 {count} msg(s)"
            print(f"  {queue_name:<30} {status}")
        except Exception as e:
            print(f"  {queue_name:<30} ❌ Error: {e}")
            queue_status[queue_name] = -1
    
    # Check poison queues
    print("\n☠️  Poison Queues:")
    poison_details = []
    for queue_name in POISON_QUEUES:
        try:
            queue = client.get_queue_client(queue_name)
            props = queue.get_queue_properties()
            count = props.approximate_message_count
            queue_status[queue_name] = count
            poison_messages += count
            
            if count > 0:
                status = f"⚠️  {count} msg(s)"
                print(f"  {queue_name:<30} {status}")
                
                # Peek at message
                messages = list(queue.peek_messages(max_messages=1))
                if messages:
                    msg = messages[0]
                    try:
                        content = json.loads(msg.content)
                        doc_id = content.get('document_id', 'unknown')
                        poison_details.append({
                            'queue': queue_name,
                            'doc_id': doc_id,
                            'content': msg.content[:100]
                        })
                    except:
                        poison_details.append({
                            'queue': queue_name,
                            'doc_id': 'parse error',
                            'content': msg.content[:100]
                        })
            else:
                print(f"  {queue_name:<30} ✅")
        except Exception as e:
            print(f"  {queue_name:<30} ⚠️  Queue may not exist")
            queue_status[queue_name] = 0
    
    # Summary
    print("\n" + "=" * 60)
    if poison_messages > 0:
        print(f"⚠️  ISSUES DETECTED: {poison_messages} message(s) in poison queues")
        print("\n💡 Poison Queue Details:")
        for detail in poison_details:
            print(f"  Queue: {detail['queue']}")
            print(f"  Doc ID: {detail['doc_id']}")
            print(f"  Preview: {detail['content']}...")
            print()
    elif total_messages > 0:
        print(f"📝 {total_messages} message(s) queued for processing")
    else:
        print("✅ All queues are empty and healthy")
    
    print("=" * 60)
    
    return queue_status, poison_messages


def send_test_message():
    """Send a test message to ingestion-uploads queue"""
    print("\n" + "=" * 60)
    print("🧪 Sending Test Message")
    print("=" * 60)
    
    conn_str = get_connection_string()
    client = QueueServiceClient.from_connection_string(conn_str)
    
    # Create test message with valid UUID
    test_doc_id = str(uuid.uuid4())
    message = {
        "document_id": test_doc_id,
        "source_storage": "supabase",
        "filename": "diagnostic-test.txt",
        "attempt": 1,
        "stage": "uploaded",
        "metadata": {
            "test": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    try:
        queue = client.get_queue_client("ingestion-uploads")
        queue.send_message(json.dumps(message))
        print(f"✅ Test message sent successfully")
        print(f"   Document ID: {test_doc_id}")
        print(f"   Timestamp: {datetime.utcnow().isoformat()}")
        return test_doc_id
    except Exception as e:
        print(f"❌ Failed to send test message: {e}")
        return None


def monitor_test_message(doc_id, timeout=60):
    """Monitor for test message processing"""
    print("\n" + "=" * 60)
    print(f"⏱️  Monitoring for {timeout} seconds...")
    print("=" * 60)
    
    conn_str = get_connection_string()
    client = QueueServiceClient.from_connection_string(conn_str)
    
    start_time = time.time()
    last_status = {}
    
    print("\nQueue Status Updates:")
    print("(Checking every 5 seconds...)")
    
    while time.time() - start_time < timeout:
        time.sleep(5)
        elapsed = int(time.time() - start_time)
        
        # Check if message moved to poison queue
        for poison_queue in POISON_QUEUES:
            try:
                queue = client.get_queue_client(poison_queue)
                messages = list(queue.peek_messages(max_messages=10))
                for msg in messages:
                    try:
                        content = json.loads(msg.content)
                        if content.get('document_id') == doc_id:
                            print(f"\n❌ [{elapsed}s] Test message found in poison queue: {poison_queue}")
                            print("   This indicates the function failed to process the message.")
                            return "FAILED"
                    except:
                        pass
            except:
                pass
        
        # Check main queue depth
        try:
            queue = client.get_queue_client("ingestion-uploads")
            props = queue.get_queue_properties()
            count = props.approximate_message_count
            
            if count != last_status.get("ingestion-uploads", -1):
                print(f"   [{elapsed}s] ingestion-uploads: {count} messages")
                last_status["ingestion-uploads"] = count
        except:
            pass
    
    print(f"\n⏱️  Monitoring timeout reached ({timeout}s)")
    return "TIMEOUT"


def provide_next_steps(result, poison_count):
    """Provide actionable next steps based on diagnostic results"""
    print("\n" + "=" * 60)
    print("📋 Next Steps")
    print("=" * 60)
    
    if result == "FAILED":
        print("\n❌ Function processing failed - message went to poison queue")
        print("\n🔍 Recommended actions:")
        print("1. Check Azure Function logs for errors:")
        print("   az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab")
        print("\n2. Or view logs in Azure Portal:")
        print("   Portal → func-raglab-uploadworkers → Functions → ingestion-worker → Monitor")
        print("\n3. Common issues to check:")
        print("   - Import errors (ModuleNotFoundError)")
        print("   - Database connection errors")
        print("   - Missing environment variables")
        print("   - Message encoding issues (should be fixed with host.json update)")
        
    elif result == "TIMEOUT":
        print("\n⏱️  Function did not process message within timeout")
        print("\n🔍 Possible causes:")
        print("1. Functions are disabled:")
        print("   - Check in Azure Portal → func-raglab-uploadworkers → Functions")
        print("   - Verify functions are not disabled")
        print("\n2. Functions are not polling:")
        print("   - Check function logs for polling activity")
        print("   - Verify AzureWebJobsStorage connection string is correct")
        print("\n3. Cold start delay:")
        print("   - First execution after deployment can take 2-3 minutes")
        print("   - Try running diagnostic again")
    
    if poison_count > 0:
        print("\n☠️  Poison queue cleanup:")
        print(f"   {poison_count} message(s) in poison queues need attention")
        print("   Review logs to identify root cause before clearing")
        print("   To clear poison queue:")
        print("   python scripts/monitor_queue_health.py --clear-poison")


def main():
    parser = argparse.ArgumentParser(description="Diagnose Azure Functions health")
    parser.add_argument("--skip-test", action="store_true", 
                       help="Skip sending test message, only check queue status")
    args = parser.parse_args()
    
    print("\n🔍 Azure Functions Diagnostic Tool")
    print(f"   Timestamp: {datetime.utcnow().isoformat()}")
    
    # Step 1: Check queue health
    queue_status, poison_count = check_queue_health()
    
    if args.skip_test:
        print("\n✅ Diagnostic complete (test skipped)")
        return
    
    # Step 2: Send test message
    test_doc_id = send_test_message()
    if not test_doc_id:
        print("\n❌ Cannot proceed without test message")
        return
    
    # Step 3: Monitor
    result = monitor_test_message(test_doc_id, timeout=60)
    
    # Step 4: Next steps
    provide_next_steps(result, poison_count)
    
    print("\n" + "=" * 60)
    print("✅ Diagnostic complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

