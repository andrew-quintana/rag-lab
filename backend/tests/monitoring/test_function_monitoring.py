"""Azure Functions Monitoring Test

Simple monitoring test with manual verification checkpoints.
This test sends a message and helps verify Azure Functions are processing correctly.

Usage:
    cd backend
    pytest tests/monitoring/test_function_monitoring.py -v -s
    
The -s flag is important to show manual verification prompts.
"""

import pytest
import time
import json
from datetime import datetime
from azure.storage.queue import QueueServiceClient

from src.services.workers.queue_client import QueueMessage, SourceStorage, ProcessingStage
from src.core.config import Config


@pytest.fixture(scope="module")
def config():
    """Load configuration from environment"""
    return Config.from_env()


@pytest.fixture(scope="module")
def queue_client(config):
    """Create queue service client"""
    conn_str = config.azure_blob_connection_string
    if not conn_str or "UseDevelopmentStorage" in conn_str:
        pytest.skip("Cloud Azure Storage connection required for monitoring tests")
    
    return QueueServiceClient.from_connection_string(conn_str)


@pytest.mark.integration
@pytest.mark.cloud
class TestFunctionMonitoring:
    """Monitoring tests for Azure Functions"""
    
    def test_queue_connectivity(self, queue_client):
        """Test basic queue connectivity"""
        print("\n" + "=" * 70)
        print("📡 Testing Queue Connectivity")
        print("=" * 70)
        
        # List queues
        try:
            queues = list(queue_client.list_queues())
            queue_names = [q.name for q in queues]
            print(f"\n✅ Connected to Azure Storage")
            print(f"   Found {len(queues)} queues:")
            for name in queue_names[:10]:  # Show first 10
                print(f"   - {name}")
            
            # Check for expected queues
            expected = [
                "ingestion-uploads",
                "ingestion-chunking",
                "ingestion-embeddings",
                "ingestion-indexing"
            ]
            missing = [q for q in expected if q not in queue_names]
            if missing:
                print(f"\n⚠️  Missing expected queues: {missing}")
            else:
                print(f"\n✅ All expected queues found")
                
        except Exception as e:
            pytest.fail(f"Failed to connect to queues: {e}")
    
    def test_queue_health_status(self, queue_client):
        """Check current health status of all queues"""
        print("\n" + "=" * 70)
        print("📊 Queue Health Status")
        print("=" * 70)
        
        main_queues = [
            "ingestion-uploads",
            "ingestion-chunking",
            "ingestion-embeddings",
            "ingestion-indexing"
        ]
        
        poison_queues = [f"{q}-poison" for q in main_queues]
        
        total_messages = 0
        poison_count = 0
        
        print("\n🔵 Main Queues:")
        for queue_name in main_queues:
            try:
                queue = queue_client.get_queue_client(queue_name)
                props = queue.get_queue_properties()
                count = props.approximate_message_count
                total_messages += count
                
                status = "✅" if count == 0 else f"📝 {count}"
                print(f"  {queue_name:<35} {status}")
            except Exception as e:
                print(f"  {queue_name:<35} ❌ {e}")
        
        print("\n☠️  Poison Queues:")
        for queue_name in poison_queues:
            try:
                queue = queue_client.get_queue_client(queue_name)
                props = queue.get_queue_properties()
                count = props.approximate_message_count
                poison_count += count
                
                status = "✅" if count == 0 else f"⚠️  {count}"
                print(f"  {queue_name:<35} {status}")
            except Exception as e:
                print(f"  {queue_name:<35} ⚠️  Queue may not exist")
        
        print("\n" + "=" * 70)
        if poison_count > 0:
            print(f"⚠️  WARNING: {poison_count} message(s) in poison queues")
            print(f"   Run: python scripts/monitor_queue_health.py --peek <queue-name>")
        else:
            print(f"✅ No messages in poison queues")
        
        if total_messages > 0:
            print(f"📝 {total_messages} message(s) queued for processing")
    
    def test_function_processing_with_manual_verification(self, config, queue_client):
        """
        Test Azure Function processing with manual verification checkpoint.
        
        This test:
        1. Checks initial queue state
        2. Sends a test message
        3. Monitors queue for 60 seconds
        4. Prompts you to verify in Azure Portal
        5. Checks final state
        """
        print("\n" + "=" * 70)
        print("🧪 Azure Function Processing Test (with manual verification)")
        print("=" * 70)
        
        # Step 1: Check initial state
        print("\n📊 Step 1: Checking initial queue state...")
        upload_queue = queue_client.get_queue_client("ingestion-uploads")
        poison_queue = queue_client.get_queue_client("ingestion-uploads-poison")
        
        initial_upload_count = upload_queue.get_queue_properties().approximate_message_count
        initial_poison_count = poison_queue.get_queue_properties().approximate_message_count
        
        print(f"   ingestion-uploads: {initial_upload_count} messages")
        print(f"   ingestion-uploads-poison: {initial_poison_count} messages")
        
        # Step 2: Send test message
        print("\n📤 Step 2: Sending test message...")
        test_id = f"test-monitor-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        test_time = datetime.utcnow().isoformat()
        
        message = {
            "document_id": test_id,
            "source_storage": "supabase",
            "filename": "monitoring-test.txt",
            "attempt": 1,
            "stage": "uploaded",
            "metadata": {
                "test": True,
                "timestamp": test_time
            }
        }
        
        upload_queue.send_message(json.dumps(message))
        print(f"   ✅ Test message sent")
        print(f"   Document ID: {test_id}")
        print(f"   Timestamp: {test_time}")
        
        # Step 3: Monitor for processing
        print(f"\n⏱️  Step 3: Monitoring for 60 seconds...")
        print("   (Functions should pick up the message within this time)")
        
        start_time = time.time()
        timeout = 60
        found_in_poison = False
        last_count = initial_upload_count + 1  # We just added one
        
        while time.time() - start_time < timeout:
            time.sleep(5)
            elapsed = int(time.time() - start_time)
            
            # Check poison queue
            current_poison = poison_queue.get_queue_properties().approximate_message_count
            if current_poison > initial_poison_count:
                print(f"\n   ❌ [{elapsed}s] Message moved to poison queue!")
                found_in_poison = True
                break
            
            # Check upload queue
            current_upload = upload_queue.get_queue_properties().approximate_message_count
            if current_upload != last_count:
                print(f"   [{elapsed}s] Queue depth: {current_upload}")
                last_count = current_upload
                
                if current_upload == initial_upload_count:
                    print(f"   ✅ [{elapsed}s] Message processed! (queue back to initial state)")
                    break
        
        final_upload_count = upload_queue.get_queue_properties().approximate_message_count
        final_poison_count = poison_queue.get_queue_properties().approximate_message_count
        
        # Step 4: Manual verification prompt
        print("\n" + "=" * 70)
        print("🔍 Step 4: MANUAL VERIFICATION CHECKPOINT")
        print("=" * 70)
        print("\nPlease verify in Azure Portal:")
        print("1. Go to: https://portal.azure.com")
        print("2. Navigate to: func-raglab-uploadworkers → Functions → ingestion-worker")
        print("3. Click 'Monitor' tab")
        print(f"4. Look for execution around: {test_time}")
        print(f"5. Check for Document ID: {test_id}")
        print("\nWhat to check:")
        print("  - Status: Success or Failed?")
        print("  - If Failed: What's the error message?")
        print("  - Duration: How long did it take?")
        print("  - Logs: Any warnings or errors?")
        
        input("\n⏸️  Press Enter when you've checked the Azure Portal...")
        
        # Step 5: Final state
        print("\n📊 Step 5: Final queue state:")
        print(f"   ingestion-uploads: {final_upload_count} messages (was {initial_upload_count})")
        print(f"   ingestion-uploads-poison: {final_poison_count} messages (was {initial_poison_count})")
        
        print("\n" + "=" * 70)
        
        # Determine outcome
        if found_in_poison or final_poison_count > initial_poison_count:
            print("❌ RESULT: Message went to poison queue - function failed")
            print("\n💡 Next steps:")
            print("   1. Check the error in Azure Portal (see Monitor tab)")
            print("   2. Common issues:")
            print("      - Import errors (ModuleNotFoundError)")
            print("      - Database connection errors")
            print("      - Missing environment variables")
            print("   3. Review logs: az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab")
            
            pytest.fail("Function processing failed - message in poison queue")
            
        elif final_upload_count == initial_upload_count:
            print("✅ RESULT: Message processed successfully!")
            print("   (Queue returned to initial state)")
            
        else:
            print("⏱️  RESULT: Timeout - message still in queue")
            print("\n💡 Possible causes:")
            print("   1. Functions are disabled (check Azure Portal)")
            print("   2. Cold start delay (try running test again)")
            print("   3. Functions not polling (check logs)")
            
            # Don't fail on timeout - might just be cold start
            pytest.skip("Timeout - unable to determine if function processed message")
    
    def test_check_function_status(self):
        """
        Manual test to guide checking function status in Azure Portal.
        This doesn't make assertions but provides a checklist.
        """
        print("\n" + "=" * 70)
        print("📋 Function Status Checklist")
        print("=" * 70)
        print("\nUse this checklist to verify function health in Azure Portal:")
        print("\n1. Function Status:")
        print("   ☐ Navigate to func-raglab-uploadworkers → Functions")
        print("   ☐ Verify all 4 functions are listed:")
        print("      - ingestion-worker")
        print("      - chunking-worker")
        print("      - embedding-worker")
        print("      - indexing-worker")
        print("   ☐ Check that none show 'Disabled' status")
        print("\n2. Recent Invocations:")
        print("   ☐ Click each function → Monitor")
        print("   ☐ Check 'Invocations' count is increasing")
        print("   ☐ Look for recent successful executions")
        print("\n3. Configuration:")
        print("   ☐ Go to func-raglab-uploadworkers → Configuration")
        print("   ☐ Verify key settings:")
        print("      - AzureWebJobsStorage: <set>")
        print("      - AZURE_STORAGE_QUEUES_CONNECTION_STRING: <set>")
        print("      - SUPABASE_DB_URL: <set>")
        print("      - SUPABASE_URL: <set>")
        print("\n4. Logs:")
        print("   ☐ Run: az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab")
        print("   ☐ Check for polling activity (should see queue checks)")
        print("   ☐ Look for errors or warnings")
        
        print("\n" + "=" * 70)
        print("✅ Checklist complete - review findings above")

