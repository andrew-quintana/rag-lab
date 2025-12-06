"""Integration test script for Azure Storage Queue operations

This script tests the queue client with real Azure Storage Queues.
Run this after creating queues (via setup_azure_queues.py or Azure Portal).
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_eval.core.config import Config
from rag_eval.services.workers.queue_client import (
    QueueMessage,
    enqueue_message,
    dequeue_message,
    peek_message,
    get_queue_length,
    delete_message,
    dequeue_message_with_receipt,
)
from rag_eval.core.logging import get_logger

logger = get_logger("scripts.test_azure_queues")


def test_queues():
    """Test all queue operations with real Azure Storage"""
    
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
    
    print("🧪 Testing Azure Storage Queues")
    print(f"   Storage Account: {connection_string.split('AccountName=')[1].split(';')[0] if 'AccountName=' in connection_string else 'Unknown'}")
    print()
    
    # Test queue names
    test_queue = "ingestion-uploads"
    queues_to_test = [
        "ingestion-uploads",
        "ingestion-chunking",
        "ingestion-embeddings",
        "ingestion-indexing",
        "ingestion-dead-letter"
    ]
    
    all_tests_passed = True
    
    # Test 1: Get queue length (should work even if empty)
    print("📊 Test 1: Get queue lengths")
    for queue_name in queues_to_test:
        try:
            length = get_queue_length(queue_name, config)
            print(f"   ✅ {queue_name}: {length} messages")
        except Exception as e:
            print(f"   ❌ {queue_name}: {e}")
            all_tests_passed = False
    print()
    
    # Test 2: Create and enqueue a test message
    print("📤 Test 2: Enqueue test message")
    test_message = QueueMessage(
        document_id="test-doc-123",
        source_storage="supabase",
        filename="test_document.pdf",
        attempt=1,
        stage="uploaded",
        metadata={"test": True, "timestamp": "2025-01-XX"}
    )
    
    try:
        enqueue_message(test_queue, test_message, config)
        print(f"   ✅ Enqueued message to {test_queue}")
        print(f"      document_id: {test_message.document_id}")
        print(f"      stage: {test_message.stage}")
    except Exception as e:
        print(f"   ❌ Failed to enqueue: {e}")
        all_tests_passed = False
        return all_tests_passed
    print()
    
    # Test 3: Get queue length (should be 1 now)
    print("📊 Test 3: Verify queue length after enqueue")
    try:
        length = get_queue_length(test_queue, config)
        print(f"   ✅ {test_queue}: {length} messages (expected: 1)")
        if length != 1:
            print(f"      ⚠️  Warning: Expected 1 message, got {length}")
            all_tests_passed = False
    except Exception as e:
        print(f"   ❌ Failed to get queue length: {e}")
        all_tests_passed = False
    print()
    
    # Test 4: Peek at message (should not remove it)
    print("👀 Test 4: Peek at message (should not remove)")
    try:
        peeked = peek_message(test_queue, config)
        if peeked:
            print(f"   ✅ Peeked message: document_id={peeked.document_id}, stage={peeked.stage}")
            if peeked.document_id != test_message.document_id:
                print(f"      ⚠️  Warning: Document ID mismatch")
                all_tests_passed = False
        else:
            print(f"   ❌ No message found (unexpected)")
            all_tests_passed = False
    except Exception as e:
        print(f"   ❌ Failed to peek: {e}")
        all_tests_passed = False
    print()
    
    # Verify queue length is still 1 after peek
    length_after_peek = get_queue_length(test_queue, config)
    if length_after_peek == 1:
        print("   ✅ Queue length still 1 after peek (message not removed)")
    else:
        print(f"   ⚠️  Queue length changed to {length_after_peek} after peek (expected: 1)")
        all_tests_passed = False
    print()
    
    # Test 5: Dequeue message with receipt
    print("📥 Test 5: Dequeue message with receipt")
    message_id = None
    pop_receipt = None
    try:
        result = dequeue_message_with_receipt(test_queue, config)
        if result:
            message, msg_id, receipt = result
            message_id = msg_id
            pop_receipt = receipt
            print(f"   ✅ Dequeued message: document_id={message.document_id}")
            print(f"      message_id: {message_id}")
            print(f"      pop_receipt: {pop_receipt[:20]}...")
            
            if message.document_id != test_message.document_id:
                print(f"      ⚠️  Warning: Document ID mismatch")
                all_tests_passed = False
        else:
            print(f"   ❌ No message dequeued (unexpected)")
            all_tests_passed = False
    except Exception as e:
        print(f"   ❌ Failed to dequeue: {e}")
        all_tests_passed = False
    print()
    
    # Test 6: Delete message
    if message_id and pop_receipt:
        print("🗑️  Test 6: Delete message")
        try:
            delete_message(test_queue, message_id, pop_receipt, config)
            print(f"   ✅ Deleted message successfully")
        except Exception as e:
            print(f"   ❌ Failed to delete: {e}")
            all_tests_passed = False
        print()
    
    # Test 7: Verify queue is empty
    print("📊 Test 7: Verify queue is empty after delete")
    try:
        length = get_queue_length(test_queue, config)
        print(f"   ✅ {test_queue}: {length} messages (expected: 0)")
        if length != 0:
            print(f"      ⚠️  Warning: Expected 0 messages, got {length}")
            all_tests_passed = False
    except Exception as e:
        print(f"   ❌ Failed to get queue length: {e}")
        all_tests_passed = False
    print()
    
    # Test 8: Test dequeue from empty queue
    print("📥 Test 8: Dequeue from empty queue (should return None)")
    try:
        result = dequeue_message(test_queue, config)
        if result is None:
            print(f"   ✅ Dequeue from empty queue returned None (correct)")
        else:
            print(f"   ⚠️  Dequeue from empty queue returned message (unexpected)")
            all_tests_passed = False
    except Exception as e:
        print(f"   ❌ Failed to dequeue: {e}")
        all_tests_passed = False
    print()
    
    if all_tests_passed:
        print("✅ All integration tests passed!")
    else:
        print("⚠️  Some tests failed or had warnings")
    
    return all_tests_passed


if __name__ == "__main__":
    success = test_queues()
    sys.exit(0 if success else 1)

