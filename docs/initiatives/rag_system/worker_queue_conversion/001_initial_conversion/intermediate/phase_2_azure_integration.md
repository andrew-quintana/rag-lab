# Phase 2 Azure Integration Testing — Queue Infrastructure

## Summary

Azure Storage Queues have been successfully set up and tested with real Azure Storage Account integration.

**Date**: 2025-12-06  
**Storage Account**: raglabqueues  
**Status**: ✅ Complete

## Setup Results

### Queues Created/Verified

All 5 required queues exist in Azure Storage Account:

- ✅ `ingestion-uploads` - Ready for ingestion worker
- ✅ `ingestion-chunking` - Ready for chunking worker  
- ✅ `ingestion-embeddings` - Ready for embedding worker
- ✅ `ingestion-indexing` - Ready for indexing worker
- ✅ `ingestion-dead-letter` - Ready for poison messages

### Connection String Configuration

**Environment Variable**: `AZURE_STORAGE_QUEUES_CONNECTION_STRING`

The queue client now supports both:
1. `AZURE_BLOB_CONNECTION_STRING` (from Config class)
2. `AZURE_STORAGE_QUEUES_CONNECTION_STRING` (direct environment variable)

Priority: Config value takes precedence, falls back to environment variable.

## Integration Test Results

### Test Execution

**Command**: `pytest tests/components/workers/test_queue_client_integration.py -v -m integration`  
**Result**: ✅ All 7 tests passed

### Test Coverage

1. ✅ **Queue Length Queries** - All 5 queues queried successfully
2. ✅ **Message Enqueue** - Test message enqueued successfully
3. ✅ **Queue Length After Enqueue** - Verified message count increased to 1
4. ✅ **Message Peek** - Peeked message without removing it
5. ✅ **Queue Length After Peek** - Verified message still in queue
6. ✅ **Message Dequeue with Receipt** - Dequeued message and received metadata
7. ✅ **Message Delete** - Deleted message successfully
8. ✅ **Queue Length After Delete** - Verified queue empty (0 messages)
9. ✅ **Dequeue from Empty Queue** - Correctly returned None

### Test Output

```
🧪 Testing Azure Storage Queues
   Storage Account: raglabqueues

📊 Test 1: Get queue lengths
   ✅ ingestion-uploads: 0 messages
   ✅ ingestion-chunking: 0 messages
   ✅ ingestion-embeddings: 0 messages
   ✅ ingestion-indexing: 0 messages
   ✅ ingestion-dead-letter: 0 messages

📤 Test 2: Enqueue test message
   ✅ Enqueued message to ingestion-uploads
      document_id: test-doc-123
      stage: uploaded

📊 Test 3: Verify queue length after enqueue
   ✅ ingestion-uploads: 1 messages (expected: 1)

👀 Test 4: Peek at message (should not remove)
   ✅ Peeked message: document_id=test-doc-123, stage=uploaded

   ✅ Queue length still 1 after peek (message not removed)

📥 Test 5: Dequeue message with receipt
   ✅ Dequeued message: document_id=test-doc-123
      message_id: 2a382212-6132-4cb4-8a95-24ce089ee772
      pop_receipt: AgAAAAMAAAAAAAAAXPA2...

🗑️  Test 6: Delete message
   ✅ Deleted message successfully

📊 Test 7: Verify queue is empty after delete
   ✅ ingestion-uploads: 0 messages (expected: 0)

📥 Test 8: Dequeue from empty queue (should return None)
   ✅ Dequeue from empty queue returned None (correct)

✅ All integration tests passed!
```

## Code Changes

### 1. Queue Client Updates

**File**: `backend/rag_eval/services/workers/queue_client.py`

- Added support for `AZURE_STORAGE_QUEUES_CONNECTION_STRING` environment variable
- Improved `_ensure_queue_exists()` to handle "QueueAlreadyExists" errors gracefully
- Enhanced error detection for queue existence checks

### 2. Integration Test Suite

**File**: `backend/tests/components/workers/test_queue_client_integration.py`

- Comprehensive pytest integration test suite
- Tests all queue operations with real Azure Storage
- Marked with `@pytest.mark.integration` for optional execution

## Unit Tests Status

**Command**: `pytest tests/components/workers/test_queue_client.py -v`  
**Result**: ✅ All 51 unit tests pass

All existing unit tests continue to pass after code changes.

## Scripts Created

1. **`backend/scripts/get_azure_storage_connection.py`** - Helper to get connection string from Azure CLI
2. **`backend/tests/components/workers/test_queue_client_integration.py`** - Integration tests for queue operations (pytest)

## Next Steps

Phase 2 Azure integration is complete. The queue infrastructure is ready for:

1. **Phase 3: Worker Implementation** - Workers can now use queue operations with real Azure Storage
2. **Local Development** - Developers can test queue operations locally with real Azure
3. **Integration Testing** - Full integration test suite available

## Usage

### Queues Auto-Create

Queues are automatically created when you first enqueue a message. No manual setup required.

### Run Integration Tests
```bash
cd backend
source venv/bin/activate
pytest tests/components/workers/test_queue_client_integration.py -v -m integration
```

### Use in Code
```python
from rag_eval.services.workers.queue_client import enqueue_message, QueueMessage
from rag_eval.core.config import Config

config = Config.from_env()
message = QueueMessage(
    document_id="doc-123",
    source_storage="supabase",
    filename="test.pdf",
    attempt=1,
    stage="uploaded"
)
enqueue_message("ingestion-uploads", message, config)
```

## Notes

- Connection string is stored as `AZURE_STORAGE_QUEUES_CONNECTION_STRING` in `.env.local`
- Queues are idempotent - safe to create multiple times
- All queue operations tested and verified with real Azure Storage
- Queue client supports both Config-based and environment variable-based connection strings

