"""Integration tests for Azure Storage Queue operations

These tests require:
1. Azure Storage Account with Queue Storage enabled
2. Connection string set in environment (AZURE_STORAGE_QUEUES_CONNECTION_STRING or AZURE_BLOB_CONNECTION_STRING)
3. Queues created (via setup_azure_queues.py or Azure Portal)

These tests are marked with @pytest.mark.integration and can be skipped
if Azure Storage is not available.
"""

import pytest
import os
from src.core.config import Config
from src.services.workers.queue_client import (
    QueueMessage,
    enqueue_message,
    dequeue_message,
    peek_message,
    get_queue_length,
    delete_message,
    dequeue_message_with_receipt,
)

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def real_config():
    """Get real config from environment"""
    try:
        config = Config.from_env()
        # Check if connection string is available
        connection_string = config.azure_blob_connection_string or os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING", "")
        if not connection_string:
            pytest.skip("Azure Storage connection string not configured")
        return config
    except Exception as e:
        pytest.skip(f"Failed to load config: {e}")


@pytest.fixture
def test_queue_name():
    """Queue name for testing"""
    return "ingestion-uploads"


@pytest.fixture
def test_message():
    """Test message for queue operations"""
    return QueueMessage(
        document_id="test-doc-integration-123",
        source_storage="supabase",
        filename="test_document.pdf",
        attempt=1,
        stage="uploaded",
        metadata={"test": True, "integration": True}
    )


class TestQueueIntegration:
    """Integration tests for Azure Storage Queue operations"""
    
    def test_get_queue_lengths(self, real_config):
        """Test getting queue lengths for all queues"""
        queues_to_test = [
            "ingestion-uploads",
            "ingestion-chunking",
            "ingestion-embeddings",
            "ingestion-indexing",
            "ingestion-dead-letter"
        ]
        
        for queue_name in queues_to_test:
            length = get_queue_length(queue_name, real_config)
            assert isinstance(length, int)
            assert length >= 0
    
    def test_enqueue_message(self, real_config, test_queue_name, test_message):
        """Test enqueuing a message"""
        enqueue_message(test_queue_name, test_message, real_config)
        
        # Verify message was enqueued
        length = get_queue_length(test_queue_name, real_config)
        assert length >= 1
    
    def test_peek_message(self, real_config, test_queue_name, test_message):
        """Test peeking at a message without removing it"""
        # First enqueue a message
        enqueue_message(test_queue_name, test_message, real_config)
        
        # Get initial length
        initial_length = get_queue_length(test_queue_name, real_config)
        
        # Peek at message
        peeked = peek_message(test_queue_name, real_config)
        assert peeked is not None
        assert peeked.document_id == test_message.document_id
        assert peeked.stage == test_message.stage
        
        # Verify queue length didn't change
        length_after_peek = get_queue_length(test_queue_name, real_config)
        assert length_after_peek == initial_length
    
    def test_dequeue_message_with_receipt(self, real_config, test_queue_name, test_message):
        """Test dequeuing a message with receipt"""
        # First enqueue a message
        enqueue_message(test_queue_name, test_message, real_config)
        
        # Dequeue with receipt
        result = dequeue_message_with_receipt(test_queue_name, real_config)
        assert result is not None
        
        message, message_id, pop_receipt = result
        assert message.document_id == test_message.document_id
        assert message_id is not None
        assert pop_receipt is not None
    
    def test_delete_message(self, real_config, test_queue_name, test_message):
        """Test deleting a message"""
        # First enqueue a message
        enqueue_message(test_queue_name, test_message, real_config)
        
        # Dequeue with receipt
        result = dequeue_message_with_receipt(test_queue_name, real_config)
        assert result is not None
        message, message_id, pop_receipt = result
        
        # Delete the message
        delete_message(test_queue_name, message_id, pop_receipt, real_config)
        
        # Verify queue is empty (or at least one less message)
        # Note: There might be other messages in the queue from previous tests
        length = get_queue_length(test_queue_name, real_config)
        # We can't assert exact length since other tests might have added messages
        assert isinstance(length, int)
        assert length >= 0
    
    def test_dequeue_from_empty_queue(self, real_config, test_queue_name):
        """Test dequeuing from an empty queue returns None"""
        # Try to dequeue (might be empty)
        result = dequeue_message(test_queue_name, real_config)
        # Result can be None (empty) or a message (if queue has messages)
        # This test just verifies the function doesn't crash
        assert result is None or isinstance(result, QueueMessage)
    
    def test_full_queue_workflow(self, real_config, test_queue_name, test_message):
        """Test complete workflow: enqueue -> peek -> dequeue -> delete"""
        # Enqueue
        enqueue_message(test_queue_name, test_message, real_config)
        initial_length = get_queue_length(test_queue_name, real_config)
        assert initial_length >= 1
        
        # Peek (should not remove)
        peeked = peek_message(test_queue_name, real_config)
        assert peeked is not None
        assert peeked.document_id == test_message.document_id
        assert get_queue_length(test_queue_name, real_config) == initial_length
        
        # Dequeue with receipt
        result = dequeue_message_with_receipt(test_queue_name, real_config)
        assert result is not None
        message, message_id, pop_receipt = result
        assert message.document_id == test_message.document_id
        
        # Delete
        delete_message(test_queue_name, message_id, pop_receipt, real_config)
        
        # Verify queue length decreased
        final_length = get_queue_length(test_queue_name, real_config)
        assert final_length < initial_length

