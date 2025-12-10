"""Comprehensive unit tests for queue client utilities

Tests message schema validation, serialization/deserialization,
and all queue operations (enqueue, dequeue, peek, delete, dead-letter).
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

logging.disable(logging.CRITICAL)

from src.core.exceptions import AzureServiceError, ValidationError
from src.services.workers.queue_client import (
    QueueMessage,
    SourceStorage,
    ProcessingStage,
    validate_message,
    serialize_message,
    deserialize_message,
    enqueue_message,
    dequeue_message,
    peek_message,
    delete_message,
    send_to_dead_letter,
    get_queue_length,
    dequeue_message_with_receipt,
)


@pytest.fixture
def mock_config():
    """Create a mock config with Azure Storage connection string"""
    config = Mock()
    config.azure_blob_connection_string = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=testkey;EndpointSuffix=core.windows.net"
    return config


@pytest.fixture
def valid_message_dict():
    """Valid message dictionary for testing"""
    return {
        "document_id": "123e4567-e89b-12d3-a456-426614174000",
        "source_storage": "supabase",
        "filename": "test_document.pdf",
        "attempt": 1,
        "stage": "uploaded",
        "metadata": {
            "tenant_id": "tenant_abc",
            "user_id": "user_123",
            "mime_type": "application/pdf"
        }
    }


@pytest.fixture
def valid_queue_message(valid_message_dict):
    """Valid QueueMessage instance for testing"""
    return QueueMessage(
        document_id=valid_message_dict["document_id"],
        source_storage=valid_message_dict["source_storage"],
        filename=valid_message_dict["filename"],
        attempt=valid_message_dict["attempt"],
        stage=valid_message_dict["stage"],
        metadata=valid_message_dict["metadata"]
    )


# Message Schema Validation Tests

def test_validate_message_valid(valid_message_dict):
    """Test validating a valid message"""
    message = validate_message(valid_message_dict)
    assert isinstance(message, QueueMessage)
    assert message.document_id == valid_message_dict["document_id"]
    assert message.source_storage == valid_message_dict["source_storage"]
    assert message.filename == valid_message_dict["filename"]
    assert message.attempt == valid_message_dict["attempt"]
    assert message.stage == valid_message_dict["stage"]
    assert message.metadata == valid_message_dict["metadata"]


def test_validate_message_missing_required_field(valid_message_dict):
    """Test validating message with missing required field"""
    del valid_message_dict["document_id"]
    with pytest.raises(ValidationError, match="Missing required fields"):
        validate_message(valid_message_dict)


def test_validate_message_empty_document_id(valid_message_dict):
    """Test validating message with empty document_id"""
    valid_message_dict["document_id"] = ""
    with pytest.raises(ValidationError, match="document_id cannot be empty"):
        validate_message(valid_message_dict)


def test_validate_message_invalid_source_storage(valid_message_dict):
    """Test validating message with invalid source_storage"""
    valid_message_dict["source_storage"] = "invalid_storage"
    with pytest.raises(ValidationError, match="source_storage must be"):
        validate_message(valid_message_dict)


def test_validate_message_invalid_stage(valid_message_dict):
    """Test validating message with invalid stage"""
    valid_message_dict["stage"] = "invalid_stage"
    with pytest.raises(ValidationError, match="stage must be one of"):
        validate_message(valid_message_dict)


def test_validate_message_attempt_less_than_one(valid_message_dict):
    """Test validating message with attempt < 1"""
    valid_message_dict["attempt"] = 0
    with pytest.raises(ValidationError, match="attempt must be >= 1"):
        validate_message(valid_message_dict)


def test_validate_message_empty_filename(valid_message_dict):
    """Test validating message with empty filename"""
    valid_message_dict["filename"] = ""
    with pytest.raises(ValidationError, match="filename cannot be empty"):
        validate_message(valid_message_dict)


def test_validate_message_without_metadata(valid_message_dict):
    """Test validating message without optional metadata"""
    del valid_message_dict["metadata"]
    message = validate_message(valid_message_dict)
    assert message.metadata is None


def test_validate_message_not_dict():
    """Test validating message that is not a dictionary"""
    with pytest.raises(ValidationError, match="Message must be a dictionary"):
        validate_message("not a dict")


def test_validate_message_invalid_field_types(valid_message_dict):
    """Test validating message with invalid field types"""
    valid_message_dict["attempt"] = "not_an_int"
    with pytest.raises(ValidationError, match="Invalid message field types"):
        validate_message(valid_message_dict)


# Message Serialization/Deserialization Tests

def test_serialize_message(valid_queue_message):
    """Test serializing a QueueMessage to JSON"""
    serialized = serialize_message(valid_queue_message)
    assert isinstance(serialized, str)
    
    # Verify it's valid JSON
    parsed = json.loads(serialized)
    assert parsed["document_id"] == valid_queue_message.document_id
    assert parsed["source_storage"] == valid_queue_message.source_storage
    assert parsed["filename"] == valid_queue_message.filename
    assert parsed["attempt"] == valid_queue_message.attempt
    assert parsed["stage"] == valid_queue_message.stage
    assert parsed["metadata"] == valid_queue_message.metadata


def test_deserialize_message(valid_message_dict):
    """Test deserializing a JSON string to QueueMessage"""
    message_str = json.dumps(valid_message_dict)
    message = deserialize_message(message_str)
    
    assert isinstance(message, QueueMessage)
    assert message.document_id == valid_message_dict["document_id"]
    assert message.source_storage == valid_message_dict["source_storage"]
    assert message.filename == valid_message_dict["filename"]
    assert message.attempt == valid_message_dict["attempt"]
    assert message.stage == valid_message_dict["stage"]
    assert message.metadata == valid_message_dict["metadata"]


def test_deserialize_message_invalid_json():
    """Test deserializing invalid JSON"""
    with pytest.raises(ValidationError, match="Invalid JSON"):
        deserialize_message("not valid json")


def test_deserialize_message_empty_string():
    """Test deserializing empty string"""
    with pytest.raises(ValidationError, match="Message string cannot be empty"):
        deserialize_message("")


def test_serialize_deserialize_roundtrip(valid_queue_message):
    """Test roundtrip serialization/deserialization"""
    serialized = serialize_message(valid_queue_message)
    deserialized = deserialize_message(serialized)
    
    assert deserialized.document_id == valid_queue_message.document_id
    assert deserialized.source_storage == valid_queue_message.source_storage
    assert deserialized.filename == valid_queue_message.filename
    assert deserialized.attempt == valid_queue_message.attempt
    assert deserialized.stage == valid_queue_message.stage
    assert deserialized.metadata == valid_queue_message.metadata


# Queue Operations Tests

@patch('src.services.workers.queue_client.QueueServiceClient')
def test_enqueue_message_success(mock_queue_service, mock_config, valid_queue_message):
    """Test successfully enqueuing a message"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    mock_queue_client.queue_name = "test-queue"
    
    enqueue_message("test-queue", valid_queue_message, mock_config)
    
    # Verify queue client was called
    mock_queue_service.from_connection_string.assert_called_once()
    mock_queue_client.create_queue.assert_called_once()
    mock_queue_client.send_message.assert_called_once()


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_enqueue_message_queue_already_exists(mock_queue_service, mock_config, valid_queue_message):
    """Test enqueuing when queue already exists"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    mock_queue_client.queue_name = "test-queue"
    
    # Simulate queue already exists error
    from azure.core.exceptions import AzureError
    mock_queue_client.create_queue.side_effect = AzureError("QueueAlreadyExists")
    
    # Should not raise error
    enqueue_message("test-queue", valid_queue_message, mock_config)
    
    # Verify send_message was still called
    mock_queue_client.send_message.assert_called_once()


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_enqueue_message_no_connection_string(mock_queue_service, valid_queue_message):
    """Test enqueuing with missing connection string"""
    config = Mock()
    config.azure_blob_connection_string = ""
    
    with pytest.raises(AzureServiceError, match="connection string is not configured"):
        enqueue_message("test-queue", valid_queue_message, config)


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_dequeue_message_success(mock_queue_service, mock_config, valid_message_dict):
    """Test successfully dequeuing a message"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Create mock Azure message
    mock_azure_message = Mock()
    mock_azure_message.content = json.dumps(valid_message_dict)
    mock_azure_message.id = "msg-123"
    mock_azure_message.pop_receipt = "receipt-123"
    
    # Mock receive_messages to return message
    mock_queue_client.receive_messages.return_value = [mock_azure_message]
    
    message = dequeue_message("test-queue", mock_config)
    
    assert message is not None
    assert isinstance(message, QueueMessage)
    assert message.document_id == valid_message_dict["document_id"]
    mock_queue_client.receive_messages.assert_called_once()


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_dequeue_message_empty_queue(mock_queue_service, mock_config):
    """Test dequeuing from empty queue"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Mock receive_messages to return empty list
    mock_queue_client.receive_messages.return_value = []
    
    message = dequeue_message("test-queue", mock_config)
    
    assert message is None


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_dequeue_message_invalid_format(mock_queue_service, mock_config):
    """Test dequeuing message with invalid format"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Create mock Azure message with invalid JSON
    mock_azure_message = Mock()
    mock_azure_message.content = "invalid json"
    
    # Mock receive_messages to return invalid message
    mock_queue_client.receive_messages.return_value = [mock_azure_message]
    
    # Should return None (skip poison message)
    message = dequeue_message("test-queue", mock_config)
    assert message is None


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_peek_message_success(mock_queue_service, mock_config, valid_message_dict):
    """Test successfully peeking at a message"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Create mock Azure message
    mock_azure_message = Mock()
    mock_azure_message.content = json.dumps(valid_message_dict)
    
    # Mock peek_messages to return message
    mock_queue_client.peek_messages.return_value = [mock_azure_message]
    
    message = peek_message("test-queue", mock_config)
    
    assert message is not None
    assert isinstance(message, QueueMessage)
    assert message.document_id == valid_message_dict["document_id"]
    mock_queue_client.peek_messages.assert_called_once()


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_peek_message_empty_queue(mock_queue_service, mock_config):
    """Test peeking at empty queue"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Mock peek_messages to return empty list
    mock_queue_client.peek_messages.return_value = []
    
    message = peek_message("test-queue", mock_config)
    
    assert message is None


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_delete_message_success(mock_queue_service, mock_config):
    """Test successfully deleting a message"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    delete_message("test-queue", "msg-123", "receipt-123", mock_config)
    
    mock_queue_client.delete_message.assert_called_once_with("msg-123", "receipt-123")


def test_delete_message_empty_message_id(mock_config):
    """Test deleting message with empty message_id"""
    with pytest.raises(ValueError, match="message_id and pop_receipt cannot be empty"):
        delete_message("test-queue", "", "receipt-123", mock_config)


def test_delete_message_empty_pop_receipt(mock_config):
    """Test deleting message with empty pop_receipt"""
    with pytest.raises(ValueError, match="message_id and pop_receipt cannot be empty"):
        delete_message("test-queue", "msg-123", "", mock_config)


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_send_to_dead_letter_success(mock_queue_service, mock_config, valid_queue_message):
    """Test successfully sending message to dead-letter queue"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    mock_queue_client.queue_name = "ingestion-dead-letter"
    
    send_to_dead_letter("ingestion-dead-letter", valid_queue_message, "Test reason", mock_config)
    
    # Verify message was enqueued
    mock_queue_client.send_message.assert_called_once()
    
    # Verify metadata was updated
    call_args = mock_queue_client.send_message.call_args[0][0]
    message_dict = json.loads(call_args)
    assert "dead_letter_reason" in message_dict["metadata"]
    assert message_dict["metadata"]["dead_letter_reason"] == "Test reason"
    assert "dead_lettered_at" in message_dict["metadata"]


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_get_queue_length_success(mock_queue_service, mock_config):
    """Test successfully getting queue length"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Mock queue properties
    mock_properties = Mock()
    mock_properties.approximate_message_count = 42
    mock_queue_client.get_queue_properties.return_value = mock_properties
    
    length = get_queue_length("test-queue", mock_config)
    
    assert length == 42
    mock_queue_client.get_queue_properties.assert_called_once()


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_get_queue_length_not_found(mock_queue_service, mock_config):
    """Test getting length of non-existent queue"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Mock queue not found
    from azure.core.exceptions import ResourceNotFoundError
    mock_queue_client.get_queue_properties.side_effect = ResourceNotFoundError("Queue not found")
    
    length = get_queue_length("test-queue", mock_config)
    
    assert length == 0


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_dequeue_message_with_receipt_success(mock_queue_service, mock_config, valid_message_dict):
    """Test dequeuing message with receipt"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Create mock Azure message
    mock_azure_message = Mock()
    mock_azure_message.content = json.dumps(valid_message_dict)
    mock_azure_message.id = "msg-123"
    mock_azure_message.pop_receipt = "receipt-123"
    
    # Mock receive_messages to return message
    mock_queue_client.receive_messages.return_value = [mock_azure_message]
    
    result = dequeue_message_with_receipt("test-queue", mock_config)
    
    assert result is not None
    message, message_id, pop_receipt = result
    assert isinstance(message, QueueMessage)
    assert message_id == "msg-123"
    assert pop_receipt == "receipt-123"


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_dequeue_message_with_receipt_empty_queue(mock_queue_service, mock_config):
    """Test dequeuing with receipt from empty queue"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Mock receive_messages to return empty list
    mock_queue_client.receive_messages.return_value = []
    
    result = dequeue_message_with_receipt("test-queue", mock_config)
    
    assert result is None


# Edge Cases and Error Handling Tests

def test_queue_message_post_init_validation():
    """Test QueueMessage __post_init__ validation"""
    # Test empty document_id
    with pytest.raises(ValidationError, match="document_id cannot be empty"):
        QueueMessage(
            document_id="",
            source_storage="supabase",
            filename="test.pdf",
            attempt=1,
            stage="uploaded"
        )
    
    # Test invalid source_storage
    with pytest.raises(ValidationError, match="source_storage must be"):
        QueueMessage(
            document_id="doc-123",
            source_storage="invalid",
            filename="test.pdf",
            attempt=1,
            stage="uploaded"
        )
    
    # Test invalid stage
    with pytest.raises(ValidationError, match="stage must be one of"):
        QueueMessage(
            document_id="doc-123",
            source_storage="supabase",
            filename="test.pdf",
            attempt=1,
            stage="invalid"
        )
    
    # Test attempt < 1
    with pytest.raises(ValidationError, match="attempt must be >= 1"):
        QueueMessage(
            document_id="doc-123",
            source_storage="supabase",
            filename="test.pdf",
            attempt=0,
            stage="uploaded"
        )


def test_queue_message_all_stages():
    """Test QueueMessage with all valid stages"""
    stages = ["uploaded", "parsed", "chunked", "embedded", "indexed"]
    for stage in stages:
        message = QueueMessage(
            document_id="doc-123",
            source_storage="supabase",
            filename="test.pdf",
            attempt=1,
            stage=stage
        )
        assert message.stage == stage


def test_queue_message_both_storage_types():
    """Test QueueMessage with both storage types"""
    for storage in ["azure_blob", "supabase"]:
        message = QueueMessage(
            document_id="doc-123",
            source_storage=storage,
            filename="test.pdf",
            attempt=1,
            stage="uploaded"
        )
        assert message.source_storage == storage


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_enqueue_message_connection_error(mock_queue_service, mock_config, valid_queue_message):
    """Test enqueuing when connection fails"""
    # Setup mocks to raise connection error
    mock_queue_service.from_connection_string.side_effect = Exception("Connection failed")
    
    with pytest.raises(AzureServiceError, match="Failed to connect"):
        enqueue_message("test-queue", valid_queue_message, mock_config)


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_dequeue_message_connection_error(mock_queue_service, mock_config):
    """Test dequeuing when connection fails"""
    # Setup mocks to raise connection error
    mock_queue_service.from_connection_string.side_effect = Exception("Connection failed")
    
    with pytest.raises(AzureServiceError, match="Failed to connect"):
        dequeue_message("test-queue", mock_config)


def test_enqueue_message_empty_queue_name(mock_config, valid_queue_message):
    """Test enqueuing with empty queue name"""
    with pytest.raises(ValueError, match="queue_name cannot be empty"):
        enqueue_message("", valid_queue_message, mock_config)


def test_dequeue_message_empty_queue_name(mock_config):
    """Test dequeuing with empty queue name"""
    with pytest.raises(ValueError, match="queue_name cannot be empty"):
        dequeue_message("", mock_config)


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_enqueue_message_queue_creation_error(mock_queue_service, mock_config, valid_queue_message):
    """Test enqueuing when queue creation fails with non-QueueAlreadyExists error"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    mock_queue_client.queue_name = "test-queue"
    
    # Simulate queue creation error (not QueueAlreadyExists)
    from azure.core.exceptions import AzureError
    mock_queue_client.create_queue.side_effect = AzureError("Some other error")
    
    with pytest.raises(AzureServiceError, match="Failed to create queue"):
        enqueue_message("test-queue", valid_queue_message, mock_config)


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_peek_message_invalid_format(mock_queue_service, mock_config):
    """Test peeking message with invalid format"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Create mock Azure message with invalid JSON
    mock_azure_message = Mock()
    mock_azure_message.content = "invalid json"
    
    # Mock peek_messages to return invalid message
    mock_queue_client.peek_messages.return_value = [mock_azure_message]
    
    # Should return None (skip poison message)
    message = peek_message("test-queue", mock_config)
    assert message is None


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_peek_message_connection_error(mock_queue_service, mock_config):
    """Test peeking when connection fails"""
    # Setup mocks to raise connection error
    mock_queue_service.from_connection_string.side_effect = Exception("Connection failed")
    
    with pytest.raises(AzureServiceError, match="Failed to connect"):
        peek_message("test-queue", mock_config)


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_delete_message_error(mock_queue_service, mock_config):
    """Test deleting message when delete fails with non-ResourceNotFoundError"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Mock delete_message to raise non-ResourceNotFoundError
    mock_queue_client.delete_message.side_effect = Exception("Delete failed")
    
    with pytest.raises(AzureServiceError, match="Failed to delete message"):
        delete_message("test-queue", "msg-123", "receipt-123", mock_config)


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_send_to_dead_letter_error(mock_queue_service, mock_config, valid_queue_message):
    """Test sending to dead-letter queue when enqueue fails"""
    # Setup mocks to raise error during enqueue
    mock_queue_service.from_connection_string.side_effect = Exception("Enqueue failed")
    
    with pytest.raises(AzureServiceError, match="Failed to send to dead-letter queue"):
        send_to_dead_letter("ingestion-dead-letter", valid_queue_message, "Test reason", mock_config)


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_get_queue_length_error(mock_queue_service, mock_config):
    """Test getting queue length when query fails with non-ResourceNotFoundError"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Mock get_queue_properties to raise non-ResourceNotFoundError
    mock_queue_client.get_queue_properties.side_effect = Exception("Query failed")
    
    with pytest.raises(AzureServiceError, match="Failed to get queue length"):
        get_queue_length("test-queue", mock_config)


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_dequeue_message_with_receipt_invalid_format(mock_queue_service, mock_config):
    """Test dequeuing with receipt when message has invalid format"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Create mock Azure message with invalid JSON
    mock_azure_message = Mock()
    mock_azure_message.content = "invalid json"
    
    # Mock receive_messages to return invalid message
    mock_queue_client.receive_messages.return_value = [mock_azure_message]
    
    # Should return None (skip poison message)
    result = dequeue_message_with_receipt("test-queue", mock_config)
    assert result is None


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_dequeue_message_with_receipt_connection_error(mock_queue_service, mock_config):
    """Test dequeuing with receipt when connection fails"""
    # Setup mocks to raise connection error
    mock_queue_service.from_connection_string.side_effect = Exception("Connection failed")
    
    with pytest.raises(AzureServiceError, match="Failed to connect"):
        dequeue_message_with_receipt("test-queue", mock_config)


@patch('src.services.workers.queue_client.QueueServiceClient')
def test_dequeue_message_with_receipt_not_found(mock_queue_service, mock_config):
    """Test dequeuing with receipt when queue not found"""
    # Setup mocks
    mock_queue_client = Mock()
    mock_queue_service.from_connection_string.return_value.get_queue_client.return_value = mock_queue_client
    
    # Mock receive_messages to raise ResourceNotFoundError
    from azure.core.exceptions import ResourceNotFoundError
    mock_queue_client.receive_messages.side_effect = ResourceNotFoundError("Queue not found")
    
    result = dequeue_message_with_receipt("test-queue", mock_config)
    assert result is None


def test_serialize_message_error(valid_queue_message):
    """Test serialization error handling"""
    # Mock asdict to raise exception
    with patch('src.services.workers.queue_client.asdict', side_effect=Exception("Serialization failed")):
        with pytest.raises(ValidationError, match="Failed to serialize message"):
            serialize_message(valid_queue_message)


def test_deserialize_message_other_error():
    """Test deserialization with other exception types"""
    # Mock json.loads to raise non-JSONDecodeError
    with patch('json.loads', side_effect=Exception("Unexpected error")):
        with pytest.raises(ValidationError, match="Failed to deserialize message"):
            deserialize_message('{"test": "value"}')


def test_validate_message_other_error():
    """Test validation with other exception types"""
    # Create a message dict that will cause an unexpected error
    # by mocking QueueMessage to raise an unexpected exception
    with patch('src.services.workers.queue_client.QueueMessage', side_effect=Exception("Unexpected error")):
        with pytest.raises(ValidationError, match="Failed to validate message"):
            validate_message({
                "document_id": "doc-123",
                "source_storage": "supabase",
                "filename": "test.pdf",
                "attempt": 1,
                "stage": "uploaded"
            })


def test_send_to_dead_letter_with_none_metadata(mock_config):
    """Test sending to dead-letter queue when message has None metadata"""
    message = QueueMessage(
        document_id="doc-123",
        source_storage="supabase",
        filename="test.pdf",
        attempt=1,
        stage="uploaded",
        metadata=None
    )
    
    with patch('src.services.workers.queue_client.enqueue_message') as mock_enqueue:
        send_to_dead_letter("ingestion-dead-letter", message, "Test reason", mock_config)
        
        # Verify metadata was set
        assert message.metadata is not None
        assert "dead_letter_reason" in message.metadata
        assert "dead_lettered_at" in message.metadata
        mock_enqueue.assert_called_once()

