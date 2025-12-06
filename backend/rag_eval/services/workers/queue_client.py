"""Azure Storage Queue client utilities for worker-queue architecture

This module provides queue operations (enqueue, dequeue, peek, delete, dead-letter)
and message schema validation for the worker-queue architecture.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import json
from enum import Enum
from datetime import datetime, timezone
from azure.storage.queue import QueueServiceClient, QueueClient
from azure.core.exceptions import AzureError, ResourceNotFoundError

from rag_eval.core.exceptions import AzureServiceError, ValidationError
from rag_eval.core.logging import get_logger

logger = get_logger("services.workers.queue_client")


class SourceStorage(str, Enum):
    """Enum for source storage backends"""
    AZURE_BLOB = "azure_blob"
    SUPABASE = "supabase"


class ProcessingStage(str, Enum):
    """Enum for processing stages"""
    UPLOADED = "uploaded"
    PARSED = "parsed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"


@dataclass
class QueueMessage:
    """Queue message schema for worker-queue architecture
    
    Attributes:
        document_id: Primary key used throughout pipeline (required)
        source_storage: Storage backend ("azure_blob" or "supabase") (required)
        filename: Original filename for logging/debugging (required)
        attempt: Retry attempt counter, incremented on retries (required)
        stage: Current processing stage (required)
        metadata: Additional metadata including tenant_id, user_id, mime_type (optional)
    """
    document_id: str
    source_storage: str
    filename: str
    attempt: int
    stage: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate message fields after initialization"""
        if not self.document_id or not self.document_id.strip():
            raise ValidationError("document_id cannot be empty")
        
        if self.source_storage not in [SourceStorage.AZURE_BLOB, SourceStorage.SUPABASE]:
            raise ValidationError(
                f"source_storage must be '{SourceStorage.AZURE_BLOB}' or '{SourceStorage.SUPABASE}'"
            )
        
        if not self.filename or not self.filename.strip():
            raise ValidationError("filename cannot be empty")
        
        if self.attempt < 1:
            raise ValidationError("attempt must be >= 1")
        
        valid_stages = [
            ProcessingStage.UPLOADED,
            ProcessingStage.PARSED,
            ProcessingStage.CHUNKED,
            ProcessingStage.EMBEDDED,
            ProcessingStage.INDEXED
        ]
        if self.stage not in valid_stages:
            raise ValidationError(
                f"stage must be one of: {', '.join(valid_stages)}"
            )


def validate_message(message: dict) -> QueueMessage:
    """Validate and create QueueMessage from dictionary
    
    Args:
        message: Dictionary with message fields
        
    Returns:
        Validated QueueMessage instance
        
    Raises:
        ValidationError: If message schema is invalid or missing required fields
    """
    if not isinstance(message, dict):
        raise ValidationError("Message must be a dictionary")
    
    required_fields = ["document_id", "source_storage", "filename", "attempt", "stage"]
    missing_fields = [field for field in required_fields if field not in message]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
    
    try:
        return QueueMessage(
            document_id=str(message["document_id"]),
            source_storage=str(message["source_storage"]),
            filename=str(message["filename"]),
            attempt=int(message["attempt"]),
            stage=str(message["stage"]),
            metadata=message.get("metadata")
        )
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid message field types: {e}") from e
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Failed to validate message: {e}") from e


def serialize_message(message: QueueMessage) -> str:
    """Serialize QueueMessage to JSON string
    
    Args:
        message: QueueMessage instance to serialize
        
    Returns:
        JSON string representation of message
        
    Raises:
        ValidationError: If message cannot be serialized
    """
    try:
        message_dict = asdict(message)
        return json.dumps(message_dict)
    except Exception as e:
        raise ValidationError(f"Failed to serialize message: {e}") from e


def deserialize_message(message_str: str) -> QueueMessage:
    """Deserialize JSON string to QueueMessage
    
    Args:
        message_str: JSON string representation of message
        
    Returns:
        QueueMessage instance
        
    Raises:
        ValidationError: If message cannot be deserialized or is invalid
    """
    if not message_str or not message_str.strip():
        raise ValidationError("Message string cannot be empty")
    
    try:
        message_dict = json.loads(message_str)
        return validate_message(message_dict)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in message: {e}") from e
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Failed to deserialize message: {e}") from e


def _get_queue_client(queue_name: str, config) -> QueueClient:
    """Get Azure Storage Queue client for a queue
    
    Args:
        queue_name: Name of the queue
        config: Application configuration with Azure Storage connection string
        
    Returns:
        QueueClient instance
        
    Raises:
        AzureServiceError: If connection fails or queue doesn't exist
        ValueError: If queue_name is empty
    """
    if not queue_name or not queue_name.strip():
        raise ValueError("queue_name cannot be empty")
    
    # Try to get connection string from config, with fallback to environment variable
    # Priority: 1) config.azure_blob_connection_string, 2) AZURE_STORAGE_QUEUES_CONNECTION_STRING env var
    connection_string = config.azure_blob_connection_string
    
    # If not in config, try environment variable directly
    if not connection_string:
        import os
        connection_string = os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING", "")
    
    if not connection_string:
        raise AzureServiceError(
            "Azure Storage connection string is not configured. "
            "Set AZURE_BLOB_CONNECTION_STRING in config or AZURE_STORAGE_QUEUES_CONNECTION_STRING in environment."
        )
    
    try:
        queue_service_client = QueueServiceClient.from_connection_string(connection_string)
        queue_client = queue_service_client.get_queue_client(queue_name)
        return queue_client
    except Exception as e:
        logger.error(f"Failed to get queue client for '{queue_name}': {e}")
        raise AzureServiceError(f"Failed to connect to queue '{queue_name}': {e}") from e


def _ensure_queue_exists(queue_client: QueueClient) -> None:
    """Ensure queue exists, create if it doesn't
    
    Args:
        queue_client: QueueClient instance
        
    Raises:
        AzureServiceError: If queue creation fails (except when queue already exists)
    """
    try:
        queue_client.create_queue()
        logger.info(f"Created queue: {queue_client.queue_name}")
    except AzureError as e:
        # Queue already exists is not an error - check error code and message
        error_code = getattr(e, 'error_code', None) or str(e)
        if ("QueueAlreadyExists" in str(e) or 
            "QUEUE_ALREADY_EXISTS" in str(error_code) or 
            "409" in str(e)):
            # Queue already exists - this is fine, just log and continue
            logger.debug(f"Queue '{queue_client.queue_name}' already exists")
            return
        else:
            logger.error(f"Failed to create queue '{queue_client.queue_name}': {e}")
            raise AzureServiceError(f"Failed to create queue: {e}") from e
    except Exception as e:
        # Check if it's a QueueAlreadyExists error in the exception message
        if "QueueAlreadyExists" in str(e) or "QUEUE_ALREADY_EXISTS" in str(e) or "409" in str(e):
            logger.debug(f"Queue '{queue_client.queue_name}' already exists")
            return
        logger.error(f"Failed to create queue '{queue_client.queue_name}': {e}")
        raise AzureServiceError(f"Failed to create queue: {e}") from e


def enqueue_message(queue_name: str, message: QueueMessage, config) -> None:
    """Enqueue a message to an Azure Storage Queue
    
    Args:
        queue_name: Name of the queue
        message: QueueMessage instance to enqueue
        config: Application configuration with Azure Storage connection string
        
    Raises:
        AzureServiceError: If enqueue fails
        ValidationError: If message is invalid
    """
    try:
        queue_client = _get_queue_client(queue_name, config)
        _ensure_queue_exists(queue_client)
        
        message_str = serialize_message(message)
        queue_client.send_message(message_str)
        logger.info(
            f"Enqueued message to '{queue_name}': document_id={message.document_id}, "
            f"stage={message.stage}, attempt={message.attempt}"
        )
    except (ValidationError, AzureServiceError, ValueError):
        raise
    except Exception as e:
        logger.error(f"Failed to enqueue message to '{queue_name}': {e}")
        raise AzureServiceError(f"Failed to enqueue message: {e}") from e


def dequeue_message(queue_name: str, config, visibility_timeout: int = 30) -> Optional[QueueMessage]:
    """Dequeue a message from an Azure Storage Queue
    
    Args:
        queue_name: Name of the queue
        config: Application configuration with Azure Storage connection string
        visibility_timeout: Message visibility timeout in seconds (default: 30)
        
    Returns:
        QueueMessage instance if message available, None if queue is empty
        
    Raises:
        AzureServiceError: If dequeue fails
    """
    try:
        queue_client = _get_queue_client(queue_name, config)
        
        messages = queue_client.receive_messages(
            messages_per_page=1,
            visibility_timeout=visibility_timeout
        )
        
        message_list = list(messages)
        if not message_list:
            return None
        
        azure_message = message_list[0]
        message = deserialize_message(azure_message.content)
        
        logger.info(
            f"Dequeued message from '{queue_name}': document_id={message.document_id}, "
            f"stage={message.stage}, attempt={message.attempt}"
        )
        
        return message
    except ValueError as e:
        # ValueError for empty queue_name should be re-raised
        if "queue_name" in str(e).lower():
            raise
        # Other ValueErrors (from validation) - log and return None to skip poison message
        logger.warning(f"Invalid message format in queue '{queue_name}', skipping")
        return None
    except ValidationError:
        # Invalid message format - log and return None to skip poison message
        logger.warning(f"Invalid message format in queue '{queue_name}', skipping")
        return None
    except ResourceNotFoundError:
        logger.warning(f"Queue '{queue_name}' not found")
        return None
    except Exception as e:
        logger.error(f"Failed to dequeue message from '{queue_name}': {e}")
        raise AzureServiceError(f"Failed to dequeue message: {e}") from e


def peek_message(queue_name: str, config) -> Optional[QueueMessage]:
    """Peek at the next message in an Azure Storage Queue without removing it
    
    Args:
        queue_name: Name of the queue
        config: Application configuration with Azure Storage connection string
        
    Returns:
        QueueMessage instance if message available, None if queue is empty
        
    Raises:
        AzureServiceError: If peek fails
    """
    try:
        queue_client = _get_queue_client(queue_name, config)
        
        messages = queue_client.peek_messages(max_messages=1)
        message_list = list(messages)
        if not message_list:
            return None
        
        azure_message = message_list[0]
        message = deserialize_message(azure_message.content)
        
        logger.debug(
            f"Peeked message from '{queue_name}': document_id={message.document_id}, "
            f"stage={message.stage}, attempt={message.attempt}"
        )
        
        return message
    except ValidationError:
        # Invalid message format - log and return None
        logger.warning(f"Invalid message format in queue '{queue_name}', skipping")
        return None
    except ResourceNotFoundError:
        logger.warning(f"Queue '{queue_name}' not found")
        return None
    except Exception as e:
        logger.error(f"Failed to peek message from '{queue_name}': {e}")
        raise AzureServiceError(f"Failed to peek message: {e}") from e


def delete_message(
    queue_name: str,
    message_id: str,
    pop_receipt: str,
    config
) -> None:
    """Delete a message from an Azure Storage Queue
    
    Args:
        queue_name: Name of the queue
        message_id: Message ID from Azure QueueMessage
        pop_receipt: Pop receipt from Azure QueueMessage
        config: Application configuration with Azure Storage connection string
        
    Raises:
        AzureServiceError: If delete fails
        ValueError: If message_id or pop_receipt is empty
    """
    if not message_id or not pop_receipt:
        raise ValueError("message_id and pop_receipt cannot be empty")
    
    try:
        queue_client = _get_queue_client(queue_name, config)
        queue_client.delete_message(message_id, pop_receipt)
        logger.info(f"Deleted message from '{queue_name}': message_id={message_id}")
    except ResourceNotFoundError:
        logger.warning(f"Message {message_id} not found in queue '{queue_name}'")
        # Not an error - message may have already been deleted
    except Exception as e:
        logger.error(f"Failed to delete message from '{queue_name}': {e}")
        raise AzureServiceError(f"Failed to delete message: {e}") from e


def send_to_dead_letter(
    queue_name: str,
    message: QueueMessage,
    reason: str,
    config
) -> None:
    """Send a message to the dead-letter queue
    
    Args:
        queue_name: Name of the dead-letter queue (typically "ingestion-dead-letter")
        message: QueueMessage instance to send
        reason: Reason for dead-lettering (for logging/debugging)
        config: Application configuration with Azure Storage connection string
        
    Raises:
        AzureServiceError: If sending to dead-letter queue fails
    """
    try:
        # Add reason to metadata for debugging
        if message.metadata is None:
            message.metadata = {}
        message.metadata["dead_letter_reason"] = reason
        message.metadata["dead_lettered_at"] = str(datetime.now(timezone.utc))
        
        enqueue_message(queue_name, message, config)
        logger.warning(
            f"Sent message to dead-letter queue '{queue_name}': document_id={message.document_id}, "
            f"stage={message.stage}, reason={reason}"
        )
    except Exception as e:
        logger.error(f"Failed to send message to dead-letter queue '{queue_name}': {e}")
        raise AzureServiceError(f"Failed to send to dead-letter queue: {e}") from e


def get_queue_length(queue_name: str, config) -> int:
    """Get the approximate number of messages in a queue
    
    Args:
        queue_name: Name of the queue
        config: Application configuration with Azure Storage connection string
        
    Returns:
        Approximate number of messages in the queue (0 if queue doesn't exist)
        
    Raises:
        AzureServiceError: If query fails
    """
    try:
        queue_client = _get_queue_client(queue_name, config)
        properties = queue_client.get_queue_properties()
        return properties.approximate_message_count or 0
    except ResourceNotFoundError:
        logger.warning(f"Queue '{queue_name}' not found")
        return 0
    except Exception as e:
        logger.error(f"Failed to get queue length for '{queue_name}': {e}")
        raise AzureServiceError(f"Failed to get queue length: {e}") from e


def dequeue_message_with_receipt(
    queue_name: str,
    config,
    visibility_timeout: int = 30
) -> Optional[tuple[QueueMessage, str, str]]:
    """Dequeue a message and return it with message_id and pop_receipt for deletion
    
    This is a convenience function that returns the message along with the Azure
    message metadata needed for deletion.
    
    Args:
        queue_name: Name of the queue
        config: Application configuration with Azure Storage connection string
        visibility_timeout: Message visibility timeout in seconds (default: 30)
        
    Returns:
        Tuple of (QueueMessage, message_id, pop_receipt) if message available,
        None if queue is empty
        
    Raises:
        AzureServiceError: If dequeue fails
    """
    try:
        queue_client = _get_queue_client(queue_name, config)
        
        messages = queue_client.receive_messages(
            messages_per_page=1,
            visibility_timeout=visibility_timeout
        )
        
        message_list = list(messages)
        if not message_list:
            return None
        
        azure_message = message_list[0]
        message = deserialize_message(azure_message.content)
        
        logger.info(
            f"Dequeued message from '{queue_name}': document_id={message.document_id}, "
            f"stage={message.stage}, attempt={message.attempt}"
        )
        
        return (message, azure_message.id, azure_message.pop_receipt)
    except ValueError as e:
        # ValueError for empty queue_name should be re-raised
        if "queue_name" in str(e).lower():
            raise
        # Other ValueErrors (from validation) - log and return None to skip poison message
        logger.warning(f"Invalid message format in queue '{queue_name}', skipping")
        return None
    except ValidationError:
        # Invalid message format - log and return None to skip poison message
        logger.warning(f"Invalid message format in queue '{queue_name}', skipping")
        return None
    except ResourceNotFoundError:
        logger.warning(f"Queue '{queue_name}' not found")
        return None
    except Exception as e:
        logger.error(f"Failed to dequeue message from '{queue_name}': {e}")
        raise AzureServiceError(f"Failed to dequeue message: {e}") from e

