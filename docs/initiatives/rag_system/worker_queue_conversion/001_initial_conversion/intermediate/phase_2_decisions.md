# Phase 2 Decisions — Queue Infrastructure

## Overview

This document captures key decisions made during Phase 2 implementation that differ from or extend the original PRD/RFC specifications.

## Decisions

### 1. Azure Storage Connection String Reuse

**Decision**: Reuse `azure_blob_connection_string` from Config for Azure Storage Queue operations.

**Rationale**: 
- Azure Storage Account connection strings work for both Blob Storage and Queue Storage services
- No need to add a separate `azure_queue_connection_string` field to Config
- Simplifies configuration management

**Impact**: Queue client uses `config.azure_blob_connection_string` to connect to queues.

### 2. QueueMessage Dataclass Naming

**Decision**: Use `QueueMessage` as the dataclass name (not `IngestionQueueMessage` or similar).

**Rationale**:
- Simple, clear name that matches the module purpose
- Avoids naming conflicts with Azure SDK's `QueueMessage` by not importing it
- Consistent with other dataclass naming in the codebase

**Impact**: All message validation and serialization uses `QueueMessage` dataclass.

### 3. Message Validation Strategy

**Decision**: Implement validation in both `QueueMessage.__post_init__` and `validate_message()` function.

**Rationale**:
- `__post_init__` ensures dataclass instances are always valid
- `validate_message()` provides explicit validation for dictionary inputs
- Dual validation ensures consistency regardless of message creation path

**Impact**: Messages are validated at creation time and when deserializing from JSON.

### 4. Error Handling for Invalid Messages

**Decision**: Return `None` for invalid messages in dequeue/peek operations instead of raising exceptions.

**Rationale**:
- Invalid messages (poison messages) should not crash workers
- Returning `None` allows workers to skip invalid messages and continue processing
- Logs warnings for observability without breaking the pipeline

**Impact**: Workers can gracefully handle corrupted queue messages.

### 5. Queue Auto-Creation

**Decision**: Automatically create queues if they don't exist when enqueuing messages.

**Rationale**:
- Simplifies deployment - queues are created on first use
- Reduces manual setup steps
- Idempotent operation (no error if queue already exists)

**Impact**: Queues are created automatically, but manual creation via Azure Portal is still recommended for production.

### 6. Dead-Letter Queue Metadata Enhancement

**Decision**: Automatically add `dead_letter_reason` and `dead_lettered_at` to message metadata when sending to dead-letter queue.

**Rationale**:
- Provides debugging context for why message was dead-lettered
- Timestamp helps with troubleshooting and retry decisions
- Metadata is preserved for manual inspection

**Impact**: Dead-letter messages include additional debugging information.

### 7. Convenience Function: dequeue_message_with_receipt

**Decision**: Add `dequeue_message_with_receipt()` function that returns message with Azure message metadata.

**Rationale**:
- Workers need message_id and pop_receipt to delete messages after processing
- Convenience function reduces boilerplate code in workers
- Maintains separation of concerns (queue operations vs. worker logic)

**Impact**: Workers can easily get message and deletion metadata in one call.

### 8. Exception Handling for Empty Queue Names

**Decision**: Raise `ValueError` for empty queue names (not `AzureServiceError`).

**Rationale**:
- Empty queue name is a validation error, not an Azure service error
- `ValueError` is more appropriate for invalid input parameters
- Consistent with Python exception handling conventions

**Impact**: Empty queue names raise `ValueError` which can be caught separately from Azure errors.

### 9. Datetime Usage

**Decision**: Use `datetime.now(timezone.utc)` instead of deprecated `datetime.utcnow()`.

**Rationale**:
- Follows Python best practices for timezone-aware datetime objects
- Avoids deprecation warnings
- More explicit about timezone handling

**Impact**: Dead-letter timestamps use timezone-aware datetime objects.

## No Decisions Required

The following aspects were implemented exactly as specified in RFC001.md:
- Queue names (`ingestion-uploads`, `ingestion-chunking`, etc.)
- Message schema structure and field types
- Queue operation signatures
- Error handling patterns (retry logic deferred to Phase 3)

## Notes

- All queue operations use mocked Azure Storage Queue SDK in tests
- Real Azure Storage Account connection required for integration testing (Phase 5)
- Queue creation via Azure Portal or Infrastructure as Code is still recommended for production

