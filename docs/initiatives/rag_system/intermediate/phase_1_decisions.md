# Phase 1 Decisions — Document Upload (Azure Blob Storage)

**Phase**: Phase 1 — Document Upload (Azure Blob Storage)  
**Date**: 2025-01-27  
**Status**: Complete

## Overview

This document captures implementation decisions made during Phase 1 that are not explicitly covered in PRD001.md or RFC001.md.

## Decisions

### Decision 1: Retry Logic Implementation

**Decision**: Implemented retry logic with exponential backoff as a private helper function `_retry_with_backoff()` within the storage module.

**Rationale**:
- Keeps retry logic close to where it's used
- Allows for consistent retry behavior across storage operations
- Can be easily extended or modified for future phases
- Follows RFC001.md requirement for retry logic with exponential backoff (3 retries max)

**Implementation Details**:
- Base delay: 1.0 second
- Exponential multiplier: 2^attempt (delays: 1s, 2s, 4s)
- Maximum retries: 3 (4 total attempts)
- Catches both `AzureError` and generic `Exception` to handle all failure modes

### Decision 2: Container Creation Strategy

**Decision**: Container creation is idempotent - checks if container exists before creating, and handles `ResourceExistsError` gracefully.

**Rationale**:
- Prevents errors when container already exists (concurrent creation scenarios)
- Aligns with RFC001.md requirement for idempotent operations
- Reduces unnecessary API calls by checking existence first

**Implementation Details**:
- Uses `get_container_properties()` to check existence
- Only creates container if `ResourceNotFoundError` is raised
- Handles `ResourceExistsError` as a no-op (container created concurrently)
- Wrapped in retry logic for transient failures

### Decision 3: Blob Overwrite Behavior

**Decision**: Upload operations use `overwrite=True` when uploading blobs.

**Rationale**:
- Allows re-uploading documents with the same document_id (useful for testing and updates)
- Prevents errors when document_id already exists
- Simplifies error handling (no need to check existence before upload)

**Implementation Details**:
- `upload_blob()` called with `overwrite=True` parameter
- If blob exists, it is replaced with new content
- Metadata is updated on overwrite

### Decision 4: Metadata Storage

**Decision**: Store document metadata (filename, document_id, upload_timestamp, content_length) as blob metadata.

**Rationale**:
- Enables retrieval of document information without downloading the blob
- Provides audit trail (upload timestamp)
- Standard Azure Blob Storage practice
- Useful for future operations (e.g., listing documents, filtering by date)

**Implementation Details**:
- Metadata stored as key-value pairs on blob
- Keys: `filename`, `document_id`, `upload_timestamp`, `content_length`
- Timestamp in ISO format (UTC)
- All metadata values are strings (Azure requirement)

### Decision 5: Error Handling Strategy

**Decision**: All errors are wrapped in `AzureServiceError` with descriptive messages, preserving original exception as cause.

**Rationale**:
- Consistent error handling across the codebase
- Clear error messages for debugging
- Preserves exception chain for troubleshooting
- Aligns with RFC001.md error handling requirements

**Implementation Details**:
- `AzureServiceError` raised for all Azure service failures
- Original exception preserved using `from e` syntax
- Validation errors (`ValueError`) are re-raised as-is (not wrapped)
- Logging at appropriate levels (info, warning, error)

### Decision 6: Download Function Implementation

**Decision**: Implemented `download_document_from_blob()` function even though it's marked as optional.

**Rationale**:
- Useful for testing and future use cases
- Completes the storage module API (upload/download pair)
- Minimal additional implementation effort
- May be needed for document reprocessing or retrieval

**Implementation Details**:
- Checks blob existence before download
- Raises `AzureServiceError` if blob not found
- Uses same retry logic as upload
- Returns raw bytes (no parsing or processing)

### Decision 7: Connection String Validation

**Decision**: Validate connection string and container name at function entry, raising `AzureServiceError` if missing.

**Rationale**:
- Fail fast with clear error messages
- Prevents confusing errors from Azure SDK when credentials are missing
- Helps developers identify configuration issues quickly

**Implementation Details**:
- Checks for empty strings (not just None)
- Raises `AzureServiceError` with descriptive message
- Validation happens before any Azure API calls

## Alternative Approaches Considered

### Alternative 1: Separate Retry Utility Module

**Rejected**: While a shared retry utility could be useful across phases, implementing it within the storage module keeps it simple and focused. Can be extracted to a shared utility in a future refactoring if needed.

### Alternative 2: Container Creation on Module Import

**Rejected**: Creating containers on import would add unnecessary overhead and complexity. Creating containers on-demand (when first upload occurs) is more efficient and follows lazy initialization principles.

### Alternative 3: Blob Existence Check Before Upload

**Rejected**: Using `overwrite=True` is simpler and more efficient than checking existence first. If future requirements need to prevent overwrites, this can be added as an optional parameter.

## Testing Decisions

### Decision 8: Mock-Based Unit Tests

**Decision**: All unit tests use mocks for Azure Blob Storage client, avoiding actual Azure API calls.

**Rationale**:
- Fast test execution
- No dependency on Azure credentials for unit tests
- Tests can run in CI/CD without Azure setup
- Allows testing of error scenarios that are difficult to reproduce with real services

### Decision 9: Separate Connection Tests

**Decision**: Connection tests are in a separate file (`test_storage_connection.py`) and warn but don't fail if credentials are missing.

**Rationale**:
- Allows developers to verify Azure setup without breaking test suite
- Follows RFC001.md requirement for connection tests
- Separates unit tests (always run) from integration tests (optional)

## Next Phase Considerations

- Storage module is ready for integration into upload pipeline
- Upload endpoint can now call `upload_document_to_blob()` before processing
- Download function available for future use cases (e.g., document retrieval, reprocessing)

---

**Document Status**: Complete  
**Last Updated**: 2025-01-27

