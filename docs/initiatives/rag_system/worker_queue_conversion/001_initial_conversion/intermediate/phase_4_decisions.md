# Phase 4 Decisions — API Integration

## Overview

This document captures key decisions made during Phase 4 implementation that differ from or extend the original PRD/RFC specifications.

## Decisions

### 1. Upload Endpoint Response Model Simplification

**Decision**: Simplified `UploadResponse` to only include `document_id`, `status`, and optional `message` (removed `chunks_created`).

**Rationale**:
- Upload endpoint now returns immediately before processing, so chunk count is not available
- Status is always `"uploaded"` at upload time
- Users can query status endpoint to track progress and get final chunk count
- Simpler response model aligns with asynchronous flow

**Impact**: Upload response no longer includes `chunks_created` field. Status endpoint should be used to track processing progress.

### 2. Status Endpoint Error Details Extraction

**Decision**: Extract error details from `documents.metadata->error_details` or `documents.metadata->error_message` for failed statuses.

**Rationale**:
- Error details are stored in document metadata by workers when processing fails
- Provides users with actionable error information
- Supports debugging and troubleshooting
- Flexible metadata structure allows for different error detail formats

**Impact**: Status endpoint returns `error_details` field when document status starts with `"failed_"`.

### 3. Delete Endpoint Graceful Degradation

**Decision**: Continue with other deletion operations even if one fails (e.g., if Azure AI Search deletion fails, still delete from chunks table, storage, and database).

**Rationale**:
- Ensures maximum cleanup even if one system is unavailable
- Prevents partial deletion scenarios where some data remains
- Logs warnings for failed deletions but continues with others
- Returns counts of successfully deleted chunks from both systems

**Impact**: Delete endpoint attempts all deletion operations and reports success/failure for each.

### 4. QueryExecutor Usage in Status Endpoint

**Decision**: Create new `QueryExecutor` instance in status endpoint to query timestamps directly from database.

**Rationale**:
- Document model doesn't include timestamp fields (`parsed_at`, `chunked_at`, etc.)
- Direct database query is more efficient than loading full document model
- Allows querying only the fields needed for status response
- Maintains separation of concerns (status endpoint vs. document service)

**Impact**: Status endpoint queries database directly for timestamp fields rather than using DocumentService.

### 5. No Backward Compatibility Flag

**Decision**: Do not implement optional synchronous path for backward compatibility.

**Rationale**:
- Asynchronous flow is the primary use case going forward
- Workers are already implemented and tested (Phase 3)
- Simplifies API surface and reduces maintenance burden
- Migration period can be handled at deployment/infrastructure level if needed

**Impact**: Upload endpoint only supports asynchronous processing via queue. No synchronous fallback path.

### 6. Import Alias for Delete Functions

**Decision**: Use import aliases to distinguish between `delete_chunks_by_document_id` from `search.py` (Azure AI Search) and `persistence.py` (chunks table).

**Rationale**:
- Both modules have functions with the same name but different purposes
- Import aliases (`delete_chunks_from_ai_search` and `delete_chunks_from_db`) make code more readable
- Avoids naming conflicts and clarifies which deletion function is being called

**Impact**: Delete endpoint uses clearly named functions for each deletion operation.

### 7. Status Query Endpoint Location

**Decision**: Add status query endpoint to `documents.py` router rather than creating a separate router.

**Rationale**:
- Status is a property of a document, so it belongs with other document endpoints
- Keeps related endpoints together for better API organization
- Reduces number of router files to maintain
- Consistent with RESTful API design patterns

**Impact**: Status endpoint is at `GET /api/documents/{document_id}/status` alongside other document endpoints.

## No Decisions Required

The following aspects were implemented exactly as specified in RFC001.md:
- Upload endpoint enqueues to `ingestion-uploads` queue
- Status endpoint returns status, timestamps, and error details
- Delete endpoint removes chunks from both chunks table and Azure AI Search
- Response models match RFC specifications (with simplification noted above)

## Notes

- All API endpoints use mocked dependencies in unit tests
- Real Azure Storage Queue connection required for integration testing (Phase 5)
- Status endpoint requires database schema with timestamp columns (`parsed_at`, `chunked_at`, `embedded_at`, `indexed_at`)
- Delete endpoint requires both chunks table and Azure AI Search to be accessible

