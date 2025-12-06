# Phase 2 Decisions — Extraction, Preprocessing, and Chunking

**Phase**: Phase 2 — Extraction, Preprocessing, and Chunking  
**Date**: 2025-01-27  
**Status**: Complete

## Overview

This document captures implementation decisions made during Phase 2 that are not explicitly covered in PRD001.md or RFC001.md.

## Decisions

### Decision 1: No Retry Logic in Ingestion Module

**Decision**: The ingestion module (`ingestion.py`) does not implement retry logic with exponential backoff.

**Rationale**:
- Azure Document Intelligence client handles retries internally
- The `begin_analyze_document()` method returns a poller that manages its own retry logic
- Adding additional retry logic would be redundant and could cause issues
- Error handling is still comprehensive (all errors wrapped in `AzureServiceError`)

**Implementation Details**:
- Errors from Azure Document Intelligence are caught and wrapped in `AzureServiceError`
- Original exception information is preserved in error messages
- Logging at appropriate levels (info for normal flow, error for failures)

**Note for Future**: If retry logic is needed for ingestion, it should be added at the poller level or as a wrapper around the entire extraction function, not at the client initialization level.

### Decision 2: LLM Chunking Fallback Strategy

**Decision**: LLM-based chunking (`chunk_text_with_llm()`) automatically falls back to fixed-size chunking on any error.

**Rationale**:
- Ensures chunking always succeeds (critical for pipeline reliability)
- Fixed-size chunking is deterministic and doesn't require external services
- Provides graceful degradation when Azure AI Foundry is unavailable
- Aligns with RFC001.md requirement for fallback behavior

**Implementation Details**:
- Catches `json.JSONDecodeError` (invalid LLM response) and falls back
- Catches all other exceptions (network errors, HTTP errors, etc.) and falls back
- Logs warnings when fallback occurs (doesn't fail silently)
- Fallback uses default parameters (chunk_size=1000, overlap=200)

**Future Consideration**: Could add configuration option to disable fallback (fail fast) for debugging purposes.

### Decision 3: Fixed-Size Chunking as Default

**Decision**: The main `chunk_text()` function defaults to fixed-size chunking (`use_llm=False`).

**Rationale**:
- Fixed-size chunking is deterministic (required for reproducible testing per PRD001.md NFR5)
- No external dependencies (faster, more reliable)
- Recommended for most use cases per RFC001.md Decision 1
- LLM chunking can be enabled when semantic coherence is critical

**Implementation Details**:
- Default parameter: `use_llm=False`
- Fixed-size chunking uses default parameters: `chunk_size=1000`, `overlap=200`
- LLM chunking parameters (`chunk_size`, `overlap`) are ignored when `use_llm=True`

### Decision 4: Text Truncation for LLM Chunking

**Decision**: Text longer than 10,000 characters is truncated when sent to LLM for chunking.

**Rationale**:
- LLM token limits and cost considerations
- Very long texts would exceed model context windows
- First 10,000 characters are usually sufficient for chunking strategy
- Full text is still chunked (truncation only affects LLM prompt)

**Implementation Details**:
- Only first 10,000 characters sent to LLM in prompt
- Full text is passed to fallback fixed-size chunking if LLM fails
- Truncation happens silently (no warning logged)

**Future Consideration**: Could make truncation limit configurable or implement sliding window approach for very long documents.

### Decision 5: Chunk ID Generation Strategy

**Decision**: Chunk IDs are generated sequentially as `chunk_0`, `chunk_1`, `chunk_2`, etc.

**Rationale**:
- Simple and predictable
- Sequential IDs are easy to debug and trace
- No external dependencies (no UUID generation needed)
- Sufficient for RAG pipeline use cases

**Implementation Details**:
- Fixed-size chunking: Sequential IDs based on chunk index
- LLM chunking: Sequential IDs based on chunk index in LLM response
- IDs are strings, not integers (for consistency with other systems)

**Future Consideration**: Could add document_id prefix (e.g., `doc_123_chunk_0`) for better traceability across documents.

### Decision 6: Metadata Preservation in Chunks

**Decision**: Chunk metadata includes chunking method, position information, and document_id.

**Rationale**:
- Enables traceability (know which method created each chunk)
- Position information useful for debugging and reconstruction
- Document_id links chunks to source document
- Metadata is extensible (can add more fields in future)

**Implementation Details**:
- Fixed-size chunks: `{"start": int, "end": int, "chunking_method": "fixed_size"}`
- LLM chunks: `{"start_index": int, "chunking_method": "azure_ai_foundry_llm"}`
- All chunks include `document_id` field (can be None)

### Decision 7: Empty Document Handling

**Decision**: Empty documents produce empty chunk lists (no single empty chunk).

**Rationale**:
- Empty chunks are not useful for retrieval
- Empty list is clearer than list with empty chunk
- Consistent with "no content = no chunks" principle

**Implementation Details**:
- `chunk_text_fixed_size("")` returns `[]`
- Empty text in LLM chunking would also produce empty list (after fallback)
- No special handling needed - algorithm naturally produces empty list

### Decision 8: Connection Test Strategy

**Decision**: Connection tests warn but don't fail if credentials are missing, and are included in the same test files as unit tests.

**Rationale**:
- Allows developers to verify Azure setup without breaking test suite
- Follows RFC001.md requirement for connection tests
- Keeps related tests together (unit tests and connection tests in same file)
- Matches pattern from Phase 1

**Implementation Details**:
- Connection tests use `pytest.skip()` with warnings if credentials missing
- Tests are in same files as unit tests (separate test classes)
- Tests attempt real Azure service calls if credentials available
- Connection status documented in test output

### Decision 9: LLM Chunking Tests Removed from Phase 2

**Decision**: LLM chunking tests removed from Phase 2 test suite - focusing only on deterministic fixed-size chunking.

**Rationale**:
- LLM chunking requires Azure AI Foundry credentials and is non-deterministic
- Phase 2 focuses on deterministic chunking for reproducible testing (NFR5 requirement)
- Fixed-size chunking is the recommended default and covers all core functionality
- LLM chunking can be tested in later phases when integration testing is appropriate

**Implementation Details**:
- Removed `TestChunkTextWithLLM` class (all LLM-based chunking tests)
- Removed `TestAzureAIFoundryConnection` class (LLM connection test)
- Kept only fixed-size chunking tests (14 tests total)
- `chunk_text_with_llm()` function remains in codebase for future use
- All tests focus on deterministic behavior validation

## Alternative Approaches Considered

### Alternative 1: Retry Logic in Ingestion Module

**Rejected**: Azure Document Intelligence client already handles retries internally. Adding additional retry logic would be redundant and could interfere with the client's own retry mechanisms.

### Alternative 2: Fail Fast for LLM Chunking Errors

**Rejected**: Automatic fallback to fixed-size chunking ensures pipeline reliability. Failing fast would break the pipeline when Azure AI Foundry is temporarily unavailable, which is not acceptable for a production-like system.

### Alternative 3: LLM Chunking as Default

**Rejected**: Fixed-size chunking is deterministic and recommended for reproducible testing. LLM chunking should be opt-in for cases where semantic coherence is critical.

### Alternative 4: UUID-Based Chunk IDs

**Rejected**: Sequential IDs are simpler, more predictable, and sufficient for RAG pipeline use cases. UUIDs would add unnecessary complexity without clear benefit.

### Alternative 5: Separate Connection Test Files

**Rejected**: Keeping connection tests in the same files as unit tests keeps related tests together and follows the pattern established in Phase 1.

## Testing Decisions

### Decision 9: Comprehensive Mock-Based Unit Tests

**Decision**: All unit tests use mocks for Azure services, avoiding actual API calls.

**Rationale**:
- Fast test execution
- No dependency on Azure credentials for unit tests
- Tests can run in CI/CD without Azure setup
- Allows testing of error scenarios that are difficult to reproduce with real services

### Decision 10: Deterministic Behavior Validation

**Decision**: Fixed-size chunking tests explicitly validate deterministic behavior (same input = same output).

**Rationale**:
- Critical requirement from PRD001.md NFR5 (Determinism)
- Ensures reproducible RAG pipeline testing
- Validates that chunking is truly deterministic

**Implementation Details**:
- Test runs chunking twice with same input
- Verifies identical chunks (text, IDs, metadata)
- Tests included in `test_rag_chunking.py`

## Next Phase Considerations

- Ingestion and chunking modules are ready for integration into upload pipeline
- Upload endpoint can call `ingest_document()` and `chunk_text()` in sequence
- Chunking produces `Chunk` objects ready for embedding generation (Phase 3)
- Fixed-size chunking is deterministic and recommended for testing

---

**Document Status**: Complete  
**Last Updated**: 2025-01-27



