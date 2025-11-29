# Phase 9.5 Decisions — Query Endpoint Testing

## Overview

This document captures key implementation decisions made during Phase 9.5 (Query Endpoint Testing) that are not already documented in [PRD001.md](./PRD001.md) or [RFC001.md](./RFC001.md).

**Status**: Complete  
**Date**: 2025-01-27  
**Related Documents**: [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md), [TODO001.md](./TODO001.md), [phase_9_decisions.md](./phase_9_decisions.md)

---

## Implementation Decisions

### 1. Simplified Mocking Strategy: Pipeline-Level Mocking

**Decision**: Mock `run_rag()` pipeline function instead of individual pipeline components.

**Rationale**:
- Query endpoint is simpler than upload endpoint (single pipeline call vs. multiple steps)
- `run_rag()` already orchestrates all pipeline components
- Reduces test complexity and maintenance
- Focuses tests on endpoint behavior, not pipeline internals
- Pipeline components already tested in previous phases

**Implementation**:
- Mock `rag_eval.api.routes.query.run_rag` directly
- Return mock `ModelAnswer` objects from mocked pipeline
- Test endpoint's request/response handling and error propagation
- Pipeline component tests remain in `test_rag_pipeline.py`

**Trade-offs**:
- ✅ Simpler test setup (one mock instead of multiple)
- ✅ Faster test execution
- ✅ Clear separation: endpoint tests vs. pipeline tests
- ⚠️ Less granular control over pipeline step failures
- ⚠️ Pipeline integration tested separately (acceptable)

**Note**: This differs from Phase 9 upload endpoint tests, which mock individual pipeline steps. The difference is justified because:
- Upload endpoint has multiple explicit steps in handler
- Query endpoint delegates to single `run_rag()` function
- Pipeline components already have comprehensive tests

---

### 2. Error Handling Strategy: Exception Propagation

**Decision**: Query endpoint propagates all exceptions from pipeline with appropriate HTTP status codes.

**Rationale**:
- Pipeline already handles error categorization
- Endpoint should preserve pipeline error semantics
- Clear error propagation for debugging
- Consistent with upload endpoint error handling

**Implementation**:
- `NotImplementedError` → `501 Not Implemented` (explicit handling)
- All other exceptions → `500 Internal Server Error` (generic handler)
- Error messages preserved from pipeline exceptions
- Full exception context logged for debugging

**Error Mapping**:
- `NotImplementedError`: 501 with "RAG pipeline not yet implemented"
- `AzureServiceError`: 500 with error message
- `ValidationError`: 500 with error message
- `DatabaseError`: 500 with error message
- `ValueError`: 500 with error message
- Generic `Exception`: 500 with error message

**Trade-offs**:
- ✅ Preserves pipeline error semantics
- ✅ Clear error categorization
- ✅ Full error context in logs
- ⚠️ Generic 500 for most errors (acceptable for API)

---

### 3. Request Validation: Pydantic Model Validation

**Decision**: Use Pydantic `QueryRequest` model for request validation.

**Rationale**:
- FastAPI automatically validates Pydantic models
- Type safety and validation built-in
- Clear API contract
- Consistent with upload endpoint pattern

**Implementation**:
- `QueryRequest` model with required `text` field
- Optional `prompt_version` field (defaults to "v1")
- FastAPI validates request before handler execution
- Invalid requests return 422 Unprocessable Entity

**Request Format**:
```json
{
  "text": "What is the coverage limit?",
  "prompt_version": "v1"
}
```

**Trade-offs**:
- ✅ Automatic validation
- ✅ Type safety
- ✅ Clear API contract
- ⚠️ Validation happens before handler (not tested in handler tests)

---

### 4. Response Format Standardization

**Decision**: Use consistent response format with required fields: `answer`, `query_id`, `prompt_version`.

**Rationale**:
- Predictable API contract
- Easy client parsing
- Comprehensive response metadata
- Consistent with upload endpoint pattern

**Implementation**:
- `QueryResponse` Pydantic model enforces response structure
- All responses include answer text and metadata
- Response fields match `ModelAnswer` fields (answer, query_id, prompt_version)

**Response Format**:
```json
{
  "answer": "The coverage limit is $500,000...",
  "query_id": "query_123",
  "prompt_version": "v1"
}
```

**Trade-offs**:
- ✅ Consistent API contract
- ✅ Comprehensive response data
- ✅ Type-safe with Pydantic validation
- ⚠️ Fixed response structure (less flexible)

---

### 5. Prompt Version Testing

**Decision**: Test query endpoint with different prompt versions.

**Rationale**:
- Query endpoint supports prompt version selection
- Important to verify version handling
- Tests prompt version parameter passing

**Implementation**:
- Test with default prompt version ("v1")
- Test with explicit prompt version ("v2")
- Verify prompt version passed correctly to pipeline
- Verify prompt version in response matches request

**Test Coverage**:
- Default prompt version (v1)
- Explicit prompt version (v2)
- Prompt version in response matches request

**Trade-offs**:
- ✅ Validates prompt version handling
- ✅ Tests parameter passing
- ⚠️ Limited to available prompt versions in test data

---

### 6. Empty Answer Handling

**Decision**: Allow empty answers in response (valid edge case).

**Rationale**:
- Empty answers are valid (LLM may return empty response)
- Endpoint should not fail on empty answers
- Tests edge case handling

**Implementation**:
- Test with empty answer text
- Verify response format still valid
- Verify empty answer doesn't cause errors

**Trade-offs**:
- ✅ Handles edge case gracefully
- ✅ Realistic scenario (LLM may return empty)
- ⚠️ Empty answers may indicate issues (but not endpoint failure)

---

### 7. Logging Strategy: Endpoint-Level Logging

**Decision**: Use standard Python logging for query endpoint observability.

**Rationale**:
- Consistent with upload endpoint logging
- Local logging sufficient for debugging
- Pipeline logging handled by `run_rag()` (Supabase logging in Phase 8)
- Endpoint logging focuses on request/response

**Implementation**:
- Logs query text on request receipt
- Logs errors with full exception context
- Uses `get_logger("api.routes.query")` for logging
- All logging at INFO level for production observability

**Logging Points**:
- Query received (query text)
- Errors with full exception context

**Trade-offs**:
- ✅ Fast logging (no database overhead)
- ✅ Sufficient for debugging
- ✅ Standard Python logging (familiar to developers)
- ⚠️ No centralized logging storage (logs in application logs only)

---

### 8. Testing Strategy: Unit and Integration Tests

**Decision**: Write comprehensive unit tests for handler function and integration tests for end-to-end flow.

**Rationale**:
- Unit tests verify handler logic with mocked pipeline
- Integration tests verify pipeline integration
- 100% error path coverage required
- Fast test execution with mocked services

**Implementation**:
- Unit tests: Test handler function directly with mocked `run_rag()`
- Integration tests: Test pipeline integration with mocked pipeline
- All error paths tested (pipeline errors, validation errors, database errors)
- Response format validation tests
- Uses pytest-asyncio for async test support

**Test Coverage**:
- ✅ Success path
- ✅ Pipeline errors (AzureServiceError)
- ✅ Validation errors (ValidationError)
- ✅ Database errors (DatabaseError)
- ✅ Generic errors (ValueError, Exception)
- ✅ NotImplementedError handling
- ✅ Different prompt versions
- ✅ Empty answer handling
- ✅ Response format validation

**Trade-offs**:
- ✅ Comprehensive test coverage
- ✅ Fast execution (mocked pipeline)
- ✅ 100% error path coverage
- ⚠️ No real pipeline tests (pipeline tested separately)

---

### 9. Async/Await Pattern

**Decision**: Use async/await for endpoint handler.

**Rationale**:
- FastAPI async support for non-blocking I/O
- Consistent with upload endpoint pattern
- Efficient FastAPI integration

**Implementation**:
- Endpoint handler is async function
- Pipeline call is synchronous (no async support in pipeline)
- Tests use pytest-asyncio for async test support

**Trade-offs**:
- ✅ Non-blocking endpoint
- ✅ Efficient FastAPI integration
- ⚠️ Pipeline call is synchronous (no async benefits for pipeline)

---

### 10. Config Mocking Strategy

**Decision**: Mock module-level config object for testing.

**Rationale**:
- Endpoint uses module-level `config = Config.from_env()`
- Tests need to mock config without loading environment
- Consistent with upload endpoint testing pattern

**Implementation**:
- Patch `rag_eval.api.routes.query.config` module-level object
- Mock config object with all required Azure credentials
- Config passed to pipeline (pipeline uses config internally)

**Trade-offs**:
- ✅ Isolated tests (no environment dependencies)
- ✅ Fast test execution
- ⚠️ Module-level config mocking (acceptable for testing)

---

## Design Patterns Used

### 1. Delegation Pattern
Query endpoint delegates to `run_rag()` pipeline function, focusing on request/response handling.

### 2. Error Propagation
Endpoint propagates pipeline errors with appropriate HTTP status codes.

### 3. Response Standardization
Consistent response format using Pydantic models for type safety.

---

## Testing Decisions

### 1. Mocked Pipeline for Unit Tests
**Decision**: Mock `run_rag()` pipeline function for fast, deterministic unit tests.

**Rationale**:
- Fast test execution
- No external dependencies
- Deterministic test results
- Tests focus on endpoint logic, not pipeline internals

### 2. Async Test Support
**Decision**: Use pytest-asyncio for async test support.

**Rationale**:
- Handler is async function
- Requires async test framework
- pytest-asyncio is standard for FastAPI testing

### 3. Error Path Coverage
**Decision**: Test all error handling paths (100% coverage).

**Rationale**:
- Ensures proper error handling
- Validates HTTP status codes
- Critical for production reliability

---

## Issues Resolved During Implementation

### 1. Config Mocking

**Issue**: Module-level config object comparison in test assertions.

**Root Cause**: Module-level config vs. fixture config created different mock objects.

**Resolution**: Verify function calls using call_args instead of exact object comparison.

**Impact**: All test assertions now pass correctly.

**Date Resolved**: 2025-01-27

---

## Future Considerations

### 1. Request Validation Testing
**Consideration**: Pydantic validation happens before handler execution.

**Current State**: Request validation not tested in handler tests (FastAPI handles it).

**Future Option**: Add FastAPI TestClient tests for request validation (requires full FastAPI stack).

### 2. Pipeline Integration Tests
**Consideration**: No real pipeline integration tests in endpoint tests.

**Current State**: Pipeline mocked in endpoint tests, tested separately in `test_rag_pipeline.py`.

**Future Option**: Add integration tests with real pipeline (requires Azure services).

### 3. Response Caching
**Consideration**: No response caching for identical queries.

**Current State**: Each query executes full pipeline.

**Future Option**: Add response caching for identical queries (performance optimization).

---

## Summary

Phase 9.5 implements query endpoint testing with a focus on:
- **Simplicity**: Pipeline-level mocking (simpler than upload endpoint)
- **Reliability**: Comprehensive error handling and validation
- **Observability**: Detailed logging for debugging
- **Testability**: Comprehensive test coverage with mocked services

All decisions align with PRD and RFC requirements, with emphasis on endpoint behavior testing rather than pipeline internals (which are tested separately).

