# Phase 7 Decisions — Pipeline Orchestration

**Date**: 2025-01-27  
**Phase**: Phase 7 — Pipeline Orchestration  
**Component**: `rag_eval/services/rag/pipeline.py`

## Overview

This document captures implementation decisions made during Phase 7 that are not explicitly covered in PRD001.md or RFC001.md.

## Key Decisions

### 1. Database Connection Management

**Decision**: Database connection is created within the pipeline and closed in a `finally` block to ensure cleanup even on errors.

**Rationale**: 
- The pipeline needs a `QueryExecutor` for prompt template loading (Phase 5)
- Database connection should be managed by the pipeline to ensure proper cleanup
- Using a `finally` block ensures connection is closed even if generation fails

**Implementation**:
- `DatabaseConnection` is created at the start of Step 3-4 (generation)
- Connection is closed in a `finally` block to handle both success and error cases
- Warning is logged if connection close fails (non-fatal)

### 2. Latency Measurement Strategy

**Decision**: Pipeline measures and logs latency for each step individually, plus total pipeline latency.

**Rationale**:
- Individual step latencies help identify bottlenecks
- Total latency provides overall pipeline performance metrics
- Logging at INFO level ensures observability in production

**Implementation**:
- Each step measures its own latency using `time.time()`
- Latency is logged immediately after each step completes
- Final log includes breakdown: `embedding=X.XXs, retrieval=X.XXs, generation=X.XXs, logging=X.XXs`

### 3. Error Handling and Propagation

**Decision**: Errors from individual steps are wrapped with context but preserve original exception types where appropriate.

**Rationale**:
- `ValidationError` and `DatabaseError` are preserved as-is for callers to handle specifically
- `AzureServiceError` is used for Azure service failures (embedding, search, generation)
- All errors are logged with full context before being raised

**Implementation**:
- Each step has its own try/except block
- Errors are logged with step context (e.g., "[Step 1/5] Query embedding generation failed")
- Original exception types are preserved where semantically meaningful
- Unexpected errors are wrapped in `AzureServiceError` with context

### 4. Query ID Preservation

**Decision**: Pipeline ensures the `query_id` in the returned `ModelAnswer` matches the query's `query_id` (or generated one).

**Rationale**:
- `generate_answer()` may generate its own query_id if the query doesn't have one
- Pipeline should ensure consistency: the answer's query_id should match the query's query_id
- This ensures traceability and consistency

**Implementation**:
- After `generate_answer()` returns, pipeline sets `answer.query_id = query_id`
- This overwrites any query_id that `generate_answer()` may have generated
- Ensures the answer always references the correct query_id

### 5. Logging Stub for Phase 8

**Decision**: Logging to Supabase is stubbed in Phase 7, with a TODO comment indicating Phase 8 implementation.

**Rationale**:
- Phase 8 will implement full Supabase logging
- Stubbing allows Phase 7 to complete without blocking on Phase 8
- Logging failures are non-fatal (warnings only) to not break the pipeline

**Implementation**:
- Step 5 logs a debug message indicating logging is stubbed
- Logging latency is still measured for consistency
- Actual logging calls are commented out with TODO for Phase 8

### 6. Config Default Handling

**Decision**: Pipeline uses `Config.from_env()` as default if config is not provided.

**Rationale**:
- Allows pipeline to be called without explicit config for convenience
- Maintains backward compatibility with existing code
- Follows the pattern established in other components

**Implementation**:
- `config: Optional[Config] = None` parameter
- If `config is None`, calls `Config.from_env()`
- This happens early in the function before any component calls

### 7. Deterministic Execution Order

**Decision**: Pipeline executes steps in a fixed, deterministic order with no parallelization.

**Rationale**:
- Deterministic execution ensures reproducible results for testing
- Serial execution simplifies error handling and debugging
- Matches the design goal of simplicity and transparency

**Implementation**:
- Steps are executed sequentially: embedding → retrieval → generation → logging
- No async/await or parallel processing
- Each step completes before the next begins

## Testing Decisions

### 1. Comprehensive Mocking Strategy

**Decision**: All external dependencies (Azure services, database) are mocked in unit tests.

**Rationale**:
- Unit tests should be fast and not depend on external services
- Mocking allows testing error scenarios that would be difficult with real services
- Tests can verify exact call arguments and data flow

**Implementation**:
- `generate_query_embedding`, `retrieve_chunks`, `generate_answer` are all mocked
- `DatabaseConnection` is mocked to avoid actual database connections
- Tests verify component calls with correct arguments

### 2. Error Path Coverage

**Decision**: All error paths are tested with 100% coverage requirement.

**Rationale**:
- Error handling is critical for production reliability
- Comprehensive error testing ensures graceful failure handling
- Meets the validation requirement of 100% error path coverage

**Implementation**:
- Tests for embedding errors, retrieval errors, generation errors
- Tests for validation errors, database errors
- Tests for empty query text, whitespace-only queries
- All error paths verified to propagate correctly

### 3. Latency Measurement Testing

**Decision**: Latency measurement is tested by mocking `time.time()` with a sequence of values.

**Rationale**:
- Verifies that latency is measured correctly
- Ensures all time.time() calls are accounted for
- Tests that latency logging works as expected

**Implementation**:
- Mock `time.time()` with a sequence of 10 values (one per call)
- Verify that all 10 calls are made
- Test passes if pipeline completes successfully with mocked time

## Documentation Decisions

### 1. Comprehensive Docstrings

**Decision**: `run_rag()` function has extensive docstring covering all aspects.

**Rationale**:
- Pipeline orchestration is complex and needs thorough documentation
- Docstring includes examples, error cases, and implementation details
- Helps future developers understand the pipeline flow

**Implementation**:
- Docstring includes purpose, execution order, latency measurement, error handling
- Includes Args, Returns, Raises sections
- Includes example usage

## Open Questions / Future Considerations

1. **Parallel Execution**: Could embedding and retrieval be parallelized? (Not implemented in Phase 7)
2. **Caching**: Should query embeddings be cached? (Not implemented in Phase 7)
3. **Retry Strategy**: Should the pipeline itself retry on transient failures? (Currently relies on component-level retries)
4. **Metrics Export**: Should latency metrics be exported to a metrics system? (Currently only logged)

## Related Documents

- [PRD001.md](./PRD001.md) - Product requirements
- [RFC001.md](./RFC001.md) - Technical design
- [TODO001.md](./TODO001.md) - Implementation tasks
- [phase_7_testing.md](./phase_7_testing.md) - Testing summary
- [phase_7_handoff.md](./phase_7_handoff.md) - Handoff documentation

