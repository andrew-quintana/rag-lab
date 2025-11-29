# Phase 6 Decisions — LLM Answer Generation

## Document Status
**Status**: Complete  
**Date**: 2025-01-27  
**Phase**: Phase 6 — LLM Answer Generation

## Overview
This document captures implementation decisions made during Phase 6 that are not already documented in PRD001.md or RFC001.md.

## Decisions

### Decision 1: REST API Approach for Azure AI Foundry Generation
**Decision**: Use REST API with `requests` library instead of `azure-ai-inference` SDK.

**Rationale**:
- Azure AI Foundry uses OpenAI-compatible REST API, which is well-documented and straightforward
- `requests` library is already a dependency and provides sufficient functionality
- REST API approach is consistent with existing embeddings and chunking implementations
- Simpler error handling and retry logic implementation
- No need for additional SDK dependencies

**Implementation**: 
- Direct HTTP POST requests to `/openai/deployments/{model}/chat/completions` endpoint
- Uses `api-version=2024-02-15-preview` for compatibility
- Headers include `Content-Type: application/json` and `api-key: {api_key}`

**Alternative Considered**: Using `azure-ai-inference` SDK  
**Rejected**: Adds unnecessary dependency when REST API is sufficient and consistent with existing patterns.

### Decision 2: Generation Parameters Configuration
**Decision**: Use fixed generation parameters (temperature: 0.1, max_tokens: 1000) with model from config.

**Rationale**:
- Temperature 0.1 provides reproducibility while allowing some variation
- Max tokens 1000 is sufficient for most RAG answers
- Model selection from config allows flexibility without code changes
- Parameters are documented and can be adjusted if needed

**Implementation**:
```python
payload = {
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.1,
    "max_tokens": 1000
}
```

**Trade-offs**:
- Fixed parameters may not be optimal for all use cases
- Can be made configurable in future if needed
- Temperature 0.1 is low but not zero (allows some non-determinism)

### Decision 3: QueryExecutor Parameter Optional
**Decision**: Make `query_executor` parameter optional in `generate_answer()`, with automatic creation if not provided.

**Rationale**:
- Provides flexibility for callers who may already have a QueryExecutor
- Allows automatic creation for convenience
- Maintains backward compatibility
- Warns when auto-creating to encourage explicit passing

**Implementation**:
```python
def generate_answer(
    query: Query,
    retrieved_chunks: List[RetrievalResult],
    prompt_version: str,
    config,
    query_executor: Optional[QueryExecutor] = None
) -> ModelAnswer:
    if query_executor is None:
        logger.warning("QueryExecutor not provided, attempting to create from config")
        # Auto-create QueryExecutor
```

**Trade-offs**:
- Auto-creation may create connection management issues
- Caller should manage connection lifecycle
- Warning encourages explicit passing

### Decision 4: Query ID Generation Strategy
**Decision**: Generate query_id automatically if missing from Query object.

**Rationale**:
- Ensures every ModelAnswer has a query_id for traceability
- Uses existing `generate_id()` utility for consistency
- Allows queries without IDs to be processed
- Maintains backward compatibility

**Implementation**:
```python
query_id = query.query_id
if not query_id:
    query_id = generate_id("query")
    logger.debug(f"Generated query_id: {query_id} for query: {query.text[:50]}...")
```

**Trade-offs**:
- Generated IDs may not match external query tracking systems
- Caller can provide query_id if needed
- Generated IDs use "query_" prefix for clarity

### Decision 5: Response Validation Strategy
**Decision**: Validate LLM response structure and content before creating ModelAnswer.

**Rationale**:
- Prevents invalid responses from propagating through pipeline
- Clear error messages help debug API issues
- Catches API response format changes early
- Ensures answer text is not empty

**Implementation**:
```python
# Validate response structure
if "choices" not in result or not result["choices"]:
    raise ValueError(f"Invalid generation API response: missing 'choices' field")

# Extract and validate answer
choice = result["choices"][0]
if "message" not in choice or "content" not in choice["message"]:
    raise ValueError(f"Invalid generation API response: missing 'content' field")

answer_text = choice["message"]["content"]
if not answer_text or not answer_text.strip():
    raise ValueError("Generated answer is empty")
```

**Error Handling**:
- Raises `ValueError` for invalid responses
- Clear error messages include response structure
- Validates answer is not empty or whitespace-only

### Decision 6: Retry Logic with Exponential Backoff
**Decision**: Use same retry pattern as embeddings (3 retries, exponential backoff).

**Rationale**:
- Consistent with existing retry patterns in codebase
- Handles transient network failures
- Exponential backoff reduces load on API
- 3 retries is sufficient for most transient errors

**Implementation**:
```python
def _retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    for attempt in range(max_retries + 1):
        try:
            return func()
        except (requests.RequestException, Exception) as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(delay)
            else:
                raise AzureServiceError(...)
```

**Trade-offs**:
- Fixed retry count may not be optimal for all scenarios
- Exponential backoff may add latency
- Can be made configurable in future if needed

### Decision 7: Timestamp Generation
**Decision**: Use `datetime.now(timezone.utc)` for ModelAnswer timestamps.

**Rationale**:
- Provides timezone-aware timestamps
- Uses UTC for consistency
- Avoids deprecated `datetime.utcnow()`
- Standard practice for timestamps

**Implementation**:
```python
from datetime import datetime, timezone

answer = ModelAnswer(
    text=answer_text,
    query_id=query_id,
    prompt_version=prompt_version,
    retrieved_chunk_ids=retrieved_chunk_ids,
    timestamp=datetime.now(timezone.utc)
)
```

**Trade-offs**:
- Requires timezone import
- UTC timestamps may need conversion for display
- Timezone-aware is better for future compatibility

### Decision 8: Non-Determinism Documentation
**Decision**: Document that LLM generation is inherently non-deterministic, even with low temperature.

**Rationale**:
- Sets expectations for users
- Explains why same prompt may produce different answers
- Temperature 0.1 helps but doesn't eliminate non-determinism
- Important for testing and evaluation

**Implementation**:
```python
"""
**Non-Determinism Note**: While temperature is set to 0.1 for reproducibility,
LLM generation is inherently non-deterministic. The same prompt may produce
slightly different answers across multiple calls. This is acceptable and expected.
"""
```

**Trade-offs**:
- Non-determinism may affect testing
- Acceptable for R&D use case
- May need deterministic mode for production (future enhancement)

### Decision 9: Error Propagation Strategy
**Decision**: Propagate specific exceptions (AzureServiceError, ValidationError, DatabaseError, ValueError) without wrapping.

**Rationale**:
- Clear error messages help debugging
- Errors are specific to the operation
- Caller can handle errors appropriately
- Consistent with other components

**Implementation**:
```python
except AzureServiceError:
    # Re-raise AzureServiceError as-is
    raise
except ValidationError:
    # Re-raise ValidationError as-is
    raise
except DatabaseError:
    # Re-raise DatabaseError as-is
    raise
except ValueError:
    # Re-raise ValueError as-is
    raise
except Exception as e:
    # Wrap unexpected errors in AzureServiceError
    raise AzureServiceError(...) from e
```

**Trade-offs**:
- Specific exceptions provide better error handling
- Unexpected errors are wrapped for safety
- Consistent with existing patterns

### Decision 10: Retrieved Chunk IDs Extraction
**Decision**: Extract chunk IDs from RetrievalResult objects for ModelAnswer metadata.

**Rationale**:
- Provides traceability for which chunks were used
- Enables analysis of retrieval quality
- Required by ModelAnswer interface
- Simple extraction from existing objects

**Implementation**:
```python
retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieved_chunks]
```

**Trade-offs**:
- Empty list if no chunks retrieved (acceptable)
- Chunk IDs must be present in RetrievalResult objects
- Simple and efficient

## Open Questions Resolved

### Q1: Should generation parameters be configurable?
**Resolution**: Fixed for now (temperature: 0.1, max_tokens: 1000). Can be made configurable in future if needed. Model is configurable via config.

### Q2: What happens if LLM response is invalid?
**Resolution**: Raise `ValueError` with clear error message. This prevents invalid responses from propagating through pipeline.

### Q3: Should query_id be required or optional?
**Resolution**: Optional - generate if missing. This provides flexibility while ensuring traceability.

### Q4: How to handle empty LLM responses?
**Resolution**: Raise `ValueError` for empty responses. This ensures ModelAnswer always has valid text.

### Q5: Should retry logic be configurable?
**Resolution**: Fixed for now (3 retries, exponential backoff). Can be made configurable in future if needed.

## Notes

- All decisions align with PRD001.md and RFC001.md requirements
- No breaking changes to existing interfaces
- Implementation follows existing patterns from previous phases
- Error handling is comprehensive and tested
- Generation parameters are documented and can be adjusted
- Non-determinism is documented and acceptable for R&D use case

---

**Related Documents**:
- [PRD001.md](./PRD001.md) - Product requirements
- [RFC001.md](./RFC001.md) - Technical design
- [TODO001.md](./TODO001.md) - Implementation tasks
- [phase_6_testing.md](./phase_6_testing.md) - Testing summary
- [phase_6_handoff.md](./phase_6_handoff.md) - Handoff documentation

