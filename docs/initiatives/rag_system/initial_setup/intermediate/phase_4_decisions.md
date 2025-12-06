# Phase 4 Decisions — Azure AI Search Integration

## Document Status
**Status**: Complete  
**Date**: 2025-01-27  
**Phase**: Phase 4 — Azure AI Search Integration

## Overview
This document captures implementation decisions made during Phase 4 that are not already documented in PRD001.md or RFC001.md.

## Decisions

### Decision 1: Vector Query Format
**Decision**: Use dictionary format for vector queries instead of `VectorizedQuery` class.

**Rationale**:
- Dictionary format is more compatible across Azure Search Documents SDK versions
- Simpler and more explicit
- Works reliably with SDK 11.4.0
- Easier to test and mock

**Implementation**:
```python
vector_query = {
    "kind": "vector",
    "vector": query_embedding,
    "k_nearest_neighbors": top_k,
    "fields": "embedding"
}
```

### Decision 2: ResourceNotFoundError Handling
**Decision**: Handle `ResourceNotFoundError` separately from retry logic - do not retry on index not found errors.

**Rationale**:
- `ResourceNotFoundError` is not a transient error - retrying won't help
- Missing index should return empty list gracefully (not fail)
- Improves user experience for empty index scenarios
- Reduces unnecessary retry attempts

**Implementation**:
- Catch `ResourceNotFoundError` before retry logic
- Return empty list when index doesn't exist
- Log warning but don't raise error

### Decision 3: Embedding Dimension Default
**Decision**: Default to 1536 dimensions for embedding vector field (matching text-embedding-3-small).

**Rationale**:
- text-embedding-3-small is the default embedding model
- 1536 dimensions is standard for this model
- Can be adjusted if different models are used
- Hardcoded for simplicity (can be made configurable in future)

**Implementation**:
- Index schema uses `vector_search_dimensions=1536`
- Documented in code comments

### Decision 4: Metadata Serialization
**Decision**: Serialize chunk metadata to JSON string for storage in Azure AI Search.

**Rationale**:
- Azure AI Search doesn't natively support complex JSON objects
- JSON string allows flexible metadata storage
- Easy to deserialize when retrieving
- Maintains type safety with proper parsing

**Implementation**:
- Store metadata as JSON string in index
- Parse JSON when retrieving chunks
- Handle JSON parsing errors gracefully (fallback to empty dict)

### Decision 5: Vector Search Algorithm Configuration
**Decision**: Use HNSW algorithm with cosine similarity metric for vector search.

**Rationale**:
- HNSW (Hierarchical Navigable Small World) is efficient for approximate nearest neighbor search
- Cosine similarity is standard for text embeddings
- Default parameters (m=4, efConstruction=400, efSearch=500) provide good balance
- Matches Azure AI Search best practices

**Implementation**:
- Configured in `_ensure_index_exists()` function
- Uses `HnswAlgorithmConfiguration` with cosine metric

### Decision 6: Empty Index Handling
**Decision**: Return empty list when index doesn't exist or is empty, rather than raising an error.

**Rationale**:
- Graceful degradation improves user experience
- Empty index is a valid state (no documents indexed yet)
- Allows query pipeline to work even before documents are uploaded
- Consistent with other "not found" scenarios

**Implementation**:
- `retrieve_chunks()` returns empty list for missing index
- `retrieve_chunks()` returns empty list for empty search results
- Logs warnings but doesn't fail

### Decision 7: Retry Logic Scope
**Decision**: Retry logic applies to all Azure AI Search operations except `ResourceNotFoundError`.

**Rationale**:
- Transient errors (network issues, rate limits) should be retried
- Permanent errors (missing index) should not be retried
- Consistent retry behavior across all operations
- Exponential backoff prevents overwhelming the service

**Implementation**:
- `_retry_with_backoff()` function handles retries
- Excludes `ResourceNotFoundError` from retries
- 3 retries max with exponential backoff (1s, 2s, 4s delays)

## Open Questions Resolved

### Q1: Should index creation be automatic or manual?
**Resolution**: Automatic with idempotent checks. Index is created on first use if it doesn't exist. Never performs destructive resets.

### Q2: How to handle schema changes?
**Resolution**: For Phase 4, schema is fixed. Future phases can implement schema versioning if needed. Current approach: create index with correct schema, never modify existing index.

### Q3: Should we support multiple indexes?
**Resolution**: Single index per configuration (out of scope for Phase 4). Can be extended in future if needed.

## Notes

- All decisions align with PRD001.md and RFC001.md requirements
- No breaking changes to existing interfaces
- Implementation follows existing patterns from Phase 3 (embeddings)
- Error handling is comprehensive and tested

---

**Related Documents**:
- [PRD001.md](./PRD001.md) - Product requirements
- [RFC001.md](./RFC001.md) - Technical design
- [TODO001.md](./TODO001.md) - Implementation tasks
- [phase_4_testing.md](./phase_4_testing.md) - Testing summary
- [phase_4_handoff.md](./phase_4_handoff.md) - Handoff documentation

