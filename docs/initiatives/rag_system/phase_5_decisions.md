# Phase 5 Decisions — Prompt Template System

## Document Status
**Status**: Complete  
**Date**: 2025-01-27  
**Phase**: Phase 5 — Prompt Template System

## Overview
This document captures implementation decisions made during Phase 5 that are not already documented in PRD001.md or RFC001.md.

## Decisions

### Decision 1: In-Memory Caching Strategy
**Decision**: Use module-level dictionary for prompt template caching.

**Rationale**:
- Simple and effective for single-process application
- No external dependencies required
- Fast lookups (O(1) dictionary access)
- Cache persists for lifetime of Python process
- Sufficient for R&D/testing use case
- No cache invalidation needed (templates are versioned)

**Implementation**:
```python
# Module-level cache
_prompt_cache: Dict[str, str] = {}

# Check cache before database query
if version in _prompt_cache:
    return _prompt_cache[version]
```

**Trade-offs**:
- Cache doesn't persist across process restarts (acceptable for R&D)
- No cache invalidation mechanism (templates are versioned, so not needed)
- Single-process only (sufficient for current architecture)

### Decision 2: Placeholder Validation
**Decision**: Validate that templates contain required placeholders (`{query}` and `{context}`) before use.

**Rationale**:
- Prevents runtime errors when formatting prompts
- Clear error messages help debug template issues
- Catches template errors early in the pipeline
- Required placeholders are well-defined in RFC001

**Implementation**:
```python
required_placeholders = ["{query}", "{context}"]
missing_placeholders = [
    placeholder for placeholder in required_placeholders
    if placeholder not in template
]
if missing_placeholders:
    raise ValidationError(f"Missing required placeholders: {missing_placeholders}")
```

**Error Handling**:
- Raises `ValidationError` with clear message
- Lists all missing placeholders
- Includes prompt version in error message for debugging

### Decision 3: Empty Chunks Handling
**Decision**: Use placeholder text "(No context retrieved)" when no chunks are provided.

**Rationale**:
- Allows prompt construction to proceed even with empty retrieval
- Clear indication to LLM that no context is available
- Prevents empty string in context (which might confuse LLM)
- Enables testing of prompt construction without retrieval

**Implementation**:
```python
if retrieved_chunks:
    context = "\n\n".join([chunk.chunk_text for chunk in retrieved_chunks])
else:
    context = "(No context retrieved)"
    logger.warning("No retrieved chunks provided for prompt construction")
```

**Trade-offs**:
- LLM will see "(No context retrieved)" text (may affect generation)
- Alternative: raise error (rejected - too strict for R&D use case)
- Alternative: empty string (rejected - might confuse LLM)

### Decision 4: Context Concatenation Strategy
**Decision**: Concatenate chunk texts with double newlines (`\n\n`) as separator.

**Rationale**:
- Double newlines provide clear separation between chunks
- Preserves readability in prompt
- Standard practice in RAG systems
- Simple and effective

**Implementation**:
```python
context_parts = [chunk.chunk_text for chunk in retrieved_chunks]
context = "\n\n".join(context_parts)
```

**Alternative Considered**:
- Single newline: Less clear separation
- Numbered chunks: Adds complexity, not needed for initial version
- Chunk metadata in context: Out of scope for Phase 5

### Decision 5: Error Propagation Strategy
**Decision**: Propagate `ValidationError` and `DatabaseError` from `load_prompt_template()` to `construct_prompt()`.

**Rationale**:
- Clear error messages help debugging
- Errors are specific to the operation (validation vs database)
- Caller can handle errors appropriately
- Consistent with other components

**Implementation**:
- `load_prompt_template()` raises `ValidationError` for missing version
- `load_prompt_template()` raises `DatabaseError` for database failures
- `construct_prompt()` propagates these errors (doesn't catch and re-raise)
- Additional `ValueError` for empty query text

### Decision 6: Query Text Validation
**Decision**: Validate query text is not empty or whitespace-only before prompt construction.

**Rationale**:
- Prevents invalid prompts from being sent to LLM
- Catches errors early in the pipeline
- Clear error message helps debugging
- Consistent with other input validation

**Implementation**:
```python
if not query.text or not query.text.strip():
    raise ValueError("Query text cannot be empty")
```

**Error Handling**:
- Raises `ValueError` (not `ValidationError`) for consistency with other input validation
- Checks both empty string and whitespace-only strings

### Decision 7: Placeholder Replacement Order
**Decision**: Replace placeholders sequentially using `str.replace()`.

**Rationale**:
- Simple and straightforward
- No risk of placeholder conflicts (different names)
- Works reliably for all cases
- No need for complex templating library

**Implementation**:
```python
prompt = template.replace("{query}", query.text)
prompt = prompt.replace("{context}", context)
```

**Alternative Considered**:
- Template library (e.g., Jinja2): Adds dependency, not needed for simple case
- Regex replacement: More complex, no benefit
- Format string: Doesn't work with `{query}` syntax (conflicts with f-strings)

## Open Questions Resolved

### Q1: Should prompt templates be cached?
**Resolution**: Yes, using in-memory dictionary. Templates are versioned, so cache invalidation not needed. Cache improves performance by avoiding repeated database queries.

### Q2: What happens if prompt version doesn't exist?
**Resolution**: Raise `ValidationError` with clear error message. This is a configuration error that should be caught early.

### Q3: Should empty chunks be allowed?
**Resolution**: Yes, use placeholder text "(No context retrieved)". This allows prompt construction to proceed even when retrieval returns no results.

### Q4: Should placeholders be validated?
**Resolution**: Yes, validate required placeholders (`{query}` and `{context}`) before use. This prevents runtime errors.

## Notes

- All decisions align with PRD001.md and RFC001.md requirements
- No breaking changes to existing interfaces
- Implementation follows existing patterns from previous phases
- Error handling is comprehensive and tested
- Caching strategy is simple and effective for R&D use case

---

**Related Documents**:
- [PRD001.md](./PRD001.md) - Product requirements
- [RFC001.md](./RFC001.md) - Technical design
- [TODO001.md](./TODO001.md) - Implementation tasks
- [phase_5_testing.md](./phase_5_testing.md) - Testing summary
- [phase_5_handoff.md](./phase_5_handoff.md) - Handoff documentation


