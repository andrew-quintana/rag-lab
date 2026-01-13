# Phase 1 Decisions — Query Generator (AI Node)

**Phase**: Phase 1 - Query Generator  
**Date**: 2025-01-XX  
**Status**: Complete

## Overview

This document captures implementation decisions made during Phase 1 that are not explicitly covered in PRD001.md or RFC001.md.

## Decisions

### 1. Chunk Sampling Strategy

**Decision**: Use random sampling from all chunks in the index rather than query-based sampling.

**Rationale**:
- Random sampling ensures diversity across different documents and content areas
- Query-based sampling would require seed queries, which defeats the purpose of automated query generation
- Azure AI Search supports `search_text="*"` to retrieve all chunks, which is efficient for sampling

**Implementation**: `sample_chunks_from_index()` function uses `search_text="*"` to retrieve chunks, then randomly samples from the results.

### 2. Query Generation Prompt

**Decision**: Use inline prompt template rather than storing in Supabase.

**Rationale**:
- Query generation is a one-time operation, not part of the RAG pipeline
- Inline prompt is simpler and doesn't require database access
- Prompt can be easily adjusted for different evaluation scenarios

**Implementation**: Prompt is constructed inline in `generate_query_from_chunk()` function with clear instructions for generating In-Corpus queries.

### 3. Error Handling for Individual Query Failures

**Decision**: Continue generating remaining queries even if individual query generation fails.

**Rationale**:
- Partial success is better than complete failure
- Individual query failures don't invalidate the entire dataset
- Logging captures failures for debugging

**Implementation**: `generate_queries_from_index()` uses try-except around individual query generation and continues with remaining queries.

### 4. Temperature for Query Generation

**Decision**: Use temperature=0.7 for query generation (higher than RAG generation's 0.1).

**Rationale**:
- Higher temperature promotes diversity in generated queries
- Query generation doesn't require strict reproducibility like RAG outputs
- Diversity is a key requirement for effective evaluation datasets

**Implementation**: `generate_query_from_chunk()` uses `temperature=0.7` when calling LLM.

### 5. Chunk Text Truncation

**Decision**: Truncate chunk text to 2000 characters in query generation prompt.

**Rationale**:
- Prevents token limit issues with very long chunks
- Most relevant information is typically in the first portion of chunks
- 2000 characters is sufficient for generating effective queries

**Implementation**: Chunk text is truncated to 2000 characters in the prompt: `{chunk.text[:2000]}`.

### 6. Directory Structure

**Decision**: Use `evaluations/in_corpus_eval/` as the default evaluation name.

**Rationale**:
- Matches the project naming convention
- Clear and descriptive
- Can be overridden via command-line argument

**Implementation**: Default `eval_name` parameter is `"in_corpus_eval"` in `main()` function.

### 7. Metadata Structure

**Decision**: Include `source_chunk_ids` as a list (even if single chunk) for consistency.

**Rationale**:
- Supports future expansion to multi-chunk queries
- Consistent structure across all queries
- Easier to process in downstream phases

**Implementation**: `source_chunk_ids` is always a list in metadata, even for single-chunk queries.

## No Decisions Required

The following aspects were implemented exactly as specified in PRD/RFC:
- JSON output structure matches RFC001.md specification
- Error handling and retry logic follow existing patterns
- Logging uses existing logger infrastructure
- Integration with Azure AI Search uses existing patterns

---
**Phase 1 Status**: ✅ Complete  
**Documentation Date**: 2025-01-XX







