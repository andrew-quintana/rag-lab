# Phase 1 Prompt — Query Generator (AI Node)

## Context

This prompt guides the implementation of **Phase 1: Query Generator (AI Node)** for the In-Corpus Evaluation Dataset Generation System. This phase implements an AI node that generates diverse, effective queries from Azure AI Search indexed documents.

**Related Documents:**
- @docs/initiatives/in_corpus_eval_setup/scoping/PRD001.md - Product requirements and functional specifications
- @docs/initiatives/in_corpus_eval_setup/scoping/RFC001.md - Technical architecture and design decisions
- @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md - Implementation breakdown (check off tasks as completed)
- @docs/initiatives/in_corpus_eval_setup/scoping/context.md - Project context and scope
- @docs/initiatives/in_corpus_eval_setup/intermediate/phase_0_handoff.md - Phase 0 handoff document

## Objectives

1. **Create Query Generator Module**: Implement AI node for generating diverse queries from Azure AI Search embeddings
2. **Sample Chunks**: Implement functionality to sample chunks from Azure AI Search index
3. **LLM Query Generation**: Use LLM to generate diverse, effective queries from chunk content
4. **Metadata Collection**: Link queries to source chunks for ground truth retrieval
5. **Output Generation**: Save queries to eval_inputs.json with proper structure

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md Phase 1 section as you complete them
- Update phase status when all implementation tasks are complete

### Validation
- **REQUIRED**: Validate all code with unit and integration tests
- **REQUIRED**: Test coverage must meet minimum 80% for query_generator.py module
- **REQUIRED**: Document any blockers or issues in fracas.md (root directory)

### Documentation
- **REQUIRED**: Create `intermediate/phase_1_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `intermediate/phase_1_testing.md` documenting test results and coverage
- **REQUIRED**: Create `intermediate/phase_1_handoff.md` summarizing what's needed for Phase 2

## Key References

### Implementation Components
- @backend/rag_eval/services/rag/search.py - Azure AI Search integration for sampling chunks
- @backend/rag_eval/services/shared/llm_providers.py - LLM provider abstraction for query generation
- @backend/rag_eval/core/config.py - Configuration management
- @backend/rag_eval/core/interfaces.py - Data structure definitions (Chunk, Query)
- Script location: `evaluations/_shared/scripts/query_generator.py`

### Directory Structure
- Create `evaluations/{eval_name}/inputs/` and `evaluations/{eval_name}/dataset/` directory structure
- Create `evaluations/_shared/scripts/query_generator.py` module
- Scripts are shared across all evaluations in `evaluations/_shared/scripts/`
- Input files go in `evaluations/{eval_name}/inputs/`
- Dataset files go in `evaluations/{eval_name}/dataset/`

## Phase 1 Tasks

### Implementation
1. Create `evaluations/{eval_name}/inputs/` and `evaluations/{eval_name}/dataset/` directory structure
2. Create `query_generator.py` module in `evaluations/_shared/scripts/` with proper imports and structure
3. Implement function to sample chunks from Azure AI Search index:
   - Use existing `retrieve_chunks()` or create new sampling function
   - Sample diverse chunks across different documents/content areas
4. Implement LLM-based query generation from chunks:
   - Use LLM provider to generate queries based on chunk content
   - Ensure queries are "In-Corpus" (answerable from indexed documents)
   - Generate diverse queries that are effective for RAG evaluation
5. Include metadata linking queries to source chunks:
   - Track source_chunk_ids for each generated query
   - Include document_id
6. Implement saving to `eval_inputs.json`:
   - Use proper JSON structure as specified in RFC001.md
   - Save to `evaluations/{eval_name}/inputs/eval_inputs.json`
7. Add error handling and retry logic:
   - Handle Azure AI Search failures
   - Handle LLM generation failures
   - Implement exponential backoff for retries
8. Add logging for query generation process:
   - Log chunk sampling progress
   - Log query generation progress
   - Log errors and warnings

### Testing
1. **REQUIRED**: Create unit tests for query generation function:
   - Test with sample chunks
   - Test query generation logic
   - Test metadata collection
   - Test JSON output structure
2. **REQUIRED**: Create integration tests with mocked Azure AI Search:
   - Mock Azure AI Search index access
   - Mock chunk sampling
   - Test end-to-end query generation
3. **REQUIRED**: Test coverage must meet minimum 80% for query_generator.py module
4. **REQUIRED**: Test with sample chunks from index
5. **REQUIRED**: Validate output JSON structure matches specification

## Success Criteria

- [ ] `query_generator.py` implemented with all required functionality
- [ ] `eval_inputs.json` generated with sample queries
- [ ] All queries are "In-Corpus" (answerable from indexed documents)
- [ ] Metadata properly links queries to source chunks
- [ ] Unit tests passing with >= 80% coverage
- [ ] Integration tests passing
- [ ] Phase 1 tasks completed and checked off in TODO001.md
- [ ] Phase 1 deliverables created in intermediate/ directory

## Next Phase

After completing Phase 1, proceed to **Phase 2: Dataset Generator** as specified in TODO001.md.

---
**Document Status**: Draft  
**Last Updated**: 2025-01-XX  
**Author**: Documentation Agent  
**Reviewers**: TBD

