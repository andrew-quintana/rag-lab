# Phase 2 Prompt — Dataset Generator

## Context

This prompt guides the implementation of **Phase 2: Dataset Generator** for the In-Corpus Evaluation Dataset Generation System. This phase implements the script that generates complete evaluation datasets from eval_inputs.json, including retrieval, prompt loading, RAG output generation, and BEIR metrics computation.

**Related Documents:**
- @docs/initiatives/in_corpus_eval_setup/scoping/PRD001.md - Product requirements and functional specifications
- @docs/initiatives/in_corpus_eval_setup/scoping/RFC001.md - Technical architecture and design decisions
- @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md - Implementation breakdown (check off tasks as completed)
- @docs/initiatives/in_corpus_eval_setup/scoping/context.md - Project context and scope
- @docs/initiatives/in_corpus_eval_setup/intermediate/phase_1_handoff.md - Phase 1 handoff document

## Objectives

1. **Create Dataset Generator Script**: Implement script to generate complete evaluation datasets
2. **Load Inputs**: Load eval_inputs.json from same directory
3. **Retrieval Query Generation**: Generate retrieval_query (with LLM sanitization if needed)
4. **Ground Truth Context Retrieval**: Retrieve context using retrieval_query for BEIR metrics
5. **Prompt Loading**: Query Supabase for system_prompt and structured_prompt
6. **RAG Output Generation**: Generate outputs using RAG pipeline
7. **BEIR Metrics Computation**: Compute BEIR metrics using ground truth chunk IDs
8. **Dataset Assembly**: Assemble complete evaluation dataset with all required fields

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md Phase 2 section as you complete them
- Update phase status when all implementation tasks are complete

### Validation
- **REQUIRED**: Validate all code with unit and integration tests
- **REQUIRED**: Test coverage must meet minimum 80% for generate_eval_dataset.py module
- **REQUIRED**: Document any blockers or issues in fracas.md (root directory)

### Documentation
- **REQUIRED**: Create `intermediate/phase_2_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `intermediate/phase_2_testing.md` documenting test results and coverage
- **REQUIRED**: Create `intermediate/phase_2_handoff.md` summarizing what's needed for Phase 3

## Key References

### Implementation Components
- @backend/rag_eval/services/rag/search.py - Retrieval functionality
- @backend/rag_eval/services/rag/generation.py - RAG output generation
- @backend/rag_eval/services/rag/pipeline.py - RAG pipeline orchestration
- @backend/rag_eval/services/evaluator/beir_metrics.py - BEIR metrics computation
- @backend/rag_eval/db/queries.py - Supabase query executor
- @backend/rag_eval/services/shared/llm_providers.py - LLM provider for query sanitization

### Data Structures
- @backend/rag_eval/core/interfaces.py - RetrievalResult, ModelAnswer, BEIRMetricsResult
- eval_inputs.json structure from Phase 1
- eval_dataset.json structure from RFC001.md

## Phase 2 Tasks

### Implementation
1. Create `generate_eval_dataset.py` script in `evaluations/_shared/scripts/` directory
2. Implement loading of `eval_inputs.json`:
   - Load from `evaluations/{eval_name}/inputs/eval_inputs.json`
   - Validate JSON structure
   - Handle missing or invalid files
3. Implement retrieval_query generation:
   - Use same as input if no sanitization needed
   - Use LLM call if input needs sanitization (e.g., remove special characters, normalize)
   - Handle LLM generation failures
4. Implement ground truth context retrieval:
   - Use retrieval_query to retrieve chunks from Azure AI Search
   - Store retrieved chunks with similarity scores
   - This serves as ground truth for BEIR metrics
5. Implement Supabase query for system_prompt:
   - Query by prompt_type and name="{type}_system" (e.g., "rag_system")
   - Handle missing prompts gracefully
   - Use QueryExecutor from @backend/rag_eval/db/queries.py
6. Implement Supabase query for structured_prompt:
   - Query by prompt_type="RAG-enabled" and live=TRUE
   - Handle missing prompts gracefully
7. Integrate with RAG pipeline for output generation:
   - Use existing `generate_answer()` or `run_rag()` functions
   - Use system_prompt and structured_prompt from Supabase
   - Use retrieval_query for retrieval
8. Implement BEIR metrics computation:
   - Use existing `compute_beir_metrics()` function
   - Use ground truth chunk IDs from eval_inputs.json metadata
   - Use retrieved chunks from retrieval step
9. Initialize LLM/human evaluation fields as null:
   - llm_correctness, llm_hallucination, llm_risk_direction, llm_risk_magnitude
   - human_correctness, human_hallucination, human_risk_direction, human_risk_magnitude
10. Implement saving to `eval_dataset.json`:
    - Use proper JSON structure as specified in RFC001.md
    - Save to `evaluations/{eval_name}/dataset/eval_dataset.json`
11. Add error handling:
    - Handle missing prompts
    - Handle failed retrievals
    - Handle RAG generation failures
    - Handle BEIR metrics computation failures
12. Add logging for dataset generation process:
    - Log progress for each input
    - Log errors and warnings
    - Log timing information

### Testing
1. **REQUIRED**: Create unit tests for dataset generation function:
   - Test loading eval_inputs.json
   - Test retrieval_query generation
   - Test prompt loading
   - Test BEIR metrics computation
   - Test dataset assembly
2. **REQUIRED**: Create integration tests with mocked Supabase and RAG pipeline:
   - Mock Supabase prompt queries
   - Mock RAG pipeline components
   - Test end-to-end dataset generation
3. **REQUIRED**: Test coverage must meet minimum 80% for generate_eval_dataset.py module
4. **REQUIRED**: Test with sample eval_inputs.json
5. **REQUIRED**: Validate output JSON structure matches specification
6. **REQUIRED**: Test BEIR metrics computation accuracy

## Success Criteria

- [ ] `generate_eval_dataset.py` implemented with all required functionality
- [ ] `eval_dataset.json` generated with all required fields
- [ ] Ground truth context properly retrieved for BEIR metrics
- [ ] System and structured prompts successfully loaded from Supabase
- [ ] RAG outputs generated successfully
- [ ] BEIR metrics computed correctly
- [ ] Unit tests passing with >= 80% coverage
- [ ] Integration tests passing
- [ ] Phase 2 tasks completed and checked off in TODO001.md
- [ ] Phase 2 deliverables created in intermediate/ directory

## Next Phase

After completing Phase 2, proceed to **Phase 3: Output Generator** as specified in TODO001.md.

---
**Document Status**: Draft  
**Last Updated**: 2025-01-XX  
**Author**: Documentation Agent  
**Reviewers**: TBD

