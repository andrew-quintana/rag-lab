# Phase 3 Prompt — Output Generator

## Context

This prompt guides the implementation of **Phase 3: Output Generator** for the In-Corpus Evaluation Dataset Generation System. This phase implements the script that generates RAG outputs for evaluation dataset entries using system prompts and structured prompts.

**Related Documents:**
- @docs/initiatives/in_corpus_eval_setup/scoping/PRD001.md - Product requirements and functional specifications
- @docs/initiatives/in_corpus_eval_setup/scoping/RFC001.md - Technical architecture and design decisions
- @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md - Implementation breakdown (check off tasks as completed)
- @docs/initiatives/in_corpus_eval_setup/scoping/context.md - Project context and scope
- @docs/initiatives/in_corpus_eval_setup/intermediate/phase_2_handoff.md - Phase 2 handoff document

## Objectives

1. **Create Output Generator Script**: Implement script to generate RAG outputs for evaluation dataset entries
2. **Load Dataset**: Load eval_dataset.json from same directory
3. **RAG Output Generation**: Generate outputs using system_prompt and structured_prompt from dataset
4. **Dataset Update**: Update output field in dataset entries
5. **Batch Processing**: Support processing multiple entries efficiently

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md Phase 3 section as you complete them
- Update phase status when all implementation tasks are complete

### Validation
- **REQUIRED**: Validate all code with unit and integration tests
- **REQUIRED**: Test coverage must meet minimum 80% for generate_eval_outputs.py module
- **REQUIRED**: Document any blockers or issues in fracas.md (root directory)

### Documentation
- **REQUIRED**: Create `intermediate/phase_3_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `intermediate/phase_3_testing.md` documenting test results and coverage
- **REQUIRED**: Create `intermediate/phase_3_handoff.md` summarizing what's needed for Phase 4

## Key References

### Implementation Components
- @backend/rag_eval/services/rag/generation.py - RAG output generation functions
- @backend/rag_eval/services/rag/pipeline.py - RAG pipeline orchestration
- @backend/rag_eval/core/interfaces.py - ModelAnswer, Query, RetrievalResult
- @backend/rag_eval/core/config.py - Configuration management

### Data Structures
- eval_dataset.json structure from Phase 2
- System prompt and structured prompt from dataset entries

## Phase 3 Tasks

### Implementation
1. Create `generate_eval_outputs.py` script in `evaluations/{eval_name}/` directory
2. Implement loading of `eval_dataset.json`:
   - Load from same directory as script
   - Validate JSON structure
   - Handle missing or invalid files
3. Implement RAG output generation:
   - For each entry in dataset:
     - Extract system_prompt and structured_prompt from entry
     - Extract retrieval_query from entry
     - Extract retrieved context from entry
     - Use RAG pipeline to generate output:
       - Option 1: Use `generate_answer()` with custom prompts
       - Option 2: Use `run_rag()` and override prompts
       - Option 3: Create custom generation function that uses prompts from dataset
   - Handle generation failures gracefully
4. Implement updating output field in dataset:
   - Update `content.output` field for each entry
   - Preserve all other fields
5. Implement saving updated dataset:
   - Save to same location as input
   - Use proper JSON formatting
6. Add error handling:
   - Handle generation failures for individual entries
   - Continue processing remaining entries on failure
   - Log errors for failed entries
7. Add logging for output generation process:
   - Log progress for each entry
   - Log errors and warnings
   - Log timing information
8. Support batch processing:
   - Process multiple entries efficiently
   - Support resuming from failures
   - Support processing subset of entries

### Testing
1. **REQUIRED**: Create unit tests for output generation function:
   - Test loading eval_dataset.json
   - Test output generation logic
   - Test dataset update logic
   - Test error handling
2. **REQUIRED**: Create integration tests with mocked RAG pipeline:
   - Mock RAG pipeline components
   - Test end-to-end output generation
   - Test with sample eval_dataset.json
3. **REQUIRED**: Test coverage must meet minimum 80% for generate_eval_outputs.py module
4. **REQUIRED**: Test with sample eval_dataset.json
5. **REQUIRED**: Validate output field updates correctly

## Success Criteria

- [ ] `generate_eval_outputs.py` implemented with all required functionality
- [ ] Outputs generated successfully for sample dataset entries
- [ ] Output field updated correctly in dataset
- [ ] Error handling works correctly for failed generations
- [ ] Batch processing works efficiently
- [ ] Unit tests passing with >= 80% coverage
- [ ] Integration tests passing
- [ ] Phase 3 tasks completed and checked off in TODO001.md
- [ ] Phase 3 deliverables created in intermediate/ directory

## Next Phase

After completing Phase 3, proceed to **Phase 4: System Prompt Management** as specified in TODO001.md.

---
**Document Status**: Draft  
**Last Updated**: 2025-01-XX  
**Author**: Documentation Agent  
**Reviewers**: TBD

