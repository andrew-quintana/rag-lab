# Phase 6 Prompt — LLM Answer Generation

## Purpose
Implement LLM answer generation using Azure AI Foundry. This completes the query pipeline by generating grounded answers from retrieved context.

## Context References

**Primary Documentation:**
- @docs/initiatives/rag_system/PRD001.md - Product requirements (FR2: Query Pipeline)
- @docs/initiatives/rag_system/RFC001.md - Technical design (Phase 4: LLM Answer Generation)
- @docs/initiatives/rag_system/TODO001.md - Phase 6 implementation tasks (check off as completed)
- @docs/initiatives/rag_system/context.md - System context

**Codebase References:**
- @backend/rag_eval/core/config.py - Azure AI Foundry configuration
- @backend/rag_eval/core/interfaces.py - Query, RetrievalResult, ModelAnswer interfaces
- @backend/rag_eval/core/exceptions.py - Error handling (AzureServiceError)
- @backend/rag_eval/services/rag/generation.py - Prompt construction (Phase 5)
- @backend/rag_eval/utils/ids.py - Query ID generation utility

**Implementation Target:**
- **EXISTING FILE**: `backend/rag_eval/services/rag/generation.py` - LLM generation function

## Phase Objectives

1. **Implement Answer Generation**: Generate answers using Azure AI Foundry LLM
2. **Prompt Integration**: Use Phase 5 prompt construction
3. **Model Configuration**: Configure generation parameters (temperature: 0.1, max_tokens: 1000)
4. **Response Assembly**: Create ModelAnswer objects with metadata
5. **Testing**: Write comprehensive unit tests with mocked Azure services

## Execution Requirements

### Checklist Management
- **REQUIRED**: Check off all [ ] checkboxes in @docs/initiatives/rag_system/TODO001.md Phase 6 section as tasks are completed
- Update TODO001.md directly when each task is finished

### Validation Requirements
- **REQUIRED**: Validate all code with unit tests before marking complete
- Test prompt construction integration
- Test ModelAnswer object creation
- Test generation parameters (temperature, max_tokens)
- Test all error handling paths (100% coverage for error paths)
- Connection tests should warn but not fail if credentials missing

### Documentation Requirements
- **REQUIRED**: Create `phase_6_decisions.md` - Document any decisions not in PRD/RFC
- **REQUIRED**: Create `phase_6_testing.md` - Document testing performed
- **REQUIRED**: Create `phase_6_handoff.md` - Document what's needed for Phase 7
- Add docstrings to all functions
- Document generation parameters and configuration
- Document model selection strategy
- Document non-determinism in LLM generation (acceptable)

## Key Implementation Tasks

### Core Implementation
- Implement `generate_answer(query: Query, retrieved_chunks: List[RetrievalResult], prompt_version: str, config) -> ModelAnswer`:
  - Load prompt template (use Phase 5 implementation)
  - Construct prompt with query and retrieved context
  - Call Azure AI Foundry (OpenAI-compatible API) for generation
  - Configure generation parameters:
    - Model: from config (default: "gpt-4o")
    - Temperature: 0.1 (for reproducibility)
    - Max tokens: 1000 (configurable)
  - Parse and validate LLM response
  - Create `ModelAnswer` object with:
    - `text`: Generated answer
    - `query_id`: From query object (generate if missing)
    - `prompt_version`: Prompt version used
    - `retrieved_chunk_ids`: List of chunk IDs from retrieval results
    - `timestamp`: Generation timestamp
  - Retry logic (3 retries, exponential backoff)
  - Error handling (raise `AzureServiceError`)

### Testing
- Unit tests for `generate_answer()` (mocked Azure AI Foundry)
- Test prompt construction integration
- Test response parsing and validation
- Test ModelAnswer object creation
- Test error handling for generation failures
- Test retry logic with exponential backoff
- Test support for multiple prompt versions
- Connection test for Azure AI Foundry (warns if credentials missing)

### Documentation
- Function docstrings
- Generation parameters and configuration documentation
- Model selection strategy documentation
- Non-determinism in LLM generation documentation (acceptable)
- Phase 6 testing summary

## Success Criteria

- [ ] `generate_answer()` function complete
- [ ] All Phase 6 checkboxes in TODO001.md are checked
- [ ] Unit tests written and passing (mocked Azure)
- [ ] Connection test implemented (warns if credentials missing)
- [ ] All error paths tested (100% coverage)
- [ ] ModelAnswer object creation validated
- [ ] phase_6_decisions.md created
- [ ] phase_6_testing.md created
- [ ] phase_6_handoff.md created
- [ ] All failures documented in fracas.md (if any)

## Dependencies

- Azure AI Foundry credentials for generation -> use virtual environment for testing
- Generation model name in config (default: "gpt-4o")
- Prompt template system (Phase 5) complete
- Query ID generation utility available

## Next Phase

Once Phase 6 is complete, proceed to **Phase 7 — Pipeline Orchestration**.

---

**Document Status**: Active  
**Last Updated**: 2025-01-27  
**Related Documents**: [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md), [TODO001.md](./TODO001.md)

