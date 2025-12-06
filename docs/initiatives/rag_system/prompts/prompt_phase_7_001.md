# Phase 7 Prompt — Pipeline Orchestration

## Purpose
Implement the complete RAG pipeline orchestration that coordinates all components into a single end-to-end query pipeline.

## Context References

**Primary Documentation:**
- @docs/initiatives/rag_system/scoping/PRD001.md - Product requirements (FR2: Query Pipeline, FR4: Python API Interface)
- @docs/initiatives/rag_system/scoping/RFC001.md - Technical design (Phase 5: Pipeline Orchestration, Interface Contracts)
- @docs/initiatives/rag_system/scoping/TODO001.md - Phase 7 implementation tasks (check off as completed)
- @docs/initiatives/rag_system/scoping/context.md - System context

**Codebase References:**
- @backend/rag_eval/core/interfaces.py - Query and ModelAnswer interfaces
- @backend/rag_eval/core/config.py - Configuration
- @backend/rag_eval/utils/ids.py - Query ID generation utility
- @backend/rag_eval/services/rag/embeddings.py - Query embedding (Phase 3)
- @backend/rag_eval/services/rag/search.py - Chunk retrieval (Phase 4)
- @backend/rag_eval/services/rag/generation.py - Answer generation (Phase 6)
- @backend/rag_eval/services/rag/logging.py - Logging (Phase 8 - can be stubbed initially)

**Implementation Target:**
- **EXISTING FILE**: `backend/rag_eval/services/rag/pipeline.py` - Pipeline orchestration

## Phase Objectives

1. **Implement Pipeline Orchestration**: Create `run_rag()` function that coordinates all components
2. **Component Integration**: Integrate embeddings, search, generation components
3. **Error Handling**: Implement graceful error handling with proper logging
4. **Latency Measurement**: Measure and log pipeline latency metrics
5. **Testing**: Write comprehensive unit tests with mocked components

## Execution Requirements

### Checklist Management
- **REQUIRED**: Check off all [ ] checkboxes in @docs/initiatives/rag_system/scoping/TODO001.md Phase 7 section as tasks are completed
- Update TODO001.md directly when each task is finished

### Validation Requirements
- **REQUIRED**: Validate all code with unit tests before marking complete
- Test end-to-end pipeline flow with mocked components
- Test component integration and data flow
- Test error propagation and handling
- Test latency measurement
- Test all error handling paths (100% coverage for error paths)

### Documentation Requirements
- **REQUIRED**: Create `phase_7_decisions.md` - Document any decisions not in PRD/RFC
- **REQUIRED**: Create `phase_7_testing.md` - Document testing performed
- **REQUIRED**: Create `phase_7_handoff.md` - Document what's needed for Phase 8
- Add docstrings to all functions
- Document pipeline flow and execution order
- Document error handling strategy
- Document latency measurement approach

## Key Implementation Tasks

### Core Implementation
- Implement `run_rag(query: Query, prompt_version: str = "v1", config: Optional[Config] = None) -> ModelAnswer`:
  - Generate query ID and timestamp if missing
  - Step 1: Generate query embedding (use Phase 3: `generate_query_embedding()`)
  - Step 2: Retrieve top-k chunks (use Phase 4: `retrieve_chunks()`)
  - Step 3: Load prompt template and construct prompt (use Phase 5: `construct_prompt()`)
  - Step 4: Generate answer (use Phase 6: `generate_answer()`)
  - Step 5: Assemble `ModelAnswer` with metadata
  - Step 6: Log to Supabase (Phase 8 - can be stubbed initially)
  - Measure and log latency metrics
  - Handle errors gracefully with proper logging
  - Ensure deterministic execution order
  - Return complete `ModelAnswer` object

### Testing
- Unit tests for `run_rag()`:
  - Test end-to-end pipeline flow with mocked components
  - Test component integration and data flow
  - Test error propagation and handling
  - Test response assembly and formatting
  - Test query ID generation
  - Test latency measurement
  - Test pipeline state management
- Integration test with real components (optional, requires all services)

### Documentation
- Function docstrings
- Pipeline flow and execution order documentation
- Error handling strategy documentation
- Latency measurement approach documentation
- Phase 7 testing summary

## Success Criteria

- [ ] `run_rag()` function complete and functional
- [ ] All Phase 7 checkboxes in TODO001.md are checked
- [ ] Unit tests written and passing (mocked components)
- [ ] All error paths tested (100% coverage)
- [ ] Pipeline flow validated end-to-end
- [ ] Latency measurement implemented
- [ ] phase_7_decisions.md created
- [ ] phase_7_testing.md created
- [ ] phase_7_handoff.md created
- [ ] All failures documented in fracas.md (if any)

## Dependencies

- All previous phases complete (embeddings, search, generation)
- Query ID generation utility available
- Logging component (Phase 8) can be stubbed initially

## Next Phase

Once Phase 7 is complete, proceed to **Phase 8 — Supabase Logging**.

---

**Document Status**: Active  
**Last Updated**: 2025-01-27  
**Related Documents**: [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md), [TODO001.md](./TODO001.md)

