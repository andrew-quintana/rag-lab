# TODO 001 — In-Corpus Evaluation Dataset Generation System

## Context

This TODO document provides the implementation breakdown for the In-Corpus Evaluation Dataset Generation System, as specified in [PRD001.md](./PRD001.md) and [RFC001.md](./RFC001.md). The system provides an end-to-end pipeline for generating evaluation datasets from Azure AI Search indexed documents.

**Current Status**: The system is currently not implemented. This TODO provides the implementation plan for building the complete dataset generation stack.

**Implementation Phases**: This TODO follows a phased implementation plan that builds the system incrementally, starting with query generation and progressing through dataset generation and output generation.

---

## Phase 0 — Context Harvest

**Status**: ⏳ Pending

### Setup Tasks
- [ ] Review adjacent components (RAG system, evaluation system, Supabase prompts)
- [ ] Review PRD001.md and RFC001.md for complete requirements understanding
- [ ] Review existing Azure AI Search integration
- [ ] Review existing Supabase prompt system
- [ ] Validate Azure AI Search index and credentials
- [ ] Validate Supabase database schema and connection
- [ ] Review evaluation system components for integration points
- [ ] **Create fracas.md** for failure tracking using FRACAS methodology

### Testing Environment Setup
- [ ] **REQUIRED**: Set up and activate backend virtual environment (`backend/venv/`)
- [ ] **REQUIRED**: Verify pytest is installed in venv (`pip install pytest pytest-cov`)
- [ ] **REQUIRED**: Verify all backend dependencies are installed (`pip install -r backend/requirements.txt`)
- [ ] **REQUIRED**: Verify pytest can discover tests (`pytest backend/tests/ --collect-only`)
- [ ] **REQUIRED**: Document venv activation command for all subsequent phases
- [ ] **REQUIRED**: All testing in subsequent phases MUST use the same venv (`backend/venv/`)
- [ ] Block: Implementation cannot proceed until Phase 0 complete

### Phase 0 Deliverables
- [ ] Created `fracas.md` for failure tracking
- [ ] Created `phase_0_testing.md` documenting testing setup validation
- [ ] Created `phase_0_decisions.md` documenting Phase 0 decisions
- [ ] Created `phase_0_handoff.md` summarizing Phase 1 entry point

---

## Phase 1 — Query Generator (AI Node)

**Component**: `evaluations/{eval_name}/query_generator.py`

**Status**: ⏳ Pending

### Implementation Tasks
- [ ] Create `evaluations/{eval_name}/` directory structure
- [ ] Create `query_generator.py` module
- [ ] Implement function to sample chunks from Azure AI Search index
- [ ] Implement LLM-based query generation from chunks
- [ ] Ensure queries are "In-Corpus" (answerable from indexed documents)
- [ ] Include metadata linking queries to source chunks
- [ ] Implement saving to `eval_inputs.json`
- [ ] Add error handling and retry logic
- [ ] Add logging for query generation process

### Testing Requirements
- [ ] **REQUIRED**: Unit tests for query generation function
- [ ] **REQUIRED**: Integration tests with mocked Azure AI Search
- [ ] **REQUIRED**: Test coverage must meet minimum 80% for query_generator.py module
- [ ] **REQUIRED**: Test with sample chunks from index
- [ ] **REQUIRED**: Validate output JSON structure

### Phase 1 Deliverables
- [ ] `query_generator.py` implemented
- [ ] `eval_inputs.json` generated with sample queries
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Test coverage >= 80%
- [ ] Created `phase_1_decisions.md` if any decisions made
- [ ] Created `phase_1_testing.md` documenting test results
- [ ] Created `phase_1_handoff.md` summarizing Phase 2 entry point

---

## Phase 2 — Dataset Generator

**Component**: `evaluations/{eval_name}/generate_eval_dataset.py`

**Status**: ⏳ Pending

### Implementation Tasks
- [ ] Create `generate_eval_dataset.py` script
- [ ] Implement loading of `eval_inputs.json`
- [ ] Implement retrieval_query generation (with LLM sanitization if needed)
- [ ] Implement ground truth context retrieval using retrieval_query
- [ ] Implement Supabase query for system_prompt (prompt_type with name="{type}_system")
- [ ] Implement Supabase query for structured_prompt (prompt_type="RAG-enabled", live=TRUE)
- [ ] Integrate with RAG pipeline for output generation
- [ ] Implement BEIR metrics computation using ground truth chunk IDs
- [ ] Initialize LLM/human evaluation fields as null
- [ ] Implement saving to `eval_dataset.json`
- [ ] Add error handling for missing prompts, failed retrievals, etc.
- [ ] Add logging for dataset generation process

### Testing Requirements
- [ ] **REQUIRED**: Unit tests for dataset generation function
- [ ] **REQUIRED**: Integration tests with mocked Supabase and RAG pipeline
- [ ] **REQUIRED**: Test coverage must meet minimum 80% for generate_eval_dataset.py module
- [ ] **REQUIRED**: Test with sample eval_inputs.json
- [ ] **REQUIRED**: Validate output JSON structure matches specification
- [ ] **REQUIRED**: Test BEIR metrics computation accuracy

### Phase 2 Deliverables
- [ ] `generate_eval_dataset.py` implemented
- [ ] `eval_dataset.json` generated with all required fields
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Test coverage >= 80%
- [ ] Created `phase_2_decisions.md` if any decisions made
- [ ] Created `phase_2_testing.md` documenting test results
- [ ] Created `phase_2_handoff.md` summarizing Phase 3 entry point

---

## Phase 3 — Output Generator

**Component**: `evaluations/{eval_name}/generate_eval_outputs.py`

**Status**: ⏳ Pending

### Implementation Tasks
- [ ] Create `generate_eval_outputs.py` script
- [ ] Implement loading of `eval_dataset.json`
- [ ] Implement RAG output generation using system_prompt and structured_prompt
- [ ] Implement updating output field in dataset
- [ ] Implement saving updated dataset
- [ ] Add error handling for generation failures
- [ ] Add logging for output generation process
- [ ] Support batch processing of multiple entries

### Testing Requirements
- [ ] **REQUIRED**: Unit tests for output generation function
- [ ] **REQUIRED**: Integration tests with mocked RAG pipeline
- [ ] **REQUIRED**: Test coverage must meet minimum 80% for generate_eval_outputs.py module
- [ ] **REQUIRED**: Test with sample eval_dataset.json
- [ ] **REQUIRED**: Validate output field updates correctly

### Phase 3 Deliverables
- [ ] `generate_eval_outputs.py` implemented
- [ ] Outputs generated for sample dataset
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Test coverage >= 80%
- [ ] Created `phase_3_decisions.md` if any decisions made
- [ ] Created `phase_3_testing.md` documenting test results
- [ ] Created `phase_3_handoff.md` summarizing Phase 4 entry point

---

## Phase 4 — System Prompt Management

**Component**: System prompts in Supabase

**Status**: ⏳ Pending

### Implementation Tasks
- [ ] Create migration `0016_add_system_prompts.sql`
- [ ] Insert basic system prompts for RAG-enabled prompts
- [ ] Ensure system prompts follow naming convention `{type}_system`
- [ ] Create optional helper script `create_system_prompts.py`
- [ ] Document system prompt querying pattern
- [ ] Test system prompt retrieval from Supabase

### Testing Requirements
- [ ] **REQUIRED**: Test migration applies successfully
- [ ] **REQUIRED**: Test system prompts can be queried by prompt_type and name
- [ ] **REQUIRED**: Test naming convention filtering works correctly
- [ ] **REQUIRED**: Verify at least one system prompt exists for RAG-enabled prompts

### Phase 4 Deliverables
- [ ] Migration created and applied
- [ ] System prompts created in Supabase
- [ ] Helper script created (if needed)
- [ ] System prompt querying tested and documented
- [ ] Created `phase_4_decisions.md` if any decisions made
- [ ] Created `phase_4_testing.md` documenting test results
- [ ] Created `phase_4_handoff.md` summarizing Phase 5 entry point

---

## Phase 5 — Terminology Update: risk_impact → risk_magnitude

**Component**: Codebase-wide terminology update

**Status**: ⏳ Pending

### Implementation Tasks
- [ ] Find all `risk_impact` references using grep
- [ ] Update `backend/rag_eval/core/interfaces.py`:
  - [ ] `JudgeEvaluationResult.risk_impact` → `risk_magnitude`
  - [ ] `MetaEvaluationResult.ground_truth_risk_impact` → `ground_truth_risk_magnitude`
  - [ ] `JudgePerformanceMetrics.risk_impact` → `risk_magnitude`
- [ ] Update `backend/rag_eval/services/evaluator/meta_eval.py`:
  - [ ] All variable names
  - [ ] All function parameters
  - [ ] All references
- [ ] Update `backend/rag_eval/services/evaluator/risk_impact.py`:
  - [ ] Function names if needed
  - [ ] All references
- [ ] Update all other files with `risk_impact` references
- [ ] Update database schemas if applicable
- [ ] Update documentation references

### Testing Requirements
- [ ] **REQUIRED**: Run all existing tests to ensure no regressions
- [ ] **REQUIRED**: Verify all `risk_impact` references updated
- [ ] **REQUIRED**: Verify no broken imports or references
- [ ] **REQUIRED**: Test coverage maintained for updated modules

### Phase 5 Deliverables
- [ ] All `risk_impact` references updated to `risk_magnitude`
- [ ] All tests passing
- [ ] No regressions introduced
- [ ] Created `phase_5_decisions.md` if any decisions made
- [ ] Created `phase_5_testing.md` documenting test results
- [ ] Created `phase_5_handoff.md` summarizing completion

---

## Success Criteria

- [ ] Query generator produces diverse, effective queries from indexed documents
- [ ] Dataset generator creates complete evaluation datasets with all required fields
- [ ] Output generator successfully generates RAG outputs for dataset entries
- [ ] System prompts properly managed in Supabase with naming convention
- [ ] All `risk_impact` references updated to `risk_magnitude`
- [ ] All tests passing with >= 80% coverage
- [ ] End-to-end pipeline works with sample data

---
**Document Status**: Draft  
**Last Updated**: 2025-01-XX  
**Author**: Documentation Agent  
**Reviewers**: TBD

