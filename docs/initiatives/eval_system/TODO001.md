# TODO 001 — RAG Evaluation MVP (LLM-as-Judge + BEIR + Meta-Eval)

## Context

This TODO document provides the implementation breakdown for the RAG Evaluation MVP system, as specified in [PRD001.md](./PRD001.md) and [RFC001.md](./RFC001.md). The evaluation system provides a quantitative framework to measure how retrieval method changes and prompt changes affect downstream RAG system performance through LLM-as-Judge evaluation, meta-evaluation for judge reliability, and BEIR-style retrieval metrics.

**Current Status**: The evaluation system is currently not implemented. This TODO provides the implementation plan for building the complete evaluation stack.

**Implementation Phases**: This TODO follows a 10-phase implementation plan that builds the system incrementally, starting with the evaluation dataset and progressing through individual LLM nodes, orchestrators, and integration.

---

## Phase 0 — Context Harvest

**Status**: ✅ Complete (2024-12-19)

### Setup Tasks
- [x] Review adjacent components in [context.md](./context.md)
- [x] Review PRD001.md and RFC001.md for complete requirements understanding
- [x] Review existing RAG system components (`rag_eval/services/rag/`)
- [x] Review existing prompt system (`rag_eval/prompts/`, `rag_eval/services/rag/generation.py`)
- [x] Validate Azure Foundry API configuration and credentials
- [x] Validate Supabase database schema and connection (for optional logging)
- [x] Review test fixtures structure (`backend/tests/fixtures/`)
- [x] Review existing evaluation components (`rag_eval/services/evaluator/`)
- [x] **Create fracas.md** for failure tracking using FRACAS methodology

### Testing Environment Setup
- [x] **REQUIRED**: Set up and activate backend virtual environment (`backend/venv/`)
- [x] **REQUIRED**: Verify pytest is installed in venv (`pip install pytest pytest-cov`)
- [x] **REQUIRED**: Verify all backend dependencies are installed (`pip install -r backend/requirements.txt`)
- [x] **REQUIRED**: Verify pytest can discover tests (`pytest backend/tests/ --collect-only`)
- [x] **REQUIRED**: Document venv activation command for all subsequent phases
- [x] **REQUIRED**: All testing in Phases 1-11 MUST use the same venv (`backend/venv/`)
- [x] Block: Implementation cannot proceed until Phase 0 complete

### Phase 0 Deliverables
- [x] Created `fracas.md` for failure tracking
- [x] Created `phase_0_testing.md` documenting testing setup validation
- [x] Created `phase_0_decisions.md` documenting Phase 0 decisions
- [x] Created `phase_0_handoff.md` summarizing Phase 1 entry point

---

## Phase 1 — Evaluation Dataset Construction (Development Task)

**Component**: `backend/tests/fixtures/evaluation_dataset/validation_dataset.json`

**Status**: ✅ Complete (2024-12-19) - 5 validation samples created and validated

### Setup Tasks
- [x] Verify `healthguard_select_ppo_plan.pdf` exists in `backend/tests/fixtures/sample_documents/`
- [x] Ensure document has been indexed via upload pipeline (to identify actual chunk IDs)
- [x] Create directory structure: `backend/tests/fixtures/evaluation_dataset/`
- [x] Review document content to identify suitable QA pair topics

### Core Implementation
- [x] Create 5 validation samples manually (not via automated generation)
- [x] For each sample, include:
  - [x] `example_id`: Unique identifier (e.g., "val_001")
  - [x] `question`: Question text covering different types (cost, coverage, eligibility, out-of-pocket max)
  - [x] `reference_answer`: Gold reference answer
  - [x] `ground_truth_chunk_ids`: List of actual chunk IDs from indexed document
  - [x] `beir_failure_scale_factor`: Float in range [0.0, 1.0] representing retrieval challenge/severity
- [x] Ensure samples cover:
  - [x] Cost-related questions (copay, deductible, coinsurance)
  - [x] Coverage questions
  - [x] Eligibility questions
  - [x] Out-of-pocket maximum questions
- [x] Store as single JSON file: `validation_dataset.json`
- [x] Validate JSON format matches `EvaluationExample` dataclass structure

### Testing Tasks
- [x] Create test file: `backend/tests/components/evaluator/test_evaluation_dataset.py`
- [x] Validate that `validation_dataset.json` exists and is properly formatted
- [x] Validate that all 5 samples have required fields
- [x] Validate that `ground_truth_chunk_ids` reference actual chunks from indexed document
- [x] Validate that `beir_failure_scale_factor` is in range [0.0, 1.0]
- [x] Validate that questions cover different types (cost, coverage, eligibility, etc.)
- [x] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [x] Document dataset structure and format
- [x] Document how to identify ground-truth chunk IDs from indexed document
- [x] Document `beir_failure_scale_factor` calculation methodology
- [x] **Phase 1 Testing Summary** for handoff to Phase 2

### Validation Requirements (Phase 1 Complete)
- [x] **REQUIRED**: All unit tests for Phase 1 must pass before proceeding to Phase 2
- [x] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluation_dataset.py -v`
- [x] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [x] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [x] **REQUIRED**: Document any test failures in fracas.md
- [x] **REQUIRED**: Phase 1 is NOT complete until all tests pass
- [x] **Status**: ✅ Complete - All tests pass (13/13), ready for Phase 2

---

## Phase 2 — Correctness LLM-Node

**Component**: `rag_eval/services/evaluator/correctness.py`

**Status**: ✅ Complete (2024-12-19) - All tests pass, 83% coverage achieved

### Setup Tasks
- [x] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [x] Create `rag_eval/services/evaluator/correctness.py` module
- [x] Ensure package `__init__.py` files are properly configured with exports
- [x] Verify imports work correctly (`from rag_eval.services.evaluator.correctness import classify_correctness`)
- [x] Review Azure Foundry API configuration for GPT-4o-mini
- [x] Set up test fixtures for mock LLM responses
- [x] Create test file: `backend/tests/components/evaluator/test_evaluator_correctness.py`

### Prompt Creation Tasks
- [x] Create prompt template for correctness classification
- [x] Prompt location: `backend/rag_eval/prompts/evaluation/correctness_prompt.md` (or store in database with `prompt_type="evaluation"`)
- [x] Prompt design:
  - [x] System instruction: "You are an expert evaluator comparing a model answer directly to a gold reference answer."
  - [x] Input placeholders: `{query}`, `{model_answer}`, `{reference_answer}`
  - [x] Output format: JSON with `correctness_binary` (bool) and `reasoning` (str)
  - [x] Include examples of correct/incorrect classifications
- [x] Test prompt template with sample inputs

### Core Implementation
- [x] Implement `classify_correctness(query: str, model_answer: str, reference_answer: str, config: Optional[Config] = None) -> bool`
  - [x] Load prompt template (from file or database)
  - [x] Construct prompt with query, model answer, reference answer
  - [x] Call Azure Foundry GPT-4o-mini with structured output (JSON)
  - [x] Set temperature=0.1 for reproducibility
  - [x] Parse JSON response to extract `correctness_binary`
  - [x] Return boolean classification
  - [x] Handle LLM failures with proper error handling (`AzureServiceError`)
  - [x] Validate inputs are non-empty (`ValueError`)
- [x] Implement `_construct_correctness_prompt(query: str, model_answer: str, reference_answer: str) -> str`
  - [x] Load prompt template
  - [x] Replace placeholders with actual values
  - [x] Return complete prompt string

### Testing Tasks
- [x] Unit tests for `classify_correctness()`
  - [x] Test binary classification: correct (true)
  - [x] Test binary classification: incorrect (false)
  - [x] Test direct comparison between model answer and reference answer
  - [x] Test edge case: empty model answer
  - [x] Test edge case: empty reference answer
  - [x] Test error handling for LLM failures
  - [x] Test input validation (empty strings)
- [x] Connection test for Azure Foundry API
  - [x] Test actual connection to Azure Foundry (warns if credentials missing, doesn't fail tests)
  - [x] Document connection status in test output
- [x] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [x] Add docstrings to all functions
  - [x] Document function signature matching RFC001 interface contract
  - [x] Document return types and error conditions
- [x] Document prompt design and rationale
- [x] Document temperature setting (0.1) for reproducibility
- [x] **Phase 2 Testing Summary** for handoff to Phase 3

### Validation Requirements (Phase 2 Complete)
- [x] **REQUIRED**: All unit tests for Phase 2 must pass before proceeding to Phase 3
- [x] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_correctness.py -v`
- [x] **REQUIRED**: Test coverage must meet minimum 80% for correctness.py module
- [x] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [x] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [x] **REQUIRED**: Document any test failures in fracas.md
- [x] **REQUIRED**: Phase 2 is NOT complete until all tests pass
- [x] **Status**: ✅ Complete - All tests pass (23/23), 83% coverage, ready for Phase 3

---

## Phase 3 — Hallucination LLM-Node

**Component**: `rag_eval/services/evaluator/hallucination.py`

**Status**: ✅ Complete (2024-12-19) - All tests pass, 86% coverage achieved

### Setup Tasks
- [x] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [x] Review `rag_eval/services/evaluator/base_evaluator.py` and `rag_eval/services/shared/llm_providers.py` to understand the base class pattern
- [x] Create `rag_eval/services/evaluator/hallucination.py` module
- [x] Create `HallucinationEvaluator` class inheriting from `BaseEvaluatorNode`
- [x] Ensure package `__init__.py` files are properly configured with exports
- [x] Verify imports work correctly
- [x] Set up test fixtures for mock LLM responses
- [x] Create test file: `backend/tests/components/evaluator/test_evaluator_hallucination.py`

### Prompt Creation Tasks
- [x] Create prompt template for hallucination classification
- [x] Prompt location: `backend/rag_eval/prompts/evaluation/hallucination_prompt.md` (or store in database)
- [x] Prompt design:
  - [x] System instruction: "You are an expert evaluator analyzing whether a model answer contains hallucinations based on grounding in retrieved evidence."
  - [x] Input placeholders: `{retrieved_context}`, `{model_answer}`
  - [x] Output format: JSON with `hallucination_binary` (bool) and `reasoning` (str)
  - [x] Emphasize: reference answer is NOT used in hallucination detection
  - [x] Include examples of hallucination detection (grounded vs. ungrounded claims)
- [x] Test prompt template with sample inputs

### Core Implementation
- [x] Implement `HallucinationEvaluator` class inheriting from `BaseEvaluatorNode`:
  - [x] Override `_construct_prompt()` method to build hallucination-specific prompt
  - [x] Implement `classify_hallucination()` method using base class `_call_llm()` and `_parse_json_response()`
  - [x] Format retrieved context (concatenate chunk texts with chunk IDs)
  - [x] Validate `hallucination_binary` field in parsed JSON response
  - [x] Return boolean classification
  - [x] Handle LLM failures with proper error handling (inherited from base class)
  - [x] Validate inputs are non-empty
- [x] Implement module-level `classify_hallucination()` function for backward compatibility:
  - [x] Create `HallucinationEvaluator` instance
  - [x] Call `classify_hallucination()` method
  - [x] Return result

### Testing Tasks
- [x] Unit tests for `classify_hallucination()`
  - [x] Test binary classification: hallucination detected (true)
  - [x] Test binary classification: no hallucination (false)
  - [x] Test grounding analysis: information not in retrieved evidence
  - [x] Test grounding analysis: information supported by evidence
  - [x] Test edge case: ambiguous grounding scenarios
  - [x] Test that reference answer is NOT used in hallucination detection
  - [x] Test edge case: zero retrieved chunks
  - [x] Test error handling for LLM failures
- [x] Connection test for Azure Foundry API
- [x] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [x] Add docstrings to all functions
- [x] Document that reference answer is NOT used in hallucination detection
- [x] Document prompt design and grounding analysis approach
- [x] **Phase 3 Testing Summary** for handoff to Phase 4

### Validation Requirements (Phase 3 Complete)
- [x] **REQUIRED**: All unit tests for Phase 3 must pass before proceeding to Phase 4
- [x] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_hallucination.py -v`
- [x] **REQUIRED**: Test coverage must meet minimum 80% for hallucination.py module
- [x] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [x] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [x] **REQUIRED**: Document any test failures in fracas.md
- [x] **REQUIRED**: Phase 3 is NOT complete until all tests pass
- [x] **Status**: ✅ Complete - All tests pass (27/27), 86% coverage, ready for Phase 4

---

## Phase 3.5 — LangGraph Infrastructure Setup

**Component**: `rag_eval/services/evaluator/graph_base.py`, `rag_eval/services/evaluator/test_graph.py`

### Setup Tasks
- [ ] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [ ] Install LangGraph dependencies: `pip install -r requirements.txt`
- [ ] Verify LangGraph and langchain-core can be imported
- [ ] Review LangGraph documentation for StateGraph patterns

### Dependency Management
- [ ] Add `langgraph>=0.2.0` to `backend/requirements.txt`
- [ ] Add `langchain-core>=0.3.0` to `backend/requirements.txt`
- [ ] Verify dependencies install correctly

### Graph Infrastructure
- [ ] Create `rag_eval/services/evaluator/graph_base.py` module
- [ ] Define `JudgeEvaluationState` TypedDict with all required fields
- [ ] Implement `validate_initial_state()` function
- [ ] Implement `get_config_from_state()` helper function
- [ ] Add docstrings and type hints

### Test Graph Implementation
- [ ] Create `rag_eval/services/evaluator/test_graph.py` module
- [ ] Implement `correctness_node()` pure function node
- [ ] Implement `hallucination_node()` pure function node
- [ ] Implement `should_continue()` conditional edge function
- [ ] Implement `create_test_graph()` function
- [ ] Implement `run_test_evaluation()` convenience function
- [ ] Ensure nodes follow LangGraph best practices (pure functions, immutable state updates)

### Package Exports
- [ ] Update `backend/rag_eval/services/evaluator/__init__.py` to export:
  - `JudgeEvaluationState`
  - `validate_initial_state`
  - `get_config_from_state`
- [ ] Maintain existing exports (classify_correctness, classify_hallucination)

### Testing Tasks
- [ ] Create test file: `backend/tests/components/evaluator/test_graph_setup.py`
- [ ] Test LangGraph import and basic functionality
- [ ] Test state class creation and validation
- [ ] Test state validation functions (success and error cases)
- [ ] Test simple graph construction and execution
- [ ] Test node functions (correctness_node, hallucination_node)
- [ ] Test conditional edge routing
- [ ] Test integration with existing evaluator classes (mocked)
- [ ] Test error handling in graph execution
- [ ] Test graph execution order
- [ ] Test config handling in state
- [ ] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [ ] Add docstrings to all functions and classes
- [ ] Document LangGraph setup and state management patterns
- [ ] Document node function pattern (pure functions wrapping evaluators)
- [ ] Create `phase_3_5_decisions.md` documenting implementation decisions
- [ ] Create `phase_3_5_testing.md` documenting test results and coverage
- [ ] Create `phase_3_5_handoff.md` summarizing setup completion and Phase 7 requirements

### Validation Requirements (Phase 3.5 Complete)
- [ ] **REQUIRED**: All unit tests for Phase 3.5 must pass before proceeding
- [ ] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_graph_setup.py -v`
- [ ] **REQUIRED**: Test coverage must meet minimum 80% for graph_base.py and test_graph.py modules
- [ ] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [ ] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [ ] **REQUIRED**: Document any test failures in fracas.md
- [ ] **REQUIRED**: Phase 3.5 is NOT complete until all tests pass
- [ ] **Status**: ⏳ Pending - Phase 3.5 cannot proceed to Phase 4 until validation complete

---

## Phase 4 — Hallucination Cost LLM-Node

**Component**: `rag_eval/services/evaluator/risk_direction.py`

**Status**: ✅ Complete (2024-12-19) - All tests pass, 87% coverage achieved

### Setup Tasks
- [x] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [x] Review `rag_eval/services/evaluator/base_evaluator.py` and `rag_eval/services/shared/llm_providers.py` to understand the base class pattern
- [x] Create `rag_eval/services/evaluator/risk_direction.py` module
- [x] Create `HallucinationCostEvaluator` class inheriting from `BaseEvaluatorNode`
- [x] Ensure package `__init__.py` files are properly configured with exports
- [x] Verify imports work correctly
- [x] Set up test fixtures for mock LLM responses
- [x] Create test file: `backend/tests/components/evaluator/test_evaluator_risk_direction.py`

### Prompt Creation Tasks
- [x] Create prompt template for cost classification
- [x] Prompt location: `backend/rag_eval/prompts/evaluation/risk_direction_prompt.md` (or store in database)
- [x] Prompt design:
  - [x] System instruction: "You are an expert evaluator classifying the cost impact direction of hallucinations."
  - [x] Input placeholders: `{model_answer}`, `{retrieved_context}`
  - [x] Output format: JSON with `risk_direction` (-1 or +1) and `reasoning` (str)
  - [x] Explain cost classification:
    - [x] -1 = Opportunity Cost: Model overestimated cost, dissuading user from seeking care
    - [x] +1 = Resource Cost: Model underestimated cost, persuading user to pursue care
  - [x] Emphasize: reference answer is NOT used (only retrieved chunks for ground truth)
  - [x] Include examples of opportunity cost vs. resource cost classifications
- [x] Test prompt template with sample inputs

### Core Implementation
- [x] Implement `HallucinationCostEvaluator` class inheriting from `BaseEvaluatorNode`:
  - [x] Override `_construct_prompt()` method to build cost classification-specific prompt
  - [x] Implement `classify_risk_direction()` method using base class `_call_llm()` and `_parse_json_response()`
  - [x] Format retrieved context
  - [x] Validate `risk_direction` field in parsed JSON response (-1 or +1)
  - [x] Return integer classification
  - [x] Handle LLM failures with proper error handling (inherited from base class)
  - [x] Validate inputs are non-empty
  - [x] Handle ambiguous cost direction cases
- [x] Implement module-level `classify_risk_direction()` function for backward compatibility:
  - [x] Create `HallucinationCostEvaluator` instance
  - [x] Call `classify_risk_direction()` method
  - [x] Return result

### Testing Tasks
- [x] Unit tests for `classify_risk_direction()`
  - [x] Test opportunity cost classification (-1): overestimated cost
  - [x] Test resource cost classification (+1): underestimated cost
  - [x] Test cost analysis for quantitative hallucinations
  - [x] Test cost analysis for non-quantitative hallucinations
  - [x] Test edge case: ambiguous cost direction
  - [x] Test that reference answer is NOT used in cost classification
  - [x] Test error handling for LLM failures
- [x] Connection test for Azure Foundry API
- [x] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [x] Add docstrings to all functions
- [x] Document cost classification logic (-1 vs. +1)
- [x] Document that reference answer is NOT used
- [x] Document prompt design and cost direction analysis
- [x] **Phase 4 Testing Summary** for handoff to Phase 5

### Validation Requirements (Phase 4 Complete)
- [x] **REQUIRED**: All unit tests for Phase 4 must pass before proceeding to Phase 5
- [x] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_risk_direction.py -v`
- [x] **REQUIRED**: Test coverage must meet minimum 80% for risk_direction.py module
- [x] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [x] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [x] **REQUIRED**: Document any test failures in fracas.md
- [x] **REQUIRED**: Phase 4 is NOT complete until all tests pass
- [x] **Status**: ✅ Complete - All tests pass (30/30), 87% coverage, ready for Phase 5

---

## Phase 5 — Cost Extraction LLM-Node

**Component**: `rag_eval/services/evaluator/cost_extraction.py`

**Status**: ✅ Complete (2024-12-19) - All tests pass, 89% coverage achieved

### Setup Tasks
- [x] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [x] Review `rag_eval/services/evaluator/base_evaluator.py` and `rag_eval/services/shared/llm_providers.py` to understand the base class pattern
- [x] Create `rag_eval/services/evaluator/cost_extraction.py` module
- [x] Create `CostExtractionEvaluator` class inheriting from `BaseEvaluatorNode`
- [x] Ensure package `__init__.py` files are properly configured with exports
- [x] Verify imports work correctly
- [x] Set up test fixtures for mock LLM responses
- [x] Create test file: `backend/tests/components/evaluator/test_evaluator_cost_extraction.py`

### Prompt Creation Tasks
- [x] Create prompt template for cost extraction
- [x] Prompt location: `backend/rag_eval/prompts/evaluation/cost_extraction_prompt.md` (or store in database)
- [x] Prompt design:
  - [x] System instruction: "You are an expert parser extracting cost information (time, money, steps) from text."
  - [x] Input placeholder: `{text}`
  - [x] Output format: JSON with `time` (optional float/str), `money` (optional float/str), `steps` (optional int/str), and `reasoning` (str)
  - [x] Include examples of cost extraction from various formats:
    - [x] Time: "2 hours", "30 minutes", "1 day"
    - [x] Money: "$500", "500 dollars", "500.00"
    - [x] Steps: "3 steps", "step 3", "third step"
  - [x] Handle missing cost information (optional fields)
- [x] Test prompt template with sample inputs

### Core Implementation
- [x] Implement `CostExtractionEvaluator` class inheriting from `BaseEvaluatorNode`:
  - [x] Override `_construct_prompt()` method to build cost extraction-specific prompt
  - [x] Implement `extract_costs()` method using base class `_call_llm()` and `_parse_json_response()`
  - [x] Parse JSON response to extract cost fields (time, money, steps)
  - [x] Return dictionary with optional cost fields
  - [x] Handle LLM failures with proper error handling (inherited from base class)
  - [x] Validate input is non-empty
- [x] Implement module-level `extract_costs()` function for backward compatibility:
  - [x] Create `CostExtractionEvaluator` instance
  - [x] Call `extract_costs()` method
  - [x] Return result

### Testing Tasks
- [x] Unit tests for `extract_costs()`
  - [x] Test extraction of time-based costs (e.g., "2 hours", "30 minutes")
  - [x] Test extraction of money-based costs (e.g., "$500", "500 dollars", "500.00")
  - [x] Test extraction of step-based costs (e.g., "3 steps", "step 3")
  - [x] Test extraction of mixed cost types from same text
  - [x] Test handling of missing cost information (optional fields)
  - [x] Test edge case: no cost information in text
  - [x] Test edge case: ambiguous cost expressions
  - [x] Test error handling for LLM failures
- [x] Connection test for Azure Foundry API
- [x] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [x] Add docstrings to all functions
- [x] Document cost extraction format and supported expressions
- [x] Document optional fields and handling of missing information
- [x] **Phase 5 Testing Summary** for handoff to Phase 6

### Validation Requirements (Phase 5 Complete)
- [x] **REQUIRED**: All unit tests for Phase 5 must pass before proceeding to Phase 6
- [x] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_cost_extraction.py -v`
- [x] **REQUIRED**: Test coverage must meet minimum 80% for cost_extraction.py module
- [x] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [x] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [x] **REQUIRED**: Document any test failures in fracas.md
- [x] **REQUIRED**: Phase 5 is NOT complete until all tests pass
- [x] **Status**: ✅ Complete - All tests pass (22/22), 89% coverage, ready for Phase 6

---

## Phase 6 — Risk Impact LLM-Node

**Component**: `rag_eval/services/evaluator/risk_impact.py`

**Status**: ✅ Complete (2024-12-19) - All tests pass, 86% coverage achieved

### Setup Tasks
- [x] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [x] Review `rag_eval/services/evaluator/base_evaluator.py` and `rag_eval/services/shared/llm_providers.py` to understand the base class pattern
- [x] Create `rag_eval/services/evaluator/risk_impact.py` module
- [x] Create `RiskImpactEvaluator` class inheriting from `BaseEvaluatorNode`
- [x] Ensure package `__init__.py` files are properly configured with exports
- [x] Verify imports work correctly
- [x] Set up test fixtures for mock LLM responses
- [x] Create test file: `backend/tests/components/evaluator/test_evaluator_risk_impact.py`

### Prompt Creation Tasks
- [x] **REQUIRED**: Store prompt template in database (`public.prompts` table)
  - [x] `prompt_type='evaluation'`
  - [x] `name='risk_impact_evaluator'`
  - [x] `version='0.1'` (or appropriate version)
  - [x] `live=true` (for live version)
- [x] Prompt design:
  - [x] System instruction: "You are an expert evaluator calculating the real-world impact magnitude of system-level deviations."
  - [x] Input placeholders: `{model_answer_cost}`, `{actual_cost}`
  - [x] Output format: JSON with `risk_impact` (float 0-3) and `reasoning` (str)
  - [x] Explain impact scale:
    - [x] 0: Minimal/no impact
    - [x] 1: Low impact
    - [x] 2: Moderate impact
    - [x] 3: High/severe impact
  - [x] **CRITICAL**: Emphasize considering mixed resource types (time, money, steps) and their relative importance
  - [x] **CRITICAL**: Evaluate impact of deviations regardless of origin (retrieval, augmentation, context ordering, prompting, model reasoning, or hallucination)
  - [x] Include examples of impact calculations for different cost differences
- [x] Test prompt template with sample inputs

### Core Implementation
- [x] Implement `RiskImpactEvaluator` class inheriting from `BaseEvaluatorNode`:
  - [x] Override `_construct_prompt()` method to build impact calculation-specific prompt
  - [x] Implement `calculate_risk_impact()` method using base class `_call_llm()` and `_parse_json_response()`
  - [x] Format cost dictionaries for prompt (JSON representation)
  - [x] Parse JSON response to extract `risk_impact` (0-3)
  - [x] Validate impact is in range [0, 3]
  - [x] Return float impact magnitude
  - [x] Handle LLM failures with proper error handling (inherited from base class)
  - [x] Validate inputs are non-empty dictionaries
- [x] Implement module-level `calculate_risk_impact()` function for backward compatibility:
  - [x] Create `RiskImpactEvaluator` instance
  - [x] Call `calculate_risk_impact()` method
  - [x] Return result

### Testing Tasks
- [x] Unit tests for `calculate_risk_impact()`
  - [x] Test impact calculation for time-based costs
  - [x] Test impact calculation for money-based costs
  - [x] Test impact calculation for step-based costs
  - [x] Test impact calculation for mixed resource types
  - [x] Test impact scaling factor range [0, 3]
  - [x] Test edge case: zero impact scenarios
  - [x] Test edge case: maximum impact scenarios
  - [x] Test error handling for LLM failures
- [x] Connection test for Azure Foundry API
- [x] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [x] Add docstrings to all functions
- [x] Document impact scale (0-3) and rationale
- [x] Document handling of mixed resource types
- [x] Document why LLM node is used (not deterministic function)
- [x] **Phase 6 Testing Summary** for handoff to Phase 7

### Validation Requirements (Phase 6 Complete)
- [x] **REQUIRED**: All unit tests for Phase 6 must pass before proceeding to Phase 7
- [x] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_risk_impact.py -v`
- [x] **REQUIRED**: Test coverage must meet minimum 80% for risk_impact.py module
- [x] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [x] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [x] **REQUIRED**: Document any test failures in fracas.md
- [x] **REQUIRED**: Phase 6 is NOT complete until all tests pass
- [x] **Status**: ✅ Complete - All tests pass (21/21), 86% coverage, ready for Phase 7

---

## Phase 7 — LLM-as-Judge Orchestrator

**Component**: `rag_eval/services/evaluator/judge.py`

### Setup Tasks
- [ ] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [ ] Create `rag_eval/services/evaluator/judge.py` module
- [ ] Ensure package `__init__.py` files are properly configured with exports
- [ ] Verify imports work correctly (correctness, hallucination, cost, impact, cost_extraction nodes)
- [ ] Review data structures: `JudgeEvaluationResult` dataclass
- [ ] Set up test fixtures for mocked LLM node responses
- [ ] Create test file: `backend/tests/components/evaluator/test_evaluator_judge.py`

### Core Implementation
- [ ] Implement `evaluate_answer_with_judge(query: str, retrieved_context: List[RetrievalResult], model_answer: str, reference_answer: str, config: Optional[Config] = None) -> JudgeEvaluationResult`
  - [ ] Validate all inputs are non-empty
  - [ ] Step 1: Call correctness LLM-node (always)
    - [ ] `correctness_binary = classify_correctness(query, model_answer, reference_answer, config)`
  - [ ] Step 2: Call hallucination LLM-node (always)
    - [ ] `hallucination_binary = classify_hallucination(retrieved_context, model_answer, config)`
  - [ ] Step 3: Conditional - if correctness is True, call cost classification node
    - [ ] `risk_direction = None`
    - [ ] `if correctness_binary: risk_direction = classify_risk_direction(model_answer, retrieved_context, config)`
  - [ ] Step 4: Conditional - if correctness is True, extract costs and calculate impact
    - [ ] `risk_impact = None`
    - [ ] `if correctness_binary:`
      - [ ] Extract costs from model answer: `model_answer_cost = extract_costs(model_answer, config)`
      - [ ] Extract costs from retrieved chunks: `chunks_text = " ".join([chunk.chunk_text for chunk in retrieved_context])`
      - [ ] `actual_cost = extract_costs(chunks_text, config)`
      - [ ] Calculate impact: `risk_impact = calculate_risk_impact(model_answer_cost, actual_cost, config)`
  - [ ] Step 5: Construct reasoning trace from all LLM node outputs
    - [ ] Collect reasoning from correctness node
    - [ ] Collect reasoning from hallucination node
    - [ ] Collect reasoning from cost node (if called)
    - [ ] Collect reasoning from impact node (if called)
    - [ ] Construct combined reasoning trace
  - [ ] Step 6: Assemble `JudgeEvaluationResult` with all fields
    - [ ] `correctness_binary`
    - [ ] `hallucination_binary`
    - [ ] `risk_direction` (optional)
    - [ ] `risk_impact` (optional)
    - [ ] `reasoning` (combined trace)
    - [ ] `failure_mode` (optional, extracted from reasoning or LLM output)
  - [ ] Handle edge cases: zero chunks, empty answers, LLM failures
  - [ ] Return `JudgeEvaluationResult` object
- [ ] Implement `_orchestrate_judge_evaluation(...)` helper function (if needed for organization)
- [ ] Implement `_construct_reasoning_trace(correctness_result: bool, hallucination_result: bool, cost_result: Optional[int], impact_result: Optional[float], node_reasonings: List[str]) -> str`
  - [ ] Combine reasoning from all invoked LLM nodes
  - [ ] Format as structured reasoning trace
  - [ ] Return combined reasoning string

### Testing Tasks
- [ ] Unit tests for `evaluate_answer_with_judge()`
  - [ ] Test deterministic script orchestration with mocked LLM calls
  - [ ] Test correctness LLM-node invocation (always called)
  - [ ] Test hallucination LLM-node invocation (always called)
  - [ ] Test conditional branching: correctness_binary true path (cost and impact nodes called)
  - [ ] Test conditional branching: correctness_binary false path (cost and impact nodes NOT called)
  - [ ] Test invocation of cost classification node when correctness is True
  - [ ] Test invocation of cost extraction node when correctness is True
  - [ ] Test invocation of impact node when correctness is True
  - [ ] Test that cost/impact nodes are NOT called when correctness is False
  - [ ] Test output schema validation (all required fields present including correctness fields)
  - [ ] Test reasoning trace construction from LLM node outputs
  - [ ] Test error handling when LLM calls fail
  - [ ] Test edge case: zero retrieved chunks
  - [ ] Test edge case: empty model answer
  - [ ] Test edge case: empty reference answer
- [ ] Integration tests with mocked LLM nodes
- [ ] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [ ] Add docstrings to all functions
- [ ] Document orchestration logic and conditional branching
- [ ] Document reasoning trace construction
- [ ] Document edge case handling
- [ ] **Phase 7 Testing Summary** for handoff to Phase 8

### Validation Requirements (Phase 7 Complete)
- [ ] **REQUIRED**: All unit tests for Phase 7 must pass before proceeding to Phase 8
- [ ] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_judge.py -v`
- [ ] **REQUIRED**: Test coverage must meet minimum 80% for judge.py module
- [ ] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [ ] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [ ] **REQUIRED**: Document any test failures in fracas.md
- [ ] **REQUIRED**: Phase 7 is NOT complete until all tests pass
- [ ] **Status**: ⏳ Pending - Phase 7 cannot proceed to Phase 8 until validation complete

---

## Phase 8 — Meta-Evaluator (Deterministic Validation)

**Component**: `rag_eval/services/evaluator/meta_eval.py`

### Setup Tasks
- [ ] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [ ] Create `rag_eval/services/evaluator/meta_eval.py` module
- [ ] Ensure package `__init__.py` files are properly configured with exports
- [ ] Verify imports work correctly
- [ ] Review data structures: `MetaEvaluationResult` dataclass
- [ ] Set up test fixtures for validation scenarios
- [ ] Create test file: `backend/tests/components/meta_eval/test_evaluator_meta_eval.py`

### Core Implementation
- [ ] Implement `meta_evaluate_judge(judge_output: JudgeEvaluationResult, retrieved_context: List[RetrievalResult], model_answer: str, reference_answer: str, extracted_costs: Optional[Dict[str, Any]] = None, actual_costs: Optional[Dict[str, Any]] = None) -> MetaEvaluationResult`
  - [ ] Validate inputs are non-empty
  - [ ] Validation 1: Validate correctness_binary
    - [ ] Compare model answer to reference answer (exact or semantic similarity)
    - [ ] If judge says `correctness_binary: true`, verify model answer matches reference
    - [ ] If judge says `correctness_binary: false`, verify model answer differs from reference
  - [ ] Validation 2: Validate hallucination_binary
    - [ ] Check if model answer claims are supported by retrieved chunks
    - [ ] If judge says `hallucination_binary: true`, verify model answer contains unsupported claims
    - [ ] If judge says `hallucination_binary: false`, verify all claims are supported
  - [ ] Validation 3: Validate risk_direction (if costs available and correctness is True)
    - [ ] Compare extracted costs vs actual costs to determine expected cost direction
    - [ ] Validate judge's `risk_direction` matches expected direction
  - [ ] Validation 4: Validate risk_impact (if costs available and correctness is True)
    - [ ] Calculate expected impact magnitude based on cost differences
    - [ ] Validate judge's `risk_impact` is within reasonable range of expected impact
  - [ ] Determine overall judge correctness (all validations pass)
  - [ ] Generate deterministic explanation of validation results
  - [ ] Return `MetaEvaluationResult` object
- [ ] Implement helper functions:
  - [ ] `_validate_correctness(judge_correctness: bool, model_answer: str, reference_answer: str) -> bool`
  - [ ] `_validate_hallucination(judge_hallucination: bool, model_answer: str, retrieved_context: List[RetrievalResult]) -> bool`
  - [ ] `_validate_cost_classification(judge_cost: Optional[int], extracted_costs: Dict[str, Any], actual_costs: Dict[str, Any]) -> bool`
  - [ ] `_validate_impact_magnitude(judge_impact: Optional[float], extracted_costs: Dict[str, Any], actual_costs: Dict[str, Any]) -> bool`
  - [ ] `_generate_explanation(validation_results: Dict[str, bool]) -> str`

### Testing Tasks
- [ ] Unit tests for `meta_evaluate_judge()`
  - [ ] Test judge_correct classification: correct correctness_binary verdict
  - [ ] Test judge_incorrect classification: incorrect correctness_binary verdict
  - [ ] Test judge_correct classification: correct hallucination_binary verdict
  - [ ] Test judge_incorrect classification: incorrect hallucination_binary verdict
  - [ ] Test validation of risk_direction against ground truth costs
  - [ ] Test validation of risk_impact against ground truth costs
  - [ ] Test deterministic explanation generation
  - [ ] Test edge case: partial judge correctness (some verdicts correct, others incorrect)
  - [ ] Test edge case: missing ground truth information
  - [ ] Test edge case: zero retrieved chunks
- [ ] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [ ] Add docstrings to all functions
- [ ] Document validation logic for each judge verdict type
- [ ] Document deterministic nature (no LLM calls)
- [ ] Document explanation generation approach
- [ ] **Phase 8 Testing Summary** for handoff to Phase 9

### Validation Requirements (Phase 8 Complete)
- [ ] **REQUIRED**: All unit tests for Phase 8 must pass before proceeding to Phase 9
- [ ] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/meta_eval/test_evaluator_meta_eval.py -v`
- [ ] **REQUIRED**: Test coverage must meet minimum 80% for meta_eval.py module
- [ ] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [ ] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [ ] **REQUIRED**: Document any test failures in fracas.md
- [ ] **REQUIRED**: Phase 8 is NOT complete until all tests pass
- [ ] **Status**: ⏳ Pending - Phase 8 cannot proceed to Phase 9 until validation complete

---

## Phase 9 — BEIR Metrics Evaluator

**Component**: `rag_eval/services/evaluator/beir_metrics.py`

### Setup Tasks
- [ ] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [ ] Create `rag_eval/services/evaluator/beir_metrics.py` module
- [ ] Ensure package `__init__.py` files are properly configured with exports
- [ ] Verify imports work correctly
- [ ] Review data structures: `BEIRMetricsResult` dataclass
- [ ] Review BEIR metric formulas (recall@k, precision@k, nDCG@k)
- [ ] Set up test fixtures for retrieval results and ground-truth chunk IDs
- [ ] Create test file: `backend/tests/components/evaluator/test_evaluator_beir_metrics.py`

### Core Implementation
- [ ] Implement `compute_beir_metrics(retrieved_chunks: List[RetrievalResult], ground_truth_chunk_ids: List[str], k: int = 5) -> BEIRMetricsResult`
  - [ ] Validate inputs are non-empty
  - [ ] Extract chunk IDs from retrieved chunks (top-k)
  - [ ] Compute recall@k
  - [ ] Compute precision@k
  - [ ] Compute nDCG@k
  - [ ] Handle edge cases: zero relevant passages, all relevant passages retrieved
  - [ ] Return `BEIRMetricsResult` object
- [ ] Implement helper functions:
  - [ ] `_compute_recall_at_k(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: List[str], k: int) -> float`
    - [ ] Formula: (Number of relevant chunks in top-k) / (Total number of relevant chunks)
  - [ ] `_compute_precision_at_k(retrieved_chunk_ids: List[str], ground_truth_chunk_ids: List[str], k: int) -> float`
    - [ ] Formula: (Number of relevant chunks in top-k) / k
  - [ ] `_compute_ndcg_at_k(retrieved_chunks: List[RetrievalResult], ground_truth_chunk_ids: List[str], k: int) -> float`
    - [ ] Formula: Normalized discounted cumulative gain using relevance scores (1 for relevant, 0 for irrelevant)

### Testing Tasks
- [ ] Unit tests for `compute_beir_metrics()`
  - [ ] Test recall@k calculation
  - [ ] Test precision@k calculation
  - [ ] Test nDCG@k calculation
  - [ ] Test metrics with ground-truth passage IDs
  - [ ] Test edge case: zero relevant passages retrieved
  - [ ] Test edge case: all relevant passages retrieved
  - [ ] Test edge case: k larger than number of retrieved chunks
  - [ ] Test edge case: empty ground-truth chunk IDs list
- [ ] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [ ] Add docstrings to all functions
- [ ] Document BEIR metric formulas
- [ ] Document edge case handling
- [ ] **Phase 9 Testing Summary** for handoff to Phase 10

### Validation Requirements (Phase 9 Complete)
- [ ] **REQUIRED**: All unit tests for Phase 9 must pass before proceeding to Phase 10
- [ ] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_beir_metrics.py -v`
- [ ] **REQUIRED**: Test coverage must meet minimum 80% for beir_metrics.py module
- [ ] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [ ] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [ ] **REQUIRED**: Document any test failures in fracas.md
- [ ] **REQUIRED**: Phase 9 is NOT complete until all tests pass
- [ ] **Status**: ⏳ Pending - Phase 9 cannot proceed to Phase 10 until validation complete

---

## Phase 10 — Evaluation Pipeline Orchestration

**Component**: `rag_eval/services/evaluator/orchestrator.py`

### Setup Tasks
- [ ] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [ ] Create `rag_eval/services/evaluator/orchestrator.py` module
- [ ] Ensure package `__init__.py` files are properly configured with exports
- [ ] Verify imports work correctly (judge, meta_eval, beir_metrics)
- [ ] Review data structures: `EvaluationExample`, `EvaluationResult` dataclasses
- [ ] Review RAG system components: `retrieve_chunks`, `generate_answer`
- [ ] Set up test fixtures for evaluation dataset and mocked RAG components
- [ ] Create test file: `backend/tests/components/evaluator/test_evaluator_orchestrator.py`

### Core Implementation
- [ ] Implement `evaluate_rag_system(evaluation_dataset: List[EvaluationExample], rag_retriever: Callable[[str, int], List[RetrievalResult]], rag_generator: Callable[[str, List[RetrievalResult]], ModelAnswer], config: Optional[Config] = None) -> List[EvaluationResult]`
  - [ ] Validate evaluation dataset is non-empty
  - [ ] Load evaluation dataset (if passed as file path, otherwise use list directly)
  - [ ] For each example in evaluation dataset:
    - [ ] Step 1: Retrieve chunks using RAG retriever
      - [ ] `retrieved_chunks = rag_retriever(example.question, k=5)`
    - [ ] Step 2: Generate answer using RAG generator
      - [ ] `model_answer = rag_generator(example.question, retrieved_chunks)`
    - [ ] Step 3: Evaluate with LLM-as-Judge
      - [ ] `judge_output = evaluate_answer_with_judge(example.question, retrieved_chunks, model_answer.text, example.reference_answer, config)`
    - [ ] Step 4: Meta-evaluate judge verdict
      - [ ] Extract costs if needed (for meta-evaluation)
      - [ ] `meta_eval_output = meta_evaluate_judge(judge_output, retrieved_chunks, model_answer.text, example.reference_answer, extracted_costs, actual_costs)`
    - [ ] Step 5: Compute BEIR metrics
      - [ ] `beir_metrics = compute_beir_metrics(retrieved_chunks, example.ground_truth_chunk_ids, k=5)`
    - [ ] Step 6: Assemble EvaluationResult
      - [ ] `result = EvaluationResult(example_id=example.example_id, judge_output=judge_output, meta_eval_output=meta_eval_output, beir_metrics=beir_metrics, timestamp=datetime.now())`
    - [ ] Handle errors gracefully with proper logging
    - [ ] Measure and log latency metrics
  - [ ] Return list of EvaluationResult objects
- [ ] Implement `_evaluate_single_example(example: EvaluationExample, rag_retriever: Callable, rag_generator: Callable, config: Config) -> EvaluationResult`
  - [ ] Extract single example evaluation logic
  - [ ] Handle errors for individual examples (don't fail entire pipeline)

### Testing Tasks
- [ ] Unit tests for `evaluate_rag_system()`
  - [ ] Test full pipeline: Retrieval → RAG generation → Judge → Meta-Eval → Metrics
  - [ ] Test pipeline with mocked RAG components
  - [ ] Test pipeline error handling and propagation
  - [ ] Test pipeline logging and observability
  - [ ] Test edge case: empty evaluation dataset
  - [ ] Test edge case: RAG retriever failure
  - [ ] Test edge case: RAG generator failure
  - [ ] Test edge case: Judge evaluation failure
- [ ] Integration tests with real RAG components (optional, requires full system setup)
- [ ] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [ ] Add docstrings to all functions
- [ ] Document pipeline flow and error handling
- [ ] Document integration with RAG system components
- [ ] Document latency measurement approach
- [ ] **Phase 10 Testing Summary** for handoff to Phase 11

### Validation Requirements (Phase 10 Complete)
- [ ] **REQUIRED**: All unit tests for Phase 10 must pass before proceeding to Phase 11
- [ ] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_orchestrator.py -v`
- [ ] **REQUIRED**: Test coverage must meet minimum 80% for orchestrator.py module
- [ ] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [ ] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [ ] **REQUIRED**: Document any test failures in fracas.md
- [ ] **REQUIRED**: Phase 10 is NOT complete until all tests pass
- [ ] **Status**: ⏳ Pending - Phase 10 cannot proceed to Phase 11 until validation complete

---

## Phase 11 — Logging and Persistence (Optional)

**Component**: `rag_eval/services/evaluator/logging.py`

**Status**: Optional - can be deferred if not needed for MVP

### Setup Tasks
- [ ] **REQUIRED**: Activate backend venv: `cd backend && source venv/bin/activate`
- [ ] Create `rag_eval/services/evaluator/logging.py` module
- [ ] Review Supabase database schema for evaluation results table
- [ ] Review existing `QueryExecutor` from `rag_eval/db/queries.py`
- [ ] Set up test fixtures for database operations
- [ ] Create test file: `backend/tests/components/evaluator/test_evaluator_logging.py`

### Core Implementation
- [ ] Implement `log_evaluation_result(result: EvaluationResult, query_executor: Optional[QueryExecutor] = None) -> Optional[str]`
  - [ ] If query_executor is None, skip logging (local-only mode)
  - [ ] Serialize EvaluationResult to JSON
  - [ ] Insert into Supabase Postgres `evaluation_results` table
  - [ ] Handle logging failures gracefully (don't fail evaluation pipeline)
  - [ ] Return result_id if successful, None otherwise
- [ ] Implement `log_evaluation_batch(results: List[EvaluationResult], query_executor: Optional[QueryExecutor] = None) -> None`
  - [ ] Batch insert evaluation results
  - [ ] Handle partial failures gracefully
  - [ ] Log batch operation status

### Testing Tasks
- [ ] Unit tests for `log_evaluation_result()`
  - [ ] Test logging with query_executor (mocked)
  - [ ] Test local-only mode (query_executor is None)
  - [ ] Test error handling for database failures
  - [ ] Test that logging failures don't fail evaluation pipeline
- [ ] Connection test for Supabase (warns if credentials missing)
- [ ] **Document any failures** in fracas.md immediately when encountered

### Documentation Tasks
- [ ] Add docstrings to all functions
- [ ] Document optional nature of logging
- [ ] Document database schema requirements
- [ ] Document local-only vs. database logging modes
- [ ] **Phase 11 Testing Summary** for final validation

### Validation Requirements (Phase 11 Complete)
- [ ] **REQUIRED**: All unit tests for Phase 11 must pass
- [ ] **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_logging.py -v`
- [ ] **REQUIRED**: Test coverage must meet minimum 80% for logging.py module
- [ ] **REQUIRED**: All test assertions must pass (no failures, no errors)
- [ ] **REQUIRED**: If tests fail, iterate on implementation until all tests pass
- [ ] **REQUIRED**: Document any test failures in fracas.md
- [ ] **REQUIRED**: Phase 11 is NOT complete until all tests pass
- [ ] **Status**: ⏳ Pending - Phase 11 cannot proceed to Initiative Completion until validation complete

---

## Initiative Completion

### Final Validation Tasks
- [ ] **REQUIRED**: Run all evaluation tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/ tests/components/meta_eval/ -v --cov=rag_eval/services/evaluator --cov-report=term-missing`
- [ ] **REQUIRED**: Verify all unit tests pass (minimum 80% coverage across all evaluator modules)
- [ ] **REQUIRED**: Verify all integration tests pass
- [ ] **REQUIRED**: Verify all connection tests documented
- [ ] **REQUIRED**: Performance validation (latency < 30 seconds per example)
- [ ] **REQUIRED**: If any tests fail, iterate on implementation until all tests pass
- [ ] **Final Testing Summary** - Comprehensive testing report across all phases
  - [ ] All unit tests pass (minimum 80% coverage)
  - [ ] All integration tests pass
  - [ ] All connection tests documented
  - [ ] Performance validation (latency < 30 seconds per example)
- [ ] **Technical Debt Documentation** - Complete technical debt catalog and remediation roadmap
  - [ ] Document any shortcuts or deferred features
  - [ ] Document known limitations
  - [ ] Document future enhancement opportunities
- [ ] Code review and quality check
- [ ] Documentation review (all docstrings, README updates)
- [ ] Stakeholder review and approval

### Success Criteria Validation
- [ ] Evaluation Coverage: 100% of evaluation components have unit tests with >80% coverage
- [ ] Pipeline Latency: Average evaluation pipeline latency < 30 seconds per example
- [ ] Judge Accuracy: Meta-Evaluator judge_correct rate > 85% (judge is reliable)
- [ ] Reproducibility: Evaluation results are reproducible for same inputs (accounting for LLM non-determinism)
- [ ] BEIR Metrics Coverage: All evaluation examples have recall@k, precision@k, nDCG@k computed

---

## Blockers
- {List current blockers and dependencies as they arise}

## Notes
- {Implementation notes and decisions made during development}

## Testing Environment Requirements

### Venv Usage
- **REQUIRED**: All backend testing MUST use the same virtual environment: `backend/venv/`
- **REQUIRED**: Activate venv before running any tests: `cd backend && source venv/bin/activate`
- **REQUIRED**: All pytest commands must be run from within the activated venv
- **REQUIRED**: Do not create separate venvs for different phases - use the same venv throughout

### Test Execution Commands
- **Unit Tests**: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluator_<component>.py -v`
- **Meta-Eval Tests**: `cd backend && source venv/bin/activate && pytest tests/components/meta_eval/test_evaluator_<component>.py -v`
- **All Evaluator Tests**: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/ tests/components/meta_eval/ -v`
- **With Coverage**: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/ tests/components/meta_eval/ -v --cov=rag_eval/services/evaluator --cov-report=term-missing`
- **All RAG Tests**: `cd backend && source venv/bin/activate && pytest tests/components/rag/ -v`
- **Connection Tests**: Run connection tests separately (they warn but don't fail if credentials missing)

### Phase Completion Requirements
- **REQUIRED**: Each phase must have all unit tests passing before proceeding to the next phase
- **REQUIRED**: Test coverage must meet minimum 80% for each module
- **REQUIRED**: If tests fail, iterate on implementation until all tests pass (may require multiple iterations)
- **REQUIRED**: Document any test failures in fracas.md
- **REQUIRED**: Phase status must be updated to "✅ Complete" only after all validation requirements pass

## FRACAS Integration
- **Failure Tracking**: All failures, bugs, and unexpected behaviors must be documented in `fracas.md`
- **Investigation Process**: Follow systematic FRACAS methodology for root cause analysis
- **Knowledge Building**: Use failure modes to build organizational knowledge and prevent recurrence
- **Status Management**: Keep failure mode statuses current and move resolved issues to historical section

**FRACAS Document Location**: `docs/initiatives/eval_system/fracas.md`

---

**Document Status**: Draft  
**Last Updated**: 2024-12-19  
**Author**: Documentation Agent  
**Reviewers**: TBD

