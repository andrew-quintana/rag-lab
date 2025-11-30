# Phase 1 Prompt — Evaluation Dataset Construction

## Context

This prompt guides the implementation of **Phase 1: Evaluation Dataset Construction** for the RAG Evaluation MVP system. This phase involves manually creating 5 validation samples for the evaluation dataset (not an automated function).

**Related Documents:**
- @docs/initiatives/eval_system/PRD001.md - Product requirements (FR8: Evaluation Dataset Construction)
- @docs/initiatives/eval_system/RFC001.md - Technical design (Phase 1: Evaluation Dataset Construction)
- @docs/initiatives/eval_system/TODO001.md - Implementation tasks (Phase 1 section - check off tasks as completed)
- @docs/initiatives/eval_system/context.md - Project context

## Objectives

1. **Create Validation Dataset**: Manually create 5 validation samples (not automated)
2. **Cover Question Types**: Ensure samples cover cost, coverage, eligibility, and out-of-pocket max questions
3. **Identify Ground Truth**: Map questions to actual chunk IDs from indexed document
4. **Validate Format**: Ensure dataset matches EvaluationExample dataclass structure

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/eval_system/TODO001.md Phase 1 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: All unit tests for Phase 1 must pass before proceeding to Phase 2
- **REQUIRED**: Run tests using venv: `cd backend && source venv/bin/activate && pytest tests/components/evaluator/test_evaluation_dataset.py -v`
- **REQUIRED**: All test assertions must pass (no failures, no errors)

### Documentation
- **REQUIRED**: Create `phase_1_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_1_testing.md` documenting testing results
- **REQUIRED**: Create `phase_1_handoff.md` summarizing what's needed for Phase 2

## Key References

### Source Document
- @backend/tests/fixtures/sample_documents/healthguard_select_ppo_plan.pdf - Source PDF for QA pairs

### Data Structures
- `EvaluationExample` dataclass (from RFC001.md):
  - `example_id: str`
  - `question: str`
  - `reference_answer: str`
  - `ground_truth_chunk_ids: List[str]`
  - `beir_failure_scale_factor: float` (range [0.0, 1.0])

### Output Location
- `backend/tests/fixtures/evaluation_dataset/validation_dataset.json`

## Phase 1 Tasks

### Setup
1. Verify `healthguard_select_ppo_plan.pdf` exists
2. Ensure document has been indexed via upload pipeline (to identify actual chunk IDs)
3. Create directory structure: `backend/tests/fixtures/evaluation_dataset/`
4. Review document content to identify suitable QA pair topics

### Dataset Creation
1. **Manually create 5 validation samples** (not via automated generation)
2. For each sample, include:
   - `example_id`: Unique identifier (e.g., "val_001")
   - `question`: Question text covering different types
   - `reference_answer`: Gold reference answer
   - `ground_truth_chunk_ids`: List of actual chunk IDs from indexed document
   - `beir_failure_scale_factor`: Float in range [0.0, 1.0]
3. Ensure samples cover:
   - Cost-related questions (copay, deductible, coinsurance)
   - Coverage questions
   - Eligibility questions
   - Out-of-pocket maximum questions
4. Store as single JSON file: `validation_dataset.json`
5. Validate JSON format matches `EvaluationExample` dataclass structure

### Testing
1. Create test file: `backend/tests/components/evaluator/test_evaluation_dataset.py`
2. Validate that `validation_dataset.json` exists and is properly formatted
3. Validate that all 5 samples have required fields
4. Validate that `ground_truth_chunk_ids` reference actual chunks from indexed document
5. Validate that `beir_failure_scale_factor` is in range [0.0, 1.0]
6. Validate that questions cover different types

### Documentation
1. Document dataset structure and format
2. Document how to identify ground-truth chunk IDs from indexed document
3. Document `beir_failure_scale_factor` calculation methodology

## Success Criteria

- [ ] 5 validation samples created and stored in `validation_dataset.json`
- [ ] All samples have complete required fields
- [ ] Samples cover all required question types
- [ ] All unit tests pass
- [ ] All Phase 1 tasks in TODO001.md checked off
- [ ] Phase 1 handoff document created

## Important Notes

- **Development Task**: This is a manual development task, not an automated function
- **Full Dataset**: Full evaluation dataset will be created by humans later; this is only for validation
- **Chunk IDs**: Must reference actual chunk IDs from indexed document (not placeholder values)

## Blockers

- **BLOCKER**: Phase 2 cannot proceed until Phase 1 validation complete
- **BLOCKER**: Document must be indexed first to identify actual chunk IDs

## Next Phase

After completing Phase 1, proceed to **Phase 2: Correctness LLM-Node** using @docs/initiatives/eval_system/prompt_phase_2_001.md

