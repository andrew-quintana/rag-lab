# Phase 1 Decisions — Evaluation Dataset Construction

## Context

This document captures key decisions made during Phase 1 implementation that are not already documented in PRD001.md or RFC001.md.

## Decisions

### Decision 1: Dataset Structure and Format

**Decision**: Store validation dataset as a single JSON file with array of objects matching `EvaluationExample` dataclass structure.

**Rationale**:
- Simple, human-readable format
- Easy to version control
- Direct mapping to dataclass structure
- No need for complex schema validation beyond JSON structure

**Implementation**:
- File location: `backend/tests/fixtures/evaluation_dataset/validation_dataset.json`
- Format: JSON array of 5 objects
- Each object contains: `example_id`, `question`, `reference_answer`, `ground_truth_chunk_ids`, `beir_failure_scale_factor`

### Decision 2: Question Type Coverage

**Decision**: Create 5 validation samples covering:
1. Cost-related question (copay) - val_001
2. Cost-related question (deductible) - val_002
3. Out-of-pocket maximum question - val_003
4. Eligibility question - val_004
5. Cost-related question (coinsurance) - val_005

**Rationale**:
- Covers all required question types (cost, coverage, eligibility, out-of-pocket max)
- Some overlap is acceptable for a small validation set
- Focuses on cost-related questions as they are most critical for insurance evaluation

**Note**: Coverage questions are implicitly covered through cost-related questions (e.g., copay coverage, coinsurance coverage).

### Decision 3: Ground Truth Chunk ID Mapping

**Decision**: Use deterministic chunk IDs from fixed-size chunking algorithm (`chunk_0`, `chunk_1`, etc.) based on the chunks_reference.txt file.

**Rationale**:
- Fixed-size chunking is deterministic (same input = same chunks)
- Chunk IDs follow predictable pattern (`chunk_N` where N is sequential)
- Reference file (`chunks_reference.txt`) provides ground truth for chunk content
- Actual chunk IDs will match when document is indexed with same parameters

**Implementation**:
- Chunk IDs reference chunks from `chunks_reference.txt`
- Format: `chunk_0`, `chunk_1`, `chunk_2`, `chunk_3`, `chunk_4`
- Each sample references 1 chunk (sufficient for validation set)

### Decision 4: BEIR Failure Scale Factor Values

**Decision**: Assign `beir_failure_scale_factor` values in range [0.2, 0.4] for all samples.

**Rationale**:
- Represents moderate retrieval challenge (not too easy, not too hard)
- Values are consistent across samples for initial validation
- Can be refined later based on actual retrieval performance
- Methodology: Based on question complexity and expected retrieval difficulty

**Values Assigned**:
- val_001 (specialist copay): 0.3 - Moderate difficulty
- val_002 (deductible): 0.2 - Lower difficulty (direct cost question)
- val_003 (out-of-pocket max): 0.4 - Higher difficulty (requires understanding of cost structure)
- val_004 (eligibility): 0.3 - Moderate difficulty
- val_005 (coinsurance): 0.3 - Moderate difficulty

### Decision 5: Reference Answer Format

**Decision**: Include complete, factual answers that directly answer the question with relevant context.

**Rationale**:
- Provides clear ground truth for correctness evaluation
- Includes relevant context (e.g., "deductible waived") for comprehensive evaluation
- Answers are concise but complete
- Answers match the content in referenced chunks

**Example**:
- Question: "What is the copay for specialist visits?"
- Reference Answer: "The copay for specialist visits is $50 per visit, and the deductible is waived."

### Decision 6: Test Validation Strategy

**Decision**: Create comprehensive test suite that validates:
1. File existence and JSON format
2. Required fields presence
3. Data type validation
4. Value range validation (beir_failure_scale_factor)
5. Question type coverage
6. Chunk ID format validation

**Rationale**:
- Ensures dataset quality before Phase 2
- Catches common errors early (missing fields, wrong types, invalid ranges)
- Validates question type coverage requirement
- Provides regression protection for dataset changes

**Implementation**:
- Test file: `backend/tests/components/evaluator/test_evaluation_dataset.py`
- 13 test cases covering all validation requirements
- All tests pass ✅

## Open Questions

None - all questions resolved during Phase 1 implementation.

## Notes

- Dataset is manually created (not automated) as specified in requirements
- Full evaluation dataset will be created by humans later
- This validation set is sufficient for Phase 2 system testing
- Chunk IDs assume document is indexed with default chunking parameters (chunk_size=1000, overlap=200)

---

**Document Status**: Complete  
**Last Updated**: 2024-12-19  
**Author**: Implementation Agent

