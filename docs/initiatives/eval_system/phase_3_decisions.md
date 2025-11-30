# Phase 3 Decisions — Hallucination LLM-Node

## Context

This document captures implementation decisions made during Phase 3 that are not explicitly covered in PRD001.md or RFC001.md.

## Decision 1: Prompt Template Storage (File-Based)

**Decision**: Store hallucination prompt template as a file (`backend/rag_eval/prompts/evaluation/hallucination_prompt.md`) rather than in the database.

**Rationale**:
- Consistent with Phase 2 (correctness prompt) approach
- Simpler for MVP: no database dependency for prompt loading
- Easier to version control: prompt changes tracked in git
- Can migrate to database later if needed (similar to RAG prompts)

**Implementation**: Prompt template loaded from file path using `_load_prompt_template()` method inherited from `BaseEvaluatorNode`.

## Decision 2: Retrieved Context Formatting with Chunk IDs

**Decision**: Format retrieved context by concatenating chunk texts with chunk IDs in format `[Chunk ID: {chunk_id}] {chunk_text}`.

**Rationale**:
- Provides traceability: chunk IDs help identify which chunks support or contradict claims
- Clear separation between chunks for LLM analysis
- Consistent with retrieval result structure
- Helps with debugging and prompt analysis

**Implementation**: `_format_retrieved_context()` method formats `List[RetrievalResult]` into string with chunk IDs.

## Decision 3: Reference Answer Explicitly NOT Used

**Decision**: Hallucination detection does NOT use reference answer - only retrieved context is used as ground truth.

**Rationale**:
- **Critical requirement**: Hallucination is about grounding, not correctness
- Reference answer may contain information not in retrieved chunks (which would be a retrieval failure, not a hallucination)
- Strict grounding analysis: model answer must be supported by what was actually retrieved
- This is explicitly emphasized in prompt template and function docstrings

**Implementation**: 
- Prompt template explicitly states "reference answer is NOT used"
- Function signature does not include `reference_answer` parameter
- Tests verify reference answer is not in prompt construction

## Decision 4: Empty Retrieved Context Handling

**Decision**: Raise `ValueError` if retrieved context is empty (zero chunks).

**Rationale**:
- Cannot perform grounding analysis without any context
- Empty context is an error condition, not a valid input
- Better to fail fast than return ambiguous result
- Consistent with input validation for empty model answer

**Implementation**: Input validation in `classify_hallucination()` checks for empty list and raises `ValueError`.

## Decision 5: Temperature Setting (0.1 for Reproducibility)

**Decision**: Use temperature=0.1 for all hallucination classification calls.

**Rationale**:
- Matches RFC001 and prompt requirements
- Consistent with Phase 2 (correctness node) temperature setting
- Improves reproducibility while acknowledging inherent LLM non-determinism
- Consistent with RAG generation temperature setting

**Implementation**: Hardcoded `temperature=0.1` in `_call_llm()` calls via base class.

## Decision 6: Error Handling Strategy

**Decision**: Re-raise `AzureServiceError` and `ValueError` as-is, wrap other exceptions in `AzureServiceError`.

**Rationale**:
- Preserves error context for debugging
- Allows callers to handle specific error types appropriately
- Consistent with Phase 2 (correctness node) error handling
- Matches existing RAG system error handling patterns

**Implementation**: Explicit exception handling in `classify_hallucination()` with re-raising of known exceptions.

## Decision 7: Max Tokens for Classification (500)

**Decision**: Use `max_tokens=500` for hallucination classification API calls.

**Rationale**:
- Classification responses are short (JSON with boolean + reasoning)
- 500 tokens sufficient for reasoning explanation
- Consistent with Phase 2 (correctness node) max tokens
- Reduces API costs compared to generation (1000 tokens)

**Implementation**: Default parameter in `_call_llm()` via base class.

## Decision 8: BaseEvaluatorNode Inheritance Pattern

**Decision**: Use `BaseEvaluatorNode` base class pattern for hallucination evaluator, similar to correctness evaluator.

**Rationale**:
- Code reuse: common functionality (prompt loading, LLM calls, JSON parsing) shared
- Consistency: all evaluation nodes follow same pattern
- Maintainability: bug fixes and improvements benefit all nodes
- Matches RFC001 Decision 5 (Base Evaluator Class Pattern)

**Implementation**: `HallucinationEvaluator` inherits from `BaseEvaluatorNode` and implements `_construct_prompt()` method.

