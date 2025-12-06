# Phase 2 Decisions — Correctness LLM-Node

## Context

This document captures implementation decisions made during Phase 2 that are not explicitly covered in PRD001.md or RFC001.md.

## Decision 1: Prompt Template Storage (File-Based)

**Decision**: Store correctness prompt template as a file (`backend/rag_eval/prompts/evaluation/correctness_prompt.md`) rather than in the database.

**Rationale**:
- Simpler for MVP: no database dependency for prompt loading
- Easier to version control: prompt changes tracked in git
- Consistent with existing RAG prompt structure (prompt_v1.md, prompt_v2.md)
- Can migrate to database later if needed (similar to RAG prompts)

**Implementation**: Prompt template loaded from file path using `_load_prompt_template()` helper function.

## Decision 2: JSON Parsing with Markdown Code Block Support

**Decision**: Parse JSON responses that may be wrapped in markdown code blocks (```json ... ```).

**Rationale**:
- LLMs sometimes wrap JSON in markdown code blocks for formatting
- More robust parsing handles both raw JSON and markdown-wrapped JSON
- Prevents test failures due to formatting differences

**Implementation**: `_call_correctness_api()` strips markdown code block markers before parsing JSON.

## Decision 3: Model Selection (gpt-4o-mini with Fallback)

**Decision**: Use `gpt-4o-mini` as the default evaluation model, with support for override via config attribute.

**Rationale**:
- Matches RFC001 specification for evaluation LLM nodes
- Cost-efficient for evaluation tasks
- Allows flexibility for testing different models via config

**Implementation**: Defaults to `"gpt-4o-mini"` if `config.azure_ai_foundry_evaluation_model` is not set.

## Decision 4: Temperature Setting (0.1 for Reproducibility)

**Decision**: Use temperature=0.1 for all correctness classification calls.

**Rationale**:
- Matches RFC001 and prompt requirements
- Improves reproducibility while acknowledging inherent LLM non-determinism
- Consistent with RAG generation temperature setting

**Implementation**: Hardcoded `temperature=0.1` in `_call_correctness_api()` calls.

## Decision 5: Error Handling Strategy

**Decision**: Re-raise `AzureServiceError` and `ValueError` as-is, wrap other exceptions in `AzureServiceError`.

**Rationale**:
- Preserves error context for debugging
- Allows callers to handle specific error types appropriately
- Consistent with existing RAG system error handling patterns

**Implementation**: Explicit exception handling in `classify_correctness()` with re-raising of known exceptions.

## Decision 6: Max Tokens for Classification (500)

**Decision**: Use `max_tokens=500` for correctness classification API calls.

**Rationale**:
- Classification responses are short (JSON with boolean + reasoning)
- 500 tokens sufficient for reasoning explanation
- Reduces API costs compared to generation (1000 tokens)

**Implementation**: Default parameter in `_call_correctness_api()`.

## Decision 7: Retry Logic (3 Retries with Exponential Backoff)

**Decision**: Implement retry logic with 3 retries and exponential backoff (1s, 2s, 4s delays).

**Rationale**:
- Handles transient network failures
- Consistent with RAG generation retry strategy
- Prevents test flakiness from temporary API issues

**Implementation**: `_retry_with_backoff()` helper function shared with RAG generation pattern.


