# Phase 11 Decisions — Logging and Persistence

## Overview

This document captures implementation decisions made during Phase 11 that are not explicitly covered in PRD001.md or RFC001.md.

## Database Schema Decisions

### JSONB Storage Format

**Decision**: All evaluation metrics are stored as JSONB columns in PostgreSQL.

**Rationale**:
- Provides flexibility for future schema changes without migrations
- Allows storing nested structures (e.g., JudgePerformanceMetrics with nested JudgeMetricScores)
- Enables efficient JSON queries using GIN indexes
- Supports extensibility without breaking existing queries

**Implementation**: All result components (JudgeEvaluationResult, MetaEvaluationResult, BEIRMetricsResult, JudgePerformanceMetrics) are serialized to JSON and stored as JSONB.

### Table Structure

**Decision**: Single `evaluation_results` table with all metrics as JSONB columns.

**Rationale**:
- Simpler schema than normalized tables
- Easier to query complete evaluation results
- JSONB provides sufficient query performance with GIN indexes
- Aligns with flexible storage requirements

**Alternative Considered**: Normalized tables with separate tables for each metric type. Rejected due to complexity and lack of clear benefit for MVP.

### Judge Performance Metrics Storage

**Decision**: Store `judge_performance_metrics` as a separate JSONB column, updated via batch UPDATE query.

**Rationale**:
- Judge performance metrics are calculated from batch results, not individual results
- Storing in metadata column allows association with batch
- Simple UPDATE query is sufficient for MVP
- Can be normalized to separate table in future if needed

**Alternative Considered**: Separate `judge_performance_metrics` table. Rejected for MVP simplicity.

## Serialization Decisions

### Custom JSON Serialization

**Decision**: Implement custom serialization helpers instead of using default `json.dumps()` with custom encoder.

**Rationale**:
- More explicit control over serialization format
- Easier to test and debug
- Clear separation of concerns (serialization vs. database operations)
- Handles dataclasses, datetime, and nested structures explicitly

**Implementation**: Helper functions `_serialize_judge_output()`, `_serialize_meta_eval_output()`, `_serialize_beir_metrics()`, `_serialize_judge_performance_metrics()`.

### Datetime Serialization

**Decision**: Serialize datetime objects to ISO format strings.

**Rationale**:
- ISO format is standard and parseable
- JSON-compatible
- Preserves timezone information
- Easy to deserialize back to datetime if needed

## Error Handling Decisions

### Graceful Degradation

**Decision**: Logging failures never raise exceptions; they are logged and the function returns None or continues.

**Rationale**:
- Evaluation pipeline should not fail due to logging issues
- Logging is optional feature
- Errors are logged for debugging
- Allows evaluation to continue in local-only mode

**Implementation**: All database operations wrapped in try-except blocks that log errors and return gracefully.

### Batch Partial Failures

**Decision**: Batch logging continues even if individual results fail to serialize or insert.

**Rationale**:
- Partial success is better than total failure
- Errors are logged for each failed result
- Successful results are still logged
- Provides visibility into which results failed

**Implementation**: Each result in batch is processed individually with error handling.

## Local-Only Mode

**Decision**: When `query_executor` is None, logging functions return immediately without attempting database operations.

**Rationale**:
- Supports local development and testing without database
- Explicit opt-in for database logging
- No performance overhead when logging is not needed
- Clear separation between local and database modes

## Testing Decisions

### Mock Strategy

**Decision**: Use Mock objects for QueryExecutor rather than real database connections in unit tests.

**Rationale**:
- Faster test execution
- No database setup required
- Tests are isolated and deterministic
- Covers all code paths without database dependencies

**Implementation**: All tests use `mock_query_executor` fixture that mocks `execute_insert()` and `execute_query()` methods.

### Coverage Target

**Decision**: Achieve 80%+ test coverage for logging.py module.

**Rationale**:
- Meets project standards
- Ensures critical paths are tested
- Validates error handling
- Provides confidence in implementation

**Result**: Achieved 97% coverage with comprehensive test suite.

## Future Considerations

### Batch Insert Optimization

**Current**: Individual INSERT statements in loop.

**Future Enhancement**: Consider using `execute_values()` or `execute_batch()` from psycopg2.extras for better performance with large batches.

### Metrics Table Normalization

**Current**: Judge performance metrics stored in JSONB column.

**Future Enhancement**: Consider separate `judge_performance_metrics` table if query patterns require it.

### Deserialization

**Current**: Focus on serialization and storage.

**Future Enhancement**: Add deserialization helpers to reconstruct EvaluationResult objects from JSONB for analysis and metrics recalculation.


