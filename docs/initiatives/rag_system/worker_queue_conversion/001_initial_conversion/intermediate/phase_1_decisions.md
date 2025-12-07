# Phase 1 Decisions — Persistence Infrastructure

## Overview

This document captures implementation decisions made during Phase 1 that are not explicitly covered in PRD001.md or RFC001.md.

## Database Schema Decisions

### Extracted Text Storage

**Decision**: Store extracted text in a `TEXT` column in the `documents` table.

**Rationale**:
- RFC001 recommends starting with database column for simplicity
- Single query to retrieve extracted text
- No additional storage service dependency
- Can migrate to storage-based approach if size becomes an issue

**Implementation**: Added `extracted_text TEXT` column to `documents` table in migration `0019_add_worker_queue_persistence.sql`.

**Future Consideration**: If extracted text size becomes a performance issue (>1MB per document), migrate to Supabase Storage or Azure Blob Storage with file-based storage.

### Chunks Table Design

**Decision**: Use a single `chunks` table with `embedding` column (JSONB) rather than separate embeddings table.

**Rationale**:
- RFC001 recommends column in chunks table for simplicity
- Single query to get chunk + embedding together
- Maintains relationship between chunks and embeddings
- Can migrate to separate table if needed for performance

**Implementation**: Created `chunks` table with:
- `chunk_id VARCHAR(255) PRIMARY KEY`
- `document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE`
- `text TEXT NOT NULL`
- `metadata JSONB`
- `embedding JSONB` (for storing embeddings as JSON array of floats)
- `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`
- Index on `document_id` for efficient queries

**Future Consideration**: If embedding storage becomes a performance issue, consider:
- Separate `embeddings` table with foreign key to chunks
- PostgreSQL vector extension (pgvector) for native vector storage
- External vector database (e.g., Pinecone, Weaviate)

### Status Column Default Value

**Decision**: Use `'uploaded'` as the default status value.

**Rationale**:
- Documents start in `uploaded` state when first created
- Aligns with status state machine progression
- Existing documents without status will default to `uploaded`

**Implementation**: `status VARCHAR(50) DEFAULT 'uploaded'` in migration.

### Timestamp Columns

**Decision**: Add separate timestamp columns for each pipeline stage (`parsed_at`, `chunked_at`, `embedded_at`, `indexed_at`).

**Rationale**:
- Enables tracking of processing time per stage
- Supports debugging and performance monitoring
- Allows querying documents by stage completion time
- More granular than single `processed_at` timestamp

**Implementation**: Added nullable `TIMESTAMP` columns for each stage in migration.

## Persistence Layer Design Decisions

### Idempotency Implementation

**Decision**: Implement idempotency checks using status-based comparison rather than checking for existing data.

**Rationale**:
- Status-based checks are simpler and more reliable
- Status progression is deterministic: `uploaded` → `parsed` → `chunked` → `embedded` → `indexed`
- Workers can safely skip processing if status is already at or beyond target stage
- Avoids race conditions from checking data existence

**Implementation**: 
- `check_document_status()` retrieves current status
- `should_process_document()` compares current status order with target status order
- Status order: `uploaded` (0) < `parsed` (1) < `chunked` (2) < `embedded` (3) < `indexed` (4)

### Chunk Persistence Strategy

**Decision**: Delete existing chunks before inserting new ones (overwrite strategy) for idempotency.

**Rationale**:
- Enables safe retry of chunking operations
- Ensures consistency: chunks always match current chunking configuration
- Simpler than checking for existing chunks and updating individually
- DELETE + INSERT is atomic within a transaction

**Implementation**: `persist_chunks()` first deletes all chunks for the document, then inserts new chunks.

### Embedding Persistence Strategy

**Decision**: Update embeddings in-place for existing chunks rather than deleting/recreating.

**Rationale**:
- Embeddings are generated after chunks exist
- Chunks should not be recreated when embeddings are updated
- UPDATE is more efficient than DELETE + INSERT for embeddings
- Maintains chunk metadata and relationships

**Implementation**: `persist_embeddings()` updates the `embedding` column for each chunk using `UPDATE` statements.

### Error Handling Strategy

**Decision**: Raise `DatabaseError` for all database-related errors, `ValueError` for validation errors.

**Rationale**:
- Consistent error handling across persistence functions
- Allows callers to distinguish between validation errors and database errors
- Aligns with existing codebase error handling patterns

**Implementation**: All functions validate inputs (raise `ValueError`) and wrap database exceptions in `DatabaseError`.

### JSONB Handling

**Decision**: Handle both JSON string and list formats for embeddings (PostgreSQL JSONB can return either).

**Rationale**:
- PostgreSQL JSONB columns can return data as strings or native Python types
- Need to handle both cases for robustness
- JSON parsing is only needed if data is a string

**Implementation**: `load_embeddings()` checks if embedding is already a list before parsing JSON.

## Testing Decisions

### Test Data Strategy

**Decision**: Use actual files (PDFs) and realistic data in unit tests instead of in-memory mock data.

**Rationale**:
- Ensures tests validate behavior with real-world data patterns
- Tests with actual extracted text, chunks, and embeddings catch real issues
- More confidence that persistence layer handles realistic data correctly
- Aligns with testing best practices for data-intensive applications

**Implementation**: 
- Tests use actual PDF files from `backend/tests/fixtures/sample_documents/`
- Extracted text represents actual document content (realistic health insurance document text)
- Chunks are generated using actual `chunk_text_fixed_size()` function from real extracted text
- Embeddings are realistic vectors (1536 dimensions) with proper value ranges
- Database operations are still mocked, but data is realistic

### Test Coverage Target

**Decision**: Achieve minimum 80% test coverage (achieved 91%).

**Rationale**:
- PRD001 requires minimum 80% coverage
- Higher coverage provides better confidence in code quality
- Edge cases and error paths are well-tested

**Implementation**: 42 comprehensive unit tests covering:
- All load/persist operations (using actual extracted text, chunks, embeddings)
- Error handling (missing data, invalid IDs, database errors)
- Idempotency checks
- Edge cases (empty data, null values, large payloads from actual files)
- Deletion operations (using actual chunks)

### Mock Strategy

**Decision**: Mock `DatabaseConnection` and `QueryExecutor` at the module level, but use actual file data.

**Rationale**:
- Tests run fast without real database connections
- Tests are deterministic and reproducible
- No external dependencies required
- Uses realistic data from actual files for better test quality
- Aligns with existing test patterns in codebase

**Implementation**: 
- All tests use `unittest.mock` to mock database operations
- All tests use actual files and realistic data (PDFs, extracted text, chunks, embeddings)
- Fixtures load actual PDF files and generate realistic test data

## Migration Decisions

### Migration File Naming

**Decision**: Use sequential migration number `0019_add_worker_queue_persistence.sql`.

**Rationale**:
- Follows existing migration naming convention
- Next sequential number after `0018_standardize_ids_to_uuid.sql`
- Descriptive name indicates purpose

**Implementation**: Created migration file in `infra/supabase/migrations/`.

### Migration Safety

**Decision**: Use `IF NOT EXISTS` and `ADD COLUMN IF NOT EXISTS` for idempotent migrations.

**Rationale**:
- Allows safe re-running of migrations
- Prevents errors if columns/tables already exist
- Supports development and deployment workflows

**Implementation**: All `ALTER TABLE` and `CREATE TABLE` statements use `IF NOT EXISTS` clauses.

## Code Organization Decisions

### Module Location

**Decision**: Place persistence module in `rag_eval/services/workers/persistence.py`.

**Rationale**:
- Persistence is worker-specific functionality
- Keeps worker-related code organized together
- Clear separation from core service modules

**Implementation**: Created `rag_eval/services/workers/` directory with `persistence.py` module.

### Function Signatures

**Decision**: All persistence functions accept `config` parameter for database connection.

**Rationale**:
- Consistent with existing codebase patterns
- Allows dependency injection for testing
- Config contains database credentials

**Implementation**: All functions follow pattern: `function_name(document_id: str, ..., config) -> ...`.

## Future Considerations

### Performance Optimization

If performance becomes an issue:
- Add connection pooling configuration tuning
- Consider pagination for large chunk/embedding loads
- Monitor database query performance and add indexes as needed
- Consider caching frequently accessed data

### Storage Migration

If database size becomes an issue:
- Migrate extracted text to Supabase Storage or Azure Blob Storage
- Consider separate embeddings table or external vector database
- Implement data archival for old documents

### Schema Evolution

If schema changes are needed:
- Use migration files following existing pattern
- Maintain backward compatibility during migration
- Test migrations on staging environment first

---

**Document Status**: Complete  
**Last Updated**: 2025-01-XX  
**Author**: Implementation Agent

