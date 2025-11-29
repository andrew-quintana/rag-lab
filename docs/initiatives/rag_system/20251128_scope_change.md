# Scope Change — Azure Blob Storage Removal

**Date**: 2025-01-27  
**Status**: Implemented

## Summary

Azure Blob Storage upload has been **removed from the project scope**. Documents are now processed in-memory without persistence to blob storage.

## Rationale

- **Simplified Architecture**: Processing documents in-memory reduces complexity and dependencies
- **Faster Processing**: No need to upload/download from blob storage during processing
- **Sufficient for R&D**: For a research and development tool, document persistence is not required
- **Reduced Costs**: Eliminates Azure Blob Storage costs

## Impact

### What Changed

1. **Upload Pipeline Flow** (Updated):
   - **Before**: Upload → Blob Storage → Extract → Chunk → Embed → Index
   - **After**: Upload → Extract → Chunk → Embed → Index (in-memory)

2. **Components**:
   - `rag_eval/services/rag/storage.py` - Still exists with tests, but **not used** in upload pipeline
   - Upload endpoint processes documents directly in-memory

3. **Documentation Updates**:
   - Phase 1 (Blob Storage) marked as out of scope in TODO001.md
   - PRD001.md updated to note blob storage removal
   - Upload pipeline documentation updated

### What Remains

- **Storage Module**: `storage.py` exists and has comprehensive tests (22 tests passing)
- **Tests**: All storage tests remain and pass
- **Code**: Storage functions are available but not integrated into upload pipeline

## Current Progress Without Blob Storage

### ✅ Completed Phases

1. **Phase 0** - Context Harvest ✅
2. **Phase 1** - Blob Storage (out of scope, but component exists) ✅
3. **Phase 2** - Extraction, Preprocessing, and Chunking ✅
   - Ingestion component: ✅ Unit tests complete (12 tests)
   - Chunking component: ✅ Unit tests complete (14 tests)
4. **Phase 3** - Embedding Generation ✅
   - Embeddings component: ✅ Unit tests complete (20 tests)
   - Connection tests: ✅ Passing (2 tests)
   - Azure AI Foundry integration: ✅ Working

### 🔄 In Progress / Next

5. **Phase 4** - Azure AI Search Integration
   - Search component: ✅ Unit tests complete
   - Connection tests: ✅ Passing
6. **Phase 5** - Prompt Template System
7. **Phase 6** - LLM Answer Generation
8. **Phase 7** - Pipeline Orchestration
9. **Phase 8** - Supabase Logging
10. **Phase 9** - Upload Pipeline Integration (no blob storage step)
11. **Phase 10** - End-to-End Testing

## Upload Pipeline (Current Implementation)

The upload pipeline now processes documents directly:

```python
# Current flow (in-memory):
1. Receive file upload
2. Read file content into memory
3. Extract text (Azure Document Intelligence)
4. Chunk text (fixed-size, deterministic)
5. Generate embeddings (Azure AI Foundry)
6. Index chunks (Azure AI Search)
7. Return success response
```

**No blob storage step** - documents are processed entirely in-memory.

## Test Status

- **Total Tests**: 96 passing, 3 skipped
- **Storage Tests**: 22 tests passing (component exists but unused)
- **Connection Tests**: 
  - ✅ Azure AI Foundry: Passing
  - ✅ Azure AI Search: Passing
  - ⚠️ Azure Blob Storage: Skipped (not configured, not needed)
  - ⚠️ Azure Document Intelligence: Skipped (not configured yet)

## Configuration

The following environment variables are **no longer required** for the upload pipeline:
- `AZURE_BLOB_CONNECTION_STRING` (optional, not used)
- `AZURE_BLOB_CONTAINER_NAME` (optional, not used)

## Migration Notes

If blob storage is needed in the future:
1. The `storage.py` module is ready and tested
2. Simply add blob storage step to upload pipeline
3. No code changes needed to storage module

---

**Document Status**: Complete  
**Last Updated**: 2025-01-27  
**Related Documents**: [TODO001.md](./TODO001.md), [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md)

