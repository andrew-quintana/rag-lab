# Test Scripts Review

## Summary

This document reviews test scripts in `backend/scripts/` to determine which should be:
1. Moved to `backend/tests/` as proper pytest tests
2. Deleted as deprecated (superseded by pytest tests)
3. Kept as utility/debugging scripts

## Test Scripts Analysis

### 1. `test_azure_pages_parameter.py`
**Type**: Investigation/Debugging script  
**Purpose**: Tests Azure Document Intelligence page extraction limits  
**Status**: ⚠️ **Keep as debugging tool**  
**Reason**: One-off investigation script for finding page limits. Not a regular test.  
**Recommendation**: Keep in scripts/ or move to `scripts/debug/` or `scripts/investigations/`

### 2. `test_batch_extraction.py`
**Type**: Manual test script  
**Purpose**: Tests batch extraction implementation  
**Status**: ⚠️ **Consider converting to pytest**  
**Reason**: Tests core functionality that should be in test suite  
**Recommendation**: Convert to pytest test in `tests/components/rag/test_rag_ingestion.py` or keep as manual verification script

### 3. `test_batch_page_ranges.py`
**Type**: Manual test script  
**Purpose**: Tests batch processing with different page ranges  
**Status**: ⚠️ **Consider converting to pytest**  
**Reason**: Tests core functionality that should be in test suite  
**Recommendation**: Convert to pytest test in `tests/components/rag/test_rag_ingestion.py` or keep as manual verification script

### 4. `test_e2e_upload.py`
**Type**: E2E test script  
**Purpose**: Tests complete RAG upload pipeline via API  
**Status**: ❌ **Likely deprecated**  
**Reason**: 
- Has corresponding pytest tests: `tests/components/api/test_upload_endpoint.py`
- Has E2E tests: `tests/components/rag/test_rag_e2e.py`
**Recommendation**: **DELETE** if pytest tests cover the same functionality

### 5. `test_extraction_chunking.py`
**Type**: Manual test script  
**Purpose**: Tests extraction and chunking workflow  
**Status**: ❌ **Likely deprecated**  
**Reason**: 
- Has comprehensive pytest tests: `tests/components/rag/test_rag_ingestion.py`
- Has comprehensive pytest tests: `tests/components/rag/test_rag_chunking.py`
**Recommendation**: **DELETE** - functionality is covered by pytest tests

### 6. `test_eval_judgments.py`
**Type**: Database test script  
**Purpose**: Tests eval_judgments table with JSONB judge_output column  
**Status**: ⚠️ **Review and potentially convert**  
**Reason**: 
- Has pytest tests for evaluator components
- But this tests database schema specifically
**Recommendation**: Review if database schema tests exist, if not, convert to pytest test

### 7. `test_meta_eval_summaries.py`
**Type**: Database test script  
**Purpose**: Tests meta eval summaries functionality  
**Status**: ⚠️ **Review and potentially convert**  
**Reason**: 
- Has pytest tests: `tests/components/meta_eval/test_evaluator_meta_eval.py`
- But this might test different aspects
**Recommendation**: Review if pytest tests cover the same functionality, if yes, DELETE

## Utility Scripts (Keep These)

These are not tests and should remain in `scripts/`:
- ✅ `setup_azure_queues.py` - Setup utility
- ✅ `get_azure_storage_connection.py` - Helper utility
- ✅ `clear_search_index.py` - Utility script
- ✅ `verify_document_intelligence_tier.py` - Verification utility
- ✅ `check_document_chunks.py` - Diagnostic utility
- ✅ `debug_extraction.py` - Debug utility
- ✅ `validate_prompt_migrations.py` - Validation utility
- ✅ `regenerate_previews.py` - Utility script

## Recommendations

### Immediate Actions

1. **DELETE** (deprecated, covered by pytest):
   - `test_e2e_upload.py` - Covered by `test_upload_endpoint.py` and `test_rag_e2e.py`
   - `test_extraction_chunking.py` - Covered by `test_rag_ingestion.py` and `test_rag_chunking.py`

2. **REVIEW** (check if pytest tests cover same functionality):
   - `test_eval_judgments.py` - Check if database schema tests exist
   - `test_meta_eval_summaries.py` - Compare with `test_evaluator_meta_eval.py`

3. **KEEP** (debugging/investigation tools):
   - `test_azure_pages_parameter.py` - Investigation script
   - `test_batch_extraction.py` - Manual verification (or convert to pytest)
   - `test_batch_page_ranges.py` - Manual verification (or convert to pytest)

### Future Improvements

- Convert `test_batch_extraction.py` and `test_batch_page_ranges.py` to pytest integration tests
- Move investigation scripts to `scripts/debug/` or `scripts/investigations/` subdirectory

