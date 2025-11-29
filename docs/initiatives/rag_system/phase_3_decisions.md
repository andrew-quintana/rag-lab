# Phase 3 Decisions — Embedding Generation

## Context

This document captures implementation decisions made during Phase 3 that are not already documented in PRD001.md or RFC001.md.

**Phase**: Phase 3 — Embedding Generation  
**Component**: `rag_eval/services/rag/embeddings.py`  
**Date**: 2025-01-27

---

## Decision 1: REST API Approach for Azure AI Foundry Embeddings

**Decision**: Use REST API with `requests` library instead of `azure-ai-inference` SDK.

**Rationale**:
- Azure AI Foundry uses OpenAI-compatible REST API, which is well-documented and straightforward
- `requests` library is already a dependency and provides sufficient functionality
- REST API approach is consistent with existing chunking implementation (`chunking.py`)
- Simpler error handling and retry logic implementation
- No need for additional SDK dependencies

**Implementation**: 
- Direct HTTP POST requests to `/openai/deployments/{model}/embeddings` endpoint
- Uses `api-version=2024-02-15-preview` for compatibility
- Headers include `Content-Type: application/json` and `api-key: {api_key}`

**Alternative Considered**: Using `azure-ai-inference` SDK  
**Rejected**: Adds unnecessary dependency when REST API is sufficient and consistent with existing patterns.

---

## Decision 2: Batch Processing Strategy

**Decision**: Process all chunks in a single batch API call rather than chunking into smaller batches.

**Rationale**:
- Azure AI Foundry embedding API supports batch processing natively
- Single API call is more efficient than multiple calls
- Simpler implementation (no batching logic needed)
- Reduces API latency and overhead
- Azure AI Foundry handles large batches efficiently

**Implementation**:
- All chunk texts are sent in a single `input` array to the embedding API
- API returns embeddings in the same order as input texts
- No manual batching or pagination required

**Limitations**:
- If Azure AI Foundry has batch size limits in the future, this may need adjustment
- Very large document sets (1000+ chunks) may need batching logic added later

**Alternative Considered**: Manual batching (e.g., 100 chunks per API call)  
**Rejected**: Unnecessary complexity when Azure handles batching efficiently.

---

## Decision 3: Embedding Dimension Validation

**Decision**: Validate embedding dimensions are consistent within a batch, but do not enforce specific dimension counts.

**Rationale**:
- Different embedding models have different dimensions (e.g., text-embedding-3-small: 1536, text-embedding-ada-002: 1536)
- Enforcing specific dimensions would require model-specific configuration
- Consistency within a batch is sufficient for vector similarity search
- More flexible for future model changes

**Implementation**:
- All embeddings in a batch must have the same dimension
- Dimension mismatch raises `ValueError` with clear error message
- No hardcoded dimension checks

**Alternative Considered**: Hardcode expected dimensions per model  
**Rejected**: Less flexible and requires maintenance when models change.

---

## Decision 4: Retry Logic Implementation

**Decision**: Implement retry logic with exponential backoff directly in embeddings module, following same pattern as `storage.py`.

**Rationale**:
- Consistent error handling pattern across RAG services
- Exponential backoff handles transient Azure service failures
- 3 retries (4 total attempts) is sufficient for most transient errors
- Base delay of 1.0 seconds provides reasonable backoff timing

**Implementation**:
- `_retry_with_backoff()` helper function (same pattern as `storage.py`)
- Retries on `requests.RequestException` and general `Exception`
- Exponential backoff: 1.0s, 2.0s, 4.0s delays
- Raises `AzureServiceError` after all retries exhausted

**Alternative Considered**: Using a retry library (e.g., `tenacity`)  
**Rejected**: Simpler to maintain consistency with existing codebase patterns.

---

## Decision 5: Connection Test Behavior

**Decision**: Connection tests should warn but not fail when credentials are missing or invalid.

**Rationale**:
- Connection tests are informational, not required for test suite to pass
- Developers may not have Azure credentials configured locally
- Tests should be runnable without external service dependencies
- Warnings provide useful feedback without breaking CI/CD

**Implementation**:
- Connection tests use `pytest.skip()` when credentials are missing
- Connection tests catch `AzureServiceError` and skip with warning when credentials are invalid
- Warnings clearly indicate that connection tests are informational only
- Test suite passes even if connection tests are skipped

**Alternative Considered**: Failing tests when credentials are invalid  
**Rejected**: Would break test suite for developers without valid credentials.

---

## Decision 6: Model Consistency Enforcement

**Decision**: Enforce model consistency via configuration (same model in config for both chunks and queries), not runtime validation.

**Rationale**:
- Configuration-based enforcement is simpler and more maintainable
- Single source of truth: `config.azure_ai_foundry_embedding_model`
- Runtime validation would require storing model name with each embedding (unnecessary overhead)
- Configuration ensures consistency at deployment time

**Implementation**:
- Both `generate_embeddings()` and `generate_query_embedding()` use `config.azure_ai_foundry_embedding_model`
- No runtime model name comparison or validation needed
- Documentation clearly states that same model must be used for chunks and queries

**Alternative Considered**: Runtime validation comparing model names  
**Rejected**: Adds complexity without significant benefit when configuration is the source of truth.

---

## Decision 7: Endpoint Format Handling

**Decision**: Strip trailing slashes from endpoint URLs to prevent double slashes in API calls.

**Rationale**:
- Azure AI Foundry endpoints may have trailing slashes
- Code constructs URLs by appending paths (e.g., `{endpoint}/openai/deployments/...`)
- Trailing slashes cause double slashes: `https://endpoint.com//openai/...`
- Double slashes can cause 404 errors or connection issues

**Implementation**:
- Added `endpoint.rstrip('/')` before constructing API URLs
- Applied to both embeddings and chunking modules
- Ensures consistent URL format regardless of endpoint configuration

**Alternative Considered**: Requiring endpoints without trailing slashes  
**Rejected**: More user-friendly to handle trailing slashes automatically.

---

## Summary

All decisions align with the existing codebase patterns and maintain simplicity while ensuring correctness. The implementation follows established patterns from Phase 1 (storage) and Phase 2 (chunking) for consistency.

**Key Principles**:
- Consistency with existing codebase patterns
- Simplicity over premature optimization
- Flexibility for future changes
- Clear error messages and documentation
- User-friendly configuration (automatic trailing slash handling)

**Validation** (2025-01-27):
- All 22 tests passing (including connection tests)
- Azure AI Foundry connection validated
- Endpoint format fix validated
- Embedding generation working with real service

---

**Document Status**: Complete  
**Last Updated**: 2025-01-27  
**Related Documents**: [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md), [TODO001.md](./TODO001.md)

