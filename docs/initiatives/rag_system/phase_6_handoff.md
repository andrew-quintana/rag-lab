# Phase 6 Handoff — LLM Answer Generation

## Document Status
**Status**: Complete  
**Date**: 2025-01-27  
**Phase**: Phase 6 — LLM Answer Generation  
**Next Phase**: Phase 7 — Pipeline Orchestration

## Phase 6 Completion Summary

### Implementation Status
✅ **Phase 6 is complete and ready for Phase 7**

All Phase 6 objectives have been achieved:
- ✅ LLM answer generation using Azure AI Foundry implemented
- ✅ Prompt construction integration (Phase 5) complete
- ✅ Generation parameters configured (temperature: 0.1, max_tokens: 1000)
- ✅ ModelAnswer object creation with metadata implemented
- ✅ Comprehensive unit tests written and passing (40 tests total)
- ✅ Connection test implemented (warns if credentials missing)
- ✅ All error paths tested (100% coverage)
- ✅ Retry logic with exponential backoff implemented

### Files Created/Modified

#### Implementation Files
- `backend/rag_eval/services/rag/generation.py` - Answer generation functions
  - `generate_answer()` - Main function for LLM answer generation
  - `_call_generation_api()` - Internal function for Azure AI Foundry API calls
  - `_retry_with_backoff()` - Retry logic with exponential backoff

#### Test Files
- `backend/tests/test_rag_generation.py` - Comprehensive unit tests (40 tests total)
  - 16 new tests for `generate_answer()` function
  - 1 connection test for Azure AI Foundry generation

#### Documentation Files
- `docs/initiatives/rag_system/phase_6_decisions.md` - Implementation decisions
- `docs/initiatives/rag_system/phase_6_testing.md` - Testing summary
- `docs/initiatives/rag_system/phase_6_handoff.md` - This document

#### Updated Files
- `docs/initiatives/rag_system/TODO001.md` - All Phase 6 tasks marked complete

## What's Ready for Phase 7

### Available Functions

#### `generate_answer(query: Query, retrieved_chunks: List[RetrievalResult], prompt_version: str, config, query_executor: Optional[QueryExecutor] = None) -> ModelAnswer`
- **Status**: ✅ Complete and tested
- **Usage**: Generate LLM answer using Azure AI Foundry
- **Features**:
  - Loads and constructs prompt using Phase 5 implementation
  - Calls Azure AI Foundry (OpenAI-compatible API) for generation
  - Configures generation parameters (temperature: 0.1, max_tokens: 1000)
  - Parses and validates LLM response
  - Creates ModelAnswer object with metadata
  - Implements retry logic with exponential backoff (3 retries max)
  - Generates query_id if missing
  - Extracts retrieved chunk IDs for traceability

### Dependencies Satisfied

#### Phase 5 Dependencies
- ✅ `construct_prompt()` - Used to build prompts before generation
- ✅ `load_prompt_template()` - Used internally by construct_prompt
- ✅ `QueryExecutor` - Used for prompt template loading

#### Phase 4 Dependencies
- ✅ `retrieve_chunks()` - Returns `List[RetrievalResult]` for prompt construction
- ✅ `RetrievalResult` interface - Used for context chunks

#### Phase 3 Dependencies
- ✅ `generate_query_embedding()` - Used by retrieve_chunks (indirect dependency)

#### Configuration
- ✅ Azure AI Foundry credentials in config:
  - `azure_ai_foundry_endpoint` (from `AZURE_AI_FOUNDRY_ENDPOINT` environment variable)
  - `azure_ai_foundry_api_key` (from `AZURE_AI_FOUNDRY_API_KEY` environment variable)
  - `azure_ai_foundry_generation_model` (from `AZURE_AI_FOUNDRY_GENERATION_MODEL`, default: "gpt-4o")
- ✅ Supabase database credentials in config:
  - `database_url` (from `DATABASE_URL` environment variable)

## Integration Points for Phase 7

### Query Pipeline Integration
The answer generation function is ready to be integrated into the query pipeline:
1. Query is embedded (Phase 3)
2. Chunks are retrieved (Phase 4)
3. Prompt is constructed (Phase 5)
4. **Answer is generated (Phase 6)** ← Ready now
5. **Pipeline orchestration (Phase 7)** ← Next phase
6. Results are logged (Phase 8)

### Usage in Phase 7
Phase 7 will use `generate_answer()` to complete the query pipeline:
```python
from rag_eval.services.rag.generation import generate_answer
from rag_eval.db.queries import QueryExecutor
from rag_eval.db.connection import DatabaseConnection

# In run_rag() function
db_conn = DatabaseConnection(config)
db_conn.connect()
query_executor = QueryExecutor(db_conn)

# Generate answer using Phase 6 function
answer = generate_answer(query, retrieved_chunks, prompt_version, config, query_executor)
# Returns: ModelAnswer object with generated text and metadata
```

## Known Limitations

### Generation Parameters
- **Fixed Parameters**: Temperature (0.1) and max_tokens (1000) are fixed
- **Model Selection**: Model is configurable via config (default: "gpt-4o")
- **Future Enhancement**: Parameters can be made configurable if needed

### Non-Determinism
- **LLM Generation**: Inherently non-deterministic, even with low temperature
- **Acceptable**: For R&D use case, non-determinism is acceptable
- **Testing**: Tests use mocked responses for deterministic testing
- **Connection Test**: May produce different results on each run (expected)

### QueryExecutor Parameter
- **Optional Parameter**: QueryExecutor is optional (auto-created if not provided)
- **Connection Management**: Caller should manage connection lifecycle
- **Warning**: Function warns when auto-creating QueryExecutor
- **Best Practice**: Pass QueryExecutor explicitly for better control

### Response Validation
- **Structure Validation**: Validates response structure (choices, message, content)
- **Content Validation**: Validates answer is not empty
- **Error Handling**: Raises ValueError for invalid responses
- **Future Enhancement**: Could add more sophisticated validation

### Retry Logic
- **Fixed Retries**: 3 retries with exponential backoff
- **Base Delay**: 1 second (doubles with each retry: 1s, 2s, 4s)
- **Future Enhancement**: Could be made configurable if needed

## Testing Status

### Unit Tests
- ✅ 16 new unit tests for `generate_answer()` function
- ✅ 24 existing tests for prompt template system (Phase 5)
- ✅ 1 connection test (warns when credentials missing)
- ✅ 100% error path coverage
- ✅ All edge cases tested
- ✅ Integration tests written

### Test Execution
```bash
cd backend
source venv/bin/activate
pytest tests/test_rag_generation.py -v
```

**Results**: 40 passed, 0 failed, 0 skipped

### Connection Tests
- Connection test implemented and ready
- Warns but doesn't fail when credentials missing
- Can be run when Azure AI Foundry credentials are configured
- Requires Supabase database for prompt template loading

## Configuration Requirements

### Required Environment Variables
For Phase 7 and beyond, ensure these are configured:
- `AZURE_AI_FOUNDRY_ENDPOINT` - Azure AI Foundry endpoint URL
  - Format: `https://your-resource.openai.azure.com`
- `AZURE_AI_FOUNDRY_API_KEY` - Azure AI Foundry API key
- `AZURE_AI_FOUNDRY_GENERATION_MODEL` - Generation model name (optional, default: "gpt-4o")
- `DATABASE_URL` - Supabase PostgreSQL connection string
  - Format: `postgresql://user:password@host:port/database`

### Azure AI Foundry Setup
- Azure AI Foundry resource must be created
- Generation model deployment must exist (e.g., "gpt-4o")
- API key must have permissions for chat completions
- Endpoint must be accessible from application

### Database Setup
- `prompt_versions` table must exist (from migrations)
- Sample prompt templates should be seeded for testing
- Table schema:
  - `version_id` (VARCHAR(255), PRIMARY KEY)
  - `version_name` (VARCHAR(100), UNIQUE, NOT NULL)
  - `prompt_text` (TEXT, NOT NULL)
  - `created_at` (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP)

## Next Phase Prerequisites

### For Phase 7 (Pipeline Orchestration)
- ✅ Answer generation complete (Phase 6)
- ✅ Prompt construction complete (Phase 5)
- ✅ Retrieval complete (Phase 4)
- ✅ Embedding generation complete (Phase 3)
- ✅ Query ID generation utility available (utils/ids.py)
- ⏳ Pipeline orchestration function to be implemented

### Integration Points
Phase 7 will orchestrate:
- `generate_query_embedding()` from Phase 3
- `retrieve_chunks()` from Phase 4
- `construct_prompt()` from Phase 5 (used by generate_answer)
- `generate_answer()` from Phase 6
- Query ID generation (utils/ids.py)
- Logging functions (Phase 8 - can be stubbed initially)

## Code Examples

### Generating Answer
```python
from rag_eval.services.rag.generation import generate_answer
from rag_eval.core.interfaces import Query, RetrievalResult
from rag_eval.db.queries import QueryExecutor
from rag_eval.db.connection import DatabaseConnection
from rag_eval.core.config import Config

config = Config.from_env()
db_conn = DatabaseConnection(config)
db_conn.connect()
query_executor = QueryExecutor(db_conn)

query = Query(text="What is the coverage limit?")
chunks = [
    RetrievalResult(
        chunk_id="chunk_1",
        similarity_score=0.9,
        chunk_text="Coverage limit is $500,000"
    )
]

answer = generate_answer(query, chunks, "v1", config, query_executor)
# Returns: ModelAnswer(
#     text="The coverage limit is $500,000.",
#     query_id="query_...",
#     prompt_version="v1",
#     retrieved_chunk_ids=["chunk_1"],
#     timestamp=datetime(...)
# )
```

### Using in Phase 7
```python
# In run_rag() function
from rag_eval.services.rag.generation import generate_answer
from rag_eval.services.rag.search import retrieve_chunks
from rag_eval.services.rag.embeddings import generate_query_embedding
from rag_eval.utils.ids import generate_id

def run_rag(query: Query, prompt_version: str = "v1", config=None):
    # Step 1: Generate query embedding
    query_embedding = generate_query_embedding(query, config)
    
    # Step 2: Retrieve chunks
    retrieved_chunks = retrieve_chunks(query, top_k=5, config=config)
    
    # Step 3: Generate answer (Phase 6)
    answer = generate_answer(query, retrieved_chunks, prompt_version, config, query_executor)
    
    return answer
```

## Documentation

### Code Documentation
- ✅ All functions have comprehensive docstrings
- ✅ Generation parameters documented
- ✅ Model selection strategy documented
- ✅ Non-determinism documented
- ✅ Error handling documented
- ✅ Retry logic documented

### External Documentation
- ✅ Phase 6 decisions documented
- ✅ Phase 6 testing documented
- ✅ This handoff document

## Validation Checklist

Before proceeding to Phase 7, verify:
- [x] All Phase 6 tests pass
- [x] All error paths tested
- [x] Connection test implemented
- [x] Documentation complete
- [x] TODO001.md updated
- [x] No known blocking issues

## Blockers

**No blockers identified** - Phase 6 is complete and ready for Phase 7.

## Questions for Phase 7

1. **Pipeline Orchestration**: Confirm pipeline flow and execution order
2. **Error Handling**: Confirm error handling strategy for pipeline failures
3. **Logging Integration**: Confirm logging strategy (Phase 8 can be stubbed initially)
4. **Latency Measurement**: Confirm latency measurement approach

## Support and Troubleshooting

### Common Issues

#### Azure AI Foundry Connection Fails
- **Check**: `AZURE_AI_FOUNDRY_ENDPOINT` environment variable is set correctly
- **Check**: `AZURE_AI_FOUNDRY_API_KEY` environment variable is set correctly
- **Check**: Azure AI Foundry resource is accessible
- **Check**: API key has permissions for chat completions

#### Generation Model Not Found
- **Check**: `AZURE_AI_FOUNDRY_GENERATION_MODEL` matches deployed model name
- **Check**: Model deployment exists in Azure AI Foundry
- **Check**: Model name is spelled correctly (case-sensitive)

#### Invalid LLM Response
- **Check**: API response structure matches expected format
- **Check**: Response contains "choices" array with "message" and "content"
- **Check**: Response content is not empty
- **Check**: API version is correct (2024-02-15-preview)

#### Prompt Template Not Found
- **Check**: Prompt version exists in `prompt_versions` table
- **Check**: Database connection is working
- **Check**: QueryExecutor is properly initialized

#### Retry Logic Issues
- **Check**: Network connectivity to Azure AI Foundry
- **Check**: API rate limits (may need to adjust retry delays)
- **Check**: API key is valid and not expired

### Debugging Tips
- Enable debug logging to see API calls and responses
- Check Azure AI Foundry logs for API errors
- Verify prompt construction with `construct_prompt()` directly
- Test with simple query and chunks first
- Use connection test to verify Azure AI Foundry connectivity

---

**Related Documents**:
- [PRD001.md](./PRD001.md) - Product requirements
- [RFC001.md](./RFC001.md) - Technical design
- [TODO001.md](./TODO001.md) - Implementation tasks
- [phase_6_decisions.md](./phase_6_decisions.md) - Implementation decisions
- [phase_6_testing.md](./phase_6_testing.md) - Testing summary

**Next Phase**: [Phase 7 — Pipeline Orchestration](./TODO001.md#phase-7--pipeline-orchestration)

