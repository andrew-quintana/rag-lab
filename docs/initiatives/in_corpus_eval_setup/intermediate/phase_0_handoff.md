# Phase 0 Handoff — In-Corpus Evaluation Dataset Generation System

**Phase**: Phase 0 - Context Harvest → Phase 1 - Query Generator  
**Date**: 2025-01-XX  
**Status**: Ready for Phase 1

## Overview

This document summarizes Phase 0 completion and provides the entry point for Phase 1 implementation.

## Phase 0 Completion Summary

### ✅ Completed Tasks

1. **Documentation Review**
   - ✅ Reviewed PRD001.md for complete requirements
   - ✅ Reviewed RFC001.md for technical architecture
   - ✅ Reviewed TODO001.md for implementation plan
   - ✅ Reviewed context.md for project context
   - ✅ Reviewed eval_system context.md for evaluation system context
   - ✅ Reviewed rag_system context.md for RAG system context

2. **Codebase Review**
   - ✅ Reviewed RAG system components (`backend/rag_eval/services/rag/`)
   - ✅ Reviewed Azure AI Search integration (`backend/rag_eval/services/rag/search.py`)
   - ✅ Reviewed RAG generation and prompt loading (`backend/rag_eval/services/rag/generation.py`)
   - ✅ Reviewed evaluation system components (`backend/rag_eval/services/evaluator/`)
   - ✅ Reviewed Supabase integration (`backend/rag_eval/db/queries.py`)
   - ✅ Reviewed prompt system in Supabase migrations
   - ✅ Reviewed BEIR metrics computation (`backend/rag_eval/services/evaluator/beir_metrics.py`)

3. **Environment Setup**
   - ✅ Virtual environment validated (`backend/venv/`)
   - ✅ pytest installed and verified
   - ✅ All backend dependencies installed
   - ✅ Test discovery validated

4. **FRACAS Setup**
   - ✅ Created `docs/initiatives/in_corpus_eval_setup/fracas.md`

5. **Intermediate Documents**
   - ✅ Created `phase_0_decisions.md`
   - ✅ Created `phase_0_testing.md`
   - ✅ Created `phase_0_handoff.md` (this document)

## Key Findings from Phase 0

### Architecture Understanding

1. **RAG System**
   - Uses Azure AI Search for vector retrieval
   - Uses Azure AI Foundry for embeddings and generation
   - Prompts stored in Supabase `prompts` table
   - Supports prompt versioning and live flags

2. **Evaluation System**
   - BEIR metrics computation available
   - LLM-as-Judge evaluation system exists
   - Meta-evaluator components available

3. **Database Schema**
   - `prompts` table supports:
     - `prompt_type` (e.g., "rag", "RAG-enabled", "evaluation")
     - `name` (for evaluation prompts, nullable for others)
     - `version` (semantic versioning)
     - `live` (boolean flag for active version)
   - System prompts can use naming convention `{type}_system`

4. **Configuration**
   - Configuration loaded from environment variables
   - Supports `.env.local` file
   - All Azure services configured via environment variables

### Environment Details

- **Python Version**: 3.13.7
- **Virtual Environment**: `backend/venv/`
- **Activation Command**: `cd backend && source venv/bin/activate`
- **Test Framework**: pytest 9.0.1 with pytest-cov 7.0.0

## Phase 1 Entry Point

### Implementation Target

**Component**: Query Generator (AI Node)  
**File**: `evaluations/_shared/scripts/query_generator.py`  
**Output**: `evaluations/{eval_name}/inputs/eval_inputs.json`

### Key Requirements

1. **Generate diverse queries** from Azure AI Search indexed documents
2. **Ensure all queries are "In-Corpus"** (answerable from indexed documents)
3. **Include metadata** linking queries to source chunks for ground truth
4. **Use LLM** for query generation based on chunk content

### Integration Points

1. **Azure AI Search**
   - Use `retrieve_chunks()` from `backend/rag_eval/services/rag/search.py`
   - Sample chunks from index
   - Use embeddings for diversity

2. **LLM Provider**
   - Use Azure AI Foundry via existing LLM provider utilities
   - Configuration available in `Config` class

3. **Output Format**
   ```json
   [
     {
       "input": "string - query for RAG evaluation",
       "metadata": {
         "source_chunk_ids": ["chunk_1", "chunk_2"],
         "document_id": "doc_123",
       }
     }
   ]
   ```

### Testing Requirements

- **Unit tests** for query generation function
- **Integration tests** with mocked Azure AI Search
- **Test coverage** >= 80% for query_generator.py module
- **Test with sample chunks** from index
- **Validate output JSON structure**

### Directory Structure

Create the following structure:
```
evaluations/
  _shared/
    scripts/
      query_generator.py
  {eval_name}/
    inputs/
      eval_inputs.json
    dataset/
      eval_dataset.json
```

### Key Functions to Implement

```python
def generate_queries_from_index(
    config: Config,
    num_queries: int = 10,
    sample_chunks: Optional[List[Chunk]] = None
) -> List[Dict[str, Any]]:
    """
    Generate diverse queries from Azure AI Search index.
    
    Args:
        config: Application configuration
        num_queries: Number of queries to generate
        sample_chunks: Optional list of chunks to use as seed (if None, samples from index)
        
    Returns:
        List of query dictionaries with input and metadata
    """
```

## Validation Checklist for Phase 1

Before starting Phase 1, ensure:

- [x] Virtual environment is activated
- [x] All dependencies are installed
- [x] pytest can discover tests
- [ ] Azure AI Search credentials are configured (validate during Phase 1)
- [ ] Supabase credentials are configured (validate during Phase 1)
- [ ] Azure AI Foundry credentials are configured (validate during Phase 1)

## Known Dependencies

1. **Azure AI Search Index**: Must exist and be accessible
2. **LLM Provider**: Azure AI Foundry must be configured
3. **Configuration**: Environment variables must be set

## Blockers

**None** - Phase 0 is complete and Phase 1 can proceed.

## Next Steps

1. **Start Phase 1 Implementation**
   - Create `evaluations/{eval_name}/` directory structure
   - Implement `query_generator.py`
   - Add unit and integration tests
   - Generate sample `eval_inputs.json`

2. **Follow Phase 1 Prompt**
   - See `docs/initiatives/in_corpus_eval_setup/prompts/prompt_phase_1_001.md`

3. **Update TODO001.md**
   - Check off Phase 1 tasks as completed

4. **Create Phase 1 Deliverables**
   - `phase_1_decisions.md` (if decisions made)
   - `phase_1_testing.md` (document test results)
   - `phase_1_handoff.md` (summarize Phase 2 entry point)

## Resources

- **PRD**: `docs/initiatives/in_corpus_eval_setup/scoping/PRD001.md`
- **RFC**: `docs/initiatives/in_corpus_eval_setup/scoping/RFC001.md`
- **TODO**: `docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md`
- **Phase 1 Prompt**: `docs/initiatives/in_corpus_eval_setup/prompts/prompt_phase_1_001.md`
- **FRACAS**: `docs/initiatives/in_corpus_eval_setup/fracas.md`

---

**Phase 0 Status**: ✅ Complete  
**Phase 1 Status**: ⏳ Ready to Start  
**Handoff Date**: 2025-01-XX

