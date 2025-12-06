# Phase 0 Prompt — Context Harvest

## Context

This prompt guides the implementation of **Phase 0: Context Harvest** for the RAG Ingestion Worker–Queue Architecture Conversion. This phase establishes the foundation by reviewing all relevant documentation and existing codebase components before beginning implementation.

**Related Documents:**
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/PRD001.md - Product requirements and functional specifications
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/RFC001.md - Technical architecture and design decisions
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md - Implementation breakdown (check off tasks as completed)
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/context.md - Project context and goals

## Objectives

1. **Review Documentation**: Understand PRD, RFC, and TODO requirements
2. **Review Codebase**: Familiarize with existing RAG system components
3. **Validate Environment**: Ensure testing environment is properly configured
4. **Create FRACAS Document**: Set up failure tracking system

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md Phase 0 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: Verify all setup tasks are complete before proceeding to Phase 1
- **REQUIRED**: Document any blockers or issues in fracas.md

### Documentation
- **REQUIRED**: Create `phase_0_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_0_testing.md` documenting any testing setup validation
- **REQUIRED**: Create `phase_0_handoff.md` summarizing what's needed for Phase 1

## Key References

### Adjacent Components
- @backend/rag_eval/services/rag/ingestion.py - Document text extraction
- @backend/rag_eval/services/rag/chunking.py - Text chunking implementation
- @backend/rag_eval/services/rag/embeddings.py - Embedding generation
- @backend/rag_eval/services/rag/search.py - Vector indexing and retrieval
- @backend/rag_eval/services/rag/storage.py - Azure Blob Storage functions
- @backend/rag_eval/services/rag/supabase_storage.py - Supabase Storage functions
- @backend/tests/fixtures/ - Test fixtures structure

### Configuration
- Azure Storage Account configuration: Review Azure resource setup
- Azure Functions configuration: Review Function App setup
- Supabase database schema: Review `infra/supabase/migrations/` for schema structure
- Azure Document Intelligence, AI Foundry, AI Search: Review configuration and credentials

## Phase 0 Tasks

### Documentation Review
1. Review @docs/initiatives/rag_system/worker_queue_conversion/scoping/PRD001.md for complete requirements
2. Review @docs/initiatives/rag_system/worker_queue_conversion/scoping/RFC001.md for technical architecture
3. Review @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md for implementation plan
4. Review @docs/initiatives/rag_system/worker_queue_conversion/scoping/context.md for project context

### Codebase Review
1. Review existing RAG system components in @backend/rag_eval/services/rag/
   - `ingestion.py` - `extract_text_from_document()` function
   - `chunking.py` - `chunk_text()` function
   - `embeddings.py` - `generate_embeddings()` function
   - `search.py` - `index_chunks()` function
   - `storage.py` - Azure Blob Storage functions
   - `supabase_storage.py` - Supabase Storage functions
2. Review existing database schema in @infra/supabase/migrations/
3. Review test fixtures structure in @backend/tests/fixtures/

### Environment Setup
1. **REQUIRED**: Set up and activate backend virtual environment (`backend/venv/`)
2. **REQUIRED**: Verify pytest is installed (`pip install pytest pytest-cov`)
3. **REQUIRED**: Verify all backend dependencies are installed (`pip install -r backend/requirements.txt`)
4. **REQUIRED**: Verify pytest can discover tests (`pytest backend/tests/ --collect-only`)
5. **REQUIRED**: Document venv activation command for all subsequent phases

### FRACAS Setup
1. Create `docs/initiatives/rag_system/worker_queue_conversion/fracas.md` for failure tracking
2. Set up FRACAS structure following template in @docs/templates/fracas_template.md (if available)

### Validation
1. Validate Azure Storage Account configuration and credentials
2. Validate Azure Functions setup (if already exists)
3. Validate Supabase database schema and connection
4. Validate Azure Document Intelligence, AI Foundry, AI Search credentials

## Success Criteria

- [ ] All documentation reviewed and understood
- [ ] All codebase components reviewed
- [ ] Testing environment fully configured and validated
- [ ] FRACAS document created
- [ ] All Phase 0 tasks in TODO001.md checked off
- [ ] Phase 0 handoff document created

## Blockers

- **BLOCKER**: Phase 1 cannot proceed until Phase 0 is complete
- **BLOCKER**: All validation requirements must pass before proceeding

## Next Phase

After completing Phase 0, proceed to **Phase 1: Persistence Infrastructure** using @docs/initiatives/rag_system/worker_queue_conversion/prompts/prompt_phase_1_001.md

