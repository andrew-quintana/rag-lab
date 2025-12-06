# Phase 0 Prompt — Context Harvest

## Context

This prompt guides the implementation of **Phase 0: Context Harvest** for the In-Corpus Evaluation Dataset Generation System. This phase establishes the foundation by reviewing all relevant documentation and existing codebase components before beginning implementation.

**Related Documents:**
- @docs/initiatives/in_corpus_eval_setup/scoping/PRD001.md - Product requirements and functional specifications
- @docs/initiatives/in_corpus_eval_setup/scoping/RFC001.md - Technical architecture and design decisions
- @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md - Implementation breakdown (check off tasks as completed)
- @docs/initiatives/in_corpus_eval_setup/scoping/context.md - Project context and scope
- @docs/initiatives/eval_system/context.md - Evaluation system context
- @docs/initiatives/rag_system/context.md - RAG system context

## Objectives

1. **Review Documentation**: Understand PRD, RFC, and TODO requirements
2. **Review Codebase**: Familiarize with existing RAG system, evaluation system, and Supabase integration
3. **Validate Environment**: Ensure testing environment is properly configured
4. **Create FRACAS Document**: Set up failure tracking system

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md Phase 0 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: Verify all setup tasks are complete before proceeding to Phase 1
- **REQUIRED**: Document any blockers or issues in fracas.md (root directory)

### Documentation
- **REQUIRED**: Create `intermediate/phase_0_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `intermediate/phase_0_testing.md` documenting any testing setup validation
- **REQUIRED**: Create `intermediate/phase_0_handoff.md` summarizing what's needed for Phase 1

## Key References

### Adjacent Components
- @backend/rag_eval/services/rag/ - Existing RAG system components
- @backend/rag_eval/services/evaluator/ - Existing evaluation system components
- @backend/rag_eval/services/rag/search.py - Azure AI Search integration
- @backend/rag_eval/services/rag/generation.py - RAG generation with prompt loading
- @backend/rag_eval/db/queries.py - Supabase query executor
- @infra/supabase/migrations/ - Database schema migrations

### Configuration
- Azure AI Search configuration: Review `rag_eval/core/config.py` or environment variables
- Supabase database schema: Review `infra/supabase/migrations/` for prompts table structure
- LLM provider configuration: Review `rag_eval/services/shared/llm_providers.py`

## Phase 0 Tasks

### Documentation Review
1. Review @docs/initiatives/in_corpus_eval_setup/scoping/PRD001.md for complete requirements
2. Review @docs/initiatives/in_corpus_eval_setup/scoping/RFC001.md for technical architecture
3. Review @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md for implementation plan
4. Review @docs/initiatives/in_corpus_eval_setup/scoping/context.md for project context
5. Review @docs/initiatives/eval_system/context.md for evaluation system context
6. Review @docs/initiatives/rag_system/context.md for RAG system context

### Codebase Review
1. Review existing RAG system components in @backend/rag_eval/services/rag/
2. Review Azure AI Search integration in @backend/rag_eval/services/rag/search.py
3. Review RAG generation and prompt loading in @backend/rag_eval/services/rag/generation.py
4. Review evaluation system components in @backend/rag_eval/services/evaluator/
5. Review Supabase integration in @backend/rag_eval/db/queries.py
6. Review prompt system in @infra/supabase/migrations/ for prompts table structure
7. Review BEIR metrics computation in @backend/rag_eval/services/evaluator/beir_metrics.py

### Environment Setup
1. **REQUIRED**: Set up and activate backend virtual environment (`backend/venv/`)
2. **REQUIRED**: Verify pytest is installed (`pip install pytest pytest-cov`)
3. **REQUIRED**: Verify all backend dependencies are installed (`pip install -r backend/requirements.txt`)
4. **REQUIRED**: Verify pytest can discover tests (`pytest backend/tests/ --collect-only`)
5. **REQUIRED**: Document venv activation command for all subsequent phases

### FRACAS Setup
1. Create `docs/initiatives/in_corpus_eval_setup/fracas.md` for failure tracking (in root directory, not intermediate/)
2. Set up FRACAS structure following template in @docs/templates/fracas_template.md (if exists)

### Validation
1. Validate Azure AI Search index exists and is accessible
2. Validate Azure AI Search credentials are configured
3. Validate Supabase database schema and connection
4. Validate prompts table structure supports system prompts
5. Validate RAG pipeline components are accessible
6. Validate evaluation system components are accessible

## Success Criteria

- [ ] All documentation reviewed and understood
- [ ] All codebase components reviewed
- [ ] Testing environment validated and documented
- [ ] FRACAS document created
- [ ] All validation checks passed
- [ ] Phase 0 tasks completed and checked off in TODO001.md
- [ ] Phase 0 deliverables created in intermediate/ directory (decisions, testing, handoff documents)

## Next Phase

After completing Phase 0, proceed to **Phase 1: Query Generator (AI Node)** as specified in TODO001.md.

---
**Document Status**: Draft  
**Last Updated**: 2025-01-XX  
**Author**: Documentation Agent  
**Reviewers**: TBD
