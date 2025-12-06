# Phase 5 Prompt — Prompt Template System

## Purpose
Implement prompt template loading and construction system. Templates are stored in Supabase and loaded by version name for flexible prompt versioning.

## Context References

**Primary Documentation:**
- @docs/initiatives/rag_system/scoping/PRD001.md - Product requirements (FR2: Query Pipeline, FR3: Logging)
- @docs/initiatives/rag_system/scoping/RFC001.md - Technical design (Phase 3: Prompt Template System, Decision 3: Prompt Template Versioning via Database)
- @docs/initiatives/rag_system/scoping/TODO001.md - Phase 5 implementation tasks (check off as completed)
- @docs/initiatives/rag_system/scoping/context.md - System context

**Codebase References:**
- @backend/rag_eval/db/queries.py - QueryExecutor interface
- @backend/rag_eval/core/interfaces.py - Query and RetrievalResult interfaces
- @backend/rag_eval/db/connection.py - Supabase connection
- @infra/supabase/migrations/0001_init.sql - Database schema (prompt_versions table)

**Implementation Target:**
- **EXISTING FILE**: `backend/rag_eval/services/rag/generation.py` - Prompt loading functions

## Phase Objectives

1. **Implement Prompt Loading**: Load prompt templates from Supabase by version name
2. **In-Memory Caching**: Cache loaded prompts to avoid repeated DB queries
3. **Prompt Construction**: Format templates with query and context placeholders
4. **Validation**: Validate prompt versions exist and templates have required placeholders
5. **Testing**: Write comprehensive unit tests with mocked Supabase

## Execution Requirements

### Checklist Management
- **REQUIRED**: Check off all [ ] checkboxes in @docs/initiatives/rag_system/scoping/TODO001.md Phase 5 section as tasks are completed
- Update TODO001.md directly when each task is finished

### Validation Requirements
- **REQUIRED**: Validate all code with unit tests before marking complete
- Test prompt template loading and caching
- Test prompt construction with placeholders
- Test validation of required placeholders ({query}, {context})
- Test all error handling paths (100% coverage for error paths)
- Connection tests should warn but not fail if credentials missing

### Documentation Requirements
- **REQUIRED**: Create `phase_5_decisions.md` - Document any decisions not in PRD/RFC
- **REQUIRED**: Create `phase_5_testing.md` - Document testing performed
- **REQUIRED**: Create `phase_5_handoff.md` - Document what's needed for Phase 6
- Add docstrings to all functions
- Document prompt template format and placeholders
- Document prompt versioning strategy
- Document caching strategy

## Key Implementation Tasks

### Core Implementation
- Implement `load_prompt_template(version: str, query_executor: QueryExecutor) -> str`:
  - Query `prompt_versions` table by `version_name`
  - Validate prompt version exists in database
  - Return `prompt_text` from database
  - Implement in-memory caching (avoid repeated DB queries)
  - Handle missing prompt versions with clear error messages
  - Error handling
- Implement `construct_prompt(query: Query, retrieved_chunks: List[RetrievalResult], prompt_version: str, query_executor: QueryExecutor) -> str`:
  - Load prompt template using `load_prompt_template()`
  - Format template with placeholders:
    - `{query}` → query text
    - `{context}` → retrieved chunks text (concatenated)
  - Validate template has required placeholders
  - Handle template formatting errors
  - Return final LLM-ready prompt

### Testing
- Unit tests for `load_prompt_template()` (mocked Supabase)
- Unit tests for `construct_prompt()` (mocked Supabase)
- Test caching behavior (avoid repeated DB queries)
- Test missing prompt version handling
- Test template placeholder replacement
- Test validation of required placeholders
- Test handling of empty retrieved chunks
- Test error handling for database failures
- Test error handling for template formatting failures
- Connection test for Supabase (warns if credentials missing)

### Documentation
- Function docstrings
- Prompt template format and placeholders documentation
- Prompt versioning strategy documentation
- Caching strategy documentation
- Phase 5 testing summary

## Success Criteria

- [ ] `load_prompt_template()` function complete with caching
- [ ] `construct_prompt()` function complete
- [ ] All Phase 5 checkboxes in TODO001.md are checked
- [ ] Unit tests written and passing (mocked Supabase)
- [ ] Connection test implemented (warns if credentials missing)
- [ ] All error paths tested (100% coverage)
- [ ] Caching behavior validated
- [ ] phase_5_decisions.md created
- [ ] phase_5_testing.md created
- [ ] phase_5_handoff.md created
- [ ] All failures documented in fracas.md (if any)

## Dependencies

- Supabase connection and `prompt_versions` table exists
- `QueryExecutor` from `rag_eval/db/queries.py` available
- Sample prompt templates in database for testing

## Next Phase

Once Phase 5 is complete, proceed to **Phase 6 — LLM Answer Generation**.

---

**Document Status**: Active  
**Last Updated**: 2025-01-27  
**Related Documents**: [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md), [TODO001.md](./TODO001.md)

