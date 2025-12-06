# Phase 4 Prompt — System Prompt Management

## Context

This prompt guides the implementation of **Phase 4: System Prompt Management** for the In-Corpus Evaluation Dataset Generation System. This phase creates and manages system prompts in Supabase with the naming convention `{type}_system`.

**Related Documents:**
- @docs/initiatives/in_corpus_eval_setup/scoping/PRD001.md - Product requirements and functional specifications
- @docs/initiatives/in_corpus_eval_setup/scoping/RFC001.md - Technical architecture and design decisions
- @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md - Implementation breakdown (check off tasks as completed)
- @docs/initiatives/in_corpus_eval_setup/scoping/context.md - Project context and scope
- @docs/initiatives/in_corpus_eval_setup/intermediate/phase_3_handoff.md - Phase 3 handoff document

## Objectives

1. **Create Migration**: Create database migration for system prompts
2. **Insert System Prompts**: Insert basic system prompts for RAG-enabled prompts
3. **Naming Convention**: Ensure system prompts follow `{type}_system` convention
4. **Helper Script**: Create optional helper script for managing system prompts
5. **Testing**: Test system prompt retrieval from Supabase

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/in_corpus_eval_setup/scoping/TODO001.md Phase 4 section as you complete them
- Update phase status when all implementation tasks are complete

### Validation
- **REQUIRED**: Test migration applies successfully
- **REQUIRED**: Test system prompts can be queried by prompt_type and name
- **REQUIRED**: Document any blockers or issues in fracas.md (root directory)

### Documentation
- **REQUIRED**: Create `intermediate/phase_4_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `intermediate/phase_4_testing.md` documenting test results
- **REQUIRED**: Create `intermediate/phase_4_handoff.md` summarizing what's needed for Phase 5

## Key References

### Implementation Components
- @infra/supabase/migrations/ - Database migration files
- @backend/rag_eval/db/queries.py - Supabase query executor
- @backend/rag_eval/services/rag/generation.py - Prompt loading functions
- Existing prompt migrations for reference

### Database Schema
- `public.prompts` table structure:
  - id (UUID)
  - version (VARCHAR)
  - prompt_type (VARCHAR)
  - name (VARCHAR, nullable)
  - prompt_text (TEXT)
  - live (BOOLEAN)
  - created_at (TIMESTAMP)

## Phase 4 Tasks

### Implementation
1. Create migration `0016_add_system_prompts.sql`:
   - Location: `infra/supabase/migrations/0016_add_system_prompts.sql`
   - Insert basic system prompts for RAG-enabled prompts
   - Follow existing migration patterns
   - Include proper comments
2. Insert system prompts:
   - For `prompt_type="rag"`, create system prompt with `name="rag_system"`
   - For `prompt_type="RAG-enabled"`, create system prompt with `name="rag_enabled_system"`
   - Ensure prompts are marked as `live=TRUE` if they should be active
   - Use semantic versioning for version field (e.g., "0.1")
3. Ensure naming convention:
   - All system prompts follow `{type}_system` pattern
   - Allows filtering with `name LIKE '%_system'` or `name CONTAINS '_system'`
4. Create optional helper script `create_system_prompts.py`:
   - Location: `evaluations/{eval_name}/create_system_prompts.py`
   - Script to create/update system prompts programmatically
   - Useful for development and testing
5. Document system prompt querying pattern:
   - Document how to query system prompts by prompt_type and name
   - Document naming convention usage
   - Update relevant documentation

### Testing
1. **REQUIRED**: Test migration applies successfully:
   - Apply migration to test database
   - Verify no errors
   - Verify system prompts are inserted
2. **REQUIRED**: Test system prompts can be queried:
   - Query by prompt_type and name
   - Test with QueryExecutor
   - Verify correct prompts are returned
3. **REQUIRED**: Test naming convention filtering:
   - Test `name LIKE '%_system'` query
   - Test filtering works correctly
4. **REQUIRED**: Verify at least one system prompt exists for RAG-enabled prompts:
   - Query for RAG-enabled system prompts
   - Verify at least one exists

## Success Criteria

- [ ] Migration created and applies successfully
- [ ] System prompts created in Supabase with correct naming convention
- [ ] System prompts can be queried by prompt_type and name
- [ ] Naming convention filtering works correctly
- [ ] At least one system prompt exists for RAG-enabled prompts
- [ ] Helper script created (if needed)
- [ ] System prompt querying documented
- [ ] Phase 4 tasks completed and checked off in TODO001.md
- [ ] Phase 4 deliverables created in intermediate/ directory

## Next Phase

After completing Phase 4, proceed to **Phase 5: Terminology Update** as specified in TODO001.md.

---
**Document Status**: Draft  
**Last Updated**: 2025-01-XX  
**Author**: Documentation Agent  
**Reviewers**: TBD

