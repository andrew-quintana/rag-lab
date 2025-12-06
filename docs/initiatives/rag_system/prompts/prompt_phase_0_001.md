# Phase 0 Prompt — Context Harvest

## Purpose
This phase establishes the foundational context and prerequisites before implementation begins. Complete all context gathering and validation tasks before proceeding to Phase 1.

## Context References

**Primary Documentation:**
- @docs/initiatives/rag_system/scoping/PRD001.md - Product requirements and acceptance criteria
- @docs/initiatives/rag_system/scoping/RFC001.md - Technical architecture and design decisions
- @docs/initiatives/rag_system/scoping/TODO001.md - Complete implementation breakdown (check off tasks as completed)
- @docs/initiatives/rag_system/scoping/context.md - System context and scope boundaries

**Codebase References:**
- @backend/rag_eval/core/config.py - Configuration system for Azure services
- @backend/rag_eval/core/interfaces.py - Core data structures (Query, ModelAnswer, Chunk, etc.)
- @backend/rag_eval/db/connection.py - Database connection setup
- @backend/rag_eval/db/models.py - Database models and schema
- @backend/rag_eval/db/queries.py - QueryExecutor interface
- @backend/rag_eval/services/rag/ingestion.py - Existing ingestion component
- @backend/rag_eval/services/rag/chunking.py - Existing chunking component

**Infrastructure:**
- @infra/supabase/migrations/0001_init.sql - Database schema
- @backend/requirements.txt - Python dependencies

## Phase Objectives

1. **Review Documentation**: Understand complete requirements from PRD, RFC, and TODO
2. **Validate Configuration**: Verify Azure service credentials and Supabase connection
3. **Review Existing Code**: Understand current implementation state
4. **Create FRACAS Document**: Set up failure tracking system
5. **Block Implementation**: Ensure Phase 0 is complete before proceeding

## Execution Requirements

### Checklist Management
- **REQUIRED**: Check off all [ ] checkboxes in @docs/initiatives/rag_system/scoping/TODO001.md Phase 0 section as tasks are completed
- Update TODO001.md directly when each task is finished

### Validation Requirements
- Validate Azure service configuration (credentials, endpoints, container names)
  - **Note**: Credentials are configured in `.env.local` file in project root
  - `Config.from_env()` automatically loads from `.env.local` or can accept custom path
- Validate Supabase database schema matches expectations
- Verify existing codebase structure matches RFC001.md architecture
- Confirm all dependencies are documented

### Documentation Requirements
- **REQUIRED**: Create `fracas.md` in @docs/initiatives/rag_system/ for failure tracking
- Document any configuration issues or blockers discovered
- Document any discrepancies between existing code and RFC001.md
- Document environment variable loading from `.env.local` file

## Key Tasks from TODO001.md

1. Review adjacent components in context.md
2. Review PRD001.md and RFC001.md for complete requirements understanding
3. Review existing codebase structure and interfaces
4. Validate Azure service configuration and credentials
5. Validate Supabase database schema and connection
6. **Create fracas.md** for failure tracking using FRACAS methodology
7. Block: Implementation cannot proceed until Phase 0 complete

## Success Criteria

- [ ] All Phase 0 checkboxes in TODO001.md are checked
- [ ] fracas.md document created and ready for use
- [ ] Azure service configuration validated (or documented as missing)
- [ ] Supabase connection validated (or documented as missing)
- [ ] Existing codebase reviewed and understood
- [ ] No blockers identified that prevent Phase 1 implementation

## Next Phase

Once Phase 0 is complete, proceed to **Phase 1 — Document Upload (Azure Blob Storage)**.

---

**Document Status**: Active  
**Last Updated**: 2025-01-27  
**Related Documents**: [PRD001.md](./PRD001.md), [RFC001.md](./RFC001.md), [TODO001.md](./TODO001.md)

