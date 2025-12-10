# Prompt: Phase 0 - Scoping & Documentation Updates

## Your Mission

Execute Phase 0 of Initiative 002: Update foundational cursor rules files to reflect the worker-queue architecture transformation implemented in Initiative 001, and establish file organization requirements for the consolidation effort.

## Context & References

**Key Documentation to Reference**:
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/PRD002.md` - Product requirements
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Technical design
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md` - Task checklist (mark tasks complete)
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/context.md` - Project context
- `@docs/initiatives/rag_system/worker_queue_conversion/001_initial_conversion/` - Initiative 001 documentation for reference

**Prerequisites**:
- Initiative 001 (Worker-Queue Architecture Implementation) must be complete
- Worker-queue architecture is implemented and functional
- Local development environment is set up

## Prerequisites Check

**Before starting, verify**:
1. Initiative 001 is complete and worker-queue architecture is functional
2. All cursor rules files exist in `.cursor/rules/`
3. You have read access to Initiative 001 documentation for context

**If any prerequisite is missing, document it and stop execution.**

---

## Task 0.1: Update Architecture Rules

**Execute**: Update `.cursor/rules/architecture_rules.md` to reflect worker-queue architecture

**Required Updates**:
- [ ] Add worker-queue architecture layer description
- [ ] Document worker infrastructure (`src/services/workers/`)
- [ ] Update layer boundaries to include Azure Functions as worker execution layer
- [ ] Document queue-based communication patterns
- [ ] Update development workflow to include local Azure Functions development with Azurite
- [ ] Add Azure Storage Queues to infrastructure components

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/context.md` - Section 10.3 for specific requirements
- `@docs/initiatives/rag_system/worker_queue_conversion/001_initial_conversion/` - For architecture details

**Success Criteria**:
- ✅ Architecture rules accurately describe worker-queue architecture
- ✅ Worker infrastructure is documented
- ✅ Azure Functions are included as execution layer
- ✅ Queue-based patterns are explained
- ✅ Local development workflow includes Azurite

**Document**: 
- Record any decisions made in `intermediate/phase_0_decisions.md`
- Note any clarifications needed from Initiative 001 documentation

---

## Task 0.2: Update Scoping Document Rules

**Execute**: Update `.cursor/rules/scoping_document.md` to reflect worker-queue architecture

**Required Updates**:
- [ ] Update "High-Level System Overview" to include worker-queue architecture
- [ ] Add worker infrastructure to codebase structure diagram
- [ ] Document Azure Functions deployment structure (will be moved to `backend/azure_functions/` in Phase 1)
- [ ] Update "Out of Scope" section (remove workers from out-of-scope list)
- [ ] Add Azure Storage Queues to infrastructure architecture section
- [ ] Update database scope to include worker status tracking (migrations 0019, 0020)
- [ ] Update "Required Scaffolding" section to include Azure Functions setup

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/context.md` - Section 10.3 for specific requirements
- `@docs/initiatives/rag_system/worker_queue_conversion/001_initial_conversion/scoping/` - For system details

**Success Criteria**:
- ✅ System overview includes worker-queue architecture
- ✅ Codebase structure diagram shows worker infrastructure
- ✅ Azure Functions deployment structure is documented
- ✅ Workers are no longer listed as out of scope
- ✅ Database scope includes worker status tracking

**Document**: 
- Record any decisions made in `intermediate/phase_0_decisions.md`
- Note any structural changes to the scoping document format

---

## Task 0.3: Update State of Development Rules

**Execute**: Update `.cursor/rules/state_of_development.md` to reflect worker-queue architecture

**Required Updates**:
- [ ] Update "CURRENT PHASE" to reflect worker-queue architecture is implemented
- [ ] Remove workers from "OUT OF SCOPE FOR THIS PHASE" section
- [ ] Add worker-queue architecture to system description
- [ ] Update "PRIMARY OBJECTIVES IN THIS PHASE" or create new section for worker-queue
- [ ] Update "LOCAL DEVELOPMENT WORKFLOW" to include Azure Functions local development
- [ ] Add validation gates for worker-queue functionality
- [ ] Update meta-rule to reflect workers are now part of the system
- [ ] Document local development with Azurite and local Supabase

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/context.md` - Section 10.3 for specific requirements
- `@docs/initiatives/rag_system/worker_queue_conversion/001_initial_conversion/intermediate/LOCAL_DEVELOPMENT.md` - For local development details

**Success Criteria**:
- ✅ Current phase reflects worker-queue implementation
- ✅ Workers are no longer out of scope
- ✅ Local development workflow includes Azure Functions
- ✅ Validation gates include worker-queue checks
- ✅ Meta-rule reflects workers as part of system

**Document**: 
- Record any decisions made in `intermediate/phase_0_decisions.md`
- Note any workflow changes or additions

---

## Task 0.4: Create Fracas Document

**Execute**: Create `fracas.md` in the root initiative directory for failure tracking

**Required Content**:
- [ ] Document structure for tracking failures (FM-XXX format)
- [ ] Include known issues from Initiative 001 (FM-001, FM-005)
- [ ] Set up tracking for new issues discovered during consolidation
- [ ] Include severity levels and status tracking

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/001_initial_conversion/fracas.md` - For format reference
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/context.md` - Section 9 for known issues

**Success Criteria**:
- ✅ Fracas document created with proper structure
- ✅ Known issues from Initiative 001 are documented
- ✅ Tracking format is clear and consistent

**Document**: 
- Fracas document is a summary document (root directory)
- Record any decisions about failure tracking format in `intermediate/phase_0_decisions.md`

---

## Task 0.5: Validate File Organization Requirements

**Execute**: Verify file organization requirements are documented in TODO002.md

**Required Verification**:
- [ ] File organization section exists in TODO002.md (Section 0.2)
- [ ] Directory purposes are clearly defined
- [ ] Naming conventions are specified
- [ ] Examples are provided for file placement

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md` - Section 0.2

**Success Criteria**:
- ✅ File organization requirements are documented
- ✅ All directory purposes are clear
- ✅ Naming conventions are specified
- ✅ Examples are helpful and accurate

**Document**: 
- Note any clarifications needed in `intermediate/phase_0_decisions.md`

---

## Task 0.6: Validation & Review

**Execute**: Validate all updates and review for consistency

**Required Actions**:
- [ ] Verify all cursor rules files accurately reflect worker-queue architecture
- [ ] Confirm file organization requirements are documented
- [ ] Review updated rules files for consistency and completeness
- [ ] Check that all references to workers are updated (no outdated "out of scope" references)
- [ ] Verify terminology is consistent across all files

**Success Criteria**:
- ✅ All cursor rules files updated correctly
- ✅ No contradictions between files
- ✅ Terminology is consistent
- ✅ File organization is clear

**Document**: 
- Record validation results in `intermediate/phase_0_testing.md`
- Note any inconsistencies found and how they were resolved

---

## Documentation Requirements

**You MUST**:
1. **Record Decisions**: Create `intermediate/phase_0_decisions.md` with:
   - Any decisions made about architecture descriptions
   - Clarifications needed from Initiative 001
   - Structural changes to cursor rules files
   - File organization clarifications

2. **Document Testing**: Create `intermediate/phase_0_testing.md` with:
   - Validation results for all cursor rules updates
   - Consistency checks performed
   - Any issues found and resolved

3. **Create Handoff**: Create `intermediate/phase_0_handoff.md` with:
   - Summary of all cursor rules updates
   - Status of file organization requirements
   - Prerequisites verified for Phase 1
   - Any blockers or concerns for Phase 1

4. **Update Tracking**: 
   - Mark Phase 0 tasks complete in `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md`
   - Check off all [ ] checkboxes in Section 0.1, 0.2, and 0.3 as tasks are completed

---

## Success Criteria

### Must Pass (Blocking) - All Required:
- [ ] All three cursor rules files updated
- [ ] Fracas document created
- [ ] File organization requirements validated
- [ ] All updates validated for consistency

**If any "Must Pass" criteria fails, document in `intermediate/phase_0_decisions.md` and address before proceeding.**

---

## Next Steps

**If all Phase 0 tasks pass**:
- Proceed to Phase 1: Code Duplication Elimination
- Reference: `prompt_phase_1_002.md`

**If Phase 0 tasks fail**:
- Document all issues in `intermediate/phase_0_decisions.md`
- Address blockers before proceeding
- Re-validate after fixes

---

## Quick Reference

**Key Files**:
- `.cursor/rules/architecture_rules.md` - Architecture documentation
- `.cursor/rules/scoping_document.md` - Scoping guidelines
- `.cursor/rules/state_of_development.md` - Development state
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md` - Task tracking
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/context.md` - Project context

**Documentation to Create**:
- `intermediate/phase_0_decisions.md` - Decisions made
- `intermediate/phase_0_testing.md` - Validation results
- `intermediate/phase_0_handoff.md` - Handoff to Phase 1
- `fracas.md` - Failure tracking (root directory)

---

**Begin execution now. Good luck!**

