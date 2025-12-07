# Prompt: Phase 1 - Code Duplication Elimination

## Your Mission

Execute Phase 1 of Initiative 002: Eliminate code duplication by updating Azure Functions to import directly from project root, removing duplicate backend code, and simplifying the build script. This phase establishes a single source of truth for all backend code.

## Context & References

**Key Documentation to Reference**:
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/PRD002.md` - Product requirements
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Technical design (Section 1)
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md` - Task checklist (mark tasks complete)
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/context.md` - Project context
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/intermediate/phase_0_handoff.md` - Phase 0 handoff

**Prerequisites** (from Phase 0):
- ✅ Cursor rules files updated
- ✅ File organization requirements documented
- ✅ Fracas document created

## Prerequisites Check

**Before starting, verify**:
1. Phase 0 is complete (check `intermediate/phase_0_handoff.md`)
2. All 4 Azure Functions exist in `infra/azure/azure_functions/`
3. Duplicate code exists in `infra/azure/azure_functions/backend/rag_eval/`
4. Build script exists at `infra/azure/azure_functions/build.sh`
5. Local development environment is functional (Azurite, local Supabase)

**If any prerequisite is missing, document it in `intermediate/phase_1_decisions.md` and stop execution.**

---

## Task 1.1: Update Function Entry Points

**Execute**: Update all 4 function entry points to import from project root using path manipulation

**Functions to Update**:
- [ ] `infra/azure/azure_functions/ingestion-worker/__init__.py`
- [ ] `infra/azure/azure_functions/chunking-worker/__init__.py`
- [ ] `infra/azure/azure_functions/embedding-worker/__init__.py`
- [ ] `infra/azure/azure_functions/indexing-worker/__init__.py`

**Required Pattern**: 
- Reference `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 1.2 for the exact implementation pattern
- Pattern must support flexible environment variable loading (Azure settings > `.env.local` > system environment > test fixtures)
- Pattern must add project root to Python path before importing backend code
- Pattern must handle environment variable loading according to the flexible strategy documented in architecture rules

**Critical Requirements**:
- [ ] Dotenv loading happens BEFORE path manipulation
- [ ] Path resolution correctly identifies project root
- [ ] All imports use `rag_eval.*` from project root
- [ ] No imports reference `backend/rag_eval/` or copied code

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 1.2 for pattern
- Current function entry points for reference

**Success Criteria**:
- ✅ All 4 functions updated with correct import pattern
- ✅ Dotenv loads before path manipulation
- ✅ Path resolution works correctly
- ✅ No references to copied code remain

**Document**: 
- Record any path resolution issues in `intermediate/phase_1_decisions.md`
- Note any deviations from the pattern and rationale

---

## Task 1.2: Test Local Function Execution

**Execute**: Test all 4 functions locally with new import structure

**Required Tests**:
- [ ] Start Azurite (`./scripts/start_azurite.sh`)
- [ ] Start local Supabase (`supabase start`)
- [ ] Test ingestion-worker function locally
- [ ] Test chunking-worker function locally
- [ ] Test embedding-worker function locally
- [ ] Test indexing-worker function locally
- [ ] Verify functions can process queue messages correctly

**Test Commands**:
```bash
# Start local services
./scripts/start_azurite.sh
supabase start

# Test functions (example for ingestion-worker)
cd infra/azure/azure_functions
func start ingestion-worker
```

**Success Criteria**:
- ✅ All 4 functions start without import errors
- ✅ Functions can import from project root
- ✅ Functions can process queue messages
- ✅ No runtime errors during execution

**Document**: 
- Record test results in `intermediate/phase_1_testing.md`
- Document any import errors or path resolution issues
- Note any fixes applied

---

## Task 1.3: Remove Duplicate Code

**Execute**: Delete duplicate backend code directory

**Required Actions**:
- [ ] Verify functions work with project root imports (from Task 1.2)
- [ ] Delete `infra/azure/azure_functions/backend/rag_eval/` directory
- [ ] Update `.gitignore` to prevent accidental commits of duplicate code
- [ ] Verify Git repository doesn't track duplicate code

**Gitignore Update**:
- [ ] Add pattern to ignore `infra/azure/azure_functions/backend/` if it doesn't exist
- [ ] Verify `.gitignore` prevents duplicate code from being committed

**Verification**:
```bash
# Check if duplicate code is tracked
git ls-files infra/azure/azure_functions/backend/rag_eval/

# Should return empty (no tracked files)
```

**Success Criteria**:
- ✅ Duplicate directory deleted
- ✅ `.gitignore` updated
- ✅ No duplicate code tracked in Git
- ✅ Functions still work after deletion

**Document**: 
- Record deletion in `intermediate/phase_1_decisions.md`
- Note any files that needed special handling

---

## Task 1.4: Simplify Build Script

**Execute**: Update build script to remove code copying logic

**Required Actions**:
- [ ] Review current `infra/azure/azure_functions/build.sh`
- [ ] Remove backend code copying logic
- [ ] Update build script to only verify prerequisites
- [ ] Ensure build script validates function entry points are correct
- [ ] Ensure build script validates `requirements.txt` is present

**Build Script Requirements**:
- [ ] Verify prerequisites (Azure CLI, Function Core Tools, etc.)
- [ ] Validate function entry points exist
- [ ] Validate `requirements.txt` exists
- [ ] Prepare deployment package without code copying
- [ ] Document new build process

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 1.3 for requirements
- Current build script for reference

**Success Criteria**:
- ✅ Build script no longer copies backend code
- ✅ Build script validates prerequisites
- ✅ Build script validates function structure
- ✅ Build process is simplified

**Document**: 
- Record build script changes in `intermediate/phase_1_decisions.md`
- Document new build process steps

---

## Task 1.5: Test Deployment Package Creation

**Execute**: Test that deployment package can be created without code copying

**Required Tests**:
- [ ] Run build script locally
- [ ] Verify deployment package is created
- [ ] Verify deployment package doesn't contain duplicate code
- [ ] Verify deployment package size is reduced
- [ ] Document package size reduction

**Test Commands**:
```bash
cd infra/azure/azure_functions
./build.sh
# Verify package contents
```

**Success Criteria**:
- ✅ Deployment package created successfully
- ✅ No duplicate code in package
- ✅ Package size is reduced (target: 50%+ reduction)
- ✅ Package structure is correct

**Document**: 
- Record package size comparison in `intermediate/phase_1_testing.md`
- Document package structure
- Note any issues with package creation

---

## Task 1.6: Validate in Azure Environment (Optional/Staging)

**Execute**: If staging environment available, validate functions work in Azure

**Required Tests** (if staging available):
- [ ] Deploy to staging/test Azure Function App
- [ ] Verify functions can import from project root in Azure
- [ ] Test that path resolution works correctly in cloud
- [ ] Verify functions process queue messages in Azure
- [ ] Monitor function logs for errors

**Note**: This is optional if staging environment is not available. Can be deferred to Phase 4.

**Success Criteria** (if executed):
- ✅ Functions deploy successfully
- ✅ Functions can import from project root in Azure
- ✅ Path resolution works in cloud environment
- ✅ Functions process messages correctly

**Document**: 
- Record Azure validation results in `intermediate/phase_1_testing.md`
- Document any cloud-specific issues found
- Note if deferred to Phase 4

---

## Documentation Requirements

**You MUST**:
1. **Record Decisions**: Create `intermediate/phase_1_decisions.md` with:
   - Path resolution approach and any issues
   - Build script changes and rationale
   - Any deviations from RFC pattern
   - Gitignore updates

2. **Document Testing**: Create `intermediate/phase_1_testing.md` with:
   - Local function execution test results
   - Deployment package creation results
   - Package size reduction metrics
   - Azure validation results (if performed)
   - Any errors encountered and fixes

3. **Create Handoff**: Create `intermediate/phase_1_handoff.md` with:
   - Summary of code duplication elimination
   - Status of function entry points
   - Build script simplification status
   - Prerequisites verified for Phase 2
   - Any blockers or concerns for Phase 2

4. **Update Tracking**: 
   - Mark Phase 1 tasks complete in `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md`
   - Check off all [ ] checkboxes in Section 1.1, 1.2, 1.3, and 1.4 as tasks are completed

5. **Update Fracas**: 
   - Document any new issues discovered in `fracas.md`
   - Update status of any existing issues if resolved

---

## Success Criteria

### Must Pass (Blocking) - All Required:
- [ ] All 4 function entry points updated
- [ ] Functions work locally with new import structure
- [ ] Duplicate code directory deleted
- [ ] Build script simplified
- [ ] Deployment package created successfully

**If any "Must Pass" criteria fails, document in `intermediate/phase_1_decisions.md` and `fracas.md`, then address before proceeding.**

---

## Next Steps

**If all Phase 1 tasks pass**:
- Proceed to Phase 2: Configuration Consolidation
- Reference: `prompt_phase_2_002.md`

**If Phase 1 tasks fail**:
- Document all issues in `intermediate/phase_1_decisions.md` and `fracas.md`
- Address blockers before proceeding
- Re-test after fixes

---

## Quick Reference

**Key Files**:
- `infra/azure/azure_functions/*-worker/__init__.py` - Function entry points
- `infra/azure/azure_functions/build.sh` - Build script
- `infra/azure/azure_functions/backend/rag_eval/` - Duplicate code (to be deleted)
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md` - Task tracking
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Technical design

**Documentation to Create**:
- `intermediate/phase_1_decisions.md` - Decisions made
- `intermediate/phase_1_testing.md` - Test results
- `intermediate/phase_1_handoff.md` - Handoff to Phase 2

---

**Begin execution now. Good luck!**

