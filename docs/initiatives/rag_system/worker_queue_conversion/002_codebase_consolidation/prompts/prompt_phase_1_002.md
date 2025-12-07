# Prompt: Phase 1 - Code Duplication Elimination & Structure Reorganization

## Your Mission

Execute Phase 1 of Initiative 002: Move Azure Functions to `backend/azure_functions/` and simplify function entry points to use direct imports from `rag_eval`. This phase establishes a single source of truth for all backend code in one location, eliminating the need for path manipulation or code copying.

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
3. Duplicate code exists in `infra/azure/azure_functions/backend/rag_eval/` (if still present)
4. Build script exists at `infra/azure/azure_functions/build.sh`
5. Local development environment is functional (Azurite, local Supabase)

**If any prerequisite is missing, document it in `intermediate/phase_1_decisions.md` and stop execution.**

---

## Task 1.1: Move Azure Functions to Backend

**Execute**: Move the entire `infra/azure/azure_functions/` directory to `backend/azure_functions/`

**Required Actions**:
- [ ] Move `infra/azure/azure_functions/` → `backend/azure_functions/`
- [ ] Verify all files moved correctly (all 4 functions, build.sh, host.json, requirements.txt, etc.)
- [ ] Update any hardcoded paths in scripts or documentation
- [ ] Remove empty `infra/azure/` directory if no longer needed

**Files to Move**:
- `infra/azure/azure_functions/ingestion-worker/`
- `infra/azure/azure_functions/chunking-worker/`
- `infra/azure/azure_functions/embedding-worker/`
- `infra/azure/azure_functions/indexing-worker/`
- `infra/azure/azure_functions/build.sh`
- `infra/azure/azure_functions/host.json`
- `infra/azure/azure_functions/requirements.txt`
- `infra/azure/azure_functions/.deployment`
- `infra/azure/azure_functions/.funcignore`
- `infra/azure/azure_functions/local.settings.json`
- Any other files in the directory

**Success Criteria**:
- ✅ All files moved to `backend/azure_functions/`
- ✅ Directory structure preserved
- ✅ No files left in old location

**Document**: 
- Record the move in `intermediate/phase_1_decisions.md`
- Note any files that needed special handling

---

## Task 1.2: Simplify Function Entry Points

**Execute**: Update all 4 function entry points to use simple direct imports (no path manipulation)

**Functions to Update**:
- [ ] `backend/azure_functions/ingestion-worker/__init__.py`
- [ ] `backend/azure_functions/chunking-worker/__init__.py`
- [ ] `backend/azure_functions/embedding-worker/__init__.py`
- [ ] `backend/azure_functions/indexing-worker/__init__.py`

**Required Pattern**: 
- Reference `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 1.2 for the exact implementation pattern
- Functions are now in `backend/azure_functions/` alongside `backend/rag_eval/`
- No path manipulation needed - direct imports work naturally
- No dotenv loading needed - Azure handles environment variables automatically

**Example Pattern:**
```python
"""Azure Function for ingestion worker

This function is triggered by messages in the ingestion-uploads queue.
It processes documents through the ingestion stage (text extraction).
"""
import json
from rag_eval.services.workers.ingestion_worker import ingestion_worker
from rag_eval.core.logging import get_logger

logger = get_logger("azure_functions.ingestion_worker")

def main(queueMessage: str) -> None:
    """
    Azure Function entry point for ingestion worker.
    
    Args:
        queueMessage: JSON string from queue trigger
    """
    try:
        # Parse queue message
        message_dict = json.loads(queueMessage)
        
        logger.info(f"Processing ingestion message for document: {message_dict.get('document_id')}")
        
        # Call worker function
        ingestion_worker(message_dict, context=None)
        
        logger.info(f"Successfully processed ingestion for document: {message_dict.get('document_id')}")
    
    except Exception as e:
        logger.error(f"Error processing ingestion message: {e}", exc_info=True)
        raise
```

**Critical Requirements**:
- [ ] Remove all path manipulation code (`Path(__file__).parent.parent...`)
- [ ] Remove dotenv loading code (Azure handles env vars)
- [ ] Use direct imports: `from rag_eval.services.workers.ingestion_worker import ingestion_worker`
- [ ] Keep function logic simple and maintainable

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 1.2 for pattern
- Current function entry points for reference

**Success Criteria**:
- ✅ All 4 functions updated with simple direct imports
- ✅ No path manipulation code remains
- ✅ No dotenv loading code remains
- ✅ Functions are clean and maintainable

**Document**: 
- Record any import issues in `intermediate/phase_1_decisions.md`
- Note any deviations from the pattern and rationale

---

## Task 1.3: Update Build Script & Configuration

**Execute**: Update build script and deployment configuration for new location

**Required Actions**:
- [ ] Update `backend/azure_functions/build.sh` paths (if needed)
- [ ] Remove backend code copying logic (no longer needed)
- [ ] Update build script to only verify prerequisites
- [ ] Update `.deployment` file if paths changed
- [ ] Update any deployment scripts that reference the old location

**Build Script Requirements**:
- [ ] Verify prerequisites (Azure CLI, Function Core Tools, etc.)
- [ ] Validate function entry points exist
- [ ] Validate `requirements.txt` exists
- [ ] No code copying needed (functions import directly from `rag_eval`)

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 1.3 for requirements
- Current build script for reference

**Success Criteria**:
- ✅ Build script updated for new location
- ✅ Build script no longer copies backend code
- ✅ Build script validates prerequisites
- ✅ Build script validates function structure

**Document**: 
- Record build script changes in `intermediate/phase_1_decisions.md`
- Document new build process steps

---

## Task 1.4: Remove Duplicate Code (if still present)

**Execute**: Delete duplicate backend code directory if it still exists

**Required Actions**:
- [ ] Check if `backend/azure_functions/backend/rag_eval/` exists (from old location)
- [ ] If it exists, delete it (no longer needed)
- [ ] Update `.gitignore` to prevent accidental commits of duplicate code
- [ ] Verify Git repository doesn't track duplicate code

**Gitignore Update**:
- [ ] Add pattern to ignore `backend/azure_functions/backend/` if it doesn't exist
- [ ] Verify `.gitignore` prevents duplicate code from being committed

**Verification**:
```bash
# Check if duplicate code is tracked
git ls-files backend/azure_functions/backend/rag_eval/

# Should return empty (no tracked files)
```

**Success Criteria**:
- ✅ Duplicate directory deleted (if it existed)
- ✅ `.gitignore` updated
- ✅ No duplicate code tracked in Git

**Document**: 
- Record deletion in `intermediate/phase_1_decisions.md`
- Note any files that needed special handling

---

## Task 1.5: Test Local Function Execution

**Execute**: Test all 4 functions locally with new structure

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
cd backend/azure_functions
func start ingestion-worker
```

**Success Criteria**:
- ✅ All 4 functions start without import errors
- ✅ Functions can import from `rag_eval` directly
- ✅ Functions can process queue messages
- ✅ No runtime errors during execution

**Document**: 
- Record test results in `intermediate/phase_1_testing.md`
- Document any import errors or issues
- Note any fixes applied

---

## Task 1.6: Test Deployment Package Creation

**Execute**: Test that deployment package can be created without code copying

**Required Tests**:
- [ ] Run build script locally
- [ ] Verify deployment package is created
- [ ] Verify deployment package doesn't contain duplicate code
- [ ] Verify deployment package size is reduced
- [ ] Document package size reduction

**Test Commands**:
```bash
cd backend/azure_functions
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

## Task 1.7: Update Deployment Scripts

**Execute**: Update any deployment scripts that reference the old location

**Scripts to Check**:
- [ ] `scripts/setup_git_deployment.sh`
- [ ] `scripts/deploy_azure_functions.sh`
- [ ] Any other scripts that reference `infra/azure/azure_functions/`

**Required Updates**:
- [ ] Update paths to reference `backend/azure_functions/`
- [ ] Update any hardcoded paths in scripts
- [ ] Test scripts still work with new location

**Success Criteria**:
- ✅ All deployment scripts updated
- ✅ Scripts reference new location
- ✅ Scripts work correctly

**Document**: 
- Record script updates in `intermediate/phase_1_decisions.md`
- Note any scripts that needed special handling

---

## Documentation Requirements

**You MUST**:
1. **Record Decisions**: Create `intermediate/phase_1_decisions.md` with:
   - Move operation details
   - Function entry point simplification approach
   - Build script changes and rationale
   - Any deviations from RFC pattern

2. **Document Testing**: Create `intermediate/phase_1_testing.md` with:
   - Local function execution test results
   - Deployment package creation results
   - Package size reduction metrics
   - Any errors encountered and fixes

3. **Create Handoff**: Create `intermediate/phase_1_handoff.md` with:
   - Summary of structure reorganization
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
- [ ] Azure Functions moved to `backend/azure_functions/`
- [ ] All 4 function entry points simplified (no path manipulation)
- [ ] Build script updated for new location
- [ ] Functions work locally with new structure
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
- `backend/azure_functions/*-worker/__init__.py` - Function entry points (new location)
- `backend/azure_functions/build.sh` - Build script (new location)
- `backend/rag_eval/services/workers/` - Worker implementations (imported by functions)
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md` - Task tracking
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Technical design

**Documentation to Create**:
- `intermediate/phase_1_decisions.md` - Decisions made
- `intermediate/phase_1_testing.md` - Test results
- `intermediate/phase_1_handoff.md` - Handoff to Phase 2

---

**Begin execution now. Good luck!**
