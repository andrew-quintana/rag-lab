# Prompt: Package Rename - platform → platform

## Your Mission

Execute the package rename from `platform/` to `platform/` to improve code organization. This rename makes the package name reflect structure rather than project goals, improving maintainability and clarity.

## Context & References

**Key Documentation to Reference**:
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/REORGANIZATION_SCOPING.md` - Reorganization scoping (Option 1)
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/PRD002.md` - Product requirements
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Technical design
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md` - Task checklist
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/context.md` - Project context

**Prerequisites**:
- Phase 1 should be complete (Azure Functions moved to `backend/azure_functions/`)
- If Phase 1 is not complete, coordinate the rename with Phase 1 execution
- All tests should be passing before starting

## Prerequisites Check

**Before starting, verify**:
1. Current package structure: `backend/platform/` exists
2. All tests are passing: `cd backend && pytest tests/ -v`
3. No uncommitted changes that would conflict
4. Backup/checkpoint current state

**If any prerequisite is missing, document it and stop execution.**

---

## Task 1: Rename Package Directory

**Execute**: Rename the package directory from `platform/` to `platform/`

**Required Actions**:
- [ ] Rename `backend/platform/` → `backend/platform/`
- [ ] Verify all subdirectories moved correctly:
  - `api/`
  - `core/`
  - `db/`
  - `services/`
  - `utils/`
  - `prompts/`
- [ ] Verify all files are present

**Success Criteria**:
- ✅ Directory renamed successfully
- ✅ All subdirectories preserved
- ✅ No files lost

**Document**: 
- Record the rename in `intermediate/package_rename_decisions.md`

---

## Task 2: Update Package Metadata

**Execute**: Update package metadata files

**Files to Update**:
- [ ] `backend/platform/__init__.py` - Update package docstring and imports
- [ ] `backend/pyproject.toml` - Update package name and pythonpath
- [ ] Any other package metadata files

**Required Changes**:
- Update package name references from `platform` to `platform`
- Update docstrings that mention package name
- Update `pyproject.toml` pythonpath if it references `platform`

**Success Criteria**:
- ✅ Package metadata updated
- ✅ `__init__.py` imports updated
- ✅ `pyproject.toml` updated

**Document**: 
- Record metadata changes in `intermediate/package_rename_decisions.md`

---

## Task 3: Update All Python Imports

**Execute**: Update all Python files to use `platform.*` instead of `platform.*`

**Scope**:
- [ ] All files in `backend/platform/` (internal imports)
- [ ] All test files in `backend/tests/`
- [ ] All scripts in `backend/scripts/`
- [ ] Azure Functions in `backend/azure_functions/` (if Phase 1 complete) or `infra/azure/azure_functions/` (if not)
- [ ] Any other Python files that import `platform`

**Import Pattern**:
```python
# Before:
from platform.core import Config
from platform.services.workers.ingestion_worker import ingestion_worker

# After:
from platform.core import Config
from platform.services.workers.ingestion_worker import ingestion_worker
```

**Required Actions**:
- [ ] Find all files with `platform` imports (use grep: `grep -r "from platform\|import platform" backend/`)
- [ ] Update each file systematically
- [ ] Update relative imports if any
- [ ] Verify no `platform` imports remain

**Success Criteria**:
- ✅ All Python imports updated
- ✅ No `platform` imports remain in code
- ✅ All imports use `platform.*`

**Document**: 
- Record import update process in `intermediate/package_rename_decisions.md`
- List any special cases or edge cases found

---

## Task 4: Update Test Configuration

**Execute**: Update test configuration files

**Files to Update**:
- [ ] `backend/pyproject.toml` - Update `pythonpath` if it references `platform`
- [ ] `backend/pytest.ini` (if exists) - Update any package references
- [ ] `backend/conftest.py` (if exists) - Update any package references
- [ ] Any test configuration files

**Required Changes**:
- Update `pythonpath` from `["."]` or `["platform"]` to ensure `platform` is findable
- Update any hardcoded `platform` references in test config

**Success Criteria**:
- ✅ Test configuration updated
- ✅ Tests can find `platform` package

**Document**: 
- Record test config changes in `intermediate/package_rename_decisions.md`

---

## Task 5: Update Scoping Documents

**Execute**: Update all scoping documents in `002_codebase_consolidation/scoping/`

**Files to Update**:
- [ ] `PRD002.md` - Update all `platform` references to `platform`
- [ ] `RFC002.md` - Update all `platform` references to `platform`
- [ ] `TODO002.md` - Update all `platform` references to `platform`
- [ ] `context.md` - Update all `platform` references to `platform`
- [ ] `REORGANIZATION_SCOPING.md` - Mark as complete, update status

**Required Changes**:
- Replace `platform` with `platform` in all documentation
- Update code examples to show `platform.*` imports
- Update file paths: `backend/platform/` → `backend/platform/`
- Update import examples throughout

**Search Pattern**:
- `platform` → `platform`
- `backend/platform/` → `backend/platform/`
- `from platform` → `from platform`
- `import platform` → `import platform`

**Success Criteria**:
- ✅ All scoping documents updated
- ✅ All code examples updated
- ✅ All file paths updated
- ✅ Documentation is consistent

**Document**: 
- Record documentation updates in `intermediate/package_rename_decisions.md`

---

## Task 6: Update Phase Prompts

**Execute**: Update all phase prompts in `002_codebase_consolidation/prompts/`

**Files to Update**:
- [ ] `prompt_phase_0_002.md` - Update all `platform` references
- [ ] `prompt_phase_1_002.md` - Update all `platform` references
- [ ] `prompt_phase_2_002.md` - Update all `platform` references
- [ ] `prompt_phase_3_002.md` - Update all `platform` references (if exists)
- [ ] `prompt_phase_4_002.md` - Update all `platform` references (if exists)

**Required Changes**:
- Replace `platform` with `platform` in all prompts
- Update import examples: `from platform.*` → `from platform.*`
- Update file paths: `backend/platform/` → `backend/platform/`
- Update code examples in prompts

**Success Criteria**:
- ✅ All phase prompts updated
- ✅ All import examples updated
- ✅ All file paths updated
- ✅ Prompts are consistent

**Document**: 
- Record prompt updates in `intermediate/package_rename_decisions.md`

---

## Task 7: Update Intermediate Documentation

**Execute**: Update intermediate documentation files

**Files to Update**:
- [ ] `intermediate/phase_0_handoff.md` - Update any `platform` references
- [ ] `intermediate/phase_1_handoff.md` - Update any `platform` references (if exists)
- [ ] `intermediate/phase_1_decisions.md` - Update any `platform` references (if exists)
- [ ] `intermediate/phase_1_testing.md` - Update any `platform` references (if exists)
- [ ] `intermediate/phase_1_refactor_summary.md` - Update any `platform` references
- [ ] Any other intermediate files with `platform` references

**Required Changes**:
- Replace `platform` with `platform` in documentation
- Update import examples
- Update file paths

**Success Criteria**:
- ✅ All intermediate documentation updated
- ✅ Historical context preserved but updated

**Document**: 
- Record intermediate doc updates in `intermediate/package_rename_decisions.md`

---

## Task 8: Update Other Documentation

**Execute**: Update other documentation files that reference `platform`

**Files to Check**:
- [ ] `README.md` - Update any `platform` references
- [ ] `.cursor/rules/architecture_rules.md` - Update `platform` references to `platform`
- [ ] `.cursor/rules/scoping_document.md` - Update `platform` references
- [ ] `.cursor/rules/state_of_development.md` - Update `platform` references
- [ ] Any other documentation files

**Required Changes**:
- Replace `platform` with `platform` in documentation
- Update code examples
- Update file paths

**Success Criteria**:
- ✅ All documentation updated
- ✅ Cursor rules files updated
- ✅ README updated if needed

**Document**: 
- Record other doc updates in `intermediate/package_rename_decisions.md`

---

## Task 9: Update Deployment Scripts

**Execute**: Update deployment and build scripts

**Files to Check**:
- [ ] `backend/azure_functions/build.sh` - Update any `platform` references (if Phase 1 complete)
- [ ] `infra/azure/azure_functions/build.sh` - Update any `platform` references (if Phase 1 not complete)
- [ ] `scripts/setup_git_deployment.sh` - Update any `platform` references
- [ ] `scripts/deploy_azure_functions.sh` - Update any `platform` references
- [ ] Any other deployment scripts

**Required Changes**:
- Update any hardcoded `platform` references in scripts
- Update file paths if they reference `platform`

**Success Criteria**:
- ✅ All deployment scripts updated
- ✅ Scripts reference `platform` instead of `platform`

**Document**: 
- Record script updates in `intermediate/package_rename_decisions.md`

---

## Task 10: Validation & Testing

**Execute**: Validate the rename and run tests

**Required Tests**:
- [ ] Verify package structure: `ls -la backend/platform/`
- [ ] Verify imports work: `python -c "from platform.core import Config; print('OK')"`
- [ ] Run all tests: `cd backend && pytest tests/ -v`
- [ ] Check for any remaining `platform` references: `grep -r "platform" backend/ --exclude-dir=__pycache__`
- [ ] Verify Azure Functions can import (if Phase 1 complete): Test function imports

**Test Commands**:
```bash
# Verify package structure
ls -la backend/platform/

# Test import
cd backend
python -c "from platform.core import Config; print('Import successful')"

# Run tests
pytest tests/ -v

# Check for remaining references
grep -r "platform" backend/ --exclude-dir=__pycache__ | grep -v ".pyc" | head -20

# If Phase 1 complete, test Azure Functions
cd backend/azure_functions
python -c "from platform.services.workers.ingestion_worker import ingestion_worker; print('OK')"
```

**Success Criteria**:
- ✅ Package structure correct
- ✅ Imports work correctly
- ✅ All tests pass
- ✅ No `platform` references remain in code
- ✅ Azure Functions can import (if applicable)

**Document**: 
- Record test results in `intermediate/package_rename_testing.md`
- Document any issues found and fixes applied

---

## Task 11: Update Fracas Document

**Execute**: Update fracas document if needed

**Files to Update**:
- [ ] `fracas.md` - Update any `platform` references to `platform`

**Required Changes**:
- Replace `platform` with `platform` in issue descriptions
- Update file paths if mentioned

**Success Criteria**:
- ✅ Fracas document updated
- ✅ Issue references are current

---

## Documentation Requirements

**You MUST**:
1. **Record Decisions**: Create `intermediate/package_rename_decisions.md` with:
   - Package rename approach
   - Files updated
   - Special cases or edge cases
   - Any deviations from plan

2. **Document Testing**: Create `intermediate/package_rename_testing.md` with:
   - Test results
   - Import validation results
   - Any errors encountered and fixes
   - Verification that no `platform` references remain

3. **Create Summary**: Create `intermediate/package_rename_summary.md` with:
   - Summary of rename operation
   - Files changed count
   - Import update count
   - Documentation update count
   - Validation results

4. **Update Tracking**: 
   - Note the rename in `scoping/TODO002.md` if there's a section for it
   - Update `REORGANIZATION_SCOPING.md` to mark Option 1 as complete

---

## Success Criteria

### Must Pass (Blocking) - All Required:
- [ ] Package directory renamed: `backend/platform/` → `backend/platform/`
- [ ] All Python imports updated: `platform.*` → `platform.*`
- [ ] All tests pass
- [ ] No `platform` references remain in code
- [ ] All scoping documents updated
- [ ] All phase prompts updated
- [ ] Package metadata updated

**If any "Must Pass" criteria fails, document in `intermediate/package_rename_decisions.md` and `fracas.md`, then address before proceeding.**

---

## Quick Reference

**Key Changes**:
- Package: `platform/` → `platform/`
- Imports: `from platform.*` → `from platform.*`
- File paths: `backend/platform/` → `backend/platform/`

**Files to Update**:
- All Python files with imports
- All test files
- All scripts
- All documentation
- All prompts
- Package metadata

**Validation**:
- Run tests: `pytest tests/ -v`
- Check for remaining references: `grep -r "platform" backend/`
- Test imports: `python -c "from platform.core import Config"`

---

## Migration Checklist

### Phase 1: Package Rename
- [ ] Rename directory
- [ ] Update package metadata
- [ ] Verify structure

### Phase 2: Code Updates
- [ ] Update all Python imports
- [ ] Update test configuration
- [ ] Update deployment scripts

### Phase 3: Documentation Updates
- [ ] Update scoping documents
- [ ] Update phase prompts
- [ ] Update intermediate docs
- [ ] Update other documentation
- [ ] Update fracas

### Phase 4: Validation
- [ ] Run all tests
- [ ] Verify imports work
- [ ] Check for remaining references
- [ ] Validate Azure Functions (if applicable)

---

**Begin execution now. Good luck!**

