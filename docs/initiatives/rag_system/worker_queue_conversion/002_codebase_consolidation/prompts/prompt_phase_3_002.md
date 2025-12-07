# Prompt: Phase 3 - Test Infrastructure Consolidation

## Your Mission

Execute Phase 3 of Initiative 002: Consolidate test infrastructure by registering pytest markers, consolidating test fixtures, and creating unified test execution scripts. This phase establishes a consistent, maintainable test infrastructure for both local and cloud testing.

## Context & References

**Key Documentation to Reference**:
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/PRD002.md` - Product requirements
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Technical design (Section 3)
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md` - Task checklist (mark tasks complete)
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/context.md` - Project context
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/intermediate/phase_2_handoff.md` - Phase 2 handoff

**Prerequisites** (from Phase 2):
- ✅ Configuration strategy documented
- ✅ Configuration files standardized
- ✅ Configuration loading works correctly

## Prerequisites Check

**Before starting, verify**:
1. Phase 2 is complete (check `intermediate/phase_2_handoff.md`)
2. Test files exist in `backend/tests/integration/`
3. Test fixtures exist in `backend/tests/fixtures/` or `backend/tests/`
4. Pytest configuration exists (`pytest.ini` or `pyproject.toml` or `conftest.py`)
5. Test execution scripts exist in `scripts/`

**If any prerequisite is missing, document it in `intermediate/phase_3_decisions.md` and stop execution.**

---

## Task 3.1: Register Pytest Markers

**Execute**: Register pytest markers to eliminate warnings

**Required Actions**:
- [ ] Review current pytest marker usage in test files
- [ ] Identify all markers used: `@pytest.mark.local`, `@pytest.mark.cloud`, `@pytest.mark.integration`, `@pytest.mark.performance`
- [ ] Register markers in `pytest.ini` or `conftest.py` or `pyproject.toml`
- [ ] Verify markers are registered (run tests and check for warnings)

**Marker Registration**:
- [ ] Register `@pytest.mark.local` - Tests that run against local resources (Azurite, local Supabase)
- [ ] Register `@pytest.mark.cloud` - Tests that run against cloud resources (Azure Functions, cloud Supabase)
- [ ] Register `@pytest.mark.integration` - Integration tests requiring real resources
- [ ] Register `@pytest.mark.performance` - Performance tests

**Registration Location Options**:
- `backend/pytest.ini` - Pytest configuration file
- `backend/conftest.py` - Pytest configuration in conftest
- `backend/pyproject.toml` - If using pyproject.toml for pytest config

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 3.1 for requirements
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/context.md` - Section 8.3 for marker descriptions
- Current test files for marker usage

**Success Criteria**:
- ✅ All markers registered
- ✅ No pytest warnings about unregistered markers
- ✅ Markers are documented

**Document**: 
- Record marker registration approach in `intermediate/phase_3_decisions.md`
- Note which file was used for registration

---

## Task 3.2: Consolidate Test Fixtures

**Execute**: Review and consolidate duplicate test fixtures

**Required Actions**:
- [ ] Review test fixtures in `backend/tests/fixtures/` and `backend/tests/`
- [ ] Identify duplicate fixtures (database, queues, configuration, etc.)
- [ ] Consolidate common fixtures into shared location
- [ ] Update tests to use consolidated fixtures
- [ ] Remove duplicate fixture code

**Common Fixtures to Consolidate**:
- [ ] Database connection fixtures
- [ ] Queue client fixtures
- [ ] Configuration fixtures
- [ ] Test data fixtures
- [ ] Cleanup fixtures

**Consolidation Approach**:
- [ ] Create or update `backend/tests/conftest.py` for shared fixtures
- [ ] Move common fixtures to shared location
- [ ] Update imports in test files
- [ ] Verify all tests still work with consolidated fixtures

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 3.2 for requirements
- Current test files for fixture usage

**Success Criteria**:
- ✅ Duplicate fixtures identified and consolidated
- ✅ All tests use consolidated fixtures
- ✅ No duplicate fixture code remains
- ✅ Tests still pass with consolidated fixtures

**Document**: 
- Record fixture consolidation decisions in `intermediate/phase_3_decisions.md`
- Document which fixtures were consolidated
- Note any test updates required

---

## Task 3.3: Create Unified Test Scripts

**Execute**: Create/update unified test execution scripts

**Required Scripts**:
- [ ] Review existing test execution scripts in `scripts/`
- [ ] Create/update `scripts/test_functions_local.sh` for local testing
- [ ] Create/update `scripts/test_functions_cloud.sh` for cloud testing
- [ ] Create `scripts/test_functions_all.sh` for running all tests with appropriate markers
- [ ] Document test execution workflows

**Script Requirements**:

**`scripts/test_functions_local.sh`**:
- [ ] Start Azurite if not running
- [ ] Verify local Supabase is running
- [ ] Run tests with `@pytest.mark.local` marker
- [ ] Run integration tests marked for local
- [ ] Provide clear output

**`scripts/test_functions_cloud.sh`**:
- [ ] Verify Azure Functions are deployed
- [ ] Verify cloud resources are accessible
- [ ] Run tests with `@pytest.mark.cloud` marker
- [ ] Run integration tests marked for cloud
- [ ] Provide clear output

**`scripts/test_functions_all.sh`**:
- [ ] Run local tests
- [ ] Run cloud tests (if environment configured)
- [ ] Provide summary of all test results
- [ ] Handle missing environments gracefully

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 3.3 for requirements
- `@docs/initiatives/rag_system/worker_queue_conversion/001_initial_conversion/intermediate/` - For existing test scripts
- Current test scripts for reference

**Success Criteria**:
- ✅ All test scripts created/updated
- ✅ Scripts use pytest markers correctly
- ✅ Scripts provide clear output
- ✅ Scripts are documented

**Document**: 
- Record script creation/updates in `intermediate/phase_3_decisions.md`
- Document script usage in test execution documentation

---

## Task 3.4: Test Consolidated Infrastructure

**Execute**: Run Phase 5 tests with consolidated infrastructure

**Required Tests**:
- [ ] Run Phase 5 local tests with consolidated infrastructure
  - [ ] Migration verification tests
  - [ ] Supabase integration tests
  - [ ] E2E pipeline tests (local markers)
- [ ] Run Phase 5 cloud tests with consolidated infrastructure (if cloud available)
  - [ ] E2E pipeline tests (cloud markers)
  - [ ] Performance tests (if applicable)
- [ ] Verify all tests pass with new structure
- [ ] Verify no pytest warnings about markers

**Test Execution**:
```bash
# Local tests
./scripts/test_functions_local.sh

# Cloud tests (if available)
./scripts/test_functions_cloud.sh

# All tests
./scripts/test_functions_all.sh
```

**Success Criteria**:
- ✅ All local tests pass
- ✅ All cloud tests pass (if executed)
- ✅ No pytest marker warnings
- ✅ Test infrastructure works correctly

**Document**: 
- Record test results in `intermediate/phase_3_testing.md`
- Document any test failures and fixes
- Note any infrastructure issues

---

## Documentation Requirements

**You MUST**:
1. **Record Decisions**: Create `intermediate/phase_3_decisions.md` with:
   - Pytest marker registration approach
   - Fixture consolidation decisions
   - Test script design decisions
   - Any test infrastructure changes

2. **Document Testing**: Create `intermediate/phase_3_testing.md` with:
   - Test execution results with consolidated infrastructure
   - Any test failures and fixes
   - Pytest marker verification results
   - Test infrastructure validation results

3. **Create Handoff**: Create `intermediate/phase_3_handoff.md` with:
   - Summary of test infrastructure consolidation
   - Pytest marker registration status
   - Fixture consolidation status
   - Test script creation status
   - Prerequisites verified for Phase 4
   - Any blockers or concerns for Phase 4

4. **Update Tracking**: 
   - Mark Phase 3 tasks complete in `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md`
   - Check off all [ ] checkboxes in Section 3.1, 3.2, 3.3, and 3.4 as tasks are completed

5. **Update Fracas**: 
   - Document any new test infrastructure issues discovered in `fracas.md`
   - Update status of any existing issues if resolved

---

## Success Criteria

### Must Pass (Blocking) - All Required:
- [ ] Pytest markers registered
- [ ] Test fixtures consolidated
- [ ] Test scripts created/updated
- [ ] All tests pass with consolidated infrastructure

**If any "Must Pass" criteria fails, document in `intermediate/phase_3_decisions.md` and `fracas.md`, then address before proceeding.**

---

## Next Steps

**If all Phase 3 tasks pass**:
- Proceed to Phase 4: Production Readiness Validation
- Reference: `prompt_phase_4_002.md`

**If Phase 3 tasks fail**:
- Document all issues in `intermediate/phase_3_decisions.md` and `fracas.md`
- Address blockers before proceeding
- Re-test after fixes

---

## Quick Reference

**Key Files**:
- `backend/pytest.ini` or `backend/conftest.py` or `backend/pyproject.toml` - Pytest configuration
- `backend/tests/fixtures/` - Test fixtures
- `backend/tests/conftest.py` - Shared fixtures
- `scripts/test_functions_*.sh` - Test execution scripts
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md` - Task tracking
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Technical design

**Documentation to Create**:
- `intermediate/phase_3_decisions.md` - Decisions made
- `intermediate/phase_3_testing.md` - Test results
- `intermediate/phase_3_handoff.md` - Handoff to Phase 4

---

**Begin execution now. Good luck!**

