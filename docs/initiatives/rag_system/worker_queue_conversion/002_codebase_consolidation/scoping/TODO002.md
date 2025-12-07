# TODO 002 — Codebase Consolidation & Production Readiness

## Context

This TODO document provides the implementation breakdown for consolidating the codebase structure, eliminating duplication, and ensuring production readiness, as specified in [PRD002.md](./PRD002.md) and [RFC002.md](./RFC002.md).

**Current Status**: The Azure Functions deployment includes duplicate backend code in `infra/azure/azure_functions/backend/rag_eval/`, and configuration management is fragmented. This TODO provides the implementation plan for consolidation.

**Implementation Phases**: This TODO follows a 5-phase implementation plan that consolidates the codebase incrementally, starting with Phase 0 (scoping and documentation updates), then progressing through code duplication elimination, configuration consolidation, test infrastructure consolidation, and production readiness validation.

---

## Phase 0 — Scoping & Documentation Updates

**Status**: ✅ Complete

### 0.1 Update Cursor Rules Files

**Purpose**: Update foundational documentation to reflect the worker-queue architecture transformation implemented in Initiative 001.

- [x] **Update `.cursor/rules/architecture_rules.md`**
  - [x] Add worker-queue architecture layer description
  - [x] Document worker infrastructure (`rag_eval/services/workers/`)
  - [x] Update layer boundaries to include Azure Functions as worker execution layer
  - [x] Document queue-based communication patterns
  - [x] Update development workflow to include local Azure Functions development with Azurite
  - [x] Add Azure Storage Queues to infrastructure components

- [x] **Update `.cursor/rules/scoping_document.md`**
  - [x] Update "High-Level System Overview" to include worker-queue architecture
  - [x] Add worker infrastructure to codebase structure diagram
  - [x] Document Azure Functions deployment structure (`infra/azure/azure_functions/`)
  - [x] Update "Out of Scope" section (remove workers from out-of-scope list)
  - [x] Add Azure Storage Queues to infrastructure architecture section
  - [x] Update database scope to include worker status tracking (migrations 0019, 0020)
  - [x] Update "Required Scaffolding" section to include Azure Functions setup

- [x] **Update `.cursor/rules/state_of_development.md`**
  - [x] Update "CURRENT PHASE" to reflect worker-queue architecture is implemented
  - [x] Remove workers from "OUT OF SCOPE FOR THIS PHASE" section
  - [x] Add worker-queue architecture to system description
  - [x] Update "PRIMARY OBJECTIVES IN THIS PHASE" or create new section for worker-queue
  - [x] Update "LOCAL DEVELOPMENT WORKFLOW" to include Azure Functions local development
  - [x] Add validation gates for worker-queue functionality
  - [x] Update meta-rule to reflect workers are now part of the system
  - [x] Document local development with Azurite and local Supabase

### 0.2 File Organization Requirements

**Purpose**: Specify where files created during implementation should be placed.

**Directory Structure:**
- **scoping/**: Initial scoping documents (PRD, RFC, TODO, context.md)
- **prompts/**: Phase execution prompts (`prompt_phase_X_002.md`)
- **intermediate/**: Phase-specific workflow documents
  - `phase_X_decisions.md` - Decisions made during each phase
  - `phase_X_testing.md` - Testing documentation for each phase
  - `phase_X_handoff.md` - Handoff documentation between phases
- **notes/**: Miscellaneous supporting documentation
  - `notes/deployment/` - Deployment guides and notes
  - `notes/troubleshooting/` - Troubleshooting guides
  - `notes/research/` - Research and exploration notes
- **Root directory**: Final summary documents only
  - `summary.md` - Final initiative summary (required)
  - `fracas.md` - Failure tracking (created in Phase 0)
  - `technical_debt.md` - Technical debt documentation (optional)

**Naming Conventions:**
- Phase documents: `phase_{phase_number}_{type}.md` (e.g., `phase_1_decisions.md`)
- Notes: Use descriptive names with underscores (e.g., `azure_functions_troubleshooting.md`)
- Prompts: `prompt_phase_{phase_number}_002.md` (serial 002 for this initiative)

**Examples:**
- Deployment verification results → `notes/deployment/verification_results.md`
- Phase 1 testing results → `intermediate/phase_1_testing.md`
- Research on code consolidation approaches → `notes/research/code_consolidation_approaches.md`

**Status**: ✅ Validated - File organization requirements are documented and clear.

### 0.3 Validation

- [x] Verify all cursor rules files accurately reflect worker-queue architecture
- [x] Confirm file organization requirements are documented
- [x] Review updated rules files for consistency and completeness

---

## Phase 1 — Code Duplication Elimination

**Status**: ⏳ Pending

### 1.1 Update Function Entry Points
- [ ] Review current function entry point import structure
- [ ] Update `infra/azure/azure_functions/ingestion-worker/__init__.py` to import from project root
- [ ] Update `infra/azure/azure_functions/chunking-worker/__init__.py` to import from project root
- [ ] Update `infra/azure/azure_functions/embedding-worker/__init__.py` to import from project root
- [ ] Update `infra/azure/azure_functions/indexing-worker/__init__.py` to import from project root
- [ ] Ensure dotenv loading happens before path manipulation
- [ ] Test local function execution with new import structure

### 1.2 Remove Duplicate Code
- [ ] Verify functions work with project root imports
- [ ] Delete `infra/azure/azure_functions/backend/rag_eval/` directory
- [ ] Update `.gitignore` to prevent accidental commits of duplicate code
- [ ] Verify Git repository doesn't track duplicate code

### 1.3 Simplify Build Script
- [ ] Review current `infra/azure/azure_functions/build.sh`
- [ ] Remove backend code copying logic
- [ ] Update build script to only verify prerequisites
- [ ] Test build script in local environment
- [ ] Document new build process

### 1.4 Testing & Validation
- [ ] Test all 4 functions locally with new import structure
- [ ] Verify functions can process queue messages correctly
- [ ] Test deployment package creation (without code copying)
- [ ] Validate functions work in Azure environment (staging/test)

---

## Phase 2 — Configuration Consolidation

**Status**: ⏳ Pending

### 2.1 Document Configuration Strategy
- [ ] Review current configuration loading approach
- [ ] Document environment variable loading precedence
- [ ] Document `.env.local` vs `local.settings.json` vs Azure settings
- [ ] Create configuration guide for local and cloud environments

### 2.2 Standardize Configuration Files
- [ ] Review and standardize `.env.local` template
- [ ] Ensure `local.settings.json` contains only Azurite/runtime settings
- [ ] Document required Azure Function App settings
- [ ] Create configuration validation scripts

### 2.3 Update Documentation
- [ ] Update `LOCAL_DEVELOPMENT.md` with consolidated configuration approach
- [ ] Update `README_LOCAL.md` with configuration details
- [ ] Create configuration migration guide if needed
- [ ] Document configuration troubleshooting steps

### 2.4 Testing & Validation
- [ ] Test configuration loading in local environment
- [ ] Test configuration loading in Azure environment
- [ ] Verify configuration precedence works correctly
- [ ] Validate all required variables are accessible

---

## Phase 3 — Test Infrastructure Consolidation

**Status**: ⏳ Pending

### 3.1 Register Pytest Markers
- [ ] Review current pytest marker usage
- [ ] Register `@pytest.mark.local` in `pytest.ini` or `conftest.py`
- [ ] Register `@pytest.mark.cloud` in `pytest.ini` or `conftest.py`
- [ ] Register `@pytest.mark.integration` if needed
- [ ] Verify markers are registered (no warnings)

### 3.2 Consolidate Test Fixtures
- [ ] Review test fixtures in `backend/tests/fixtures/`
- [ ] Identify duplicate fixtures
- [ ] Consolidate common fixtures (database, queues, etc.)
- [ ] Update tests to use consolidated fixtures
- [ ] Remove duplicate fixture code

### 3.3 Create Unified Test Scripts
- [ ] Review existing test execution scripts
- [ ] Create/update `scripts/test_functions_local.sh` for local testing
- [ ] Create/update `scripts/test_functions_cloud.sh` for cloud testing
- [ ] Create `scripts/test_functions_all.sh` for running all tests
- [ ] Document test execution workflows

### 3.4 Testing & Validation
- [ ] Run Phase 5 local tests with consolidated infrastructure
- [ ] Run Phase 5 cloud tests with consolidated infrastructure
- [ ] Verify all tests pass with new structure
- [ ] Document any test infrastructure changes

---

## Phase 4 — Production Readiness Validation

**Status**: ⏳ Pending

### 4.1 Complete Phase 5 Local Testing
- [ ] Run all Phase 5 local tests
- [ ] Verify migration tests pass
- [ ] Verify Supabase integration tests pass
- [ ] Verify E2E pipeline tests pass
- [ ] Document local test results

### 4.2 Complete Phase 5 Cloud Testing
- [ ] Deploy consolidated codebase to Azure Functions
- [ ] Run all Phase 5 cloud tests
- [ ] Verify queue triggers work correctly
- [ ] Verify performance tests pass
- [ ] Document cloud test results

### 4.3 Resolve Identified Issues
- [ ] Review all issues from Phase 5 testing
- [ ] Fix FM-001 (batch result persistence query error) if still present
- [ ] Fix FM-005 (Azure Functions queue trigger issue) if still present
- [ ] Fix any new issues discovered during consolidation
- [ ] Update `fracas.md` with resolved issues

### 4.4 Performance Validation
- [ ] Run performance tests in cloud environment
- [ ] Validate throughput requirements
- [ ] Validate latency requirements
- [ ] Document performance test results

### 4.5 Production Deployment Verification
- [ ] Deploy to production environment
- [ ] Verify all functions are deployed correctly
- [ ] Test end-to-end pipeline in production
- [ ] Monitor function logs for errors
- [ ] Validate production stability

### 4.6 Re-validate Cursor Rules Files
- [ ] **Re-validate `.cursor/rules/architecture_rules.md`**
  - [ ] Verify it accurately describes consolidated code structure (no duplicate code references)
  - [ ] Verify layer boundaries reflect single source of truth for backend code
  - [ ] Verify development workflow matches consolidated deployment process
  - [ ] Update if any inaccuracies are found (but note: files are NOT modified as part of consolidation work)

- [ ] **Re-validate `.cursor/rules/scoping_document.md`**
  - [ ] Verify codebase structure diagram reflects consolidated structure (no duplicate backend/rag_eval/)
  - [ ] Verify deployment process description matches simplified build script
  - [ ] Verify configuration management section reflects consolidated approach
  - [ ] Update if any inaccuracies are found

- [ ] **Re-validate `.cursor/rules/state_of_development.md`**
  - [ ] Verify development workflow matches consolidated local development process
  - [ ] Verify configuration loading strategy matches consolidated approach
  - [ ] Verify deployment process description is accurate
  - [ ] Update if any inaccuracies are found

**Note**: These files are documentation/guidance files. They are updated once in Phase 0 to reflect worker-queue architecture, then re-validated in Phase 4 to ensure they remain accurate after consolidation. They are NOT modified as part of the consolidation work itself unless inaccuracies are discovered during validation.

---

## Success Criteria Checklist

### Codebase Consolidation
- [ ] Duplicate `backend/rag_eval/` code removed from `infra/azure/azure_functions/backend/`
- [ ] Functions import directly from project root
- [ ] Build script simplified (no code copying)
- [ ] Deployment package size reduced
- [ ] Single source of truth verified

### Configuration Consolidation
- [ ] Configuration loading strategy documented
- [ ] Configuration precedence clearly defined
- [ ] Local and cloud configuration standardized
- [ ] All configuration sources documented

### Test Infrastructure Consolidation
- [ ] Pytest markers registered
- [ ] Test fixtures consolidated
- [ ] Unified test execution scripts created
- [ ] Test infrastructure documented

### Production Readiness
- [ ] All Phase 5 local tests pass
- [ ] All Phase 5 cloud tests pass
- [ ] Performance requirements validated
- [ ] All identified issues resolved
- [ ] Production deployment verified and stable

---

## Notes

- Consolidation should be done incrementally with testing at each phase
- Maintain backward compatibility where possible
- Document all changes for future reference
- Test thoroughly in staging before production deployment

---

**Last Updated**: 2025-12-07  
**Related Initiative**: 001 (Worker-Queue Architecture Implementation)  
**Status**: Ready for implementation

