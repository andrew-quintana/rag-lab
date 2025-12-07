# Prompt: Phase 2 - Configuration Consolidation

## Your Mission

Execute Phase 2 of Initiative 002: Consolidate configuration management by unifying environment variable loading strategy, standardizing configuration files, and documenting configuration precedence. This phase establishes a single, clear approach to configuration across local and cloud environments.

## Context & References

**Key Documentation to Reference**:
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/PRD002.md` - Product requirements
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Technical design (Section 2)
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md` - Task checklist (mark tasks complete)
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/context.md` - Project context
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/intermediate/phase_1_handoff.md` - Phase 1 handoff

**Prerequisites** (from Phase 1):
- ✅ Function entry points updated to import from project root
- ✅ Duplicate code removed
- ✅ Build script simplified

## Prerequisites Check

**Before starting, verify**:
1. Phase 1 is complete (check `intermediate/phase_1_handoff.md`)
2. Function entry points load `.env.local` via dotenv
3. `local.settings.json` exists in `infra/azure/azure_functions/`
4. `.env.local` exists in project root
5. Azure Function App settings are accessible (for documentation)

**If any prerequisite is missing, document it in `intermediate/phase_2_decisions.md` and stop execution.**

---

## Task 2.1: Document Configuration Strategy

**Execute**: Review and document environment variable loading precedence and strategy

**Required Documentation**:
- [ ] Review current configuration loading approach in function entry points
- [ ] Document environment variable loading precedence:
  - Azure Function App settings (highest priority)
  - `.env.local` (loaded via dotenv in function entry points)
  - `local.settings.json` (only for Azurite/runtime settings)
- [ ] Document when each configuration source is used
- [ ] Create configuration guide for local and cloud environments

**Configuration Precedence** (from RFC002.md):
1. **Azure Function App Settings** (production/cloud)
2. **`.env.local`** (local development, loaded via dotenv)
3. **`local.settings.json`** (only Azurite connection string and runtime settings)

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 2.1 for strategy
- Current function entry points for implementation details
- `@docs/initiatives/rag_system/worker_queue_conversion/001_initial_conversion/intermediate/LOCAL_DEVELOPMENT.md` - For local setup details

**Success Criteria**:
- ✅ Configuration precedence clearly documented
- ✅ Loading strategy explained
- ✅ Local vs cloud configuration documented
- ✅ Guide is clear and actionable

**Document**: 
- Create configuration guide in `notes/deployment/configuration_guide.md`
- Record any clarifications needed in `intermediate/phase_2_decisions.md`

---

## Task 2.2: Standardize Configuration Files

**Execute**: Review and standardize `.env.local` and `local.settings.json`

**Required Actions**:
- [ ] Review `.env.local` template/example
- [ ] Ensure `.env.local` contains all application configuration (database, Azure AI services, etc.)
- [ ] Ensure `local.settings.json` contains ONLY:
  - `AzureWebJobsStorage` (Azurite connection string)
  - `AZURE_STORAGE_QUEUES_CONNECTION_STRING` (Azurite connection string)
  - `FUNCTIONS_WORKER_RUNTIME` (python)
- [ ] Document required Azure Function App settings for production
- [ ] Create configuration validation scripts

**Configuration File Requirements**:

**`.env.local`** (project root):
- All application configuration
- Database connection strings
- Azure AI service endpoints and keys
- Supabase configuration
- Any other application settings

**`local.settings.json`** (`infra/azure/azure_functions/`):
- ONLY Azurite connection strings
- ONLY runtime settings
- NO application configuration

**Azure Function App Settings** (production):
- All application configuration (secrets, endpoints, etc.)
- Same variables as `.env.local` but as Azure settings

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 2.2 for requirements
- Current configuration files for reference

**Success Criteria**:
- ✅ `.env.local` template is standardized
- ✅ `local.settings.json` contains only Azurite/runtime settings
- ✅ Required Azure settings are documented
- ✅ Configuration files are properly separated

**Document**: 
- Record configuration file changes in `intermediate/phase_2_decisions.md`
- Document required Azure settings in `notes/deployment/azure_function_app_settings.md`

---

## Task 2.3: Create Configuration Validation Scripts

**Execute**: Create scripts to validate configuration

**Required Scripts**:
- [ ] Create script to validate `.env.local` has all required variables
- [ ] Create script to validate `local.settings.json` has only allowed variables
- [ ] Create script to validate Azure Function App settings (if possible)
- [ ] Document validation script usage

**Script Requirements**:
- [ ] Check for required environment variables
- [ ] Validate variable formats where applicable
- [ ] Provide clear error messages
- [ ] Can be run as part of local development setup

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 2.2 mentions validation scripts

**Success Criteria**:
- ✅ Validation scripts created
- ✅ Scripts check required variables
- ✅ Scripts provide helpful error messages
- ✅ Scripts are documented

**Document**: 
- Record script location and usage in `intermediate/phase_2_decisions.md`
- Document script usage in configuration guide

---

## Task 2.4: Update Documentation

**Execute**: Update existing documentation with consolidated configuration approach

**Required Updates**:
- [ ] Update `LOCAL_DEVELOPMENT.md` (if exists) with consolidated configuration approach
- [ ] Update `README_LOCAL.md` (if exists) with configuration details
- [ ] Create configuration migration guide if needed (for existing setups)
- [ ] Document configuration troubleshooting steps

**Documentation Locations**:
- Check `@docs/initiatives/rag_system/worker_queue_conversion/001_initial_conversion/intermediate/LOCAL_DEVELOPMENT.md`
- Check project root for `README.md` or similar
- Create new documentation in `notes/deployment/` if needed

**Reference**: 
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Section 2.3 for documentation requirements

**Success Criteria**:
- ✅ Existing documentation updated
- ✅ Configuration approach clearly explained
- ✅ Troubleshooting steps documented
- ✅ Migration guide created if needed

**Document**: 
- Record documentation updates in `intermediate/phase_2_decisions.md`
- Note any documentation that needs to be created

---

## Task 2.5: Test Configuration Loading

**Execute**: Test configuration loading in local and cloud environments

**Required Tests**:
- [ ] Test configuration loading in local environment
  - [ ] Verify `.env.local` loads correctly
  - [ ] Verify `local.settings.json` doesn't override `.env.local` for app config
  - [ ] Verify Azurite connection string from `local.settings.json` works
- [ ] Test configuration loading in Azure environment (if staging available)
  - [ ] Verify Azure Function App settings are used
  - [ ] Verify environment variables are accessible
  - [ ] Verify configuration precedence works correctly
- [ ] Verify all required variables are accessible in both environments

**Test Approach**:
- Run functions locally and verify configuration loads
- Check function logs for configuration issues
- Test with missing variables to verify error handling
- Test with Azure settings if staging available

**Success Criteria**:
- ✅ Configuration loads correctly in local environment
- ✅ Configuration loads correctly in Azure (if tested)
- ✅ Configuration precedence works as documented
- ✅ All required variables are accessible

**Document**: 
- Record test results in `intermediate/phase_2_testing.md`
- Document any configuration loading issues
- Note any fixes applied

---

## Documentation Requirements

**You MUST**:
1. **Record Decisions**: Create `intermediate/phase_2_decisions.md` with:
   - Configuration precedence decisions
   - Configuration file standardization decisions
   - Validation script design decisions
   - Documentation update decisions

2. **Document Testing**: Create `intermediate/phase_2_testing.md` with:
   - Configuration loading test results (local)
   - Configuration loading test results (Azure, if tested)
   - Validation script test results
   - Any configuration issues found and resolved

3. **Create Handoff**: Create `intermediate/phase_2_handoff.md` with:
   - Summary of configuration consolidation
   - Configuration strategy documentation status
   - Configuration files standardization status
   - Prerequisites verified for Phase 3
   - Any blockers or concerns for Phase 3

4. **Update Tracking**: 
   - Mark Phase 2 tasks complete in `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md`
   - Check off all [ ] checkboxes in Section 2.1, 2.2, 2.3, and 2.4 as tasks are completed

5. **Update Fracas**: 
   - Document any new configuration issues discovered in `fracas.md`
   - Update status of any existing issues if resolved

---

## Success Criteria

### Must Pass (Blocking) - All Required:
- [ ] Configuration strategy documented
- [ ] Configuration files standardized
- [ ] Configuration loading works in local environment
- [ ] Documentation updated

**If any "Must Pass" criteria fails, document in `intermediate/phase_2_decisions.md` and `fracas.md`, then address before proceeding.**

---

## Next Steps

**If all Phase 2 tasks pass**:
- Proceed to Phase 3: Test Infrastructure Consolidation
- Reference: `prompt_phase_3_002.md`

**If Phase 2 tasks fail**:
- Document all issues in `intermediate/phase_2_decisions.md` and `fracas.md`
- Address blockers before proceeding
- Re-test after fixes

---

## Quick Reference

**Key Files**:
- `.env.local` - Application configuration (project root)
- `infra/azure/azure_functions/local.settings.json` - Azurite/runtime settings
- Function entry points - Configuration loading implementation
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/TODO002.md` - Task tracking
- `@docs/initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/scoping/RFC002.md` - Technical design

**Documentation to Create**:
- `notes/deployment/configuration_guide.md` - Configuration guide
- `notes/deployment/azure_function_app_settings.md` - Azure settings documentation
- `intermediate/phase_2_decisions.md` - Decisions made
- `intermediate/phase_2_testing.md` - Test results
- `intermediate/phase_2_handoff.md` - Handoff to Phase 3

---

**Begin execution now. Good luck!**

