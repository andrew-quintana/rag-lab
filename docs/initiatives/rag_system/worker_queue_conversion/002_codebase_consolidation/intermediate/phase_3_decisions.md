# Phase 3 Decisions - Test Infrastructure Consolidation

**Date**: 2025-12-09  
**Phase**: 3 - Test Infrastructure Consolidation  
**Status**: ✅ Complete

## Summary

Phase 3 successfully consolidated test infrastructure by registering pytest markers, consolidating test fixtures, and creating unified test execution scripts. This establishes a consistent, maintainable test infrastructure for both local and cloud testing.

---

## Task 3.1: Pytest Marker Registration

### Decision: Register Markers in pyproject.toml

**Approach**: Registered all pytest markers in `backend/pyproject.toml` under `[tool.pytest.ini_options]`.

**Rationale**:
- `pyproject.toml` already contains pytest configuration
- Centralized configuration location
- Consistent with existing project structure
- No need for separate `pytest.ini` file

### Markers Registered

1. **`integration`** - Already registered
   - Marks tests as integration tests requiring real resources
   - Description: "marks tests as integration tests (deselect with '-m \"not integration\"')"

2. **`local`** - ✅ Newly registered
   - Marks tests that run against local resources (Azurite, local Supabase)
   - Description: "marks tests that run against local resources (Azurite, local Supabase)"

3. **`cloud`** - ✅ Newly registered
   - Marks tests that run against cloud resources (Azure Functions, cloud Supabase)
   - Description: "marks tests that run against cloud resources (Azure Functions, cloud Supabase)"

4. **`performance`** - ✅ Newly registered
   - Marks tests as performance tests (throughput, latency validation)
   - Description: "marks tests as performance tests (throughput, latency validation)"

### Files Modified

- `backend/pyproject.toml` - Added `local`, `cloud`, and `performance` markers to `[tool.pytest.ini_options]` section

### Verification

- Markers are registered in pytest configuration
- No pytest warnings about unregistered markers expected when running tests
- Markers are documented with clear descriptions

---

## Task 3.2: Test Fixture Consolidation

### Decision: Create Shared conftest.py for Integration Test Fixtures

**Approach**: Created `backend/tests/conftest.py` to consolidate common integration test fixtures.

**Rationale**:
- Eliminates duplication of `config`, `db_conn`, and `is_local` fixtures across multiple test files
- Provides single source of truth for shared fixtures
- Maintains consistency across integration tests
- Unit test fixtures (like `mock_config`) remain in individual test files since they may have test-specific configurations

### Fixtures Consolidated

1. **`config`** (module-scoped)
   - Loads configuration from environment using `Config.from_env()`
   - Previously duplicated in:
     - `test_phase5_e2e_pipeline.py`
     - `test_supabase_phase5.py`
     - `test_phase5_performance.py`

2. **`db_conn`** (module-scoped)
   - Creates database connection using config fixture
   - Skips tests if `DATABASE_URL` not set
   - Previously duplicated in:
     - `test_phase5_e2e_pipeline.py`
     - `test_supabase_phase5.py`

3. **`is_local`** (module-scoped)
   - Determines if running in local development mode (Azurite)
   - Uses helper function `_is_local_development()`
   - Previously duplicated in `test_phase5_e2e_pipeline.py`

4. **Helper Function: `_is_local_development()`**
   - Checks if connection string is `UseDevelopmentStorage=true`
   - Moved from `test_phase5_e2e_pipeline.py` to `conftest.py`

### Fixtures NOT Consolidated

**Rationale**: Unit test fixtures remain in individual test files because:
- They may have test-specific mock configurations
- They're used only within specific test modules
- Consolidating them would reduce flexibility

Examples:
- `mock_config` fixtures in unit test files
- Test-specific fixtures like `test_pdf_path`, `test_document_id`

### Files Created

- `backend/tests/conftest.py` - Shared fixtures for integration tests

### Issue Discovered and Fixed

**Issue**: Initial implementation had `DatabaseConnection` imported at module level, causing import errors when psycopg2 wasn't available or when using system Python instead of venv.

**Fix Applied**: Changed to lazy import inside `db_conn` fixture:
```python
@pytest.fixture(scope="module")
def db_conn(config):
    if not config.database_url:
        pytest.skip("DATABASE_URL not set - skipping integration tests")
    # Import here to avoid import errors when psycopg2 is not available
    from src.db.connection import DatabaseConnection
    return DatabaseConnection(config)
```

**Learning**: Use lazy imports for optional dependencies in conftest.py to avoid breaking test collection.

### Files Updated

- `backend/tests/integration/test_phase5_e2e_pipeline.py`
  - Removed: `config`, `db_conn`, `is_local` fixtures, `_is_local_development()` function
  - Now uses: Fixtures from `conftest.py`

- `backend/tests/integration/test_supabase_phase5.py`
  - Removed: `config`, `db_conn` fixtures
  - Now uses: Fixtures from `conftest.py`

- `backend/tests/integration/test_phase5_performance.py`
  - Removed: `config` fixture
  - Now uses: Fixtures from `conftest.py`

- `backend/tests/components/evaluator/test_prompt_database_integration.py`
  - Updated: `query_executor` fixture now uses `db_conn` from `conftest.py` instead of creating its own

### Verification

- All test files updated to use consolidated fixtures
- No duplicate fixture code remains in integration test files
- Tests should still pass with consolidated fixtures (to be verified in Task 3.4)

---

## Task 3.3: Unified Test Scripts

### Decision: Create/Update Three Test Execution Scripts

**Approach**: Created unified test execution scripts that use pytest markers correctly and provide clear output.

### Scripts Created/Updated

#### 1. `scripts/test_functions_local.sh` - ✅ Updated

**Purpose**: Run local tests against Azurite and local Supabase

**Features**:
- Checks prerequisites (Supabase, Azurite, `.env.local`, `local.settings.json`)
- **NEW**: Automatically starts Azurite if not running
- Verifies queues exist in Azurite
- Runs tests with `@pytest.mark.local and @pytest.mark.integration` markers
- **UPDATED**: Now runs all integration tests with local marker (not just `test_phase5_e2e_pipeline.py`)
- Provides clear output and exit codes

**Changes from Previous Version**:
- Added automatic Azurite startup if not running
- Changed to run all integration tests with local marker (broader scope)
- Improved error messages and output formatting

#### 2. `scripts/test_functions_cloud.sh` - ✅ Created

**Purpose**: Run cloud tests against Azure Functions and cloud Supabase

**Features**:
- Verifies Azure Functions deployment (if Azure CLI available)
- Checks cloud resource accessibility
- Validates required environment variables are set
- Runs tests with `@pytest.mark.cloud and @pytest.mark.integration` markers
- Provides clear output and exit codes
- Handles missing environment gracefully

**New Script**: Created from scratch following RFC002 Section 3.3 requirements

#### 3. `scripts/test_functions_all.sh` - ✅ Created

**Purpose**: Run both local and cloud tests with comprehensive summary

**Features**:
- Runs local tests first
- Runs cloud tests second (if environment configured)
- Provides summary of all test results
- Handles missing cloud environment gracefully (skips cloud tests if not configured)
- Returns appropriate exit codes based on test results

**New Script**: Created from scratch following RFC002 Section 3.3 requirements

### Script Design Decisions

1. **Marker Usage**: Scripts use `-m "local and integration"` and `-m "cloud and integration"` to ensure tests are both integration tests and appropriately marked for environment

2. **Prerequisite Checking**: Scripts check prerequisites before running tests to provide clear error messages

3. **Graceful Degradation**: Cloud scripts handle missing environment variables gracefully

4. **Exit Codes**: All scripts return appropriate exit codes for CI/CD integration

5. **Output Formatting**: Clear section headers and status messages for easy reading

### Files Created

- `scripts/test_functions_cloud.sh` - Cloud test execution script
- `scripts/test_functions_all.sh` - Comprehensive test execution script

### Files Updated

- `scripts/test_functions_local.sh` - Enhanced with automatic Azurite startup and broader test scope

### Verification

- All scripts are executable (chmod +x applied)
- Scripts use pytest markers correctly
- Scripts provide clear output
- Scripts are documented with comments

---

## Test Infrastructure Changes Summary

### Configuration Changes

1. **Pytest Configuration** (`backend/pyproject.toml`)
   - Added 3 new markers: `local`, `cloud`, `performance`
   - All markers now registered and documented

### Code Changes

1. **Shared Fixtures** (`backend/tests/conftest.py`)
   - Created new file with consolidated fixtures
   - 3 fixtures consolidated: `config`, `db_conn`, `is_local`
   - 1 helper function consolidated: `_is_local_development()`

2. **Test Files Updated**
   - 4 integration test files updated to use consolidated fixtures
   - Removed duplicate fixture code
   - Maintained test-specific fixtures where appropriate

### Script Changes

1. **Test Execution Scripts**
   - 1 script updated: `test_functions_local.sh`
   - 2 scripts created: `test_functions_cloud.sh`, `test_functions_all.sh`
   - All scripts use pytest markers correctly

---

## Decisions Not Made / Future Considerations

1. **Unit Test Fixture Consolidation**: Decided to keep unit test fixtures (like `mock_config`) in individual test files for flexibility. Could be consolidated in future if patterns emerge.

2. **Test-Specific Fixtures**: Fixtures like `test_pdf_path` and `test_document_id` remain in test files since they're specific to those tests.

3. **Performance Test Execution**: Performance tests remain skipped until cloud functions are stable (as per existing practice).

---

## Next Steps

1. **Task 3.4**: Test consolidated infrastructure with Phase 5 tests
2. **Verification**: Run tests to ensure consolidated fixtures work correctly
3. **Documentation**: Complete phase_3_testing.md and phase_3_handoff.md

---

**Last Updated**: 2025-12-09  
**Status**: ✅ Phase 3 Decisions Complete

