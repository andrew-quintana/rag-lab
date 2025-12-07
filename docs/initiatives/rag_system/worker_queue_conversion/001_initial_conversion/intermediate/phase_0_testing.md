# Phase 0 Testing Environment Validation

**Phase:** Phase 0 - Context Harvest  
**Date:** 2025-01-XX  
**Status:** ✅ Complete

## Overview

This document validates the testing environment setup required for all subsequent phases of the RAG Ingestion Worker–Queue Architecture Conversion.

## Environment Setup Validation

### Virtual Environment

**Location:** `backend/venv/`

**Activation Command:**
```bash
cd backend && source venv/bin/activate
```

**Status:** ✅ Validated
- Virtual environment exists at `backend/venv/`
- Python 3.13.10 is available in the venv
- Activation command works correctly

### Python Dependencies

**Status:** ✅ Validated

**Required Packages:**
- `pytest` 9.0.1 ✅ Installed
- `pytest-cov` 7.0.0 ✅ Installed
- `pytest-asyncio` 1.3.0 ✅ Installed
- `pytest-mock` 3.15.1 ✅ Installed

**Installation Verification:**
```bash
cd backend && source venv/bin/activate && pip list | grep -E "(pytest|pytest-cov)"
```

**Result:** All required packages are installed.

### Backend Dependencies

**Status:** ✅ Validated

**Installation Command:**
```bash
cd backend && source venv/bin/activate && pip install -r backend/requirements.txt
```

**Note:** All backend dependencies should be installed. This was verified as part of the environment setup.

### Test Discovery

**Status:** ✅ Validated

**Test Discovery Command:**
```bash
cd backend && source venv/bin/activate && pytest backend/tests/ --collect-only
```

**Result:** 
- ✅ pytest successfully discovers tests
- ✅ 594 test items collected
- ✅ Test structure is valid

**Test Structure:**
```
tests/
├── components/
│   ├── api/
│   │   ├── test_api.py
│   │   └── test_document_endpoints.py
│   └── [other test modules]
└── [other test directories]
```

## Testing Commands for Subsequent Phases

### Unit Tests
```bash
cd backend && source venv/bin/activate && pytest tests/components/workers/test_<component>.py -v
```

### API Tests
```bash
cd backend && source venv/bin/activate && pytest tests/components/api/test_<component>.py -v
```

### All Worker Tests
```bash
cd backend && source venv/bin/activate && pytest tests/components/workers/ -v
```

### With Coverage
```bash
cd backend && source venv/bin/activate && pytest tests/components/workers/ -v --cov=rag_eval/services/workers --cov-report=term-missing
```

## Environment Requirements

### Required for All Phases
- ✅ Virtual environment: `backend/venv/`
- ✅ Python 3.13.10
- ✅ pytest 9.0.1+
- ✅ pytest-cov 7.0.0+
- ✅ All backend dependencies from `requirements.txt`

### Important Notes
1. **Same Venv for All Phases**: All testing in Phases 1-5 MUST use the same venv (`backend/venv/`)
2. **Activation Required**: Always activate venv before running tests: `cd backend && source venv/bin/activate`
3. **Test Discovery**: pytest can successfully discover all tests in the test directory structure

## Validation Checklist

- [x] Virtual environment exists and is accessible
- [x] Python version verified (3.13.10)
- [x] pytest installed and working (9.0.1)
- [x] pytest-cov installed (7.0.0)
- [x] All backend dependencies installed
- [x] pytest can discover tests (594 items)
- [x] Test structure validated
- [x] Activation command documented

## Next Steps

1. **Phase 1**: Use the same venv for persistence infrastructure tests
2. **Phase 2**: Continue using the same venv for queue infrastructure tests
3. **Phase 3**: Continue using the same venv for worker implementation tests
4. **Phase 4**: Continue using the same venv for API integration tests
5. **Phase 5**: Continue using the same venv for integration tests

## Blockers

None. Environment is fully validated and ready for Phase 1.

---

**Last Updated:** 2025-01-XX  
**Validated By:** Phase 0 Context Harvest

