# Phase 0 Testing — In-Corpus Evaluation Dataset Generation System

**Phase**: Phase 0 - Context Harvest  
**Date**: 2025-01-XX  
**Status**: Complete

## Overview

This document documents the testing environment setup validation performed during Phase 0.

## Testing Environment Setup

### Virtual Environment

**Status**: ✅ Validated

**Details**:
- Virtual environment location: `backend/venv/`
- Python version: 3.13.7
- Activation command: `cd backend && source venv/bin/activate`

**Validation Steps**:
1. Verified virtual environment exists
2. Activated virtual environment
3. Confirmed Python version

### Test Framework

**Status**: ✅ Validated

**Details**:
- Framework: pytest 9.0.1
- Coverage tool: pytest-cov 7.0.0
- Additional plugins: pytest-asyncio, pytest-mock

**Validation Steps**:
1. Verified pytest is installed
2. Verified pytest-cov is installed
3. Ran `pytest tests/ --collect-only` to verify test discovery
4. Confirmed tests can be discovered successfully

**Test Discovery Results**:
- Tests discovered successfully
- Multiple test modules found:
  - `tests/components/api/test_api.py`
  - `tests/components/api/test_document_endpoints.py`
  - `tests/components/api/test_query_endpoint.py`
  - And others

### Backend Dependencies

**Status**: ✅ Validated

**Details**:
- Dependencies file: `backend/requirements.txt`
- Installation: All dependencies installed successfully

**Validation Steps**:
1. Verified `requirements.txt` exists
2. Ran `pip install -r requirements.txt`
3. Confirmed all dependencies installed without errors

**Key Dependencies**:
- FastAPI and server components
- Database (psycopg2-binary)
- Azure SDKs (AI Inference, Document Intelligence, Search, Blob Storage)
- Utilities (python-dotenv, requests)
- Supabase client
- LangGraph for orchestration
- Scientific computing (scipy)

## Integration Validation (Deferred)

The following validations are deferred to Phase 1 when actual implementation begins:

### Azure AI Search

**Status**: ⏳ Deferred to Phase 1

**Planned Validation**:
- Verify index exists and is accessible
- Verify credentials are configured
- Test basic query functionality

### Supabase Database

**Status**: ⏳ Deferred to Phase 1

**Planned Validation**:
- Verify database connection
- Verify prompts table structure
- Test prompt retrieval functionality

### RAG Pipeline Components

**Status**: ⏳ Deferred to Phase 1

**Planned Validation**:
- Verify RAG pipeline components are accessible
- Test retrieval functionality
- Test generation functionality

### Evaluation System Components

**Status**: ⏳ Deferred to Phase 1

**Planned Validation**:
- Verify evaluation system components are accessible
- Test BEIR metrics computation
- Test evaluation pipeline

## Test Coverage Requirements

**Status**: ⏳ Not Applicable (Phase 0)

**Note**: Test coverage requirements will be enforced starting in Phase 1:
- Minimum 80% coverage for each module
- Unit tests for all key functions
- Integration tests with mocked dependencies

## Testing Commands

### Activate Virtual Environment
```bash
cd backend && source venv/bin/activate
```

### Run Tests
```bash
# Discover tests
pytest tests/ --collect-only

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=rag_eval --cov-report=html
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Environment Configuration

**Configuration Source**: Environment variables loaded from `.env.local`

**Key Configuration Areas**:
- Database (Supabase Postgres)
- Azure AI Foundry (endpoint, API key, models)
- Azure AI Search (endpoint, API key, index name)
- Azure Document Intelligence
- Azure Blob Storage

**Note**: Configuration validation will be performed during Phase 1 when actual integration testing begins.

## Summary

✅ **Testing environment is properly configured and validated**

All Phase 0 testing requirements have been met:
- Virtual environment exists and is functional
- pytest is installed and can discover tests
- All backend dependencies are installed
- Test discovery works correctly

**Next Steps**: Proceed to Phase 1 with confidence that the testing environment is ready.

---

**Next Phase**: Phase 1 - Query Generator (AI Node)

