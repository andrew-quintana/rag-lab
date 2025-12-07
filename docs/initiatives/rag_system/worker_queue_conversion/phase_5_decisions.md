# Phase 5 Decisions

## Overview

This document captures key decisions made during Phase 5: Integration Testing & Migration implementation that are not already documented in PRD001.md or RFC001.md.

## Decision 1: Azure Functions Deployment Structure

**Decision**: Deploy all workers as separate functions within a single Function App

**Rationale**:
- Single Function App simplifies deployment and configuration
- Shared environment variables and Application Insights
- Easier monitoring and management
- Cost-effective (Consumption Plan)

**Alternatives Considered**:
- Separate Function Apps per worker (more complex, higher cost)
- Single function with routing (less modular, harder to scale independently)

**Status**: ✅ Implemented

## Decision 2: Test Data Constraint

**Decision**: Limit integration and performance tests to first 6 pages of test PDF

**Rationale**:
- Azure Document Intelligence free-tier budget constraints
- Tests should be representative but cost-effective
- 6 pages provides sufficient coverage (3 batches of 2 pages)

**Implementation**:
- Use `slice_pdf_to_batch()` to extract first 6 pages
- Document constraint in all test files
- Skip tests if PDF not available

**Status**: ✅ Implemented

## Decision 3: Integration Test Approach

**Decision**: Create comprehensive integration tests that can run with or without deployed Azure Functions

**Rationale**:
- Some tests require Azure Functions to be deployed
- Some tests can validate database/queue operations independently
- Allows incremental testing during deployment

**Implementation**:
- Mark tests requiring Azure Functions with `@pytest.mark.skip`
- Provide clear instructions for running post-deployment
- Separate tests for database operations vs. full pipeline

**Status**: ✅ Implemented

## Decision 4: Migration Strategy

**Decision**: Gradual migration with feature flags and parallel operation

**Rationale**:
- Minimizes risk of breaking production
- Allows validation before full cutover
- Enables rollback if issues arise
- Provides time for team to adapt

**Implementation**:
- Feature flag to control path selection
- Percentage-based traffic shifting
- Monitor both paths during transition
- Documented in `phase_5_migration_strategy.md`

**Status**: ✅ Documented

## Decision 5: Verification Scripts

**Decision**: Create standalone verification scripts for migrations and deployment

**Rationale**:
- Allows verification without running full test suite
- Useful for deployment validation
- Can be run by operations team
- Provides clear pass/fail status

**Implementation**:
- `verify_phase5_migrations.py` for database migrations
- Clear output with pass/fail indicators
- Can be run independently

**Status**: ✅ Implemented

## Decision 6: Azure Functions Entry Points

**Decision**: Create thin wrapper functions in Azure Functions that call worker functions

**Rationale**:
- Keeps worker logic separate from Azure Functions runtime
- Allows workers to be tested independently
- Easier to maintain and update
- Supports local testing

**Implementation**:
- Each function has `__init__.py` that parses queue message and calls worker
- Workers remain in `backend/rag_eval/services/workers/`
- Functions add backend to Python path

**Status**: ✅ Implemented

## Decision 7: Performance Test Scope

**Decision**: Focus performance tests on key metrics with budget constraints

**Rationale**:
- Limited by Azure Document Intelligence budget
- Focus on critical metrics (processing time, throughput, queue depth)
- Use realistic but constrained test data

**Implementation**:
- Limit to 3-5 documents per performance test
- Focus on worker processing time and queue depth
- Skip tests requiring Azure Functions (documented for post-deployment)

**Status**: ✅ Implemented

## Decision 8: Documentation Structure

**Decision**: Create separate documentation files for each major aspect of Phase 5

**Rationale**:
- Clear separation of concerns
- Easier to maintain and update
- Better organization for different audiences

**Files Created**:
- `phase_5_migration_guide.md` - Database migration instructions
- `phase_5_azure_functions_deployment.md` - Deployment guide
- `phase_5_migration_strategy.md` - Migration execution plan
- `phase_5_decisions.md` - This document
- `phase_5_testing.md` - Testing documentation
- `phase_5_handoff.md` - Completion summary

**Status**: ✅ Implemented

## Decision 9: Supabase Integration Tests

**Decision**: Create dedicated integration tests for Supabase operations

**Rationale**:
- Workers depend heavily on Supabase for persistence
- Need to validate real database operations
- Separate from Azure Functions tests
- Can run independently

**Implementation**:
- `test_supabase_phase5.py` with comprehensive database tests
- Tests use real database connections
- Clean up after tests
- Marked with `@pytest.mark.integration`

**Status**: ✅ Implemented

## Decision 10: Queue Trigger Configuration

**Decision**: Use Azure Storage Queue triggers with connection string from environment

**Rationale**:
- Standard Azure Functions pattern
- Simple configuration
- Works with Consumption Plan
- Easy to test locally

**Implementation**:
- `function.json` files configure queue triggers
- Connection string from `AzureWebJobsStorage` setting
- Queue names match RFC specification

**Status**: ✅ Implemented

## Open Questions / Future Decisions

1. **Key Vault Integration**: Should all secrets be moved to Key Vault? (Recommended for production)
2. **Auto-Scaling Configuration**: What are optimal scaling parameters? (To be determined post-deployment)
3. **Dead-Letter Queue Handling**: How should dead-letter messages be handled? (Manual inspection initially)
4. **Monitoring Dashboards**: What custom dashboards are needed? (To be created based on metrics)

## Notes

- All decisions align with PRD001.md and RFC001.md requirements
- Decisions prioritize simplicity and maintainability
- Budget constraints are respected throughout
- Migration strategy prioritizes safety and validation

