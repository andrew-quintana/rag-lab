# Phase 5 Prompt — Integration Testing & Migration

## Context

This prompt guides the implementation of **Phase 5: Integration Testing & Migration** for the RAG Ingestion Worker–Queue Architecture Conversion. This phase involves deploying workers as Azure Functions, running integration tests with real Azure resources, and executing the migration strategy.

**Related Documents:**
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/PRD001.md - Product requirements
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/RFC001.md - Technical design (Integration Testing & Migration section)
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md - Implementation tasks (Phase 5 section - check off tasks as completed)
- @docs/initiatives/rag_system/worker_queue_conversion/scoping/context.md - Project context

## Objectives

1. **Azure Functions Deployment**: Deploy all workers as Azure Functions with queue triggers
2. **Integration Testing**: Run end-to-end tests with real Azure Storage Queues
3. **Performance Testing**: Validate throughput and latency requirements
4. **Migration Strategy**: Execute gradual migration from synchronous to asynchronous path

## Execution Requirements

### Checklist Modifications
- **REQUIRED**: Check off completed tasks in @docs/initiatives/rag_system/worker_queue_conversion/scoping/TODO001.md Phase 5 section as you complete them
- Update phase status when all validation requirements are met

### Validation
- **REQUIRED**: Azure Functions deployed and configured
- **REQUIRED**: All integration tests pass with real Azure Storage Queues
- **REQUIRED**: Performance tests validate throughput requirements
- **REQUIRED**: Migration strategy executed successfully

### Documentation
- **REQUIRED**: Create `phase_5_decisions.md` if any decisions are made that aren't in PRD/RFC
- **REQUIRED**: Create `phase_5_testing.md` documenting testing results
- **REQUIRED**: Create `phase_5_handoff.md` summarizing completion and any remaining work

## Key References

### Azure Resources
- Azure Function App (Consumption Plan)
- Azure Storage Account and Queues
- Azure Application Insights
- Azure Key Vault

### Workers
- @backend/rag_eval/services/workers/ingestion_worker.py
- @backend/rag_eval/services/workers/chunking_worker.py
- @backend/rag_eval/services/workers/embedding_worker.py
- @backend/rag_eval/services/workers/indexing_worker.py

### Test Data Constraint
- **IMPORTANT**: Tests that use actual PDF pages must use only the first 6 pages of `docs/inputs/scan_classic_hmo.pdf` to avoid exceeding Azure Document Intelligence budget

## Phase 5 Tasks

### Azure Functions Deployment
1. Create Azure Function App (Consumption Plan)
2. Deploy all workers as Azure Functions:
   - `ingestion-worker` function with queue trigger for `ingestion-uploads`
   - `chunking-worker` function with queue trigger for `ingestion-chunking`
   - `embedding-worker` function with queue trigger for `ingestion-embeddings`
   - `indexing-worker` function with queue trigger for `ingestion-indexing`
3. Configure queue triggers for each worker
4. Set up Application Insights for monitoring
5. Configure environment variables and Key Vault integration
6. Test Azure Functions deployment and queue trigger configuration

### Integration Tests (Post-Deployment)
1. **REQUIRED**: End-to-end pipeline flow with real Azure Storage Queues:
   - Test message passing between stages through actual queues
   - Test failure scenarios and dead-letter handling with real queues
   - Test status transitions through complete pipeline
   - Test concurrent document processing across multiple workers
   - Test queue depth handling under load
   - Test Azure Functions queue trigger behavior
   - Test worker scaling and concurrency
   - **Test Data Constraint**: Use only first 6 pages of `docs/inputs/scan_classic_hmo.pdf` for tests that process actual PDFs
2. **Document any failures** in fracas.md immediately when encountered

### Performance Testing (Post-Deployment)
1. **REQUIRED**: Test worker processing time under load (real Azure Functions)
2. **REQUIRED**: Test queue depth handling with real Azure Storage Queues
3. **REQUIRED**: Test concurrent document processing across multiple function instances
4. **REQUIRED**: Validate throughput meets requirements
5. **REQUIRED**: Monitor Azure Functions cold start latency
6. **REQUIRED**: **Test Data Constraint**: Performance tests should limit to first 6 pages of test PDF to stay within budget

### Migration Strategy
1. **REQUIRED**: Gradual migration: run both paths in parallel
   - Introduce queues and workers without turning off synchronous path
   - For new uploads, prefer enqueuing message instead of synchronous processing
   - Gradually move UI/API endpoints to rely on document `status`
   - Monitor and validate worker behavior via Application Insights
2. **REQUIRED**: Deprecate synchronous path once stable
   - Remove or deprecate direct "do everything in one request" ingestion paths
   - Update documentation to reflect asynchronous architecture

### Documentation
1. Document Azure Functions deployment process
2. Document queue trigger configuration
3. Document Application Insights setup and monitoring
4. Document migration strategy and timeline
5. Document performance test results

## Success Criteria

- [ ] Azure Functions deployed and configured
- [ ] All integration tests pass with real Azure Storage Queues
- [ ] Performance tests validate throughput requirements
- [ ] Migration strategy executed successfully
- [ ] Synchronous path deprecated or removed
- [ ] All Phase 5 tasks in TODO001.md checked off
- [ ] Phase 5 handoff document created

## Important Notes

- **Test Data Constraint**: Critical budget constraint - use only first 6 pages of test PDF for any tests that process actual PDFs
- **Integration Tests**: Must be run post-deployment with real Azure resources
- **Performance Tests**: Must be run post-deployment with real Azure Functions
- **Migration Strategy**: Gradual migration allows validation before full cutover
- **Monitoring**: Application Insights provides observability for worker behavior

## Blockers

- **BLOCKER**: Azure Functions must be deployed before integration testing
- **BLOCKER**: All previous phases must be complete before deployment

## Next Steps

After completing Phase 5, proceed to **Initiative Completion** tasks:
- Final validation of all tests
- Technical debt documentation
- Summary document creation
- Stakeholder review and approval

