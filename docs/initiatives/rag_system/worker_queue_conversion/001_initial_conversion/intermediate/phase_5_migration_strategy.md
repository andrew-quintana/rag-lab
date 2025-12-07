# Phase 5: Migration Strategy

## Overview

This document outlines the migration strategy for transitioning from synchronous to asynchronous worker-queue architecture. The migration follows a gradual approach to minimize risk and allow validation before full cutover.

## Migration Principles

1. **Gradual Migration**: Run both paths in parallel during transition
2. **No Breaking Changes**: Maintain backward compatibility during migration
3. **Validation First**: Validate worker behavior before deprecating synchronous path
4. **Observability**: Monitor both paths during migration period
5. **Rollback Capability**: Ability to revert to synchronous path if needed

## Migration Phases

### Phase 1: Parallel Operation (Week 1-2)

**Objective**: Deploy workers and queues without turning off synchronous path

**Actions**:
1. Apply database migrations (0019, 0020)
2. Deploy Azure Functions with all workers
3. Configure queues and triggers
4. Set up Application Insights monitoring
5. Keep synchronous upload endpoint active

**Validation**:
- Verify workers can process messages
- Verify queues are functioning
- Monitor Application Insights for errors
- Test with small number of documents

**Success Criteria**:
- [ ] Workers deployed and processing messages
- [ ] No errors in Application Insights
- [ ] Synchronous path still functional
- [ ] Both paths can process documents independently

### Phase 2: Gradual Traffic Shift (Week 3-4)

**Objective**: Gradually shift new uploads to asynchronous path

**Actions**:
1. Add feature flag to control path selection
2. Start routing small percentage (10-20%) of new uploads to queue
3. Monitor worker performance and error rates
4. Gradually increase percentage (50%, 80%, 100%)
5. Update UI/API endpoints to show document status

**Feature Flag Example**:
```python
# In upload endpoint
USE_ASYNC_PATH = os.getenv("USE_ASYNC_PATH", "false").lower() == "true"
ASYNC_PATH_PERCENTAGE = float(os.getenv("ASYNC_PATH_PERCENTAGE", "0"))

if USE_ASYNC_PATH and random.random() < ASYNC_PATH_PERCENTAGE:
    # Enqueue to async path
    enqueue_message("ingestion-uploads", message, config)
else:
    # Use synchronous path
    process_synchronously(file, config)
```

**Validation**:
- Monitor error rates for both paths
- Compare processing times
- Verify status tracking works correctly
- Test status query endpoint

**Success Criteria**:
- [ ] 100% of new uploads use async path
- [ ] Error rates acceptable (< 1%)
- [ ] Status tracking accurate
- [ ] UI shows document status correctly

### Phase 3: Status-Based UI Updates (Week 5)

**Objective**: Update UI/API to rely on document status

**Actions**:
1. Update frontend to poll document status
2. Show processing progress through pipeline stages
3. Display error messages for failed stages
4. Add retry functionality for failed documents
5. Remove dependency on synchronous response

**Validation**:
- Test status polling works correctly
- Verify UI updates reflect actual status
- Test error handling and display
- Verify retry functionality

**Success Criteria**:
- [ ] UI shows document status correctly
- [ ] Users can track processing progress
- [ ] Error messages displayed appropriately
- [ ] Retry functionality works

### Phase 4: Deprecation (Week 6-7)

**Objective**: Deprecate synchronous path once stable

**Actions**:
1. Monitor async path for 1-2 weeks with 100% traffic
2. Verify stability and performance
3. Remove or deprecate synchronous processing code
4. Update documentation
5. Remove feature flags

**Validation**:
- Monitor error rates and performance
- Verify no regressions
- Test complete pipeline end-to-end
- Review Application Insights metrics

**Success Criteria**:
- [ ] Async path stable for 1-2 weeks
- [ ] Error rates acceptable
- [ ] Performance meets requirements
- [ ] Synchronous path removed or deprecated
- [ ] Documentation updated

## Monitoring During Migration

### Key Metrics to Monitor

1. **Error Rates**
   - Target: < 1% failure rate
   - Monitor per worker and overall pipeline
   - Alert on spikes

2. **Processing Time**
   - Target: < 30 seconds per worker
   - Target: < 5 minutes end-to-end
   - Compare async vs sync times

3. **Queue Depth**
   - Target: Average < 10 messages per queue
   - Alert if depth > 50

4. **Status Accuracy**
   - Verify status transitions are correct
   - Monitor for stuck documents

5. **Throughput**
   - Verify constant throughput despite batch constraints
   - Monitor documents processed per hour

### Application Insights Queries

```kusto
// Error rate by worker
traces
| where severityLevel >= 3
| summarize count() by operation_Name, bin(timestamp, 1h)
| order by timestamp desc

// Processing time by worker
traces
| where operation_Name contains "worker"
| summarize avg(duration), max(duration) by operation_Name

// Queue depth over time
customMetrics
| where name == "QueueDepth"
| summarize avg(value) by bin(timestamp, 1h)
```

## Rollback Plan

If issues arise during migration:

1. **Immediate Rollback**: Set feature flag to route 100% traffic to synchronous path
2. **Investigate**: Review Application Insights logs and metrics
3. **Fix Issues**: Address root cause
4. **Re-test**: Validate fixes before resuming migration
5. **Resume**: Gradually shift traffic back to async path

## Risk Mitigation

### Risk: Worker Failures
- **Mitigation**: Dead-letter queue for failed messages
- **Mitigation**: Retry logic with exponential backoff
- **Mitigation**: Monitor error rates and alert on spikes

### Risk: Performance Degradation
- **Mitigation**: Monitor processing times
- **Mitigation**: Scale workers based on queue depth
- **Mitigation**: Optimize worker code if needed

### Risk: Data Loss
- **Mitigation**: Idempotent workers prevent duplicate processing
- **Mitigation**: Status tracking ensures consistency
- **Mitigation**: Database transactions ensure atomicity

### Risk: Queue Backlog
- **Mitigation**: Auto-scaling based on queue depth
- **Mitigation**: Monitor queue depth and alert
- **Mitigation**: Manual scaling if needed

## Timeline Summary

| Week | Phase | Activities | Success Criteria |
|------|-------|-----------|------------------|
| 1-2 | Parallel Operation | Deploy workers, keep sync path | Workers processing, no errors |
| 3-4 | Gradual Traffic Shift | Route 10% → 100% to async | 100% async, error rates acceptable |
| 5 | Status-Based UI | Update UI for status tracking | UI shows status correctly |
| 6-7 | Deprecation | Remove sync path, update docs | Stable for 1-2 weeks, sync removed |

## Post-Migration

After successful migration:

1. **Documentation**: Update all documentation to reflect async architecture
2. **Training**: Train team on new architecture and monitoring
3. **Optimization**: Optimize worker performance based on metrics
4. **Scaling**: Configure auto-scaling based on observed patterns
5. **Monitoring**: Set up alerts and dashboards for ongoing monitoring

## Success Criteria

Migration is complete when:

- [ ] 100% of new uploads use async path
- [ ] Synchronous path deprecated or removed
- [ ] Error rates < 1%
- [ ] Processing times meet requirements
- [ ] UI shows document status correctly
- [ ] Documentation updated
- [ ] Team trained on new architecture
- [ ] Monitoring and alerts configured

