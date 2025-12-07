# Prompt: Phase 5.5 - Application Insights Monitoring Validation

## Your Mission

Execute Phase 5 of Phase 5 integration testing: Application Insights Monitoring Validation. You must verify that all function executions, errors, and custom metrics are properly tracked in Application Insights. Document all results, monitoring gaps, and telemetry issues as you progress.

## Context & References

**Key Documentation to Reference**:
- `phase_5_testing.md` - Testing overview and test locations
- `phase_5_azure_functions_deployment.md` - Application Insights setup
- `phase_5_handoff.md` - Phase 5 completion summary
- `fracas.md` - Document all failures immediately
- `scoping/TODO001.md` - Task checklist (mark tasks complete)

**Prerequisites** (from Phase 5.4):
- ✅ Performance tests completed
- ✅ Functions processing documents successfully
- ✅ Application Insights resource exists

## Prerequisites Check

**Before starting, verify**:
1. Application Insights resource `ai-rag-evaluator` exists
2. Application Insights connection configured in Function App
3. Access to Azure Portal for Application Insights
4. Functions have been executing (to generate telemetry)
5. Azure CLI installed and authenticated

**If any prerequisite is missing, document it in `fracas.md` and stop execution.**

---

## Task 5.1: Telemetry Verification

**Execute**:
1. Navigate to Azure Portal → Application Insights → `ai-rag-evaluator`
2. Review Application Insights dashboard
3. Check for function execution logs
4. Verify custom metrics tracked

**Verify metrics**:
- ✅ Function execution count
- ✅ Function execution duration
- ✅ Error rate
- ✅ Queue depth (if custom metrics implemented)
- ✅ Status transition events (if custom events implemented)
- ✅ Document processing events (if custom events implemented)

**Azure Portal Queries**:
```kusto
// Function execution count
traces
| where operation_Name startswith "ingestion-worker"
| summarize count() by bin(timestamp, 1h)

// Error rate
traces
| where severityLevel >= 3
| summarize count() by operation_Name, bin(timestamp, 1h)

// Processing time
traces
| where operation_Name contains "worker"
| summarize avg(duration), max(duration) by operation_Name
```

**Success Criteria**:
- ✅ All function executions logged
- ✅ Error tracking functional
- ✅ Performance metrics available
- ✅ Custom events captured (if implemented)
- ✅ Telemetry appears in real-time (within 1-2 minutes)

**Document**: 
- Screenshot Application Insights dashboard (save to appropriate location)
- Note any missing metrics in `phase_5_testing.md`
- Document monitoring gaps in `fracas.md`
- Record query results and metrics observed

**Reference**: `phase_5_azure_functions_deployment.md` - Step 1.3: Create Application Insights

---

## Task 5.2: Error Monitoring

**Execute**:
1. Review Application Insights for errors
2. Test error scenarios and verify they appear in logs
3. Verify error details include context (document_id, stage, etc.)

**Error Scenarios to Test**:
- Send invalid message format (should appear in logs)
- Process document with missing file (should log error)
- Simulate transient Azure service error (should log retry attempts)

**Azure Portal Queries**:
```kusto
// Recent errors
exceptions
| where timestamp > ago(1h)
| order by timestamp desc
| project timestamp, type, message, operation_Name

// Error rate by function
exceptions
| summarize count() by operation_Name, bin(timestamp, 1h)
| order by timestamp desc
```

**Verify**:
- ✅ Errors appear in Application Insights
- ✅ Error details include sufficient context:
  - Document ID
  - Stage (ingestion, chunking, embedding, indexing)
  - Error message
  - Stack trace (if available)
- ✅ Error severity levels correct
- ✅ Error correlation with function executions

**Success Criteria**:
- ✅ Errors appear in Application Insights
- ✅ Error details include sufficient context
- ✅ Alerts configured (if applicable)
- ✅ Error tracking is functional

**Document**: 
- Document error tracking setup, alert configuration in `phase_5_testing.md`
- Document any monitoring gaps in `fracas.md`
- Include screenshots of error logs
- Record error query results

---

## Task 5.3: Custom Metrics Verification

**Execute**:
1. Check for custom metrics in Application Insights
2. Verify custom metrics are tracked (if implemented)
3. Review metric values and trends

**Custom Metrics to Verify** (if implemented):
- Queue depth per queue
- Document processing status counts
- Worker processing time per stage
- Retry counts
- Batch processing metrics

**Azure Portal Queries**:
```kusto
// Custom metrics (if implemented)
customMetrics
| where timestamp > ago(1h)
| summarize avg(value), max(value), min(value) by name, bin(timestamp, 1h)
| order by timestamp desc
```

**Success Criteria**:
- ✅ Custom metrics tracked (if implemented)
- ✅ Metric values are reasonable
- ✅ Metrics update in real-time

**Document**: 
- Document custom metrics status in `phase_5_testing.md`
- Note any missing custom metrics in `fracas.md`
- Record metric values observed

---

## Task 5.4: Function Execution Tracking

**Execute**:
1. Process a test document through the pipeline
2. Verify function executions appear in Application Insights
3. Verify execution details are captured

**Azure Portal Queries**:
```kusto
// Function executions
traces
| where operation_Name contains "worker"
| where timestamp > ago(1h)
| order by timestamp desc
| project timestamp, operation_Name, message, severityLevel

// Execution duration
traces
| where operation_Name contains "worker"
| where timestamp > ago(1h)
| summarize avg(duration), max(duration), min(duration) by operation_Name
```

**Verify**:
- ✅ All function executions logged
- ✅ Execution duration tracked
- ✅ Function names correct
- ✅ Execution context included (document_id, stage, etc.)

**Success Criteria**:
- ✅ All function executions tracked
- ✅ Execution details captured
- ✅ Telemetry appears in real-time

**Document**: 
- Document function execution tracking in `phase_5_testing.md`
- Include screenshots of execution logs
- Record execution query results

---

## Task 5.5: Alert Configuration (Optional)

**Execute**:
1. Review Application Insights alert configuration
2. Verify alerts are configured (if applicable)
3. Test alert triggers (if possible)

**Alerts to Verify** (if configured):
- High error rate alerts
- Function execution failures
- Queue depth alerts
- Performance degradation alerts

**Success Criteria**:
- ✅ Alerts configured (if applicable)
- ✅ Alert thresholds reasonable
- ✅ Alert notifications working (if configured)

**Document**: 
- Document alert configuration in `phase_5_testing.md`
- Note any missing alerts in `fracas.md`

---

## Success Criteria

### Must Pass (Blocking) - All Required:
- [ ] Function executions logged in Application Insights
- [ ] Errors tracked in Application Insights
- [ ] Basic telemetry functional

### Should Pass (Non-Blocking) - Document Results:
- [ ] Custom metrics tracked (if implemented)
- [ ] Alerts configured (if applicable)
- [ ] Telemetry appears in real-time
- [ ] Error context sufficient for debugging

**If any "Must Pass" criteria fails, document in `fracas.md` and stop execution.**

---

## Documentation Requirements

**You MUST**:
1. **Record Results**: Update `phase_5_testing.md` with:
   - Telemetry verification results
   - Error monitoring status
   - Custom metrics status
   - Function execution tracking status
   - Alert configuration status
   - Screenshots of Application Insights dashboard

2. **Document Failures**: Immediately document any failures in `fracas.md` with:
   - Failure description
   - Missing metrics or telemetry
   - Steps to reproduce
   - Impact assessment
   - Severity (Critical/High/Medium/Low)

3. **Update Tracking**: 
   - Mark Phase 5 tasks complete in `scoping/TODO001.md`
   - Update `phase_5_handoff.md` with Phase 5 status

---

## Next Steps

**If all Phase 5 tasks pass**:
- Proceed to Phase 5.6: Production Readiness Validation
- Reference: `prompt_phase_5_6_001.md`

**If Phase 5 tasks fail**:
- Document all issues in `fracas.md`
- Address critical blockers before proceeding
- Re-run Phase 5 tests after fixes

---

## Quick Reference

**Application Insights Queries**:
- Function executions: `traces | where operation_Name contains "worker"`
- Errors: `exceptions | where timestamp > ago(1h)`
- Custom metrics: `customMetrics | where timestamp > ago(1h)`

**Key Files**:
- `phase_5_testing.md` - Test results
- `fracas.md` - Failure tracking
- `phase_5_handoff.md` - Handoff document
- `scoping/TODO001.md` - Task tracking

**Azure Portal Navigation**:
- Application Insights → `ai-rag-evaluator` → Logs
- Application Insights → `ai-rag-evaluator` → Metrics
- Application Insights → `ai-rag-evaluator` → Alerts

---

**Begin execution now. Good luck!**

