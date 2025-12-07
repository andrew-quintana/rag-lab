# FRACAS.md - Failure Reporting, Analysis, and Corrective Actions System

**Initiative:** Codebase Consolidation & Production Readiness (002)  
**Status:** Active  
**Date Started:** 2025-12-07  
**Last Updated:** 2025-12-07  
**Maintainer:** Development Team

## 📋 **How to Use This Document**

This document serves as a comprehensive failure tracking system for Initiative 002: Codebase Consolidation & Production Readiness. Use it to:

1. **Document new failures** as they occur during consolidation/testing
2. **Track investigation progress** and findings
3. **Record root cause analysis** and solutions
4. **Maintain a knowledge base** of known issues and fixes

### **Documentation Guidelines:**
- **Be specific** about symptoms, timing, and context
- **Include evidence** (logs, error messages, screenshots)
- **Update status** as investigation progresses
- **Link related failures** when applicable
- **Record both successful and failed solutions**

---

## 🚨 **Active Failure Modes**

### **FM-001: Batch Result Persistence Query Error** (From Initiative 001)
- **Severity**: Low
- **Status**: ⚠️ Known issue (Non-blocking)
- **First Observed**: 2025-01-XX (Initiative 001)
- **Last Updated**: 2025-12-07

**Symptoms:**
- `test_batch_result_persistence` test fails with "tuple index out of range" error
- Error occurs in `get_completed_batches` function when querying chunks table
- 12 of 13 Supabase integration tests pass

**Observations:**
- Error occurs during batch metadata persistence test
- Query execution fails with tuple index out of range
- Other persistence operations work correctly
- Non-blocking for Phase 5 testing - other tests pass

**Investigation Notes:**
- Issue appears to be in `get_completed_batches` function in `rag_eval/services/workers/persistence.py`
- May be related to query result handling or batch metadata structure
- Non-blocking for consolidation work

**Root Cause:**
Under investigation (carried over from Initiative 001)

**Solution:**
Pending (low priority, non-blocking)

**Evidence:**
- Test output: `ERROR rag_eval.db.queries:queries.py:32 Query execution failed: tuple index out of range`
- Test: `tests/integration/test_supabase_phase5.py::TestBatchMetadata::test_batch_result_persistence`
- Reference: Initiative 001 fracas.md FM-001

---

### **FM-005: Azure Functions Not Processing Queue Messages** (From Initiative 001)
- **Severity**: Critical
- **Status**: ✅ **RESOLVED** (requires redeployment)
- **First Observed**: 2025-01-XX (Initiative 001)
- **Last Updated**: 2025-12-07

**Symptoms:**
- Messages queued in `ingestion-uploads` queue but not being processed
- No function execution logs in Application Insights
- Test `test_azure_functions_queue_trigger_behavior` fails (timeout waiting for processing)

**Observations:**
- Queue depths show messages but functions not processing
- Application Insights: No function execution traces
- Function App status: Running, all functions enabled

**Investigation Notes:**
- Root cause identified in Initiative 001: Message encoding issue
- Azure Functions queue triggers cannot decode messages without `dataType: "string"` in function.json
- Solution implemented: Added `dataType: "string"` to all function.json bindings

**Root Cause:**
Message encoding issue - Azure Functions queue triggers cannot decode messages without explicit `dataType: "string"` setting in function.json.

**Solution:**
**IMPLEMENTED** (Initiative 001):
1. ✅ Added `"dataType": "string"` to all function.json bindings
2. ✅ Updated queue_client.py to send messages as plain text strings
3. ✅ Simplified deserialize_message to handle plain text

**Next Steps**:
1. **Redeploy Azure Functions** to apply function.json changes
2. **Clear existing queue messages** (they were sent with wrong encoding)
3. **Test with new messages** after redeployment
4. **Verify functions process messages** successfully

**Evidence:**
- Reference: Initiative 001 fracas.md FM-005
- Analysis document: `001/worker_logs_analysis.md`
- Fix document: `001/MESSAGE_ENCODING_FIX.md`

**Impact:**
- **Critical**: Messages accumulating in queues but not being processed
- Blocks end-to-end pipeline testing
- Production risk: Documents would not be processed if deployed

**Status for Initiative 002:**
- Solution implemented in Initiative 001
- Requires redeployment to verify fix
- Will be validated during Phase 4: Production Readiness Validation

---

## 🔧 **Resolved Failure Modes**

*No resolved failure modes specific to Initiative 002 at this time. This section will be populated as issues are resolved.*

---

## 📝 **New Failure Documentation Template**

Use this template when documenting new failures:

```markdown
### **FM-XXX: [Failure Name]**
- **Severity**: [Low/Medium/High/Critical]
- **Status**: [🔍 Under Investigation | ⚠️ Known issue | 🔧 Fix in progress | ✅ Resolved]
- **First Observed**: [YYYY-MM-DD]
- **Last Updated**: [YYYY-MM-DD]

**Symptoms:**
- [Specific error messages or behaviors]
- [When the failure occurs]
- [What functionality is affected]

**Observations:**
- [What you noticed during testing]
- [Patterns or timing of the failure]
- [Any error messages or logs]

**Investigation Notes:**
- [Steps taken to investigate]
- [Hypotheses about the cause]
- [Tests performed or attempted]
- [Files or components involved]

**Root Cause:**
[The actual cause once identified, or "Under investigation" if unknown]

**Solution:**
[How the issue was fixed, or "Pending" if not yet resolved]

**Evidence:**
- [Code changes made]
- [Log entries or error messages]
- [Test results or screenshots]

**Related Issues:**
- [Links to related failures or issues]

**Impact:**
- [What functionality is affected]
- [Blocking status for consolidation work]
```

---

## 🔍 **Failure Tracking Guidelines**

### **When to Document a Failure:**
- Any unexpected behavior or error during consolidation/testing
- Code duplication issues discovered
- Configuration management problems
- Build/deployment script failures
- Import path resolution issues
- Test infrastructure problems
- Performance issues or slow responses
- Service unavailability or crashes
- Data inconsistencies or corruption

### **What to Include:**
1. **Immediate Documentation**: Record symptoms and context as soon as possible
2. **Evidence Collection**: Screenshots, logs, error messages, stack traces
3. **Reproduction Steps**: Detailed steps to reproduce the issue
4. **Environment Details**: OS, Python version, service versions, configuration
5. **Impact Assessment**: What functionality is affected and severity

### **Investigation Process:**
1. **Initial Assessment**: Determine severity and impact
2. **Data Gathering**: Collect logs, error messages, and context
3. **Hypothesis Formation**: Develop theories about the root cause
4. **Testing**: Attempt to reproduce and isolate the issue
5. **Root Cause Analysis**: Identify the actual cause
6. **Solution Development**: Implement and test fixes
7. **Documentation**: Update the failure record with findings

### **Status Updates:**
- **🔍 Under Investigation**: Issue is being analyzed and tested
- **⚠️ Known issue**: Issue understood, workaround available
- **🔧 Fix in progress**: Solution being implemented
- **✅ Resolved**: Issue has been resolved and verified
- **Won't Fix**: Issue is known but not planned to be addressed

---

## 📈 **System Health Metrics**

### **Current Status:**
- Phase 0: ✅ Complete (Scoping & Documentation Updates)
- Code Duplication: ⚠️ Present (to be eliminated in Phase 1)
- Configuration Management: ⚠️ Fragmented (to be consolidated in Phase 2)
- Test Infrastructure: ⚠️ Partially unified (to be consolidated in Phase 3)
- Production Readiness: ⏳ Pending (Phase 4)

### **Known Limitations:**
- Duplicate backend code exists in `infra/azure/azure_functions/backend/rag_eval/`
- Configuration split between `.env.local`, `local.settings.json`, and Azure settings
- Test markers not registered (causes warnings)
- FM-001 and FM-005 from Initiative 001 need resolution

---

## 🔍 **Investigation Areas**

### **High Priority:**
1. Code duplication elimination (Phase 1)
2. Configuration consolidation (Phase 2)
3. FM-005 resolution (requires Azure Functions redeployment)

### **Medium Priority:**
1. Test infrastructure consolidation (Phase 3)
2. Build script simplification

### **Low Priority:**
1. FM-001 resolution (non-blocking batch metadata issue)

---

## 📝 **Testing Notes**

### **Phase 0 Testing:**
- ✅ Cursor rules files updated to reflect worker-queue architecture
- ✅ File organization requirements validated
- ✅ Documentation structure established

### **Next Test Session:**
- [ ] Phase 1: Code duplication elimination testing
- [ ] Phase 2: Configuration consolidation testing
- [ ] Phase 3: Test infrastructure consolidation testing
- [ ] Phase 4: Production readiness validation

---

**Last Updated**: 2025-12-07  
**Next Review**: After each phase completion  
**Maintainer**: Development Team

