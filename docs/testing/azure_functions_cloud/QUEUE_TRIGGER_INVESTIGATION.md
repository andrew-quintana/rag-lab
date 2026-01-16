# Queue Trigger Investigation Report

**Date**: 2026-01-14  
**Issue**: Azure Functions queue trigger not processing messages  
**Status**: 🔍 **INVESTIGATING**

---

## 🔍 Current Situation

### Observations

1. **Message Status**:
   - ✅ Message successfully enqueued to `ingestion-uploads` queue
   - ❌ Message remains in queue (not being processed)
   - ✅ Message NOT in poison queue (function hasn't attempted to process it)

2. **Function App Status**:
   - ✅ Function app is running (`State: Running`)
   - ✅ All 4 functions registered (ingestion-worker, chunking-worker, embedding-worker, indexing-worker)
   - ✅ Functions are enabled (`IsDisabled: False`)
   - ✅ Runtime: Python 3.12

3. **Configuration**:
   - ✅ `AzureWebJobsStorage` connection string is set correctly
   - ✅ Function binding uses `"connection": "AzureWebJobsStorage"`
   - ✅ Queue name matches: `ingestion-uploads`
   - ✅ All environment variables configured

4. **Database**:
   - ✅ Cloud Supabase connected
   - ✅ Document record exists
   - ❌ Status still "uploaded" (no processing occurred)
   - ❌ `chunks_created: 0`

5. **Logging**:
   - ❌ No logs in Application Insights
   - ❌ No traces from ingestion-worker
   - ❌ No exceptions logged
   - ⚠️ Application Insights connection string is configured

---

## 🔧 Root Cause Analysis

### Hypothesis 1: Function Runtime Not Starting
**Likelihood**: High  
**Evidence**: No logs at all, not even startup logs

**Possible Causes**:
- Module import error during function initialization
- Missing dependency causing silent failure
- Python runtime error preventing function host from starting

**Investigation Steps**:
1. Check Function App logs via Azure Portal (Log Stream)
2. Check Kudu console for runtime errors
3. Test with minimal function to isolate issue

### Hypothesis 2: Queue Trigger Not Bound
**Likelihood**: Medium  
**Evidence**: Function is registered but trigger not firing

**Possible Causes**:
- Function host not detecting queue trigger binding
- Connection string format issue
- Queue trigger extension not loaded

**Investigation Steps**:
1. Verify `host.json` has correct extensions configuration
2. Check if queue trigger extension is installed
3. Verify connection string format matches Azure Functions requirements

### Hypothesis 3: Application Insights Not Logging
**Likelihood**: Low  
**Evidence**: Function may be running but logs not appearing

**Possible Causes**:
- Logging configuration issue
- Application Insights sampling filtering out logs
- Logs delayed (Application Insights has 2-5 minute lag)

**Investigation Steps**:
1. Check `host.json` logging configuration
2. Verify Application Insights sampling settings
3. Wait longer and check again

---

## 📋 Immediate Next Steps

### Priority 1: Check Function Runtime Logs

**Via Azure Portal**:
1. Navigate to: https://portal.azure.com
2. Go to: Function App → `func-raglab-uploadworkers` → Monitor → Log stream
3. Look for:
   - Function host startup messages
   - Python runtime errors
   - Module import errors
   - Queue trigger binding messages

**Via Azure CLI** (if available):
```bash
# Check function app logs
az webapp log tail \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab
```

### Priority 2: Verify host.json Configuration

Check if `host.json` has correct extensions:

```json
{
  "version": "2.0",
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[3.*, 4.0.0)"
  },
  "logging": {
    "applicationInsights": {
      "samplingSettings": {
        "isEnabled": false
      }
    }
  }
}
```

### Priority 3: Test with Minimal Function

Create a test function to verify runtime:

```python
# test-queue/__init__.py
import azure.functions as func
import logging

def main(msg: func.QueueMessage) -> None:
    logging.info(f"TEST: Function executed! Message: {msg.get_body().decode('utf-8')}")
```

### Priority 4: Check Kudu Console

1. Navigate to: `https://func-raglab-uploadworkers.scm.azurewebsites.net`
2. Go to: Debug console → CMD
3. Check: `/home/site/wwwroot/` directory structure
4. Verify: `backend/src` directory exists
5. Check: Function logs in `/home/LogFiles/`

---

## 🔍 Diagnostic Commands

### Check Function Status
```bash
az functionapp function list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --output table
```

### Check App Settings
```bash
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[?contains(name, 'AzureWebJobsStorage') || contains(name, 'SUPABASE') || contains(name, 'DATABASE')]"
```

### Check Queue Status
```bash
cd /Users/aq_home/1Projects/rag_evaluator
source backend/venv/bin/activate
export AZURE_STORAGE_QUEUES_CONNECTION_STRING="..."
python scripts/monitor_queue_health.py
```

### Manually Trigger Function (if HTTP trigger exists)
```bash
# If we add an HTTP trigger for testing
curl -X POST https://func-raglab-uploadworkers.azurewebsites.net/api/ingestion-worker \
  -H "Content-Type: application/json" \
  -d '{"document_id": "test"}'
```

---

## 📊 Success Criteria

- [ ] Function host starts successfully
- [ ] Queue trigger binds correctly
- [ ] Function executes when message is enqueued
- [ ] Logs appear in Application Insights
- [ ] Database status updates to "processing"
- [ ] First batch is processed
- [ ] Message re-enqueued for continuation (time-based limiting)

**Current Status**: 0/7 criteria met

---

## 🎯 Recommended Action Plan

1. **Immediate**: Access Azure Portal Log Stream to see real-time errors
2. **Short-term**: Verify `host.json` and function binding configuration
3. **If still failing**: Create minimal test function to isolate runtime vs. code issues
4. **Long-term**: Set up proper logging and monitoring pipeline

---

## 📝 Notes

- Deployment succeeded without errors
- All infrastructure components are correctly configured
- The issue appears to be in the function runtime or trigger binding
- No error messages are being captured, suggesting silent failure during initialization

**Critical**: Need access to Azure Portal Log Stream for real-time debugging. This is the fastest way to identify the root cause.

---

**Next Update**: After checking Azure Portal Log Stream

---

## 📋 Handoff Created

**For Next Agent**: A comprehensive handoff document has been created:
👉 **See**: [`HANDOFF_NEXT_AGENT_QUEUE_TRIGGER.md`](./HANDOFF_NEXT_AGENT_QUEUE_TRIGGER.md)

This document contains:
- Complete problem description
- Step-by-step debugging guide
- All testing commands
- Common issues and solutions
- Success criteria
- Recommended approach
