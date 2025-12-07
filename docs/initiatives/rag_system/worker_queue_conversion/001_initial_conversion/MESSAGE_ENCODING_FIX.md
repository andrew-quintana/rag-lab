# Message Encoding Fix for Azure Functions Queue Triggers

**Date**: 2025-01-XX  
**Issue**: FM-005 - Azure Functions not processing queue messages  
**Status**: ✅ **FIXED** (requires redeployment)

## Problem

Azure Functions were not processing queue messages. Application Insights logs showed:
- "Message decoding has failed! Check MessageEncoding settings."
- 18 messages queued in `ingestion-uploads` but not processed
- 0 function invocations in last 24 hours

## Root Cause

Azure Functions Python queue triggers require explicit `dataType` configuration in `function.json`. Without this setting, Azure Functions tries to auto-detect message encoding and fails to decode plain JSON strings.

## Solution

### 1. Updated function.json Files

Added `"dataType": "string"` to all queue trigger bindings:

**Files Updated**:
- `infra/azure/azure_functions/ingestion-worker/function.json`
- `infra/azure/azure_functions/chunking-worker/function.json`
- `infra/azure/azure_functions/embedding-worker/function.json`
- `infra/azure/azure_functions/indexing-worker/function.json`

**Before**:
```json
{
  "name": "queueMessage",
  "type": "queueTrigger",
  "direction": "in",
  "queueName": "ingestion-uploads",
  "connection": "AzureWebJobsStorage"
}
```

**After**:
```json
{
  "name": "queueMessage",
  "type": "queueTrigger",
  "direction": "in",
  "queueName": "ingestion-uploads",
  "connection": "AzureWebJobsStorage",
  "dataType": "string"
}
```

### 2. Verified Message Encoding

Confirmed that `queue_client.py` sends messages as plain text JSON strings, which is correct for `dataType: "string"`.

## Deployment Required

**⚠️ IMPORTANT**: Functions must be redeployed for changes to take effect.

### Option 1: Git Deployment (Recommended)
```bash
# If using Git deployment, push changes
git add infra/azure/azure_functions/*/function.json
git commit -m "Fix: Add dataType string to queue trigger bindings"
git push origin main
```

### Option 2: Manual Deployment
```bash
cd infra/azure/azure_functions
./build.sh
func azure functionapp publish func-raglab-uploadworkers --python
```

## Post-Deployment Steps

1. **Clear Existing Queue Messages**
   - The 18 messages currently in `ingestion-uploads` were sent with the old encoding
   - They will fail to decode even after redeployment
   - Clear them or let them expire (7-day TTL)

2. **Test with New Messages**
   - Send a test message after redeployment
   - Verify function processes it successfully
   - Check Application Insights for successful invocations

3. **Verify Function Execution**
   - Check Azure Portal → Function App → ingestion-worker → Invocations
   - Should see successful invocations
   - Check Application Insights for execution logs

## Verification

After redeployment, verify:
- ✅ Functions appear in Azure Portal with updated bindings
- ✅ New messages are processed successfully
- ✅ Application Insights shows function executions
- ✅ No "Message decoding has failed" errors

## Related Issues

- **FM-005**: Azure Functions not processing queue messages (RESOLVED)
- **Phase 5.3 Test Failure**: `test_azure_functions_queue_trigger_behavior` (should pass after redeployment)

