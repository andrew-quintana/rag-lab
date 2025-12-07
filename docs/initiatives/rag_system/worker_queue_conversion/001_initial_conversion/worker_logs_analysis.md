# Azure Functions Worker Logs Analysis

**Date**: 2025-01-XX  
**Analysis Date**: 2025-01-XX  
**Status**: ⚠️ **ISSUE IDENTIFIED** - Functions not processing queue messages

## Summary

Analysis of Azure Functions logs reveals that:
- ✅ Function App is running and healthy
- ✅ Functions are deployed and enabled
- ✅ Queue configuration is correct
- ❌ **Functions are NOT processing messages from queues**
- ❌ **18 messages are queued in `ingestion-uploads` but not being processed**

## Findings

### 1. Function App Status
- **State**: Running ✅
- **Functions Deployed**: 4/4 (ingestion-worker, chunking-worker, embedding-worker, indexing-worker) ✅
- **Functions Enabled**: All functions enabled ✅
- **Host Status**: Active (lease renewals every 12 seconds) ✅

### 2. Queue Status
- **ingestion-uploads**: 18 messages queued (not processing) ❌
- **ingestion-chunking**: 0 messages ✅
- **ingestion-embeddings**: 0 messages ✅
- **ingestion-indexing**: 0 messages ✅

### 3. Application Insights Logs
- **Function Executions**: 0 in last 24 hours ❌
- **Exceptions**: 0 in last 24 hours
- **Traces**: Only host lease renewal logs (no function execution logs) ❌
- **Last Activity**: Host lease renewals (ongoing, every 12 seconds)

### 4. Configuration Verification
- **AzureWebJobsStorage**: ✅ Configured correctly
- **Queue Name**: ✅ Matches function.json (`ingestion-uploads`)
- **Connection**: ✅ Points to correct storage account (`raglabqueues`)
- **Function Bindings**: ✅ Queue trigger configured correctly

## Root Cause Analysis

### ✅ **ROOT CAUSE IDENTIFIED**: Message Encoding Issue

**Evidence from Application Insights**:
- Multiple errors: "Message decoding has failed! Check MessageEncoding settings."
- Functions are receiving messages but failing to decode them
- 18 messages queued but not processed due to decoding failures

**Root Cause**:
- Azure Functions Python queue triggers require `"dataType": "string"` in function.json bindings
- Without this setting, Azure Functions tries to auto-detect message encoding and fails
- Messages are being sent as plain JSON strings, but Functions can't decode them without explicit dataType configuration

**Solution Implemented**:
1. ✅ Added `"dataType": "string"` to all function.json bindings
2. ✅ Updated queue_client.py to send messages as plain text strings
3. ✅ Simplified message deserialization

**Action Required**:
- **Redeploy Azure Functions** to apply function.json changes
- **Clear existing queue messages** (they were sent with wrong encoding)
- **Test with new messages** after redeployment

### Other Possible Causes (Investigated)

1. **Function Code Import Errors** (Not the issue)
   - No import errors found in logs
   - Functions are loading successfully

2. **Queue Trigger Not Activating** (Not the issue)
   - Triggers are activating (messages are being received)
   - Issue is with message decoding, not trigger activation

3. **Function Deployment Issues** (Not the issue)
   - Functions are deployed correctly
   - Code is accessible

4. **Environment Variable Issues** (Not the issue)
   - No startup failures observed
   - Functions are initializing successfully

## Evidence

### Queue Depths (2025-01-XX)
```
ingestion-uploads: 18 messages
ingestion-chunking: 0 messages
ingestion-embeddings: 0 messages
ingestion-indexing: 0 messages
```

### Application Insights Query Results
- **Traces with "ingestion" or "worker"**: Only host lease renewal logs
- **Function executions**: None in last 24 hours
- **Exceptions**: None in last 24 hours
- **Last function execution**: Not found

### Function Configuration
```json
{
  "name": "ingestion-worker",
  "language": "python",
  "isDisabled": false,
  "bindings": [
    {
      "name": "queueMessage",
      "type": "queueTrigger",
      "direction": "in",
      "queueName": "ingestion-uploads",
      "connection": "AzureWebJobsStorage"
    }
  ]
}
```

## Impact

- **Critical**: Messages are accumulating in queues but not being processed
- **Test Failure**: `test_azure_functions_queue_trigger_behavior` fails because functions don't process messages
- **Production Risk**: If deployed to production, documents would not be processed

## Recommended Actions

### Immediate (High Priority)

1. **Check Function Logs for Import Errors**
   ```bash
   # Access function logs via Azure Portal
   # Navigate to: Function App → ingestion-worker → Monitor → Logs
   ```

2. **Verify Function Code Deployment**
   - Check if backend code is included in deployment package
   - Verify `build.sh` correctly copies backend code
   - Check function directory structure in Azure

3. **Test Function Manually**
   - Use Azure Portal to test function with sample queue message
   - Check for error messages in function execution logs

4. **Check Application Insights for Startup Errors**
   - Look for exceptions during function startup
   - Check for import errors or missing dependencies

### Investigation Steps

1. **Review Function Deployment**
   - Check deployment logs for errors
   - Verify all files are included in deployment
   - Check if `requirements.txt` dependencies are installed

2. **Test Queue Trigger Manually**
   - Send test message via Azure Portal
   - Monitor function execution in real-time
   - Check if function is triggered
   - Check for error messages

3. **Check Function Runtime Logs**
   - Access Kudu console or SCM site
   - Check function host logs
   - Look for Python import errors

4. **Verify Environment Variables**
   - Ensure all required variables are set
   - Check for missing variables that might cause silent failures

## Next Steps

1. **Access Azure Portal** → Function App → ingestion-worker → Monitor → Logs
2. **Check for error messages** in function execution logs
3. **Test function manually** with a sample queue message
4. **Review deployment package** to ensure backend code is included
5. **Check Application Insights** for startup exceptions

## Documentation

- **Issue**: FM-005 (to be created in fracas.md)
- **Severity**: Critical
- **Status**: Under Investigation
- **Related**: Phase 5.3 test failure (`test_azure_functions_queue_trigger_behavior`)

