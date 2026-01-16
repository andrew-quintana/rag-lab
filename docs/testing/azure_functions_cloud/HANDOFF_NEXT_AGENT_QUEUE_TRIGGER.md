# Agent Handoff: Azure Functions Queue Trigger Debugging

**Date**: 2026-01-14  
**Previous Agent**: Auto  
**Status**: ⚠️ **BLOCKED - Queue Trigger Not Firing**  
**Priority**: 🔴 **HIGH** - Blocking E2E cloud testing

---

## 🎯 Primary Objective

**Fix Azure Functions queue trigger so that messages in the `ingestion-uploads` queue are processed by the `ingestion-worker` function.**

---

## 📋 Current Situation Summary

### ✅ What's Working

1. **Infrastructure Setup** (100% Complete):
   - ✅ Cloud Supabase database deployed and configured
   - ✅ All 20 migrations applied successfully
   - ✅ Azure Functions deployed (`func-raglab-uploadworkers`)
   - ✅ All 4 functions registered and enabled
   - ✅ Environment variables configured correctly
   - ✅ Azure Blob Storage configured
   - ✅ Test PDF uploaded to blob storage
   - ✅ Build script fixed (includes `backend/src` directory)
   - ✅ `ModuleNotFoundError` resolved

2. **Configuration Verified**:
   - ✅ `AzureWebJobsStorage` connection string is set
   - ✅ Function binding configuration is correct
   - ✅ Queue name matches: `ingestion-uploads`
   - ✅ Application Insights connection string configured
   - ✅ All Supabase and Azure service keys configured

3. **Function App Status**:
   - ✅ Function app is running (`State: Running`)
   - ✅ Runtime: Python 3.12
   - ✅ Functions are enabled (`IsDisabled: False`)

### ❌ What's Not Working

1. **Queue Trigger**:
   - ❌ Messages enqueued to `ingestion-uploads` are NOT being processed
   - ❌ Messages remain in queue (not consumed)
   - ❌ No function execution detected
   - ❌ No logs in Application Insights (not even startup logs)

2. **Database State**:
   - ❌ Document status still "uploaded" (should be "processing")
   - ❌ `chunks_created: 0` (no batches processed)

3. **Logging**:
   - ❌ No traces in Application Insights
   - ❌ No exceptions logged
   - ❌ No function execution logs

---

## 🔍 Key Observations

### Critical Finding
**Messages are NOT going to poison queue** - This means the function is not even attempting to process them. The queue trigger is not firing at all.

### Configuration Check Results
- ✅ Function binding: `"connection": "AzureWebJobsStorage"` ✓
- ✅ `AzureWebJobsStorage` environment variable: Set ✓
- ✅ Queue name in binding: `ingestion-uploads` ✓
- ✅ Queue exists and has messages ✓
- ✅ Function code deployed correctly ✓

### What This Suggests
The issue is likely:
1. **Function host not starting** (silent failure during initialization)
2. **Queue trigger extension not loading** (binding not recognized)
3. **Runtime error preventing function host from binding triggers**

---

## 📁 Key Files and Locations

### Function Code
- **Entry Point**: `backend/azure_functions/ingestion-worker/__init__.py`
- **Function Binding**: `backend/azure_functions/ingestion-worker/function.json`
- **Host Config**: `backend/azure_functions/host.json`
- **Build Script**: `backend/azure_functions/build.sh`

### Worker Code
- **Worker Implementation**: `backend/src/services/workers/ingestion_worker.py`
- **Persistence**: `backend/src/services/workers/persistence.py`
- **RAG Services**: `backend/src/services/rag/ingestion.py`

### Documentation
- **Setup Summary**: `docs/testing/azure_functions_cloud/SETUP_COMPLETE_SUMMARY.md`
- **Investigation Report**: `docs/testing/azure_functions_cloud/QUEUE_TRIGGER_INVESTIGATION.md`
- **Environment Variables**: `docs/setup/ENVIRONMENT_VARIABLES.md`

### Scripts
- **Queue Monitoring**: `scripts/monitor_queue_health.py`
- **Cloud Setup**: `scripts/setup_cloud_supabase.sh`

---

## 🔧 Immediate Action Plan

### Step 1: Access Azure Portal Log Stream (CRITICAL)

**This is the fastest way to see what's happening:**

1. Navigate to: https://portal.azure.com
2. Go to: Resource Groups → `rag-lab` → `func-raglab-uploadworkers`
3. Click: **Monitor** → **Log stream**
4. Look for:
   - Function host startup messages
   - Python runtime errors
   - Module import errors
   - Queue trigger binding messages
   - Any ERROR or WARNING messages

**Expected Output:**
- If function host is starting: You'll see "Host started" messages
- If there's an import error: You'll see Python traceback
- If queue trigger is binding: You'll see "Found the following functions" with queue triggers listed

**If you see nothing**: Function host may not be starting at all (check Kudu console)

### Step 2: Check Kudu Console

1. Navigate to: `https://func-raglab-uploadworkers.scm.azurewebsites.net`
2. Go to: **Debug console** → **CMD**
3. Check directory structure:
   ```bash
   cd /home/site/wwwroot
   ls -la
   ls -la backend/
   ls -la backend/src/
   ```
4. Check function logs:
   ```bash
   cd /home/LogFiles
   ls -la
   tail -100 Application/Functions/Function/*.log
   ```

### Step 3: Verify Function Host Configuration

Check if `host.json` is correct:

```bash
cd /Users/aq_home/1Projects/rag_evaluator/backend/azure_functions
cat host.json
```

**Current `host.json` has:**
- ✅ `extensionBundle` configured
- ✅ `logging` configured with Application Insights
- ✅ `queues` extension settings
- ⚠️ Extension bundle version: `[4.*, 5.0.0)` - verify this is compatible

### Step 4: Test with Minimal Function

If logs show nothing, create a minimal test function to isolate the issue:

**Create**: `backend/azure_functions/test-queue/__init__.py`
```python
import azure.functions as func
import logging

def main(msg: func.QueueMessage) -> None:
    logging.info("TEST: Function executed successfully!")
    logging.info(f"TEST: Message body: {msg.get_body().decode('utf-8')}")
```

**Create**: `backend/azure_functions/test-queue/function.json`
```json
{
  "scriptFile": "__init__.py",
  "bindings": [
    {
      "name": "queueMessage",
      "type": "queueTrigger",
      "direction": "in",
      "queueName": "ingestion-uploads",
      "connection": "AzureWebJobsStorage",
      "dataType": "string"
    }
  ]
}
```

**Deploy and test:**
```bash
cd /Users/aq_home/1Projects/rag_evaluator/backend/azure_functions
./build.sh
func azure functionapp publish func-raglab-uploadworkers --python
```

If this works, the issue is in the `ingestion-worker` code. If it doesn't, the issue is in the runtime/configuration.

---

## 🧪 Testing Commands

### Check Queue Status
```bash
cd /Users/aq_home/1Projects/rag_evaluator
source backend/venv/bin/activate
export AZURE_STORAGE_QUEUES_CONNECTION_STRING="DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=raglabqueues;AccountKey=***REDACTED***;BlobEndpoint=https://raglabqueues.blob.core.windows.net/;FileEndpoint=https://raglabqueues.file.core.windows.net/;QueueEndpoint=https://raglabqueues.queue.core.windows.net/;TableEndpoint=https://raglabqueues.table.core.windows.net/"
python scripts/monitor_queue_health.py
```

### Enqueue Test Message
```bash
cd /Users/aq_home/1Projects/rag_evaluator
source backend/venv/bin/activate
python3 << 'PYTHON_SCRIPT'
from azure.storage.queue import QueueClient
import json

queue_client = QueueClient(
    account_url="https://raglabqueues.queue.core.windows.net/",
    queue_name="ingestion-uploads",
    credential="***REDACTED***"
)

message = {
    "document_id": "f7df526a-97df-4e85-836d-b3478a72ae6d",
    "source_storage": "azure_blob",
    "filename": "scan_classic_hmo.pdf",
    "attempt": 1,
    "stage": "uploaded",
    "metadata": {"test": "queue-trigger-debug"}
}

queue_client.send_message(json.dumps(message))
print(f"✅ Message enqueued: {message['document_id']}")
PYTHON_SCRIPT
```

### Check Function App Status
```bash
az functionapp show \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "{state:state, runtime:siteConfig.linuxFxVersion, enabled:enabled, status:availabilityState}" \
  --output table
```

### Check Environment Variables
```bash
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[?contains(name, 'AzureWebJobsStorage') || contains(name, 'SUPABASE') || contains(name, 'DATABASE')]" \
  --output table
```

### Check Database Status
```bash
cd /Users/aq_home/1Projects/rag_evaluator
export $(grep -v '^#' .env.prod | xargs)
psql "$DATABASE_URL" << 'SQL'
SELECT 
    id,
    filename,
    status,
    chunks_created,
    metadata->>'test' as test_tag
FROM documents
WHERE id = 'f7df526a-97df-4e85-836d-b3478a72ae6d';
SQL
```

---

## 🔍 Common Issues and Solutions

### Issue 1: Function Host Not Starting
**Symptoms**: No logs at all, not even startup messages

**Possible Causes**:
- Module import error in `__init__.py`
- Missing dependency in `requirements.txt`
- Python syntax error

**Solution**:
1. Check Azure Portal Log Stream for Python errors
2. Check Kudu console for runtime errors
3. Verify all imports in `ingestion-worker/__init__.py` are available

### Issue 2: Queue Trigger Extension Not Loading
**Symptoms**: Function host starts but queue trigger doesn't fire

**Possible Causes**:
- Extension bundle version incompatible
- Queue trigger extension not installed
- Connection string format issue

**Solution**:
1. Verify `extensionBundle` in `host.json`
2. Check if extension bundle is downloaded (Kudu console)
3. Try explicit extension installation

### Issue 3: Connection String Format
**Symptoms**: Trigger not binding, no error messages

**Possible Causes**:
- Connection string format not recognized
- Storage account permissions issue

**Solution**:
1. Verify `AzureWebJobsStorage` format matches Azure Functions requirements
2. Check storage account access keys
3. Try using connection string name instead of full string

### Issue 4: Message Encoding Mismatch
**Symptoms**: Function executes but can't parse message

**Possible Causes**:
- `host.json` has `"messageEncoding": "base64"` but function expects string
- Message format doesn't match expected JSON

**Solution**:
1. Check `host.json` queue settings
2. Verify message format matches function expectations
3. Test with base64 encoded vs. plain string

---

## 📊 Success Criteria

### Immediate (Queue Trigger Fix)
- [ ] Function host starts successfully (visible in logs)
- [ ] Queue trigger binds correctly (visible in logs)
- [ ] Function executes when message is enqueued
- [ ] Logs appear in Application Insights
- [ ] No messages stuck in queue

### Full E2E Test
- [ ] Database status updates to "processing"
- [ ] First batch of document is processed
- [ ] Time-based limiting re-enqueues message for continuation
- [ ] Multiple batches processed across executions
- [ ] Full document processed end-to-end
- [ ] No messages in poison queue

---

## 🎯 Recommended Approach

### Phase 1: Diagnosis (30-60 minutes)
1. **Access Azure Portal Log Stream** - This will immediately show what's wrong
2. **Check Kudu Console** - Verify file structure and check logs
3. **Review `host.json`** - Ensure configuration is correct
4. **Check Function Binding** - Verify `function.json` is correct

### Phase 2: Isolation (30-60 minutes)
1. **Create Minimal Test Function** - Isolate runtime vs. code issues
2. **Test with Simple Queue Message** - Verify trigger works
3. **Compare Working vs. Non-Working** - Identify differences

### Phase 3: Fix (30-120 minutes)
1. **Apply Fix** - Based on findings from Phase 1 & 2
2. **Redeploy** - Test fix
3. **Validate** - Run full E2E test

### Phase 4: Validation (30 minutes)
1. **Enqueue Test Message** - Verify trigger fires
2. **Monitor Processing** - Check database updates
3. **Verify Time-Based Limiting** - Check message re-enqueuing
4. **Document Solution** - Update relevant docs

---

## 📝 Important Notes

### Environment Variables
- **Local**: `.env.local` (local Supabase)
- **Cloud**: `.env.prod` (cloud Supabase)
- **Azure Functions**: Configured via `az functionapp config appsettings set`

### Test Document
- **ID**: `f7df526a-97df-4e85-836d-b3478a72ae6d`
- **Filename**: `scan_classic_hmo.pdf`
- **Location**: Azure Blob Storage container `documents`
- **Database**: Cloud Supabase (record exists)

### Connection Strings
- **Queue Storage**: `AZURE_STORAGE_QUEUES_CONNECTION_STRING` (in `.env.prod`)
- **Blob Storage**: `AZURE_BLOB_CONNECTION_STRING` (in Azure Functions)
- **Queue Trigger**: `AzureWebJobsStorage` (in Azure Functions)

### Key Insights from Previous Work
1. **Build script was fixed** - `backend/src` is now included in deployment
2. **ModuleNotFoundError resolved** - Code structure is correct
3. **All infrastructure configured** - Issue is in runtime/trigger binding
4. **No error messages** - Suggests silent failure during initialization

---

## 🚨 Critical: What to Check First

**If you can only do one thing, do this:**

1. **Open Azure Portal** → Function App → `func-raglab-uploadworkers` → **Monitor** → **Log stream**
2. **Enqueue a test message** (use command above)
3. **Watch the log stream** - You'll immediately see:
   - If function host is starting
   - If there are import errors
   - If queue trigger is binding
   - What's actually happening

**This single action will reveal 90% of the issue.**

---

## 📚 Related Documentation

- **Environment Variables**: `docs/setup/ENVIRONMENT_VARIABLES.md`
- **Setup Summary**: `docs/testing/azure_functions_cloud/SETUP_COMPLETE_SUMMARY.md`
- **Investigation Report**: `docs/testing/azure_functions_cloud/QUEUE_TRIGGER_INVESTIGATION.md`
- **Time-Based Limiting**: `docs/testing/azure_functions_cloud/TIME_BASED_LIMITING_IMPLEMENTATION.md`

---

## ✅ Expected Outcome

After fixing the queue trigger:
1. Messages in `ingestion-uploads` queue are processed automatically
2. Function execution logs appear in Application Insights
3. Database status updates from "uploaded" to "processing"
4. Document batches are processed with time-based limiting
5. Full E2E cloud testing can proceed

---

**Good luck! The infrastructure is ready, we just need to get the trigger working.** 🚀

---

**Last Updated**: 2026-01-14  
**Next Agent**: Please update this document with your findings and solution
