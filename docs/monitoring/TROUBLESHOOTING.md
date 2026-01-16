# Azure Functions Troubleshooting Playbook

**Structured guide for diagnosing and fixing common Azure Functions issues**

## How to Use This Playbook

1. Find your issue in the table of contents
2. Follow the **Diagnosis** steps to confirm the issue
3. Apply the **Fix** that matches your situation
4. Verify the fix worked

---

## Table of Contents

1. [Messages Going to Poison Queue](#1-messages-going-to-poison-queue)
2. [Functions Not Processing Messages](#2-functions-not-processing-messages)
3. [Import Errors (ModuleNotFoundError)](#3-import-errors-modulenotfounderror)
4. [Database Connection Errors](#4-database-connection-errors)
5. [Message Decoding Errors](#5-message-decoding-errors)
6. [Functions Showing as Disabled](#6-functions-showing-as-disabled)
7. [Slow Processing / Timeouts](#7-slow-processing--timeouts)
8. [Configuration Issues](#8-configuration-issues)
9. [Deployment Issues](#9-deployment-issues)
10. [Local Development Issues](#10-local-development-issues)
11. [Using Application Insights for Diagnosis](#11-using-application-insights-for-diagnosis)

---

## 1. Messages Going to Poison Queue

### Symptoms
- Messages appear in `*-poison` queues
- Warning in monitoring: `⚠️ X message(s) in poison queues`
- Azure logs show: `Moving message to queue 'ingestion-uploads-poison'`

### Diagnosis

**Step 1:** Check poison queue contents
```bash
python scripts/monitor_queue_health.py --peek ingestion-uploads-poison
```

**Step 2:** Check Azure Function logs for errors

**Option A: Application Insights (Recommended)**
```bash
# View recent errors
make logs-errors

# View errors from specific time window
TIME_WINDOW=2h make logs-errors

# Query specific operation (get operation_Id from error logs)
./scripts/query_app_insights.sh operation 30d <operation_id>
```

**Option B: Azure CLI Log Stream**
```bash
az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab
```

Look for Python exceptions around the time the message was moved to poison queue.

### Common Causes & Fixes

#### Cause 1: Import Errors
**Log shows:** `ModuleNotFoundError: No module named 'src'`

**Fix:** Update function's `__init__.py` to include sys.path manipulation:
```python
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from src.services.workers.ingestion_worker import ingestion_worker
```

Then redeploy:
```bash
cd backend/azure_functions
func azure functionapp publish func-raglab-uploadworkers --python
```

#### Cause 2: Database Connection Error
**Log shows:** `psycopg2.OperationalError: connection to server at "..." failed`

**Fix:** Verify `DATABASE_URL` in Function App settings:
```bash
# Check current value
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[?name=='DATABASE_URL'].value" -o tsv

# Update if needed (with ngrok URL)
az functionapp config appsettings set \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --settings "DATABASE_URL=postgresql://..."
```

#### Cause 3: Missing Environment Variable
**Log shows:** `KeyError: 'SUPABASE_URL'` or similar

**Fix:** Add missing environment variable in Azure Portal:
1. Go to: func-raglab-uploadworkers → Configuration
2. Click: **+ New application setting**
3. Add the missing setting
4. Click: **Save**

Or via CLI:
```bash
az functionapp config appsettings set \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --settings "SUPABASE_URL=https://..."
```

#### Cause 4: Message Decoding Error
**Log shows:** `Message decoding has failed! Check MessageEncoding settings`

**Fix:** See [#5 Message Decoding Errors](#5-message-decoding-errors)

### After Fixing

1. Clear poison queue:
   ```bash
   python scripts/monitor_queue_health.py --clear-poison
   ```

2. Send test message:
   ```bash
   python scripts/diagnose_functions.py
   ```

3. Verify success in logs

---

## 2. Functions Not Processing Messages

### Symptoms
- Messages sit in queue without being processed
- No invocations showing in Azure Portal
- Diagnostic timeout: `⏱️ Function did not process message within timeout`

### Diagnosis

**Step 1:** Check queue status
```bash
python scripts/monitor_queue_health.py
```

**Step 2:** Check if functions are disabled
```bash
az functionapp function list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[].{Name:name, Disabled:config.disabled}" -o table
```

**Note:** On Linux, Azure CLI may incorrectly show functions with hyphens as disabled. Check logs for polling activity instead:

```bash
az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab | grep -i poll
```

Should see:
```
Poll for function 'ingestion-worker' on queue 'ingestion-uploads'...
```

### Fixes

#### Fix 1: Functions Are Disabled

**If truly disabled** (not just CLI display bug):
1. Go to Azure Portal → func-raglab-uploadworkers → Functions
2. Click on disabled function
3. Click **Enable**

#### Fix 2: Cold Start Delay

**After fresh deployment**, functions can take 2-3 minutes to start:
- **Wait 3 minutes**
- Check logs for polling activity
- Try diagnostic again

#### Fix 3: Wrong Connection String

**Check AzureWebJobsStorage**:
```bash
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[?name=='AzureWebJobsStorage'].value" -o tsv
```

Should be a valid Azure Storage connection string (not `UseDevelopmentStorage=true`).

#### Fix 4: Restart Function App

```bash
az functionapp restart --name func-raglab-uploadworkers --resource-group rag-lab
```

Wait 2-3 minutes, then test:
```bash
python scripts/diagnose_functions.py
```

---

## 3. Import Errors (ModuleNotFoundError)

### Symptoms
- Log shows: `ModuleNotFoundError: No module named 'src'`
- Messages go to poison queue immediately

### Root Cause
Function can't find the `src` package because `backend/` is not in Python path.

### Fix

**For each function** (`ingestion-worker`, `chunking-worker`, etc.), update `__init__.py`:

```python
import sys
from pathlib import Path

# Add backend directory to Python path for imports
# With Git deployment, functions are in backend/azure_functions/ and src is in backend/src/
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from src.services.workers.ingestion_worker import ingestion_worker
from src.core.logging import get_logger

logger = get_logger("azure_functions.ingestion_worker")

def main(queueMessage: str) -> None:
    # ... rest of function
```

**Deploy:**
```bash
cd backend/azure_functions
func azure functionapp publish func-raglab-uploadworkers --python
```

**Verify:**
```bash
python scripts/diagnose_functions.py
```

---

## 4. Database Connection Errors

### Symptoms
- Log shows: `psycopg2.OperationalError: connection to server failed`
- Messages go to poison queue after database operations

### Diagnosis

**Check DATABASE_URL**:
```bash
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[?name=='DATABASE_URL'].value" -o tsv
```

**Common issues:**
- Points to `127.0.0.1:54322` (local) from cloud
- ngrok tunnel is down
- Wrong credentials

### Fixes

#### Fix 1: Using Local Supabase (via ngrok)

**Ensure ngrok tunnel is running:**
```bash
# Check if ngrok is running
ps aux | grep ngrok

# If not, start it
./scripts/start_ngrok_tunnel.sh
```

**Update DATABASE_URL:**
```bash
./scripts/update_azure_db_url.sh
```

**Verify ngrok URL:**
```bash
curl https://db.raglab.ngrok.io  # Should connect
```

#### Fix 2: Using Cloud Supabase

**Update DATABASE_URL to cloud:**
```bash
az functionapp config appsettings set \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --settings "DATABASE_URL=postgresql://postgres:[password]@[project].supabase.co:5432/postgres"
```

#### Fix 3: Test Connection

```bash
# Get DATABASE_URL
DB_URL=$(az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[?name=='DATABASE_URL'].value" -o tsv)

# Test connection
psql "$DB_URL" -c "SELECT 1"
```

---

## 5. Message Decoding Errors

### Symptoms
- Log shows: `Message decoding has failed! Check MessageEncoding settings`
- Log shows: `MessageId=... Moving message to queue '...-poison'`

### Root Cause
Azure Functions expects Base64-encoded messages, but `host.json` doesn't specify encoding.

### Fix

**Update `backend/azure_functions/host.json`:**
```json
{
  "version": "2.0",
  "extensions": {
    "queues": {
      "messageEncoding": "base64",
      "maxDequeueCount": 5,
      "visibilityTimeout": "00:10:00"
    }
  },
  "logging": {
    "applicationInsights": {
      "samplingSettings": {
        "isEnabled": false
      }
    },
    "logLevel": {
      "default": "Information",
      "Function": "Information"
    }
  }
}
```

**Deploy using manual method** (ensures file is synced):
```bash
cd backend/azure_functions
func azure functionapp publish func-raglab-uploadworkers --python
```

**Verify deployment:**
```bash
# Clear poison queue
python scripts/monitor_queue_health.py --clear-poison --yes

# Send test
python scripts/diagnose_functions.py
```

---

## 6. Functions Showing as Disabled

### Symptoms
- Azure CLI shows: `Disabled: True`
- But logs show functions are polling queues

### Root Cause
**Known Azure CLI bug** on Linux: CLI incorrectly reports functions with hyphens in their names as disabled.

### Verification

**Check logs for actual polling activity:**
```bash
az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab | grep -i poll
```

If you see:
```
Poll for function 'ingestion-worker' on queue 'ingestion-uploads' with ClientRequestId '...' found 0 messages
```

**Functions ARE running** (despite CLI saying disabled).

### Fix

**If functions are truly disabled:**
1. Go to Azure Portal
2. Navigate to: func-raglab-uploadworkers → Functions → [function-name]
3. Click **Enable** button

**If it's just the CLI bug:**
- No fix needed
- Functions are working
- Verify with logs or diagnostic script

---

## 7. Slow Processing / Timeouts

### Symptoms
- Functions take >60 seconds to process messages
- Timeout errors in logs
- Diagnostic script times out

### Diagnosis

**Check function timeout in `host.json`:**
```bash
grep -A 2 "functionTimeout" backend/azure_functions/host.json
```

Should be: `"functionTimeout": "00:10:00"` (10 minutes)

**Check logs for slow operations:**
```bash
az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab
```

Look for:
- Database query times
- External API call times
- Large file processing

### Fixes

#### Fix 1: Increase Timeout

**Update `host.json`:**
```json
{
  "functionTimeout": "00:10:00"  // 10 minutes
}
```

#### Fix 2: Optimize Database Queries

- Add indexes
- Use connection pooling
- Reduce query complexity

#### Fix 3: Cold Start Optimization

**Reduce function size:**
- Remove unnecessary dependencies
- Use lazy imports
- Optimize package.json

---

## 8. Configuration Issues

### Symptoms
- Settings not taking effect
- Functions using wrong configuration
- Environment variables not found

### Diagnosis

**List all settings:**
```bash
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[].{Name:name, Value:value}" -o table
```

**Check specific setting:**
```bash
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[?name=='DATABASE_URL'].value" -o tsv
```

### Fixes

#### Fix 1: Update Single Setting

```bash
az functionapp config appsettings set \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --settings "KEY=value"
```

#### Fix 2: Sync from .env.local

```bash
./scripts/configure_function_app_env.sh
```

#### Fix 3: Restart After Config Changes

```bash
az functionapp restart --name func-raglab-uploadworkers --resource-group rag-lab
```

---

## 9. Deployment Issues

### Symptoms
- Git push succeeds but functions don't update
- Manual deployment fails
- Functions show old code

### Fixes

#### Fix 1: Manual Deployment (Most Reliable)

```bash
cd backend/azure_functions
func azure functionapp publish func-raglab-uploadworkers --python
```

This ensures all files are synced.

#### Fix 2: Check Deployment Logs

**Azure Portal:**
1. Go to: func-raglab-uploadworkers → Deployment Center
2. Click: **Logs**
3. Check for errors

#### Fix 3: Verify Files Deployed

**Check deployed files:**
```bash
az functionapp deployment source show \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab
```

#### Fix 4: Clear Deployment Cache

```bash
az functionapp deployment source delete \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab

# Redeploy
cd backend/azure_functions
func azure functionapp publish func-raglab-uploadworkers --python
```

---

## 10. Local Development Issues

### Symptoms
- Functions won't start locally
- Import errors locally
- Queue errors locally

### Fixes

#### Fix 1: Verify Local Setup

```bash
# Check Azurite is running
lsof -i :10000

# Start if not running
./scripts/start_azurite.sh

# Check Supabase is running
cd infra/supabase && supabase status

# Start if not running
supabase start
```

#### Fix 2: Verify local.settings.json

**File:** `backend/azure_functions/local.settings.json`

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python"
  }
}
```

#### Fix 3: Test Locally

```bash
cd backend/azure_functions
func start
```

Should see:
```
Functions:
  chunking-worker: queueTrigger
  embedding-worker: queueTrigger
  indexing-worker: queueTrigger
  ingestion-worker: queueTrigger
```

---

## 11. Using Application Insights for Diagnosis

### Overview

Application Insights provides comprehensive telemetry for Azure Functions. Use KQL queries to diagnose issues faster than streaming logs.

### Quick Diagnosis Workflow

**Step 1:** Check for recent errors
```bash
make logs-errors
```

**Step 2:** If errors found, get operation_Id and query full trace
```bash
# Get operation_Id from error logs, then:
./scripts/query_app_insights.sh operation 30d <operation_id>
```

**Step 3:** Check function execution patterns
```bash
make logs-functions
```

### Common Diagnostic Queries

#### Find All Errors in Last 24 Hours
```bash
TIME_WINDOW=24h make logs-errors
```

#### Find Database Connection Failures
```bash
az monitor app-insights query \
  --app ai-rag-evaluator \
  --resource-group rag-lab \
  --analytics-query "union traces | union exceptions | where timestamp > ago(24h) | where message contains 'database' or message contains 'psycopg2' | where severityLevel >= 3 | project timestamp, message, operation_Id" \
  -o table
```

#### Find Queue Trigger Activity
```bash
make logs-queue
```

#### Find Specific Function Failures
```bash
az monitor app-insights query \
  --app ai-rag-evaluator \
  --resource-group rag-lab \
  --analytics-query "union traces | union exceptions | where timestamp > ago(24h) | where operation_Name contains 'ingestion-worker' | where severityLevel >= 3 | project timestamp, message, operation_Id" \
  -o table
```

### Getting Operation ID from Logs

When you see an error, note the `operation_Id` field. Use it to get the complete trace:

```bash
./scripts/query_app_insights.sh operation 30d <operation_id>
```

This shows the complete execution trace for that specific function invocation.

### Comparing Working vs. Failing Executions

1. **Get operation_Id from a successful execution:**
   ```bash
   make logs-functions
   # Note operation_Id from successful execution
   ```

2. **Get operation_Id from a failed execution:**
   ```bash
   make logs-errors
   # Note operation_Id from failed execution
   ```

3. **Compare the traces:**
   ```bash
   ./scripts/query_app_insights.sh operation 30d <successful_operation_id>
   ./scripts/query_app_insights.sh operation 30d <failed_operation_id>
   ```

4. **Look for differences** in:
   - Environment variables
   - Database connection attempts
   - Message processing steps
   - Error locations

### When to Use Application Insights vs. Log Stream

**Use Application Insights (KQL queries) when:**
- ✅ Looking for patterns across multiple executions
- ✅ Analyzing historical data
- ✅ Finding specific operation traces
- ✅ Comparing different time periods
- ✅ Need structured query results

**Use Log Stream when:**
- ✅ Watching real-time execution
- ✅ Debugging current issue
- ✅ Need immediate feedback
- ✅ Testing changes in real-time

**See [TELEMETRY_GUIDE.md](TELEMETRY_GUIDE.md) for complete telemetry documentation.**

---

## Quick Reference: Diagnostic Commands

```bash
# Check queue health
python scripts/monitor_queue_health.py

# Full diagnostic
python scripts/diagnose_functions.py

# Application Insights telemetry (KQL queries)
make logs-errors          # View recent errors
make logs-functions       # View function execution logs
make logs-queue          # View queue-related logs
make logs-ingestion      # View ingestion worker logs

# Query specific operation
./scripts/query_app_insights.sh operation 30d <operation_id>

# Stream logs (real-time)
az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab

# Check function status
az functionapp function list --name func-raglab-uploadworkers --resource-group rag-lab

# Restart function app
az functionapp restart --name func-raglab-uploadworkers --resource-group rag-lab

# Manual deployment
cd backend/azure_functions && func azure functionapp publish func-raglab-uploadworkers --python
```

---

## Escalation Path

If issues persist after trying these fixes:

1. **Collect diagnostic information:**
   ```bash
   python scripts/diagnose_functions.py > diagnostic_output.txt
   az functionapp log download --name func-raglab-uploadworkers --resource-group rag-lab
   ```

2. **Check Azure service health:**
   - Visit: https://status.azure.com/

3. **Review Azure Functions known issues:**
   - Check: https://github.com/Azure/azure-functions-host/issues

4. **Contact support:**
   - Azure Portal → Help + support → New support request

