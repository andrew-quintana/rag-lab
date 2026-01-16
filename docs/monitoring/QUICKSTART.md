# Azure Functions Monitoring - Quick Start Guide

**5-minute guide to monitoring and diagnosing Azure Functions**

## Prerequisites

- Azure CLI installed and logged in (`az login`)
- Python environment with dependencies installed
- `.env.local` configured with Azure Storage connection string

## Quick Health Check (30 seconds)

Check if your Azure Functions are healthy:

```bash
python scripts/monitor_queue_health.py
```

**What you'll see:**
- ✅ Green checkmarks = healthy, no messages
- 📝 Blue numbers = messages queued (normal)
- ⚠️ Yellow warnings = poison queue has messages (needs attention)
- ❌ Red errors = connectivity or configuration issues

**Interpreting results:**
- **All green** → Everything is healthy
- **Messages in main queues** → Functions are processing (normal)
- **Messages in poison queues** → Functions are failing (investigate)

## Full Diagnostic (2 minutes)

Run a comprehensive diagnostic that sends a test message:

```bash
python scripts/diagnose_functions.py
```

This script will:
1. Check queue status
2. Send a test message
3. Monitor for 60 seconds
4. Tell you if it succeeded, failed, or timed out
5. Provide next steps

## Common Scenarios

### Scenario 1: Everything is Healthy ✅

```
Queue Health Status
==================================================
ingestion-uploads:        ✅ Empty
ingestion-uploads-poison: ✅ Empty
...
✅ All queues are empty and healthy
```

**Action:** None needed. Functions are working correctly.

---

### Scenario 2: Messages in Poison Queue ⚠️

```
Queue Health Status
==================================================
ingestion-uploads:        ✅ Empty
ingestion-uploads-poison: ⚠️  3 message(s)
...
⚠️ WARNING: 3 message(s) in poison queues
```

**Action:** Investigate the failures.

**Steps:**
1. **Peek at the poison queue** to see what failed:
   ```bash
   python scripts/monitor_queue_health.py --peek ingestion-uploads-poison
   ```

2. **Check Azure Function logs** for the error:
   ```bash
   az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab
   ```

3. **Look for common errors** (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)):
   - `ModuleNotFoundError: No module named 'src'` → Import path issue
   - `psycopg2.OperationalError` → Database connection issue
   - `Message decoding has failed` → host.json configuration issue

4. **After fixing**, clear the poison queue:
   ```bash
   python scripts/monitor_queue_health.py --clear-poison
   ```

---

### Scenario 3: Functions Not Processing Messages ⏱️

```
Test message sent successfully
⏱️  Monitoring timeout reached (60s)
⏱️  Function did not process message within timeout
```

**Possible causes:**
- Functions are disabled
- Cold start delay (2-3 minutes after deployment)
- Functions not polling queues

**Steps:**
1. **Check function status** in Azure Portal:
   - Go to: Portal → func-raglab-uploadworkers → Functions
   - Verify functions are not "Disabled"

2. **Check logs for polling activity**:
   ```bash
   az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab | grep -i poll
   ```
   
   Should see lines like:
   ```
   Poll for function 'ingestion-worker' on queue 'ingestion-uploads'...
   ```

3. **If no polling activity**, restart the function app:
   ```bash
   az functionapp restart --name func-raglab-uploadworkers --resource-group rag-lab
   ```

4. **Wait 2-3 minutes** for cold start, then retry:
   ```bash
   python scripts/diagnose_functions.py
   ```

---

## Watch Mode - Live Monitoring

Monitor queues in real-time (refreshes every 10 seconds):

```bash
python scripts/monitor_queue_health.py --watch
```

Press `Ctrl+C` to stop.

**Use case:** Keep this running while testing or debugging to see queue changes in real-time.

---

## Application Insights Telemetry (KQL Queries)

### Quick Log Queries

```bash
# View recent function execution logs
make logs-functions

# View recent errors
make logs-errors

# View queue-related logs
make logs-queue

# View ingestion worker logs
make logs-ingestion

# Custom time window
TIME_WINDOW=30m make logs-functions
```

### Query Specific Operation

```bash
# Query by operation_Id (from error logs or Application Insights)
./scripts/query_app_insights.sh operation 30d <operation_id>
```

**See [TELEMETRY_GUIDE.md](TELEMETRY_GUIDE.md) for complete telemetry documentation.**

---

## Azure Portal Quick Links

### View Function Logs
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to: **func-raglab-uploadworkers**
3. Select: **Functions** → Choose a function (e.g., `ingestion-worker`)
4. Click: **Monitor** tab
5. View recent invocations and their status

### View Application Insights Logs
1. Go to: **Application Insights** → `ai-rag-evaluator`
2. Click: **Logs** (in left sidebar)
3. Run KQL queries to analyze telemetry
4. Or use scripts: `make logs-*` commands

### Check Function Configuration
1. Go to: **func-raglab-uploadworkers**
2. Click: **Configuration**
3. Verify key settings:
   - `AzureWebJobsStorage`
   - `DATABASE_URL`
   - `SUPABASE_URL`

### Stream Live Logs
1. Go to: **func-raglab-uploadworkers**
2. Click: **Log stream** (in left sidebar)
3. Watch live logs in real-time

---

## Common Commands Reference

### Queue Monitoring
```bash
# Quick health check
python scripts/monitor_queue_health.py

# Full diagnostic with test message
python scripts/diagnose_functions.py

# Watch mode (live updates)
python scripts/monitor_queue_health.py --watch

# Peek at specific queue
python scripts/monitor_queue_health.py --peek ingestion-uploads-poison

# Clear poison queues
python scripts/monitor_queue_health.py --clear-poison
```

### Application Insights Telemetry (KQL)
```bash
# View function execution logs
make logs-functions

# View recent errors
make logs-errors

# View queue-related logs
make logs-queue

# View ingestion worker logs
make logs-ingestion

# Custom time window
TIME_WINDOW=30m make logs-functions

# Query specific operation
./scripts/query_app_insights.sh operation 30d <operation_id>
```

### Azure CLI Commands
```bash
# Stream Azure Function logs
az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab

# Check function status
az functionapp function list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[].{Name:name, Status:config.disabled}" -o table

# Restart function app
az functionapp restart --name func-raglab-uploadworkers --resource-group rag-lab
```

---

## Monitoring Test Suite

Run pytest-based monitoring tests:

```bash
cd backend
pytest tests/monitoring/test_function_monitoring.py -v -s
```

The `-s` flag shows manual verification prompts.

Tests include:
- Queue connectivity check
- Queue health status
- Function processing with manual verification
- Function status checklist

---

## When to Use Each Tool

| Tool | When to Use | Time |
|------|-------------|------|
| `monitor_queue_health.py` | Quick check if everything is OK | 10s |
| `diagnose_functions.py` | Comprehensive test with test message | 2min |
| `monitor_queue_health.py --watch` | Live monitoring during debugging | Continuous |
| `az functionapp log tail` | See detailed logs and errors | Continuous |
| Monitoring tests | Automated verification in CI/CD | 2-5min |

---

## Next Steps

- **Telemetry & KQL Queries?** See [TELEMETRY_GUIDE.md](TELEMETRY_GUIDE.md)
- **Troubleshooting?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Deployment?** See [.cursorrules-azure-functions](../../.cursorrules-azure-functions)
- **Configuration?** See [configuration_guide.md](../initiatives/rag_system/worker_queue_conversion/002_codebase_consolidation/notes/deployment/configuration_guide.md)

---

## Tips

💡 **Run health check before starting work** - Catch issues early

💡 **Keep logs streaming during testing** - See issues immediately

💡 **Don't clear poison queues without investigating** - You'll lose the error information

💡 **Cold starts are normal** - First run after deployment takes 2-3 minutes

💡 **Use watch mode during active development** - See queue changes in real-time

