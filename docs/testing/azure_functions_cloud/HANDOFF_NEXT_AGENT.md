# Agent Handoff: Azure Functions RAG Pipeline Debugging

**Date**: 2026-01-13  
**Status**: 🔴 CRITICAL BUG BLOCKING E2E PIPELINE  
**Priority**: HIGH - Production Blocker

---

## Your Mission

Fix the Azure Functions ingestion worker to successfully process multi-page PDF documents through the RAG pipeline, then validate the complete end-to-end flow from upload to query.

---

## Current Situation

### ✅ What's Working
- **Infrastructure**: Azure Functions deployed and operational
- **Environment**: ngrok tunnels active for local Supabase access
- **Component Tests**: chunking-worker validated successfully with synthetic data
- **Monitoring**: Comprehensive tools created and validated
- **Queue System**: All queues operational and accessible

### ❌ Critical Bug Blocking Progress
**Ingestion Worker**: Failing on multi-page PDF processing with Azure Document Intelligence API error

```
Error: (InvalidArgument) Invalid argument.
Inner error: {
    "code": "InvalidParameter",
    "message": "The parameter pages is invalid: The page range..."
}
```

**Impact**: All multi-page PDF documents (2+ pages) fail and go to poison queue

**Test Document**:
- **ID**: `f7df526a-97df-4e85-836d-b3478a72ae6d`
- **File**: `scan_classic_hmo.pdf` (232 pages, 2.5 MB)
- **Status**: Stuck at 'uploaded', in poison queue
- **Location**: Poison queue `ingestion-uploads-poison`

---

## Task Breakdown

### Phase 1: Fix the Page Range Bug (HIGH PRIORITY)

#### 1.1 Investigate the Issue

**Location**: `backend/src/services/workers/ingestion_worker.py` and `backend/src/services/rag/ingestion.py`

**What to find**:
```python
# Look for code that formats page parameters for Azure Document Intelligence API
# Current (broken) implementation likely looks like:
pages_param = f"{start_page}-{end_page}"  # e.g., "3-5"

# Possible fixes:
# Option A: Use 0-indexed pages
pages_param = f"{start_page-1}-{end_page-1}"  # e.g., "2-4"

# Option B: Use comma-separated list
pages_param = ",".join(str(p) for p in range(start_page, end_page + 1))  # e.g., "3,4,5"

# Option C: Use 1-indexed with different format
pages_param = f"pages={start_page}-{end_page}"

# Option D: Check API version - parameter name may have changed
```

**Investigation Steps**:
1. Search codebase for Azure Document Intelligence API calls:
   ```bash
   cd /Users/aq_home/1Projects/rag_evaluator
   grep -r "DocumentAnalysis" backend/src/services/
   grep -r "begin_analyze_document" backend/src/services/
   grep -r "pages" backend/src/services/rag/
   ```

2. Check Azure Document Intelligence SDK documentation:
   ```bash
   # Look at the exact API being used
   grep -r "from azure.ai.documentintelligence" backend/src/
   ```

3. Review Application Insights logs for exact error:
   ```bash
   az monitor app-insights query \
     --app ai-rag-evaluator \
     --resource-group rag-lab \
     --analytics-query "exceptions | where timestamp > ago(1h) | where message contains 'InvalidParameter' | take 1 | project details" \
     --output json | python3 -m json.tool
   ```

4. Check what parameters are actually being sent:
   ```bash
   az monitor app-insights query \
     --app ai-rag-evaluator \
     --resource-group rag-lab \
     --analytics-query "traces | where timestamp > ago(1h) | where message contains 'Processing batch' | take 5 | project message"
   ```

#### 1.2 Implement the Fix

**Files to modify**:
- `backend/src/services/rag/ingestion.py` - Text extraction logic
- `backend/src/services/workers/ingestion_worker.py` - Worker orchestration

**Testing strategy**:
```python
# Test locally first if possible
# Create a small test:
# 1. 1-page PDF (should work - no batching)
# 2. 3-page PDF (will trigger batching with pages 1-2, 3-3)
# 3. Full 232-page PDF (comprehensive test)
```

**Validation checklist**:
- [ ] Code compiles without errors
- [ ] Linter passes
- [ ] Logic handles edge cases (single page, odd number of pages)
- [ ] Page ranges are correctly formatted per Azure API docs

#### 1.3 Deploy the Fix

```bash
cd /Users/aq_home/1Projects/rag_evaluator

# Deploy to Azure Functions
./scripts/deploy_azure_functions.sh

# Wait for deployment
echo "Waiting 90 seconds for functions to restart..."
sleep 90
```

#### 1.4 Verify the Fix

```bash
# 1. Clear poison queue
cd /Users/aq_home/1Projects/rag_evaluator
source backend/venv/bin/activate
export AZURE_STORAGE_CONNECTION_STRING="<from .env.local>"
python scripts/monitor_queue_health.py --clear-poison --yes

# 2. Re-enqueue the test document
python3 << 'EOF'
import json
import base64
from azure.storage.queue import QueueServiceClient

conn_str = "DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=raglabqueues;AccountKey=***REDACTED***;BlobEndpoint=https://raglabqueues.blob.core.windows.net/;FileEndpoint=https://raglabqueues.file.core.windows.net/;QueueEndpoint=https://raglabqueues.queue.core.windows.net/;TableEndpoint=https://raglabqueues.table.core.windows.net/"

message = {
    "document_id": "f7df526a-97df-4e85-836d-b3478a72ae6d",
    "source_storage": "supabase",
    "filename": "scan_classic_hmo.pdf",
    "attempt": 1,
    "stage": "uploaded",
    "metadata": {}
}

message_b64 = base64.b64encode(json.dumps(message).encode()).decode()
client = QueueServiceClient.from_connection_string(conn_str)
queue = client.get_queue_client("ingestion-uploads")
queue.send_message(message_b64)
print("✅ Message enqueued for retry")
EOF

# 3. Monitor progress (NEW TOOL!)
python scripts/check_document_progress.py f7df526a-97df-4e85-836d-b3478a72ae6d
```

**Success Criteria for Fix**:
- [ ] Worker executions show `success=True` in Application Insights
- [ ] Document status changes from 'uploaded' → 'parsed'
- [ ] Extracted text exists in database
- [ ] Document progresses to next queue (ingestion-chunking)
- [ ] NO messages in poison queue
- [ ] Application Insights shows "completed batch X/Y" logs

---

### Phase 2: Complete E2E Validation (AFTER Phase 1 Success)

#### 2.1 Monitor Full Pipeline

```bash
# Use the new comprehensive monitoring tool
python scripts/check_document_progress.py --watch f7df526a-97df-4e85-836d-b3478a72ae6d

# Or use pipeline health check
python scripts/pipeline_health_check.py --watch
```

**Expected Timeline**:
- Ingestion: 15-30 minutes (232 pages, 116 batches)
- Chunking: 2-5 minutes
- Embedding: 5-10 minutes (depends on chunk count)
- Indexing: 2-5 minutes

**Total**: 25-50 minutes for full pipeline

#### 2.2 Validate Each Stage

**After Ingestion Completes**:
```bash
# Check status
python scripts/check_document_progress.py f7df526a-97df-4e85-836d-b3478a72ae6d

# Expected: status='parsed', chunks_created=0 (not yet chunked)
```

**After Chunking Completes**:
```bash
# Expected: status='chunked', chunks_created > 0
```

**After Embedding Completes**:
```bash
# Expected: status='embedded'
```

**After Indexing Completes**:
```bash
# Expected: status='indexed'
```

#### 2.3 Verify Azure AI Search Index

```bash
# Check document count in search index
SEARCH_KEY=$(az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[?name=='AZURE_SEARCH_API_KEY'].value" -o tsv)

curl -s -X GET \
  "https://rag-lab-search.search.windows.net/indexes/rag-lab-search/docs/\$count?api-version=2023-11-01" \
  -H "api-key: $SEARCH_KEY"

# Expected: count > 0 (should show indexed chunks)
```

#### 2.4 Test Query Retrieval

```bash
# Start backend API if not running
cd /Users/aq_home/1Projects/rag_evaluator/backend
source venv/bin/activate
export AZURE_STORAGE_QUEUES_CONNECTION_STRING="<from .env.local>"
python -m uvicorn src.api.main:app --port 8000 &

# Wait for API to start
sleep 5

# Test query with HMO-related questions
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "What is covered under the HMO plan?",
    "prompt_version": "v1"
  }' | jq '.'

# Expected: Response with chunks from scan_classic_hmo.pdf
```

**Validation checklist**:
- [ ] Query returns results
- [ ] Results contain text from scan_classic_hmo.pdf
- [ ] Results include relevance scores
- [ ] Response time reasonable (< 5 seconds)

---

### Phase 3: Create Test Report

Update `AZURE_FUNCTIONS_TEST_RESULTS.md` with:

1. **Bug Fix Details**:
   - What was wrong
   - What the fix was
   - Why it works now

2. **E2E Test Results**:
   - Timeline for each stage
   - Final document status
   - Search index statistics
   - Query test results

3. **Worker Performance**:
   - Success rates for each worker
   - Processing times
   - Any warnings or issues

4. **Recommendations**:
   - Monitoring best practices
   - Performance optimizations identified
   - Future improvements

---

## New Monitoring Tools You Have

### 1. Document Progress Checker
```bash
# Check specific document
python scripts/check_document_progress.py <document_id>

# Check most recent
python scripts/check_document_progress.py --latest

# Watch mode (auto-refresh every 30s)
python scripts/check_document_progress.py --watch <document_id>
```

**Shows**:
- ✅ Database status and time in current state
- ✅ Current pipeline stage
- ✅ Queue position (if queued)
- ✅ Poison queue check
- ✅ Recent errors from Application Insights
- ✅ Overall assessment (processing/stuck/failed/complete)

### 2. Pipeline Health Check
```bash
# Overall health check
python scripts/pipeline_health_check.py

# Verbose mode
python scripts/pipeline_health_check.py --verbose

# Watch mode
python scripts/pipeline_health_check.py --watch
```

**Shows**:
- ✅ All queue statistics
- ✅ Poison queue alerts
- ✅ Stuck documents (>30min in same status)
- ✅ Worker success rates (last hour)
- ✅ Document distribution by status
- ✅ Overall system health assessment

### 3. Queue Monitor (existing, improved)
```bash
python scripts/monitor_queue_health.py
python scripts/monitor_queue_health.py --watch
python scripts/monitor_queue_health.py --clear-poison --yes
```

---

## Key Files and Locations

### Code to Fix
- `backend/src/services/rag/ingestion.py` - Text extraction implementation
- `backend/src/services/workers/ingestion_worker.py` - Worker orchestration

### Configuration
- `.env.local` - Local environment variables
- `backend/azure_functions/host.json` - Azure Functions configuration

### Monitoring Scripts (NEW!)
- `scripts/check_document_progress.py` - Document-specific monitoring
- `scripts/pipeline_health_check.py` - Overall system health
- `scripts/monitor_queue_health.py` - Queue monitoring
- `scripts/generate_test_data.py` - Create synthetic test data

### Documentation
- `AZURE_FUNCTIONS_TEST_RESULTS.md` - Test results (update after success)
- `CRITICAL_BUG_FOUND.md` - Bug details and analysis
- `RETROSPECTIVE_CLOUD_TESTING.md` - Lessons learned
- `HANDOFF_NEXT_AGENT.md` - This file

### Test Data
- Document ID: `f7df526a-97df-4e85-836d-b3478a72ae6d`
- File: `docs/inputs/scan_classic_hmo.pdf`
- Status: In poison queue, needs retry after fix

---

## Debugging Resources

### Application Insights Queries

**Recent exceptions**:
```bash
az monitor app-insights query \
  --app ai-rag-evaluator \
  --resource-group rag-lab \
  --analytics-query "exceptions | where timestamp > ago(1h) | order by timestamp desc | take 5 | project timestamp, outerMessage"
```

**Worker execution results**:
```bash
az monitor app-insights query \
  --app ai-rag-evaluator \
  --resource-group rag-lab \
  --analytics-query "requests | where timestamp > ago(1h) | where name contains 'ingestion-worker' | project timestamp, success, duration | order by timestamp desc"
```

**Document-specific logs**:
```bash
az monitor app-insights query \
  --app ai-rag-evaluator \
  --resource-group rag-lab \
  --analytics-query "traces | where timestamp > ago(1h) | where message contains 'f7df526a-97df-4e85-836d-b3478a72ae6d' | order by timestamp desc | take 20"
```

**Progress tracking**:
```bash
az monitor app-insights query \
  --app ai-rag-evaluator \
  --resource-group rag-lab \
  --analytics-query "traces | where message contains 'Progress' or message contains 'completed batch' | order by timestamp desc | take 10"
```

### Azure Functions Management

**List functions**:
```bash
az functionapp list --resource-group rag-lab --query "[].{name:name, state:state}"
```

**Check environment variables**:
```bash
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --output table
```

**Restart functions**:
```bash
az functionapp restart --name func-raglab-uploadworkers --resource-group rag-lab
```

### Queue Management

**Check queue**:
```bash
az storage message peek \
  --queue-name ingestion-uploads \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
  --num-messages 5
```

**Clear queue**:
```bash
az storage message clear \
  --queue-name ingestion-uploads \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING"
```

---

## Success Criteria

### Minimum Viable Success
- [ ] Ingestion worker processes test PDF without errors
- [ ] Document progresses from 'uploaded' to 'parsed'
- [ ] At least one batch successfully extracts text
- [ ] No messages in poison queue for test document

### Full Success
- [ ] Complete E2E pipeline: uploaded → parsed → chunked → embedded → indexed
- [ ] Document fully indexed in Azure AI Search (count > 0)
- [ ] Query API returns results from the test document
- [ ] All worker executions show success=True
- [ ] Pipeline health check shows "healthy" status
- [ ] Test results documented in updated report

### Stretch Goals
- [ ] Test with additional PDF documents (varying sizes)
- [ ] Measure and document processing times
- [ ] Identify performance optimization opportunities
- [ ] Add automated integration tests

---

## Important Reminders

### Don't Make the Same Mistakes

1. **✅ DO**: Check database status changes, not just activity logs
2. **✅ DO**: Check poison queues within first 60 seconds
3. **✅ DO**: Verify worker execution success (success=True/False)
4. **✅ DO**: Look for completion logs, not just start logs
5. **✅ DO**: Use multiple verification methods (database + queues + logs)

6. **❌ DON'T**: Assume "Processing batch X" means progress
7. **❌ DON'T**: Assume no exceptions means success
8. **❌ DON'T**: Wait 30 minutes before checking poison queues
9. **❌ DON'T**: Trust early activity logs without state verification

### Use the New Tools

The monitoring tools were specifically designed to avoid the false positives that occurred in initial testing:

```bash
# Quick health check (30 seconds)
python scripts/pipeline_health_check.py

# Document-specific check (for test document)
python scripts/check_document_progress.py f7df526a-97df-4e85-836d-b3478a72ae6d

# Watch mode for continuous monitoring
python scripts/check_document_progress.py --watch f7df526a-97df-4e85-836d-b3478a72ae6d
```

These tools check **state**, not just **activity**.

---

## Environment Setup

```bash
# Navigate to project
cd /Users/aq_home/1Projects/rag_evaluator

# Activate venv
source backend/venv/bin/activate

# Set Azure connection string
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=raglabqueues;AccountKey=***REDACTED***;BlobEndpoint=https://raglabqueues.blob.core.windows.net/;FileEndpoint=https://raglabqueues.file.core.windows.net/;QueueEndpoint=https://raglabqueues.queue.core.windows.net/;TableEndpoint=https://raglabqueues.table.core.windows.net/"

# Verify ngrok tunnels are running
curl -s http://127.0.0.1:4040/api/tunnels | python3 -c "import sys, json; data=json.load(sys.stdin); print('\\n'.join([t['public_url'] + ' -> ' + t['config']['addr'] for t in data.get('tunnels', [])]))"

# Expected output:
# tcp://4.tcp.us-cal-1.ngrok.io:18029 -> localhost:54322
# https://....ngrok-free.dev -> localhost:54321
```

---

## Questions or Issues?

### If stuck on the bug fix:
1. Read `CRITICAL_BUG_FOUND.md` for detailed analysis
2. Check Azure Document Intelligence SDK documentation
3. Review Application Insights for exact error messages
4. Test hypothesis locally before deploying

### If monitoring shows unexpected behavior:
1. Use `python scripts/check_document_progress.py <doc_id>` for details
2. Check poison queues first
3. Review Application Insights worker execution results
4. Verify ngrok tunnels are still active

### If E2E test fails:
1. Identify which stage failed (use progress checker)
2. Check that specific worker's logs
3. Verify queue transitions happened
4. Check for messages in poison queues

---

## Timeline Estimate

- **Bug Investigation**: 30-60 minutes
- **Implement Fix**: 30-60 minutes
- **Deploy & Test**: 15-30 minutes
- **E2E Validation**: 30-60 minutes (waiting for processing)
- **Documentation**: 15-30 minutes

**Total**: 2-4 hours end-to-end

---

## Final Notes

This handoff represents ~3 hours of investigation, testing, and tool development. The infrastructure is solid, the monitoring is comprehensive, and the bug is well-documented. You have everything you need to:

1. Fix the page range parameter bug
2. Validate the fix works
3. Complete the E2E test
4. Document the success

The test document is ready, the tools are ready, and the path forward is clear. 

**Good luck! 🚀**

---

**Previous Agent**: Testing complete, bug identified, monitoring improved, documentation comprehensive
**Status**: Ready for next agent to debug and iterate
**Handoff Date**: 2026-01-13 17:30 UTC

