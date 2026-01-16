# ETL Pipeline Testing - Resume Prompt

**Date**: 2026-01-16  
**Status**: ✅ **READY FOR TESTING** - Database connections fixed (Supabase REST API)

---

## Context

The ETL pipeline has been migrated to use **Supabase REST API** for all database operations, resolving previous connection issues. All persistence functions now use `SupabaseDatabaseService` instead of direct PostgreSQL connections.

**Previous Issues (Resolved)**:
- ❌ Direct PostgreSQL connection failures
- ❌ Connection pooler authentication issues
- ❌ Password/hostname mismatches

**Current Status**:
- ✅ All database operations use Supabase REST API
- ✅ No connection strings needed
- ✅ More reliable in serverless environment
- ✅ Code deployed to Azure Functions

---

## ETL Pipeline Overview

The pipeline processes documents through these stages:

1. **Upload** → Document uploaded via API, stored in Supabase Storage
2. **Ingestion** → Extract text from PDF using Azure Document Intelligence
3. **Chunking** → Split text into chunks
4. **Embedding** → Generate vector embeddings for chunks
5. **Indexing** → Index chunks to Azure AI Search

Each stage is handled by an Azure Function worker that:
- Reads messages from Azure Storage Queue
- Processes the document
- Updates document status in database (via Supabase REST API)
- Enqueues message to next stage queue

---

## Testing Objectives

### Primary Goals

1. **End-to-End Pipeline Validation**
   - Upload a test document
   - Verify all stages complete successfully
   - Confirm document reaches "indexed" status
   - Verify chunks are in Azure AI Search

2. **Database Operations Verification**
   - Confirm all status updates work via REST API
   - Verify extracted text is persisted
   - Verify chunks are stored correctly
   - Verify metadata updates work

3. **Queue Health Monitoring**
   - Monitor all queues during processing
   - Verify no messages go to poison queues
   - Confirm proper message flow between stages

4. **Error Handling**
   - Test retry logic
   - Verify dead-letter queue handling
   - Confirm error logging

---

## Test Plan

### Phase 1: Pre-Test Verification

**1.1 Verify Environment Configuration**
```bash
# Check Azure Functions are deployed
az functionapp list --resource-group rag-lab --query "[].name"

# Verify environment variables are set
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[?name=='SUPABASE_URL' || name=='SUPABASE_ANON_KEY' || name=='SUPABASE_DB_URL' || name=='SUPABASE_DB_PASSWORD']"
```

**1.2 Verify Queue Health**
```bash
# Check all queues are empty/healthy
python3 scripts/monitor_queue_health.py
```

**Expected**: All main queues empty, no poison queue messages

**1.3 Verify Database Connection**
```bash
# Send a test message and check logs for REST API success
python3 scripts/send_test_message.py

# Monitor logs for REST API initialization
make logs-ingestion TIME_WINDOW=2m | grep -E "(REST API|Supabase REST API|Got document status)"
```

**Expected**: Logs show "Supabase REST API client initialized" and successful status checks

---

### Phase 2: Single Document E2E Test

**2.1 Upload Test Document**

Upload a small test PDF (1-2 pages) via the API:

```bash
# Use the upload endpoint
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@test-document.pdf" \
  -H "Authorization: Bearer $API_KEY"
```

**Verify**:
- Document record created in database (status: "uploaded")
- File stored in Supabase Storage
- Message enqueued to `ingestion-uploads` queue

**2.2 Monitor Pipeline Execution**

```bash
# Monitor all queues
python3 scripts/monitor_queue_health.py --watch

# Monitor ingestion worker logs
make logs-ingestion TIME_WINDOW=5m

# Monitor chunking worker logs
make logs-chunking TIME_WINDOW=5m

# Monitor embedding worker logs
make logs-embedding TIME_WINDOW=5m

# Monitor indexing worker logs
make logs-indexing TIME_WINDOW=5m
```

**Expected Flow**:
1. `ingestion-uploads` → Message processed → Message to `ingestion-chunking`
2. `ingestion-chunking` → Message processed → Message to `ingestion-embeddings`
3. `ingestion-embeddings` → Message processed → Message to `ingestion-indexing`
4. `ingestion-indexing` → Message processed → No more messages

**2.3 Verify Database Updates**

Check document status progression:
```sql
-- In Supabase SQL Editor or via REST API
SELECT id, filename, status, chunks_created, 
       metadata->>'ingestion' as ingestion_meta
FROM documents
WHERE filename = 'test-document.pdf'
ORDER BY upload_timestamp DESC
LIMIT 1;
```

**Expected Status Progression**:
- `uploaded` → `parsing` → `parsed` → `chunking` → `chunked` → `embedding` → `embedded` → `indexing` → `indexed`

**2.4 Verify Data Persistence**

```sql
-- Check extracted text exists
SELECT id, filename, 
       LENGTH(extracted_text) as text_length,
       chunks_created
FROM documents
WHERE filename = 'test-document.pdf';

-- Check chunks exist
SELECT COUNT(*) as chunk_count
FROM chunks
WHERE document_id = (
  SELECT id FROM documents WHERE filename = 'test-document.pdf' LIMIT 1
);
```

**Expected**:
- `extracted_text` is not null and has content
- `chunks_created` > 0
- Chunks exist in database

**2.5 Verify Azure AI Search Index**

```bash
# Query Azure AI Search to verify chunks are indexed
# (Use Azure Portal or Azure CLI)
az search document search \
  --service-name <search-service> \
  --index-name <index-name> \
  --query-text "test" \
  --query-parameters "searchFields=text"
```

**Expected**: Chunks from test document appear in search results

---

### Phase 3: Error Handling Tests

**3.1 Test Invalid Document**

Upload a corrupted/invalid file and verify:
- Error is caught and logged
- Document status set to `failed_*`
- Message goes to poison queue after retries
- Error details in document metadata

**3.2 Test Large Document**

Upload a large PDF (>50MB) and verify:
- File size validation works
- Appropriate error message
- Document status reflects failure

**3.3 Test Network Interruption**

Simulate network issues and verify:
- Retry logic works
- Exponential backoff
- Dead-letter queue after max retries

---

### Phase 4: Load Testing

**4.1 Multiple Documents**

Upload 5-10 documents simultaneously and verify:
- All documents process successfully
- No queue congestion
- No database connection issues
- All reach "indexed" status

**4.2 Monitor Performance**

Track:
- Time per stage
- Queue processing rate
- Database operation latency
- Any errors or retries

---

## Success Criteria

### ✅ Pipeline Completes Successfully

- [ ] Document uploads successfully
- [ ] All 4 worker stages execute
- [ ] Document reaches "indexed" status
- [ ] No messages in poison queues
- [ ] Chunks indexed to Azure AI Search

### ✅ Database Operations Work

- [ ] Status updates via REST API succeed
- [ ] Extracted text persisted correctly
- [ ] Chunks stored in database
- [ ] Metadata updates work
- [ ] No connection errors in logs

### ✅ Error Handling Works

- [ ] Invalid documents handled gracefully
- [ ] Retry logic functions correctly
- [ ] Dead-letter queue receives failed messages
- [ ] Error details logged appropriately

### ✅ Performance Acceptable

- [ ] Single document processes in < 5 minutes
- [ ] Multiple documents process concurrently
- [ ] No queue backlog
- [ ] Database operations complete quickly

---

## Monitoring Commands

### Queue Health
```bash
# Check queue status
python3 scripts/monitor_queue_health.py

# Watch queues in real-time
python3 scripts/monitor_queue_health.py --watch
```

### Application Insights Logs
```bash
# Ingestion worker
make logs-ingestion TIME_WINDOW=5m

# Chunking worker
make logs-chunking TIME_WINDOW=5m

# Embedding worker
make logs-embedding TIME_WINDOW=5m

# Indexing worker
make logs-indexing TIME_WINDOW=5m

# All errors
make logs-errors TIME_WINDOW=10m
```

### Database Verification
```bash
# Check document status via Supabase Dashboard
# Or use REST API directly
curl -X GET \
  "https://<project>.supabase.co/rest/v1/documents?select=*&order=upload_timestamp.desc&limit=5" \
  -H "apikey: <service-role-key>" \
  -H "Authorization: Bearer <service-role-key>"
```

---

## Troubleshooting

### If Pipeline Stalls

1. **Check Queue Status**
   ```bash
   python3 scripts/monitor_queue_health.py
   ```
   - Are messages stuck in a queue?
   - Are there poison queue messages?

2. **Check Worker Logs**
   ```bash
   make logs-ingestion TIME_WINDOW=10m | grep -E "(Error|Failed|Exception)"
   ```
   - Look for database connection errors
   - Look for REST API failures
   - Check for retry attempts

3. **Check Database**
   - Verify document status in Supabase
   - Check if status is stuck at a particular stage
   - Review metadata for error details

4. **Check Azure Functions**
   ```bash
   az functionapp list-functions --name func-raglab-uploadworkers --resource-group rag-lab
   az functionapp show --name func-raglab-uploadworkers --resource-group rag-lab --query "state"
   ```

### If Database Operations Fail

1. **Verify REST API Configuration**
   - Check `SUPABASE_URL` is set correctly
   - Check `SUPABASE_ANON_KEY` is valid
   - Verify anonymous key has proper permissions

2. **Check REST API Logs**
   ```bash
   make logs-ingestion TIME_WINDOW=5m | grep -E "(REST API|Supabase|HTTP Request)"
   ```
   - Look for HTTP 200 OK responses
   - Check for authentication errors
   - Verify API requests are reaching Supabase

3. **Test REST API Directly**
   ```bash
   # Test status check
   curl -X GET \
     "https://<project>.supabase.co/rest/v1/documents?select=status&id=eq.<document-id>" \
     -H "apikey: <anon-key>" \
     -H "Authorization: Bearer <anon-key>"
   ```

---

## Next Steps After Successful Testing

1. **Document Results**
   - Record test results
   - Note any issues found
   - Document performance metrics

2. **Production Readiness**
   - Verify all success criteria met
   - Review error handling
   - Confirm monitoring is in place

3. **Optimization**
   - Identify bottlenecks
   - Optimize slow stages
   - Improve error messages

---

## Test Checklist

Use this checklist when running tests:

- [ ] Pre-test verification complete
- [ ] Test document uploaded
- [ ] Ingestion stage completes
- [ ] Chunking stage completes
- [ ] Embedding stage completes
- [ ] Indexing stage completes
- [ ] Document status is "indexed"
- [ ] Extracted text persisted
- [ ] Chunks stored in database
- [ ] Chunks indexed to Azure AI Search
- [ ] No poison queue messages
- [ ] Error handling tested
- [ ] Load testing completed
- [ ] Performance acceptable
- [ ] All success criteria met

---

**Ready to begin testing!** Start with Phase 1 (Pre-Test Verification) and proceed through each phase systematically.
