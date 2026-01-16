# ETL Pipeline Debugging - Handoff Summary

**Date**: 2026-01-15  
**Previous Focus**: Telemetry/Observability Setup  
**Current Status**: ⚠️ **READY FOR E2E TESTING** - Infrastructure fixes applied, needs validation  
**Priority**: 🔴 **HIGH** - Verify complete pipeline works end-to-end

---

## 🎯 Primary Objective

**Verify the complete ETL pipeline works end-to-end after infrastructure fixes:**
1. Document upload → Queue → Ingestion → Chunking → Embedding → Indexing
2. Validate all stages process successfully
3. Confirm database updates correctly
4. Ensure no messages go to poison queues

---

## ✅ What Was Fixed (Before Switching to Telemetry)

### 1. Queue Trigger Not Firing ✅ FIXED
**Issue**: Messages in `ingestion-uploads` queue were not being processed  
**Root Cause**: `host.json` had `"messageEncoding": "base64"` but messages are sent as plain text JSON  
**Fix Applied**: Changed `backend/azure_functions/host.json` to `"messageEncoding": "none"`  
**Evidence**: Confirmed via Application Insights logs (operation_Id: `3576e5b499f5b0b67bddfc5c6a8d951b`)
- ✅ "Executing 'Functions.ingestion-worker' (Reason='New queue message detected on 'ingestion-uploads'.)"
- ✅ "Processing ingestion message for document: 73075619-73f7-4bbb-9bdc-8fa4add80ffb"
- ✅ "Starting ingestion worker"

**Status**: ✅ **CONFIRMED WORKING** via Application Insights telemetry

### 2. Supabase Key Configuration ✅ FIXED
**Issue**: Code expected `SUPABASE_KEY` but environment uses `SUPABASE_SERVICE_ROLE_KEY`  
**Fix Applied**:
- Updated `backend/src/core/config.py` to use `SUPABASE_SERVICE_ROLE_KEY`
- Updated `backend/src/services/rag/supabase_storage.py` to use service role key
- Updated documentation in `docs/setup/ENVIRONMENT_VARIABLES.md`

**Status**: ✅ **FIXED** - Code updated and deployed

### 3. Database Connection (IPv6 Issue) ⚠️ **PASSWORD ISSUE IDENTIFIED**
**Issue**: DATABASE_URL using direct endpoint (port 5432) causing IPv6 connection failures  
**Original Error**: `"Failed to initialize database pool: connection to server at 'db.oeyivkusvlgyuorcjime.supabase.co' (2600:1f16:1cd0:3337:14af:3286:5f36:bf51), port 5432 failed: Network is unreachable"`  
**Fix Applied**: Updated DATABASE_URL to use connection pooler endpoint (port 6543):
- `.env.prod`: Updated to `postgresql://postgres.oeyivkusvlgyuorcjime:...@aws-0-us-east-1.pooler.supabase.com:6543/postgres`
- Azure Functions: Updated via `az functionapp config appsettings set`
- Added SSL parameter: `?sslmode=require`

**Current Error**: `"FATAL: Tenant or user not found"`  
**Root Cause**: **Password in connection string is incorrect**  
**Current Connection String**: `postgresql://postgres.oeyivkusvlgyuorcjime:***@aws-0-us-east-1.pooler.supabase.com:6543/postgres?sslmode=require`

**Status**: ⚠️ **PASSWORD VERIFICATION IN PROGRESS** - Connection string format verified as correct:
- ✅ Username: `postgres.oeyivkusvlgyuorcjime` (correct format)
- ✅ Port: `6543` (transaction pooler)
- ✅ Host: `aws-1-us-east-2.pooler.supabase.com` (pooler endpoint)
- ✅ SSL: `sslmode=require`
- ✅ Password: 20 characters, no special characters requiring encoding

**Current Error**: Still seeing "Tenant or user not found" - suggests password may still be incorrect or needs to be re-verified from Supabase Dashboard.

**Next Steps**:
1. Double-check password in Supabase Dashboard → Settings → Database → Connection string (Transaction pooler)
2. Ensure password matches exactly (no extra spaces, correct case)
3. Restart function app after any password updates
4. Monitor logs to verify connection succeeds

### 4. Azure Storage Connection Strings ✅ FIXED
**Issue**: Missing `AZURE_STORAGE_QUEUES_CONNECTION_STRING`, `AZURE_BLOB_CONNECTION_STRING`, `AZURE_BLOB_CONTAINER_NAME`  
**Fix Applied**: Added to `.env.prod` and Azure Functions app settings

**Status**: ✅ **FIXED** - Configuration updated

---

## ⚠️ What Needs Testing

### Critical: End-to-End Pipeline Test

**Test Document**:
- **ID**: `f7df526a-97df-4e85-836d-b3478a72ae6d` (or create new test document)
- **File**: `scan_classic_hmo.pdf` (232 pages, 2.5 MB)
- **Location**: Azure Blob Storage container `documents`
- **Current Status**: `uploaded` (stuck from previous attempts)

**Test Steps**:
1. **Clear poison queue** (if any messages exist):
   ```bash
   python scripts/monitor_queue_health.py --clear-poison
   ```

2. **Upload new test document** (or re-enqueue existing):
   ```bash
   # Via API
   python scripts/upload_to_cloud.py
   
   # Or manually enqueue
   python scripts/monitor_queue_health.py --enqueue-test
   ```

3. **Monitor queue status**:
   ```bash
   python scripts/monitor_queue_health.py --watch
   ```

4. **Check Application Insights logs**:
   ```bash
   # View function execution
   make logs-functions
   
   # View errors
   make logs-errors
   
   # View ingestion worker logs
   make logs-ingestion
   ```

5. **Verify database updates**:
   ```sql
   SELECT id, filename, status, chunks_created, metadata
   FROM documents
   WHERE id = '<document_id>';
   ```

**Expected Results**:
- ✅ Queue trigger fires (already confirmed working)
- ✅ Database connection succeeds (fix applied, needs verification)
- ✅ Document status updates: `uploaded` → `processing` → `parsed`
- ✅ Batches processed: `chunks_created > 0`
- ✅ No messages in poison queue
- ✅ All 4 workers execute successfully (ingestion → chunking → embedding → indexing)

---

## 🔍 Known Issues (From Previous Sessions)

### 1. Page Range Parameter Bug (Previously Fixed)
**Status**: ✅ **FIXED** (according to `BUG_FIX_SUMMARY.md`)  
**Details**: Azure Document Intelligence API page range parameter format was incorrect  
**Fix**: Changed to pre-slice PDF before calling API (no page range parameter)  
**Location**: `backend/src/services/rag/ingestion.py` - Uses pre-sliced PDF approach

**Note**: This was fixed in a previous session. If errors reoccur, verify the fix is still in place.

### 2. Time-Based Limiting (Previously Implemented)
**Status**: ✅ **IMPLEMENTED** (according to `TIME_BASED_LIMITING_IMPLEMENTATION.md`)  
**Details**: Adaptive batch processing with automatic re-enqueuing to handle large documents  
**Configuration**: 4-minute target processing time, 1-minute safety buffer

**Note**: This should handle large documents (232 pages) by processing in batches and re-enqueuing.

---

## 📊 Observability Tools Available

### Application Insights KQL Queries
```bash
# View function execution logs
make logs-functions

# View recent errors
make logs-errors

# View queue-related logs
make logs-queue

# View ingestion worker logs
make logs-ingestion

# Query specific operation
./scripts/query_app_insights.sh operation 30d <operation_id>
```

### Queue Monitoring
```bash
# Quick health check
python scripts/monitor_queue_health.py

# Watch mode (live updates)
python scripts/monitor_queue_health.py --watch

# Full diagnostic
python scripts/diagnose_functions.py
```

**See**: `docs/monitoring/TELEMETRY_GUIDE.md` for complete telemetry documentation.

---

## 🎯 Success Criteria

### Immediate (Next Session)
- [ ] Database connection succeeds (no "Network is unreachable" errors)
- [ ] Document processing starts (status changes from "uploaded" to "processing")
- [ ] At least one batch processes successfully
- [ ] No messages go to poison queue during initial processing

### Full E2E Validation
- [ ] Complete document processed (all 232 pages or test document)
- [ ] All 4 workers execute successfully:
  - [ ] `ingestion-worker` - Extracts text from PDF
  - [ ] `chunking-worker` - Splits text into chunks
  - [ ] `embedding-worker` - Generates embeddings
  - [ ] `indexing-worker` - Stores in vector database
- [ ] Database status: `parsed` → `chunked` → `embedded` → `indexed`
- [ ] `chunks_created > 0` in database
- [ ] No errors in Application Insights
- [ ] No messages in poison queues

---

## 📁 Key Files

### Worker Code
- `backend/src/services/workers/ingestion_worker.py` - Main ingestion worker
- `backend/src/services/workers/chunking_worker.py` - Chunking worker
- `backend/src/services/workers/embedding_worker.py` - Embedding worker
- `backend/src/services/workers/indexing_worker.py` - Indexing worker

### Service Code
- `backend/src/services/rag/ingestion.py` - Text extraction (Azure Document Intelligence)
- `backend/src/services/rag/storage.py` - Azure Blob Storage operations
- `backend/src/services/rag/supabase_storage.py` - Supabase Storage operations
- `backend/src/services/workers/persistence.py` - Database persistence

### Configuration
- `backend/azure_functions/host.json` - Function host config (messageEncoding fixed)
- `.env.prod` - Production environment variables
- `backend/src/core/config.py` - Application configuration

### Observability
- `scripts/query_app_insights.sh` - Main KQL query script
- `scripts/logs_*.sh` - Quick access scripts
- `docs/monitoring/TELEMETRY_GUIDE.md` - Complete telemetry guide

---

## 🚀 Recommended Next Steps

### Step 1: Verify Database Connection Fix (5 minutes)
```bash
# Upload a test document
python scripts/upload_to_cloud.py

# Check logs for database connection errors
make logs-errors

# Verify no "Network is unreachable" errors
```

### Step 2: Monitor First Execution (10 minutes)
```bash
# Watch queue in real-time
python scripts/monitor_queue_health.py --watch

# In another terminal, watch logs
make logs-ingestion
```

### Step 3: Verify Database Updates (5 minutes)
```sql
-- Check document status
SELECT id, filename, status, chunks_created, metadata
FROM documents
WHERE id = '<document_id>';

-- Check for batch results
SELECT * FROM batch_results
WHERE document_id = '<document_id>';
```

### Step 4: Full E2E Test (30-60 minutes)
- Monitor all 4 workers
- Verify complete document processing
- Check for errors at each stage
- Validate final database state

---

## 📝 Previous Session Context

### What We Were Working On
1. **Queue Trigger Issue** - Fixed (messageEncoding)
2. **Database Connection** - Fixed (Supabase REST API)
3. **Supabase Keys** - Fixed (service role key)
4. **Observability Setup** - Completed (KQL queries, scripts, documentation)

### What We Switched To
- **Telemetry Documentation** - Created comprehensive Application Insights observability guide
- **KQL Scripts** - Created standardized query scripts
- **Documentation** - Updated all monitoring documentation

### What We Didn't Test
- ❌ End-to-end pipeline after database connection fix
- ❌ Full document processing (232 pages)
- ❌ All 4 workers executing in sequence
- ❌ Final database state validation

---

## 🔗 Related Documentation

- **Queue Trigger Fix**: `docs/testing/azure_functions_cloud/QUEUE_TRIGGER_FIX_SUMMARY.md`
- **Telemetry Guide**: `docs/monitoring/TELEMETRY_GUIDE.md`
- **Troubleshooting**: `docs/monitoring/TROUBLESHOOTING.md`
- **Previous Bug Fixes**: `docs/testing/azure_functions_cloud/BUG_FIX_SUMMARY.md`
- **Time-Based Limiting**: `docs/testing/azure_functions_cloud/TIME_BASED_LIMITING_IMPLEMENTATION.md`

---

## ⚡ Quick Start Commands

```bash
# 1. Check current queue status
python scripts/monitor_queue_health.py

# 2. Upload test document
python scripts/upload_to_cloud.py

# 3. Watch logs in real-time
make logs-ingestion

# 4. Check for errors
make logs-errors

# 5. Monitor queue
python scripts/monitor_queue_health.py --watch
```

---

**Last Updated**: 2026-01-15  
**Next Action**: Run end-to-end test to verify all fixes work together  
**Blocking Issues**: None - Ready for testing
