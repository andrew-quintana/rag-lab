# Critical Bug: Ingestion Worker Failing on Real PDFs

**Severity**: 🔴 CRITICAL - Blocks all document processing  
**Discovered**: 2026-01-13 17:15 UTC  
**Status**: NOT FIXED

---

## Summary

The Azure Functions ingestion worker **fails to process real PDF documents** due to an incorrect page range parameter format when calling Azure Document Intelligence API. Documents get stuck in poison queues and never progress beyond the 'uploaded' status.

---

## Error Details

### Error Message
```
Error: (InvalidArgument) Invalid argument.
Code: InvalidArgument
Message: Invalid argument.
Inner error: {
    "code": "InvalidParameter",
    "message": "The parameter pages is invalid: The page range..."
}
```

### Location
- **Function**: `ingestion-worker`
- **Service**: Azure Document Intelligence API call
- **Phase**: Batch text extraction

### Reproduction
1. Upload any PDF with 2+ pages (triggers batch processing)
2. Document is enqueued to `ingestion-uploads`
3. Worker downloads file successfully
4. Worker detects page count correctly
5. **Worker fails when calling Document Intelligence API with page range parameter**
6. After 5 retry attempts, document moves to poison queue
7. Document remains stuck at 'uploaded' status forever

---

## Test Case That Revealed the Bug

**Document**: `scan_classic_hmo.pdf`  
**ID**: `f7df526a-97df-4e85-836d-b3478a72ae6d`  
**Size**: 2,544,678 bytes (2.5 MB)  
**Pages**: 232 pages  
**Batches**: 116 batches (batch_size=2)

### Worker Execution Log
```
2026-01-13T17:12:40 - Processing document
2026-01-13T17:12:43 - Downloaded 2544678 bytes
2026-01-13T17:12:43 - Detected 232 pages
2026-01-13T17:12:46 - Generated 116 batches
2026-01-13T17:12:46 - Processing batch 0 (pages 1-3)
2026-01-13T17:12:53 - Processing batch 1 (pages 3-5)  ← FAILS HERE
2026-01-13T17:13:44 - Executed 'Functions.ingestion-worker' (Failed, Duration=30706ms)
2026-01-13T17:14:32 - Executed 'Functions.ingestion-worker' (Failed, Duration=33969ms)
```

### Failure Stats
- **Success Rate**: 0/8 executions (0%)
- **All executions marked**: `Success=False`
- **Final destination**: `ingestion-uploads-poison` queue
- **Azure Search index count**: 0 documents

---

## Root Cause Analysis

### Suspected Issue
The page range parameter format being passed to Azure Document Intelligence API is incorrect. The API expects a specific format (e.g., "1-3" or perhaps "0-2" for 0-indexed), but the worker is passing an incompatible format.

### Evidence
1. Batch 0 processes "pages 1-3" - attempts extraction
2. Batch 1 processes "pages 3-5" - API rejects with InvalidParameter
3. The page range "3-5" triggers the error every time
4. Multiple retries all fail at the same batch with the same error

### Possible Causes
1. **Page indexing mismatch**: Document Intelligence API expects 0-indexed pages, worker sending 1-indexed
2. **Range format**: API expects different format (e.g., comma-separated "3,4,5" instead of "3-5")
3. **Boundary issue**: Overlapping ranges (batch 0 ends at page 3, batch 1 starts at page 3)
4. **API version change**: Page parameter format changed in newer API version

---

## Impact

### Affected Functionality
- ❌ **All real PDF uploads** - Documents with 2+ pages will fail
- ❌ **E2E pipeline** - Cannot validate full pipeline until fixed
- ❌ **Production readiness** - System cannot process real documents

### Workarounds
- ✅ **Chunking worker** - Still works with synthetic data (tested successfully)
- ⚠️ **Single-page PDFs** - Might work if they skip batch processing
- ❌ **Multi-page PDFs** - No workaround available

---

## Fix Required

### Files to Investigate
1. `backend/src/services/workers/ingestion_worker.py`
2. `backend/src/services/rag/ingestion.py`
3. Azure Document Intelligence API call parameters

### What to Check
```python
# Look for code like this:
pages_param = f"{start_page}-{end_page}"  # Current (broken?)
# Should it be:
pages_param = f"{start_page},{start_page+1},{end_page}"  # Comma-separated?
# Or:
pages_param = f"{start_page-1}-{end_page-1}"  # 0-indexed?
```

### Testing After Fix
1. Clear poison queue: `python scripts/monitor_queue_health.py --clear-poison --yes`
2. Re-enqueue test document: Use existing document ID `f7df526a-97df-4e85-836d-b3478a72ae6d`
3. Monitor logs: Watch for successful batch processing
4. Verify status updates: Document should progress to 'parsed'
5. Validate E2E: Check full pipeline through to indexing

---

## Monitoring Commands

### Check Current Status
```bash
# Check poison queue
python scripts/monitor_queue_health.py --peek ingestion-uploads-poison

# Check document status
python -c "from src.core.config import Config; from src.db.connection import DatabaseConnection; from src.db.documents import DocumentService; config=Config.from_env(); db=DatabaseConnection(config); db.connect(); svc=DocumentService(db); doc=svc.get_document('f7df526a-97df-4e85-836d-b3478a72ae6d'); print(f'Status: {doc.status}'); db.close()"

# Check recent errors
az monitor app-insights query \
  --app ai-rag-evaluator \
  --resource-group rag-lab \
  --analytics-query "exceptions | where timestamp > ago(10m) | order by timestamp desc | take 3"
```

---

## Related Issues

- Queue management bug (old invalid UUIDs blocking queue) - **FIXED**
- UUID validation in diagnostic scripts - **FIXED**
- Storage path convention mismatch - **FIXED**
- **Page range parameter format** - **NOT FIXED** ← THIS ISSUE

---

## Next Actions

1. 🔍 **Investigate** Document Intelligence API documentation for correct page parameter format
2. 🐛 **Fix** page range parameter in ingestion worker code
3. 🧪 **Test** locally if possible with Document Intelligence API
4. 🚀 **Deploy** corrected code to Azure Functions
5. ✅ **Validate** with test document `f7df526a-97df-4e85-836d-b3478a72ae6d`
6. 📊 **Monitor** full E2E pipeline completion

**Priority**: HIGH - This blocks all real document processing

