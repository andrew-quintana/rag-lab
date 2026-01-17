# Batch Processing Failure Analysis

**Date**: 2026-01-16  
**Document**: `scan_classic_hmo.pdf` (232 pages, 2.5 MB)  
**Document ID**: `2fe231cd-40fa-47b1-8d00-59c88e3c9621`  
**Status**: ⚠️ **STUCK** - Processing failed

---

## Failure Mode Analysis

### Current State

**Document Status**:
- Status: `uploaded` (stuck, not progressing)
- Parsing Status: `in_progress`
- Pages: 232 total
- Batches: 116 total (2 pages per batch)
- Batches Completed: 7 (non-sequential: 1, 5, 8, 11, 15, 19, 20)
- Last Successful Page: 24
- Next Batch: 12
- Errors: None in metadata

**Queue Status**:
- Main queues: All empty
- Poison queue: 6 messages (including this document)
- Message dequeue count: 1 (tried once, failed, moved to poison)

**Logs**:
- No function execution logs in Application Insights
- No exceptions logged
- Only queue polling requests visible
- No ingestion worker traces

---

## Root Cause Hypothesis

### Hypothesis 1: Function Timeout Before Re-Enqueue ⚠️ **MOST LIKELY**

**Evidence**:
- Time-based limiting is implemented (4 min processing + 1 min buffer)
- Function timeout: 10 minutes (configured in host.json)
- Only 7 batches completed before failure
- Message in poison queue (function failed/timed out)
- No logs suggest function was killed mid-execution

**What Likely Happened**:
1. Function started processing batches
2. Processed 7 batches (non-sequential suggests some failed/retried)
3. Function execution exceeded timeout (10 minutes) before time-based limiting could re-enqueue
4. Function was killed by Azure Functions runtime
5. Message moved to poison queue after max retries

**Why Time-Based Limiting Didn't Help**:
- If batches are taking longer than expected, function might timeout before reaching the time check
- Initial batches might be slow (cold start, API latency)
- Time check happens BEFORE each batch, but if a batch is already running when timeout occurs, it gets killed

### Hypothesis 2: Batch Processing Errors

**Evidence**:
- Non-sequential batch completion (1, 5, 8, 11, 15, 19, 20)
- Some batches completed, others didn't
- Suggests intermittent failures

**Possible Causes**:
- Azure Document Intelligence API rate limiting
- Network timeouts during API calls
- PDF slicing errors for certain page ranges
- Memory issues with large PDF

### Hypothesis 3: Function Not Executing

**Evidence**:
- No function execution logs
- Only queue polling visible
- Function might not be triggering

**Less Likely** because:
- Some batches were completed (function did run)
- Message is in poison queue (function tried and failed)

---

## Diagnostic Steps Taken

1. ✅ Checked poison queue - Found document message
2. ✅ Checked document metadata - No errors recorded
3. ✅ Checked Application Insights - No execution logs
4. ✅ Checked queue status - All empty
5. ✅ Verified time-based limiting code exists
6. ✅ Checked function timeout configuration

---

## Recommended Solutions

### Solution 1: Reduce Time Limit More Aggressively

**Problem**: 4 minutes might be too long if batches are slow

**Fix**: Reduce `MAX_EXECUTION_TIME_SECONDS` to 180 seconds (3 minutes)

```python
MAX_EXECUTION_TIME_SECONDS = 180  # 3 minutes (was 240)
SAFETY_BUFFER_SECONDS = 60        # 1 minute buffer
```

**Rationale**: 
- If batches take 10-15 seconds each, 3 minutes = ~12-18 batches
- More frequent re-enqueuing = less risk of timeout
- More executions but safer

### Solution 2: Add Timeout Protection in Batch Loop

**Problem**: Time check happens before batch, but batch might exceed remaining time

**Fix**: Add timeout check INSIDE batch processing (e.g., during API call)

```python
# In batch processing loop
if time.time() - execution_start_time > MAX_EXECUTION_TIME_SECONDS:
    # Emergency exit - save progress and re-enqueue
    logger.warning("Emergency timeout - saving progress and re-enqueuing")
    # Save current progress
    # Re-enqueue
    return
```

### Solution 3: Process Fewer Batches Per Execution

**Problem**: Too many batches attempted per execution

**Fix**: Add hard limit on batches per execution (e.g., max 10 batches)

```python
MAX_BATCHES_PER_EXECUTION = 10

# In batch loop
if batches_processed_this_execution >= MAX_BATCHES_PER_EXECUTION:
    logger.info(f"Reached batch limit ({MAX_BATCHES_PER_EXECUTION}). Re-enqueuing...")
    # Re-enqueue
    return
```

### Solution 4: Check Azure Functions Plan

**Problem**: Consumption plan has strict timeout limits

**Fix**: Verify function timeout setting and consider Premium plan for large documents

```bash
az functionapp config show \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query functionTimeout
```

---

## Immediate Actions

1. **Check Function Execution History**:
   ```bash
   az functionapp function show \
     --name func-raglab-uploadworkers \
     --resource-group rag-lab \
     --function-name ingestion-worker
   ```

2. **Check Azure Portal Logs**:
   - Navigate to Function App → ingestion-worker → Monitor → Logs
   - Look for timeout errors or execution failures

3. **Test with Smaller Document**:
   - Use 4-10 page document to verify batch processing works
   - If it works, the issue is specific to large documents

4. **Review Time-Based Limiting Logic**:
   - Verify it's actually re-enqueuing
   - Check if time calculation is correct
   - Ensure re-enqueue happens before timeout

---

## Next Steps

1. **Reduce time limit** to 3 minutes (180 seconds)
2. **Add emergency timeout check** inside batch processing
3. **Add batch count limit** (max 10 batches per execution)
4. **Redeploy** and test with same document
5. **Monitor** execution logs to verify re-enqueuing works

---

## Code Changes Needed

### File: `backend/src/services/workers/ingestion_worker.py`

**Change 1**: Reduce time limit
```python
MAX_EXECUTION_TIME_SECONDS = 180  # 3 minutes (reduced from 240)
```

**Change 2**: Add batch count limit
```python
MAX_BATCHES_PER_EXECUTION = 10  # Hard limit on batches per execution
```

**Change 3**: Add emergency timeout check
```python
# Inside batch processing, after API call starts
if time.time() - execution_start_time > MAX_EXECUTION_TIME_SECONDS:
    logger.warning("Emergency timeout during batch processing")
    # Save progress and re-enqueue
    return
```

---

**Last Updated**: 2026-01-16
