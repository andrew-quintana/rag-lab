# Time-Based Execution Limiting Implementation

**Date**: 2026-01-13  
**Status**: ✅ **IMPLEMENTED** | ⏸️ **TESTING BLOCKED BY INFRASTRUCTURE**  
**Purpose**: Prevent Azure Functions timeout on large document processing

---

## Problem Statement

Azure Functions Consumption Plan has a **5-10 minute execution time limit**. Processing a 232-page PDF with 2-page batches (116 batches) requires ~20-30 minutes, causing timeouts.

---

## Solution: Time-Based Batch Limiting

### Architecture

**Resumable Processing Pattern**:
```
Execution 1: Process batches for 4 minutes → Save progress → Re-enqueue
Execution 2: Resume from last batch → Process for 4 minutes → Re-enqueue
Execution 3: Resume → Process → Re-enqueue
...
Execution N: Complete remaining batches → Mark document as parsed
```

###Key Features

1. **Time-Based** (not count-based)
   - Monitors elapsed execution time
   - Tracks average batch processing time
   - Estimates if next batch will fit in remaining time
   - Stops before timeout and re-enqueues

2. **Adaptive**
   - Learns actual batch processing speed
   - Adjusts dynamically to document complexity
   - 20% buffer for variance

3. **Safety Margins**
   - Target: 4 minutes processing (240 seconds)
   - Buffer: 1 minute for cleanup (60 seconds)
   - Total: 5 minutes per execution
   - Configured timeout: 10 minutes (max for Consumption plan)

4. **Resumable**
   - Saves progress after each batch
   - Re-enqueues with same document ID
   - Idempotent - skips completed batches
   - Can recover from crashes

---

## Implementation Details

### Configuration Constants

```python
# In ingestion_worker.py (line ~309)
MAX_EXECUTION_TIME_SECONDS = 240  # 4 minutes of processing
SAFETY_BUFFER_SECONDS = 60        # 1 minute for cleanup/final operations
MIN_TIME_PER_BATCH = 5            # Minimum estimate for first batch
```

### Time Tracking

**Variables tracked per execution**:
- `execution_start_time`: When execution began
- `batches_processed_this_execution`: Count of batches in this run
- `batch_times[]`: List of actual batch durations
- `avg_batch_time`: Rolling average of batch processing time

### Decision Logic

Before processing each batch:

```python
elapsed_time = time.time() - execution_start_time

# Estimate time for next batch
if batch_times:
    avg_batch_time = sum(batch_times) / len(batch_times)
    estimated_next_batch_time = avg_batch_time * 1.2  # 20% buffer
else:
    estimated_next_batch_time = MIN_TIME_PER_BATCH

time_remaining = MAX_EXECUTION_TIME_SECONDS - elapsed_time

# Check if we have time for another batch
if time_remaining < (estimated_next_batch_time + SAFETY_BUFFER_SECONDS):
    # Not enough time - re-enqueue and exit
    enqueue_continuation_message()
    return
else:
    # Process batch
    process_batch()
```

### Logging

**Key log messages to monitor**:

```
"Starting time-limited batch processing (max 240s, safety buffer 60s)"

"Processing batch X (pages Y-Z) - batch N this execution, 
 45.2s elapsed, 194.8s remaining"

"Time limit approaching: 235.5s elapsed, 4.5s remaining, 
 need 12.3s for next batch. Processed 25 batches this execution. 
 Re-enqueuing for continuation from batch 26..."

"Progress saved: 25/116 batches completed. Continuation message enqueued. 
 Average batch time: 8.7s"

"All batches completed. Processed 15 batches this execution in 128.3s. 
 Average batch time: 8.6s"
```

---

## Expected Behavior

### For 232-Page Document (116 batches)

**Scenario**: Each batch takes ~7-10 seconds

| Execution | Batches | Time | Total Progress |
|-----------|---------|------|----------------|
| 1 | 25-30 | ~4min | 25% |
| 2 | 25-30 | ~4min | 50% |
| 3 | 25-30 | ~4min | 75% |
| 4 | 25-30 | ~4min | 100% |

**Total time**: ~16-20 minutes (vs. timeout at 5-10 min without limiting)

### For Smaller Documents

- **10 pages (5 batches)**: 1 execution, ~40 seconds
- **50 pages (25 batches)**: 1 execution, ~3 minutes
- **100 pages (50 batches)**: 2 executions, ~8 minutes total

---

## Testing & Validation

### How to Test

1. **Upload test PDF to Azure Blob Storage**:
   ```bash
   az storage blob upload \
     --account-name raglabqueues \
     --container-name documents \
     --name test-doc.pdf \
     --file your-test.pdf \
     --auth-mode key
   ```

2. **Enqueue message**:
   ```python
   message = {
       "document_id": "<uuid>",
       "source_storage": "azure_blob",
       "filename": "test-doc.pdf",
       "attempt": 1,
       "stage": "uploaded",
       "metadata": {}
   }
   ```

3. **Monitor Application Insights**:
   ```bash
   az monitor app-insights query \
     --app ai-rag-evaluator \
     --resource-group rag-lab \
     --analytics-query "traces | where timestamp > ago(10m) | 
       where message contains 'Time limit' or message contains 'elapsed' or 
       message contains 'Re-enqueuing for continuation' | 
       order by timestamp desc | take 20"
   ```

4. **Watch for re-enqueue logs**:
   - Should see "Time limit approaching" messages
   - Should see "Re-enqueuing for continuation" messages
   - Should see multiple executions handling different batch ranges

### Success Criteria

- ✅ Each execution completes within 5 minutes
- ✅ No timeout errors in Application Insights
- ✅ Progress saved between executions
- ✅ Document eventually reaches 'parsed' status
- ✅ All batches processed (no skipped pages)
- ✅ Continuation messages appear in queue
- ✅ Final batch count matches expected (num_pages / batch_size)

---

## Configuration Tuning

### Adjust Processing Window

To change the processing time per execution:

```python
# More conservative (safer but more executions)
MAX_EXECUTION_TIME_SECONDS = 180  # 3 minutes
SAFETY_BUFFER_SECONDS = 120       # 2 minute buffer

# More aggressive (fewer executions but riskier)
MAX_EXECUTION_TIME_SECONDS = 360  # 6 minutes
SAFETY_BUFFER_SECONDS = 30        # 30 second buffer
```

### Azure Functions Timeout

Maximum timeout for Consumption plan:

```json
// backend/azure_functions/host.json
{
  "functionTimeout": "00:10:00"  // 10 minutes (max)
}
```

For Premium plan (unlimited):

```json
{
  "functionTimeout": "00:00:00"  // No limit
}
```

---

## Code Changes

###Files Modified

**`backend/src/services/workers/ingestion_worker.py`**:
- Added time tracking constants (lines ~308-315)
- Added execution time monitoring before batch loop (lines ~317-323)
- Added time check before each batch (lines ~349-379)
- Added batch timing tracking after each batch (lines ~451-458)
- Added final timing summary (lines ~470-476)

###Key Functions

1. **Time calculation**:
   ```python
   elapsed_time = time.time() - execution_start_time
   time_remaining = MAX_EXECUTION_TIME_SECONDS - elapsed_time
   ```

2. **Batch time estimation**:
   ```python
   avg_batch_time = sum(batch_times) / len(batch_times)
   estimated_next_batch_time = avg_batch_time * 1.2  # 20% buffer
   ```

3. **Continuation enqueue**:
   ```python
   continuation_message = QueueMessage(
       document_id=document_id,
       source_storage=source_storage,
       filename=filename,
       attempt=1,  # Reset - this is continuation, not retry
       stage=ProcessingStage.UPLOADED,
       metadata=message.metadata
   )
   enqueue_message("ingestion-uploads", continuation_message, config)
   return  # Exit this execution
   ```

---

## Benefits

### vs. Fixed Batch Count

**Old approach** (e.g., "process 10 batches per execution"):
- ❌ Doesn't adapt to document complexity
- ❌ Fast batches waste time (could do more)
- ❌ Slow batches risk timeout

**Time-based approach**:
- ✅ Adapts to actual performance
- ✅ Maximizes work per execution
- ✅ Safer - always stops before timeout

### vs. Premium Plan

**Premium Plan** (~$150/month):
- ✅ Unlimited execution time
- ❌ Higher cost

**Time-based limiting** (stays on Consumption):
- ✅ Works on Consumption plan (cheaper)
- ✅ Still completes large documents
- ✅ Auto-scales with load
- ⚠️  Takes longer (multiple executions)

---

## Limitations & Considerations

### 1. Multiple Executions

**Impact**: A 200-page document might trigger 4-5 separate function executions

**Mitigation**: Executions are queued and processed automatically

### 2. Queue Visibility

**Issue**: Message becomes visible again after 10 minutes (configured timeout)

**Mitigation**: Worker completes within timeout, so message is deleted before reappearing

### 3. Database Connections

**Issue**: Each execution opens new database connections

**Mitigation**: Connection pooling handles this efficiently

### 4. Cost

**Consumption plan billing**:
- Charged per execution-second
- 4 executions × 4 minutes each = 16 minutes total
- Still cheaper than 1 execution × 20 minutes (which would timeout)

---

## Troubleshooting

### "Document stuck in 'uploaded' status"

**Possible causes**:
1. All executions timing out (check MAX_EXECUTION_TIME_SECONDS)
2. Messages going to poison queue (check poison queue)
3. Database connection issues (check ngrok tunnel if using Supabase)
4. Blob not found in Azure Storage (verify upload)

**Debug**:
```bash
# Check poison queues
python scripts/monitor_queue_health.py

# Check Application Insights for errors
az monitor app-insights query --app ai-rag-evaluator --resource-group rag-lab \
  --analytics-query "exceptions | where timestamp > ago(1h) | take 10"

# Check document progress
python scripts/check_document_progress.py <document_id>
```

### "Not seeing re-enqueue logs"

**Possible causes**:
1. Batches completing too fast (adjust MAX_EXECUTION_TIME_SECONDS lower to force re-enqueue)
2. Document has few pages (completes in single execution)
3. Deployment didn't take effect (re-deploy)

**Verify deployment**:
```bash
./scripts/deploy_azure_functions.sh
# Wait 90 seconds
# Re-test
```

### "Still getting timeouts"

**Solutions**:
1. Reduce MAX_EXECUTION_TIME_SECONDS to be more conservative
2. Increase SAFETY_BUFFER_SECONDS for more margin
3. Check Azure Functions timeout setting in host.json
4. Consider Premium plan for unlimited time

---

## Future Improvements

### 1. Dynamic Safety Buffer

Adjust buffer based on variance in batch times:

```python
if len(batch_times) > 5:
    variance = max(batch_times) - min(batch_times)
    dynamic_buffer = SAFETY_BUFFER_SECONDS + (variance * 0.5)
```

### 2. Progress Reporting

Add metadata to track execution count:

```python
"metadata": {
    "execution_count": 3,
    "total_time_spent": 720,  # seconds
    "estimated_completion_time": "2026-01-13T19:30:00Z"
}
```

### 3. Parallel Processing

If Azure functions can run multiple instances:
- Split document into chunks
- Process chunks in parallel
- Merge results

### 4. Batch Size Optimization

Dynamically adjust batch size based on document characteristics:
- Small pages → larger batches
- Complex pages → smaller batches

---

## Summary

✅ **Implemented**: Time-based execution limiting with adaptive batch processing  
✅ **Deployed**: Code live in Azure Functions  
✅ **Configuration**: 4 min processing + 1 min buffer = 5 min target  
⏸️  **Testing**: Blocked by ngrok tunnel instability  
📋 **Next Steps**: Test with stable infrastructure (Azure Blob + stable database)

**Expected Result**: 232-page document completes in 4-5 executions over 16-20 minutes without timeout errors.

---

**Implementation Date**: 2026-01-13  
**Deployed To**: func-raglab-uploadworkers (Azure Functions)  
**Status**: Ready for testing once infrastructure is stable
