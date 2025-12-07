# Prompt: Phase 5.4 - Performance Testing

## Your Mission

Execute Phase 4 of Phase 5 integration testing: Performance Testing. You must validate that the worker-queue architecture meets performance requirements including worker processing time, queue depth handling, concurrent processing, and cold start latency using **real Azure Functions**. Document all results, metrics, and performance issues as you progress.

## Context & References

**Key Documentation to Reference**:
- `phase_5_testing.md` - Testing overview and test locations
- `phase_5_azure_functions_deployment.md` - Azure Functions deployment guide
- `scoping/PRD001.md` - Product requirements (performance section)
- `phase_5_handoff.md` - Phase 5 completion summary
- `fracas.md` - Document all failures immediately
- `scoping/TODO001.md` - Task checklist (mark tasks complete)

**Prerequisites** (from Phase 5.3):
- ✅ End-to-end pipeline tests passed
- ✅ All workers processing documents successfully
- ✅ Queue triggers working correctly

## Prerequisites Check

**Before starting, verify**:
1. Azure Functions deployed and running (from Phase 5.1)
2. All queues accessible and functional
3. Supabase database accessible
4. Azure AI Search accessible
5. Test PDF available: `docs/inputs/scan_classic_hmo.pdf` (use only first 6 pages)
6. Application Insights accessible for monitoring

**If any prerequisite is missing, document it in `fracas.md` and stop execution.**

---

## Task 4.1: Worker Processing Time

**Execute**:
```bash
cd backend
source venv/bin/activate
pytest tests/integration/test_phase5_performance.py::TestWorkerProcessingTime -v -m integration -m performance
```

**Measure and verify**:
- ⏱️ **Ingestion Worker**: < 30 seconds per document
- ⏱️ **Chunking Worker**: < 10 seconds per document
- ⏱️ **Embedding Worker**: < 60 seconds per document (depends on chunk count)
- ⏱️ **Indexing Worker**: < 30 seconds per document
- ⏱️ **Total Pipeline**: < 5 minutes per document

**Test Scenarios**:
- Process single document and measure each worker's processing time
- Process documents of varying sizes
- Verify processing time scales linearly with document size
- Test with different chunk sizes and overlap settings

**Success Criteria**:
- ✅ All workers meet time requirements
- ✅ Processing time scales linearly with document size
- ✅ No performance degradation under load
- ✅ Processing times are consistent across multiple runs

**Document**: 
- Record processing times, document sizes in `phase_5_testing.md`
- Create performance graphs if possible (processing time vs document size)
- Document any performance issues in `fracas.md`
- Include worker-by-worker breakdown of processing times

**Reference**: `scoping/PRD001.md` - Performance requirements section

---

## Task 4.2: Queue Depth and Throughput

**Execute**:
```bash
pytest tests/integration/test_phase5_performance.py::TestQueueDepthHandling -v -m integration -m performance
```

**Test Scenarios**:
- Upload 10 documents simultaneously
- Monitor queue depths for each stage (`ingestion-uploads`, `ingestion-chunking`, `ingestion-embeddings`, `ingestion-indexing`)
- Verify queues drain efficiently
- Verify no queue backlog > 20 messages
- Measure queue drain rate (messages per minute)

**Measurements**:
- Queue depth at each stage over time
- Time to drain queues completely
- Maximum queue depth reached
- Average queue depth during processing
- Throughput (documents processed per minute)

**Success Criteria**:
- ✅ Average queue depth < 10 messages
- ✅ Maximum queue depth < 20 messages
- ✅ Queues drain within 10 minutes for 10 documents
- ✅ No message loss
- ✅ Queue depth decreases steadily (no backlog accumulation)

**Document**: 
- Record queue depths over time, drain rates in `phase_5_testing.md`
- Create queue depth graphs if possible (depth vs time)
- Document any queue backlog issues in `fracas.md`
- Include throughput measurements

---

## Task 4.3: Concurrent Processing

**Execute**:
```bash
pytest tests/integration/test_phase5_performance.py::TestConcurrentProcessing -v -m integration -m performance
```

**Test Scenarios**:
- Upload 5 documents simultaneously
- Verify multiple function instances process in parallel
- Verify no resource contention
- Verify all documents complete successfully
- Measure concurrent execution count
- Measure total processing time vs sequential processing

**Measurements**:
- Number of concurrent function instances
- Concurrent execution duration
- Total processing time for batch
- Throughput improvement vs sequential processing
- Resource utilization (if available)

**Success Criteria**:
- ✅ Multiple function instances active simultaneously
- ✅ Concurrent processing improves throughput
- ✅ No data corruption or cross-contamination
- ✅ All documents process successfully
- ✅ Throughput scales with concurrent instances

**Document**: 
- Record concurrent instance count, throughput improvement in `phase_5_testing.md`
- Document any concurrency issues in `fracas.md`
- Include comparison of concurrent vs sequential processing times
- Record maximum concurrent instances observed

---

## Task 4.4: Cold Start Latency

**Execute**:
```bash
pytest tests/integration/test_phase5_performance.py::TestColdStartLatency -v -m integration -m performance
```

**Test Scenarios**:
- Wait 10 minutes (allow functions to scale to zero)
- Send message to trigger cold start
- Measure time from message enqueue to function execution start
- Measure time from execution start to first log/telemetry
- Compare cold start vs warm start latency

**Measurements**:
- Cold start latency (message enqueue to execution start)
- Warm start latency (subsequent invocations)
- Function initialization time
- Total cold start duration

**Success Criteria**:
- ✅ Cold start latency < 30 seconds
- ✅ Subsequent invocations have minimal latency (< 5 seconds)
- ✅ Cold start latency is acceptable for use case
- ✅ Warm start latency is consistent

**Document**: 
- Record cold start times, warm start times in `phase_5_testing.md`
- Document any cold start issues in `fracas.md`
- Include comparison of cold vs warm start times
- Note any patterns in cold start behavior

---

## Task 4.5: Throughput Validation

**Execute**:
```bash
pytest tests/integration/test_phase5_performance.py::TestThroughput -v -m integration -m performance
```

**Test Scenarios**:
- Process multiple documents in sequence
- Measure total throughput (documents per hour)
- Verify throughput meets requirements
- Test under sustained load
- Monitor for performance degradation over time

**Measurements**:
- Documents processed per hour
- Average processing time per document
- Peak throughput achieved
- Sustained throughput over extended period

**Success Criteria**:
- ✅ Throughput meets requirements (if specified in PRD)
- ✅ No performance degradation under sustained load
- ✅ Throughput is consistent over time

**Document**: 
- Record throughput measurements in `phase_5_testing.md`
- Document any throughput issues in `fracas.md`

---

## Test Data Constraints

**CRITICAL**: All tests processing actual PDFs must use only the **first 6 pages** of `docs/inputs/scan_classic_hmo.pdf` to avoid exceeding Azure Document Intelligence budget.

**Implementation**:
- Use `slice_pdf_to_batch()` helper function
- Document constraint in all test results
- Skip tests if PDF not available

**Reference**: `phase_5_testing.md` - "Test Data Constraints" section

---

## Success Criteria

### Must Pass (Blocking) - All Required:
- [ ] All workers meet time requirements
- [ ] Total pipeline time < 5 minutes per document
- [ ] Queue depth handling works under load
- [ ] Concurrent processing functional

### Should Pass (Non-Blocking) - Document Results:
- [ ] Cold start latency acceptable (< 30 seconds)
- [ ] Throughput meets requirements
- [ ] Performance scales linearly with load
- [ ] No performance degradation over time

**If any "Must Pass" criteria fails, document in `fracas.md` and stop execution.**

---

## Documentation Requirements

**You MUST**:
1. **Record Results**: Update `phase_5_testing.md` with:
   - Worker processing times
   - Queue depth measurements
   - Concurrent processing metrics
   - Cold start latency measurements
   - Throughput measurements
   - Performance graphs (if possible)

2. **Document Failures**: Immediately document any failures in `fracas.md` with:
   - Failure description
   - Performance metrics that failed
   - Steps to reproduce
   - Impact assessment
   - Severity (Critical/High/Medium/Low)

3. **Update Tracking**: 
   - Mark Phase 4 tasks complete in `scoping/TODO001.md`
   - Update `phase_5_handoff.md` with Phase 4 status

---

## Next Steps

**If all Phase 4 tasks pass**:
- Proceed to Phase 5.5: Application Insights Monitoring Validation
- Reference: `prompt_phase_5_5_001.md`

**If Phase 4 tasks fail**:
- Document all issues in `fracas.md`
- Address critical blockers before proceeding
- Re-run Phase 4 tests after fixes

---

## Quick Reference

**Test Files**:
- `backend/tests/integration/test_phase5_performance.py` - Performance tests

**Key Files**:
- `phase_5_testing.md` - Test results
- `fracas.md` - Failure tracking
- `phase_5_handoff.md` - Handoff document
- `scoping/TODO001.md` - Task tracking

**Performance Targets**:
- Ingestion Worker: < 30 seconds
- Chunking Worker: < 10 seconds
- Embedding Worker: < 60 seconds
- Indexing Worker: < 30 seconds
- Total Pipeline: < 5 minutes

---

**Begin execution now. Good luck!**

