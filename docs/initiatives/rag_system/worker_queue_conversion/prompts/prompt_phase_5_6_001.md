# Prompt: Phase 5.6 - Production Readiness Validation

## Your Mission

Execute Phase 6 of Phase 5 integration testing: Production Readiness Validation. You must verify data integrity, rollback capabilities, and overall system readiness for production deployment. Document all results, validation findings, and readiness issues as you progress.

## Context & References

**Key Documentation to Reference**:
- `phase_5_testing.md` - Testing overview and test locations
- `phase_5_migration_strategy.md` - Migration execution plan and rollback procedures
- `phase_5_handoff.md` - Phase 5 completion summary
- `fracas.md` - Document all failures immediately
- `scoping/TODO001.md` - Task checklist (mark tasks complete)

**Prerequisites** (from Phase 5.5):
- ✅ Application Insights monitoring validated
- ✅ All previous phase tests completed
- ✅ System functioning end-to-end

## Prerequisites Check

**Before starting, verify**:
1. All previous phase tests passed (or documented)
2. System functioning end-to-end
3. Access to Supabase database
4. Access to Azure AI Search
5. Test PDF available: `docs/inputs/scan_classic_hmo.pdf` (use only first 6 pages)
6. Rollback procedures documented

**If any prerequisite is missing, document it in `fracas.md` and stop execution.**

---

## Task 6.1: Data Integrity Verification

**Execute**:
1. Upload test document through complete pipeline
2. Verify data at each stage:
   - Extracted text matches source
   - Chunks created correctly
   - Embeddings generated (non-zero vectors)
   - Search results return correct document

**Verification Steps**:

### 1. Extracted Text Verification
```sql
-- Query Supabase to verify extracted text
SELECT id, filename, status, 
       LENGTH(extracted_text) as text_length,
       parsed_at
FROM documents
WHERE id = '<document_id>';
```

**Verify**:
- ✅ Extracted text is not empty
- ✅ Text length is reasonable for document size
- ✅ Text content matches source document (spot check)
- ✅ Text persisted correctly in database

### 2. Chunks Verification
```sql
-- Query Supabase to verify chunks
SELECT document_id, COUNT(*) as chunk_count,
       AVG(LENGTH(text)) as avg_chunk_length,
       MIN(LENGTH(text)) as min_chunk_length,
       MAX(LENGTH(text)) as max_chunk_length
FROM chunks
WHERE document_id = '<document_id>'
GROUP BY document_id;
```

**Verify**:
- ✅ Chunks created (count > 0)
- ✅ Chunk text is not empty
- ✅ Chunk sizes are reasonable
- ✅ Chunk metadata stored correctly
- ✅ Chunks linked to correct document_id

### 3. Embeddings Verification
```sql
-- Query Supabase to verify embeddings
SELECT document_id, COUNT(*) as embedding_count,
       embedding IS NOT NULL as has_embedding
FROM chunks
WHERE document_id = '<document_id>'
GROUP BY document_id, has_embedding;
```

**Verify**:
- ✅ Embeddings generated (non-null)
- ✅ Embedding vectors are non-zero
- ✅ Embedding dimensions correct (1536 for text-embedding-3-small)
- ✅ Embeddings stored correctly in JSONB format

### 4. Search Results Verification
```python
# Test search functionality
# Verify document is searchable in Azure AI Search
# Query with terms from the document
# Verify document appears in search results
```

**Verify**:
- ✅ Document indexed in Azure AI Search
- ✅ Document appears in search results
- ✅ Search results are accurate
- ✅ Document metadata correct in search index

**Success Criteria**:
- ✅ No data loss through pipeline
- ✅ Data integrity maintained at all stages
- ✅ Search results accurate
- ✅ All data persisted correctly

**Document**: 
- Record data integrity verification results in `phase_5_testing.md`
- Document any data integrity issues in `fracas.md`
- Include verification query results
- Record search test results

---

## Task 6.2: Rollback Testing

**Execute**:
1. Document current synchronous path still functional
2. Verify can disable async path if needed
3. Verify can re-enable synchronous processing
4. Test rollback procedures

**Rollback Scenarios to Test**:

### 1. Synchronous Path Availability
- Verify synchronous processing path still exists (if applicable)
- Test synchronous upload endpoint (if available)
- Verify synchronous path processes documents correctly

### 2. Feature Flag Testing (if implemented)
- Test feature flag to disable async path
- Verify system falls back to synchronous processing
- Test feature flag to re-enable async path

### 3. Queue Disabling
- Test disabling queue triggers (if possible)
- Verify system handles queue unavailability gracefully
- Test re-enabling queue triggers

### 4. Rollback Procedure Documentation
- Review rollback procedures in `phase_5_migration_strategy.md`
- Verify rollback steps are documented
- Test rollback procedure (if safe to do so)

**Success Criteria**:
- ✅ Rollback plan documented
- ✅ Synchronous path still available (if applicable)
- ✅ Feature flags functional (if implemented)
- ✅ Rollback procedures tested and verified

**Document**: 
- Record rollback testing results in `phase_5_testing.md`
- Document any rollback issues in `fracas.md`
- Verify rollback procedures are documented

**Reference**: `phase_5_migration_strategy.md` - Rollback procedures section

---

## Task 6.3: Production Readiness Checklist

**Execute**: Review and verify all production readiness criteria

**Checklist Items**:

### Infrastructure
- [ ] All Azure resources provisioned and configured
- [ ] All environment variables configured
- [ ] Application Insights monitoring enabled
- [ ] Error alerts configured (if applicable)
- [ ] Backup and disaster recovery plan documented

### Functionality
- [ ] All workers deployed and functional
- [ ] Queue triggers working correctly
- [ ] Status transitions working correctly
- [ ] Error handling functional
- [ ] Retry logic working correctly

### Data
- [ ] Database migrations applied
- [ ] Data integrity verified
- [ ] Search indexing working correctly
- [ ] Data persistence working correctly

### Performance
- [ ] Performance requirements met
- [ ] Queue depth handling acceptable
- [ ] Concurrent processing functional
- [ ] Cold start latency acceptable

### Monitoring
- [ ] Application Insights telemetry functional
- [ ] Error tracking functional
- [ ] Performance metrics available
- [ ] Alerts configured (if applicable)

### Documentation
- [ ] Deployment guide complete
- [ ] Migration guide complete
- [ ] Rollback procedures documented
- [ ] Troubleshooting guide available

**Success Criteria**:
- ✅ All critical checklist items verified
- ✅ Production readiness confirmed
- ✅ Any gaps documented

**Document**: 
- Record production readiness checklist results in `phase_5_testing.md`
- Document any gaps in `fracas.md`
- Update `phase_5_handoff.md` with readiness status

---

## Task 6.4: Final System Validation

**Execute**: Run complete end-to-end validation

**Validation Steps**:
1. Upload test document
2. Monitor complete pipeline execution
3. Verify all stages complete successfully
4. Verify data integrity at each stage
5. Verify document searchable
6. Verify monitoring captures all events
7. Verify no errors in logs

**Success Criteria**:
- ✅ Complete pipeline executes successfully
- ✅ All data persisted correctly
- ✅ Document searchable
- ✅ No errors in logs
- ✅ Monitoring captures all events

**Document**: 
- Record final validation results in `phase_5_testing.md`
- Document any issues in `fracas.md`

---

## Success Criteria

### Must Pass (Blocking) - All Required:
- [ ] Data integrity verified at all stages
- [ ] Search results accurate
- [ ] Rollback procedures documented and tested
- [ ] Production readiness checklist complete

### Should Pass (Non-Blocking) - Document Results:
- [ ] All monitoring functional
- [ ] All documentation complete
- [ ] Performance requirements met
- [ ] Error handling verified

**If any "Must Pass" criteria fails, document in `fracas.md` and stop execution.**

---

## Documentation Requirements

**You MUST**:
1. **Record Results**: Update `phase_5_testing.md` with:
   - Data integrity verification results
   - Rollback testing results
   - Production readiness checklist results
   - Final system validation results

2. **Document Failures**: Immediately document any failures in `fracas.md` with:
   - Failure description
   - Validation gaps
   - Steps to reproduce
   - Impact assessment
   - Severity (Critical/High/Medium/Low)

3. **Update Tracking**: 
   - Mark Phase 6 tasks complete in `scoping/TODO001.md`
   - Update `phase_5_handoff.md` with Phase 6 status and overall completion
   - Create final summary of all test results

---

## Final Steps

**Once all Phase 6 tasks complete**:

1. **Update Handoff Document**: 
   - Update `phase_5_handoff.md` with complete test results summary
   - Include overall status (Pass/Fail/Partial)
   - List all blocking issues
   - Provide recommendations for next steps

2. **Update Task Tracking**:
   - Mark all Phase 5 testing tasks complete in `scoping/TODO001.md`
   - Note any incomplete tasks
   - Update validation requirements

3. **Create Final Summary**:
   - Summarize all test results across all phases
   - List all failures from `fracas.md`
   - Provide recommendations for next steps
   - Document production readiness status

4. **If All Tests Pass**:
   - Proceed with migration strategy execution (see `phase_5_migration_strategy.md`)
   - Update production documentation
   - Schedule stakeholder review

**Reference**: `phase_5_migration_strategy.md` - Complete migration execution plan

---

## Quick Reference

**Key Files**:
- `phase_5_testing.md` - Test results
- `fracas.md` - Failure tracking
- `phase_5_handoff.md` - Handoff document
- `scoping/TODO001.md` - Task tracking
- `phase_5_migration_strategy.md` - Migration and rollback procedures

**SQL Queries** (for data verification):
- Documents: `SELECT * FROM documents WHERE id = '<document_id>';`
- Chunks: `SELECT * FROM chunks WHERE document_id = '<document_id>';`
- Status: `SELECT id, status, parsed_at, chunked_at, embedded_at, indexed_at FROM documents WHERE id = '<document_id>';`

---

**Begin execution now. Good luck!**

