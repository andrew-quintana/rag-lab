# Prompt: Phase 5.2 - Azure Functions Deployment Validation

## Your Mission

Execute Phase 2 of Phase 5 integration testing: Azure Functions Deployment Validation. You must verify that all worker functions are properly deployed, configured, and can be triggered via Azure Storage Queue messages. Document all results, failures, and configuration issues as you progress.

## Context & References

**Key Documentation to Reference**:
- `phase_5_azure_functions_deployment.md` - Azure Functions deployment guide
- `GIT_DEPLOYMENT.md` - Git-based deployment guide (recommended approach)
- `DEPLOYMENT_GUIDE.md` - Complete deployment guide
- `phase_5_testing.md` - Testing overview and test locations
- `phase_5_handoff.md` - Phase 5 completion summary
- `../../setup/environment_variables.md` - Environment variable reference
- `ENV_VARIABLE_MAPPING.md` - Environment variable name mapping
- `fracas.md` - Document all failures immediately
- `scoping/TODO001.md` - Task checklist (mark tasks complete)

**Prerequisites** (from Phase 5.1):
- ✅ Database migrations 0019 and 0020 applied to production Supabase
- ✅ Supabase integration tests passed (12/13 tests)
- ⚠️ Azure Functions deployment status to be verified

## Prerequisites Check

**Before starting, verify**:
1. Azure Function App `func-raglab-uploadworkers` exists (see `phase_5_azure_functions_deployment.md`)
2. Azure CLI installed and authenticated (`az login`)
3. Access to Azure Portal for Application Insights
4. Azure Storage Account `raglabqueues` accessible
5. Test PDF available: `docs/inputs/scan_classic_hmo.pdf` (use only first 6 pages)
6. (Optional) Git deployment configured - Check if functions are deployed via Git:
   ```bash
   az functionapp deployment source show \
     --name func-raglab-uploadworkers \
     --resource-group rag-lab \
     --query "{Repo:repoUrl, Branch:branch}" -o json
   ```

**If any prerequisite is missing, document it in `fracas.md` and stop execution.**

---

## Task 2.1: Function App Health Check

**Execute**:
```bash
# Check Function App status
az functionapp show \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "{Name:name, State:state, Location:location, Runtime:siteConfig.linuxFxVersion, FunctionsVersion:siteConfig.functionsExtensionVersion}"

# Verify all functions are deployed
az functionapp function list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[].{Name:name, Language:language, Status:status}" -o table

# (Optional) Check Git deployment status
az functionapp deployment source show \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "{Repo:repoUrl, Branch:branch, Status:status}" -o json
```

**Verify**:
- ✅ Function App state is "Running"
- ✅ All 4 worker functions listed: `ingestion-worker`, `chunking-worker`, `embedding-worker`, `indexing-worker`
- ✅ Python runtime version 3.12 (or 3.11)
- ✅ Functions Extension Version ~4
- ✅ All functions show "Enabled" status
- (Optional) Git deployment configured and active

**Document**: 
- Record status in `phase_5_testing.md` under "Phase 2: Azure Functions Deployment Validation"
- If any function is missing, document in `fracas.md` with severity "Critical"
- Include Function App details (name, state, location, runtime)
- Note deployment method (Git-based or manual)

**Reference**: 
- `phase_5_azure_functions_deployment.md` - Step 7: Testing Deployment
- `GIT_DEPLOYMENT.md` - Git-based deployment guide

---

## Task 2.2: Environment Variables Verification

**Execute**:
```bash
# List all environment variables
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[].{Name:name}" -o table

# Check for specific critical variables (without exposing values)
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[?contains(name, 'DATABASE') || contains(name, 'SUPABASE') || contains(name, 'AZURE_AI') || contains(name, 'AZURE_SEARCH') || contains(name, 'AZURE_DOCUMENT') || contains(name, 'AZURE_BLOB') || contains(name, 'AZURE_STORAGE_QUEUES') || contains(name, 'AZURE_STORAGE_CONNECTION')].{Name:name}" -o table
```

**Verify all required variables are present**:
- `AzureWebJobsStorage` (storage connection string) - **REQUIRED**
- `AZURE_STORAGE_QUEUES_CONNECTION_STRING` - **REQUIRED** (or `AZURE_STORAGE_CONNECTION_STRING` mapped to this)
- `DATABASE_URL` - **REQUIRED**
- `SUPABASE_URL` - **REQUIRED**
- `SUPABASE_KEY` - **REQUIRED**
- `AZURE_AI_FOUNDRY_ENDPOINT` - **REQUIRED**
- `AZURE_AI_FOUNDRY_API_KEY` - **REQUIRED**
- `AZURE_SEARCH_ENDPOINT` - **REQUIRED**
- `AZURE_SEARCH_API_KEY` - **REQUIRED**
- `AZURE_SEARCH_INDEX_NAME` - **REQUIRED**
- `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` - **REQUIRED**
- `AZURE_DOCUMENT_INTELLIGENCE_API_KEY` - **REQUIRED**
- `AZURE_BLOB_CONNECTION_STRING` (if used) - Optional
- `AZURE_BLOB_CONTAINER_NAME` (if used) - Optional
- `APPLICATIONINSIGHTS_CONNECTION_STRING` or `AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING` - Recommended

**Note**: Variable names may differ between `.env.local` and Azure Functions. See `ENV_VARIABLE_MAPPING.md` for mapping details.

**Document**: 
- List all found variables in `phase_5_testing.md`
- List any missing variables in `fracas.md` with severity:
  - **Critical**: Missing required variables (blocks function execution)
  - **High**: Missing recommended variables (affects monitoring/functionality)
- If critical variables are missing, stop and document in `fracas.md`

**Reference**: 
- `../../setup/environment_variables.md` for variable descriptions
- `ENV_VARIABLE_MAPPING.md` for variable name mapping
- `phase_5_azure_functions_deployment.md` - Step 2: Configure Environment Variables

---

## Task 2.3: Queue Trigger Configuration Test

**Execute**:
```bash
# Get storage account connection string
STORAGE_CONN_STRING=$(az storage account show-connection-string \
  --name raglabqueues \
  --resource-group rag-lab \
  --query connectionString -o tsv)

# Send test message to ingestion-uploads queue
az storage message put \
  --queue-name ingestion-uploads \
  --content '{"document_id":"test-trigger-001","source_storage":"supabase","filename":"test.pdf","attempt":1,"stage":"uploaded"}' \
  --account-name raglabqueues \
  --connection-string "$STORAGE_CONN_STRING"

# Monitor function execution (watch for 60 seconds)
az functionapp log tail \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --timeout 60
```

**Verify**:
- ✅ Message successfully enqueued (check queue length)
- ✅ `ingestion-worker` function triggered within 30 seconds
- ✅ Function execution appears in logs
- ✅ No errors in function execution logs
- ✅ Function processes message (or fails gracefully with expected error if test document doesn't exist)

**Alternative Verification** (if logs don't show execution):
```bash
# Check Application Insights for function execution
# Navigate to Azure Portal → Application Insights → ai-rag-evaluator
# Query: traces | where operation_Name contains "ingestion-worker" | order by timestamp desc | take 10
```

**Document**: 
- Record trigger latency in `phase_5_testing.md`
- Record message enqueue time and function trigger time
- Document any errors in `fracas.md` with:
  - Error messages
  - Function execution logs
  - Queue message status
- If function doesn't trigger, document in `fracas.md` with severity "Critical" and stop

**Reference**: `phase_5_azure_functions_deployment.md` - Step 7.1: Test Queue Trigger

---

## Task 2.4: Function Configuration Verification

**Execute**:
```bash
# Check function configuration for each worker
for func in ingestion-worker chunking-worker embedding-worker indexing-worker; do
  echo "=== Checking $func ==="
  az functionapp function show \
    --name func-raglab-uploadworkers \
    --resource-group rag-lab \
    --function-name $func \
    --query "{Name:name, Language:language, Config:config}" -o json
done
```

**Verify**:
- ✅ All functions have queue trigger bindings configured
- ✅ Queue names match expected names:
  - `ingestion-worker` → `ingestion-uploads`
  - `chunking-worker` → `ingestion-chunking`
  - `embedding-worker` → `ingestion-embeddings`
  - `indexing-worker` → `ingestion-indexing`
- ✅ Batch size and max dequeue count configured appropriately

**Document**: 
- Record function configurations in `phase_5_testing.md`
- Document any misconfigurations in `fracas.md`

---

## Success Criteria

### Must Pass (Blocking) - All Required:
- [ ] Function App is running
- [ ] All 4 worker functions are deployed and enabled
- [ ] All critical environment variables are configured
- [ ] Queue trigger successfully triggers at least one function

### Should Pass (Non-Blocking) - Document Results:
- [ ] All recommended environment variables configured
- [ ] Function configurations verified
- [ ] Application Insights connection configured
- [ ] Git deployment configured (if using Git-based deployment)

**If any "Must Pass" criteria fails, document in `fracas.md` and stop execution.**

---

## Documentation Requirements

**You MUST**:
1. **Record Results**: Update `phase_5_testing.md` with:
   - Function App status
   - Deployed functions list
   - Environment variables status
   - Queue trigger test results
   - Trigger latency measurements
   - Deployment method (Git-based or manual)

2. **Document Failures**: Immediately document any failures in `fracas.md` with:
   - Failure description
   - Steps to reproduce
   - Error messages
   - Impact assessment
   - Severity (Critical/High/Medium/Low)

3. **Update Tracking**: 
   - Mark Phase 2 tasks complete in `scoping/TODO001.md`
   - Update `phase_5_handoff.md` with Phase 2 status

---

## Next Steps

**If all Phase 2 tasks pass**:
- Proceed to Phase 5.3: End-to-End Pipeline Integration Testing
- Reference: `prompt_phase_5_3_001.md`

**If Phase 2 tasks fail**:
- Document all issues in `fracas.md`
- Address critical blockers before proceeding
- Re-run Phase 2 tests after fixes

---

## Quick Reference

**Azure CLI Commands**:
- Function App status: `az functionapp show --name func-raglab-uploadworkers --resource-group rag-lab`
- List functions: `az functionapp function list --name func-raglab-uploadworkers --resource-group rag-lab`
- Environment variables: `az functionapp config appsettings list --name func-raglab-uploadworkers --resource-group rag-lab`
- Function logs: `az functionapp log tail --name func-raglab-uploadworkers --resource-group rag-lab`
- Git deployment status: `az functionapp deployment source show --name func-raglab-uploadworkers --resource-group rag-lab`

**Key Files**:
- `phase_5_testing.md` - Test results
- `fracas.md` - Failure tracking
- `phase_5_handoff.md` - Handoff document
- `scoping/TODO001.md` - Task tracking
- `GIT_DEPLOYMENT.md` - Git-based deployment guide
- `ENV_VARIABLE_MAPPING.md` - Environment variable mapping

---

**Begin execution now. Good luck!**
