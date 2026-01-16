# E2E Cloud/Production Testing Gaps - Comprehensive Checklist

**Date**: 2026-01-14  
**Status**: 🔴 **BLOCKING ISSUES IDENTIFIED**  
**Priority**: 🔥 **CRITICAL - BLOCKS PRODUCTION DEPLOYMENT**

---

## 📋 Gap Summary

| Category | Critical | High | Medium | Total |
|----------|----------|------|--------|-------|
| **Deployment** | 1 | 1 | 1 | 3 |
| **Infrastructure** | 1 | 2 | 1 | 4 |
| **Testing** | 0 | 2 | 2 | 4 |
| **Monitoring** | 0 | 1 | 2 | 3 |
| **Documentation** | 0 | 0 | 3 | 3 |
| **TOTAL** | **2** | **6** | **9** | **17** |

---

## 🔴 CRITICAL GAPS (Blocks All Testing)

### GAP-001: Deployment Package Missing Backend Code
**Priority**: 🔥 CRITICAL  
**Status**: ❌ BLOCKING  
**Impact**: Functions crash on startup, no processing possible

**Problem**:
```bash
# Current deployment only uploads azure_functions/
func azure functionapp publish func-raglab-uploadworkers --python
# Result: /home/site/wwwroot/backend/ does NOT exist
# Effect: ModuleNotFoundError: No module named 'src'
```

**Evidence**:
- Application Insights: "ModuleNotFoundError: No module named 'src'" (every execution)
- Function execution duration: 3-13ms (crashes before user code runs)
- Queue messages stuck in 10-minute visibility timeout

**Solution**:
```bash
# Option A: Fix build.sh to copy backend code
cd backend/azure_functions
cat > build.sh << 'EOF'
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Copying backend code to deployment package..."
rm -rf "$SCRIPT_DIR/backend"
mkdir -p "$SCRIPT_DIR/backend"
cp -r "$PROJECT_ROOT/backend/src" "$SCRIPT_DIR/backend/"
echo "✓ Backend code ready for deployment"
EOF

chmod +x build.sh
./build.sh
func azure functionapp publish func-raglab-uploadworkers --python
```

**Validation**:
- [ ] Run build.sh locally
- [ ] Verify backend/ directory created in azure_functions/
- [ ] Deploy to Azure
- [ ] Check logs for "ModuleNotFoundError" - should be gone
- [ ] Enqueue test message
- [ ] Verify processing starts within 2 minutes

**Estimated Effort**: 30 minutes  
**Owner**: DevOps / Backend Team  
**Blocking**: ALL e2e testing

---

### GAP-002: Database Connection Unstable (ngrok Tunnel)
**Priority**: 🔥 CRITICAL  
**Status**: ⚠️ INTERMITTENT  
**Impact**: Functions can run but database operations fail

**Problem**:
```python
DATABASE_URL = "postgresql://postgres:postgres@2.tcp.us-cal-1.ngrok.io:12473/postgres"
# ngrok tunnel to local Supabase
# Issues:
#   - Tunnel URL changes on restart
#   - Free tier has connection limits
#   - Tunnel can drop randomly
#   - Must manually update Azure Functions config when URL changes
```

**Evidence**:
- Earlier logs: "connection to server at 4.tcp.us-cal-1.ngrok.io port 18029 failed"
- Requires manual config updates: `az functionapp config appsettings set ...`
- Not sustainable for testing or production

**Solutions**:

**Option A: Azure PostgreSQL** (RECOMMENDED for production)
```bash
# Create Azure PostgreSQL
az postgres flexible-server create \
  --name rag-lab-postgres \
  --resource-group rag-lab \
  --location eastus \
  --admin-user pgadmin \
  --admin-password <strong-password> \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --storage-size 32 \
  --version 14

# Update DATABASE_URL
az functionapp config appsettings set \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --settings DATABASE_URL="postgresql://pgadmin:<password>@rag-lab-postgres.postgres.database.azure.com:5432/postgres?sslmode=require"
```

**Option B: Persistent ngrok Tunnel** (for testing only)
```bash
# Use ngrok reserved domain (requires paid plan)
ngrok tcp --region=us --authtoken=<token> 54322
# Static URL: tcp://1.tcp.ngrok.io:12345
```

**Option C: Local Supabase with Public IP** (not recommended)
- Requires firewall configuration
- Security risk
- Not suitable for production

**Validation**:
- [ ] Database accessible from Azure
- [ ] Run database migrations
- [ ] Test connection from Azure Functions
- [ ] Verify no connection drops over 1 hour
- [ ] Document connection string

**Estimated Effort**: 2-4 hours (Azure PostgreSQL) or 30 min (persistent ngrok)  
**Owner**: Infrastructure Team  
**Blocking**: Sustained e2e testing

---

## 🟠 HIGH PRIORITY GAPS (Limits Testing Capability)

### GAP-003: Queue Message Visibility Timeout Too Long
**Priority**: 🟠 HIGH  
**Status**: ⚠️ SLOWS TESTING  
**Impact**: Failed messages invisible for 10 minutes, slows iteration

**Problem**:
```json
// host.json
{
  "extensions": {
    "queues": {
      "visibilityTimeout": "00:10:00"  // ← 10 minutes!
    }
  }
}
```

**Impact**:
- Message picked up but function crashes
- Message invisible for 10 minutes
- Cannot retry quickly
- Slows debugging cycle

**Solution**:
```json
// host.json
{
  "extensions": {
    "queues": {
      "visibilityTimeout": "00:01:00",  // ← 1 minute for testing
      "maxDequeueCount": 3  // Fail faster, move to poison queue
    }
  }
}
```

**For Production**:
```json
{
  "extensions": {
    "queues": {
      "visibilityTimeout": "00:05:00",  // 5 minutes
      "maxDequeueCount": 5
    }
  }
}
```

**Validation**:
- [ ] Update host.json
- [ ] Redeploy functions
- [ ] Trigger test failure
- [ ] Verify message visible again in 1 minute (not 10)

**Estimated Effort**: 15 minutes  
**Owner**: Backend Team

---

### GAP-004: No Deployment Smoke Test
**Priority**: 🟠 HIGH  
**Status**: ❌ MISSING  
**Impact**: Deployment issues discovered late, during testing

**Problem**:
- Deploy code
- Assume it works
- Discover issues hours later when testing
- No early warning of deployment problems

**Solution**:
```bash
#!/bin/bash
# File: backend/azure_functions/deploy_and_test.sh

set -e

echo "1. Building deployment package..."
./build.sh

echo "2. Deploying to Azure..."
func azure functionapp publish func-raglab-uploadworkers --python

echo "3. Waiting for deployment to stabilize..."
sleep 60

echo "4. Running smoke test..."

# Test 1: Check function registration
echo "  - Checking function registration..."
FUNC_COUNT=$(az functionapp function list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "length(@)" -o tsv)

if [ "$FUNC_COUNT" -lt 4 ]; then
  echo "    ❌ FAILED: Expected 4 functions, found $FUNC_COUNT"
  exit 1
fi
echo "    ✅ All 4 functions registered"

# Test 2: Check for startup errors
echo "  - Checking for startup errors..."
ERRORS=$(az monitor app-insights query \
  --app ai-rag-evaluator \
  --resource-group rag-lab \
  --analytics-query "exceptions | where timestamp > ago(5m) | where outerMessage contains 'ModuleNotFoundError' or outerMessage contains 'ImportError' | count" \
  --query "tables[0].rows[0][0]" -o tsv)

if [ "$ERRORS" != "0" ]; then
  echo "    ❌ FAILED: Found $ERRORS import errors in last 5 minutes"
  echo "    Run this to see errors:"
  echo "    az monitor app-insights query --app ai-rag-evaluator --resource-group rag-lab --analytics-query \"exceptions | where timestamp > ago(5m) | take 10\""
  exit 1
fi
echo "    ✅ No import errors detected"

# Test 3: Enqueue test message and verify pickup
echo "  - Testing queue trigger..."
# (Add test message enqueue and verification)

echo ""
echo "✅ Deployment smoke test PASSED"
echo "   Functions are deployed and operational"
```

**Validation**:
- [ ] Create deploy_and_test.sh script
- [ ] Run after each deployment
- [ ] Document expected output
- [ ] Add to CI/CD pipeline

**Estimated Effort**: 2 hours  
**Owner**: DevOps Team

---

### GAP-005: No Automated Rollback Strategy
**Priority**: 🟠 HIGH  
**Status**: ❌ MISSING  
**Impact**: Bad deployments can break production, no easy recovery

**Problem**:
- Deploy broken code
- Functions crash
- No automated rollback
- Manual intervention required

**Solution**:
```bash
# Azure Functions supports deployment slots

# 1. Deploy to staging slot
az functionapp deployment slot create \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --slot staging

# 2. Deploy to staging
func azure functionapp publish func-raglab-uploadworkers --slot staging --python

# 3. Run smoke tests on staging
./smoke_test.sh --slot staging

# 4. If tests pass, swap to production
az functionapp deployment slot swap \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --slot staging

# 5. If issues detected, swap back
az functionapp deployment slot swap \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --slot staging --action swap
```

**Validation**:
- [ ] Create staging slot
- [ ] Test deployment to staging
- [ ] Test swap operation
- [ ] Test rollback operation
- [ ] Document process

**Estimated Effort**: 3 hours  
**Owner**: DevOps Team

---

### GAP-006: Environment Variables Not Documented
**Priority**: 🟠 HIGH  
**Status**: ⚠️ INCOMPLETE  
**Impact**: Difficult to reproduce environment, error-prone setup

**Problem**:
- 23 environment variables configured
- No single source of truth
- No validation script
- Easy to miss required variables

**Solution**:
```bash
# File: backend/azure_functions/env_template.sh
#!/bin/bash
# Azure Functions Environment Variables Template
# Copy this file to env_local.sh and fill in values
# Run: source env_local.sh

# Functions Runtime
export FUNCTIONS_WORKER_RUNTIME="python"
export FUNCTIONS_EXTENSION_VERSION="~4"

# Azure Storage (required)
export AzureWebJobsStorage="<connection-string>"  # REQUIRED
export AZURE_STORAGE_QUEUES_CONNECTION_STRING="<connection-string>"  # REQUIRED
export AZURE_BLOB_CONNECTION_STRING="<connection-string>"  # REQUIRED
export AZURE_BLOB_CONTAINER_NAME="documents"  # REQUIRED
export AZURE_STORAGE_ENDPOINT="https://raglabqueues.queue.core.windows.net/"
export AZURE_STORAGE_API_KEY="<api-key>"
export AZURE_QUEUE_STORAGE_NAME="ingestion-uploads"
export AZURE_DOCUMENT_STORAGE_NAME="documents"

# Database (required)
export DATABASE_URL="<postgresql-url>"  # REQUIRED
export SUPABASE_URL="<supabase-url>"  # Optional
export SUPABASE_KEY="<supabase-key>"  # Optional

# Azure AI/Search (required)
export AZURE_AI_FOUNDRY_ENDPOINT="<endpoint>"  # REQUIRED
export AZURE_AI_FOUNDRY_API_KEY="<key>"  # REQUIRED
export AZURE_AI_FOUNDRY_EMBEDDING_MODEL="text-embedding-3-small"
export AZURE_AI_FOUNDRY_GENERATION_MODEL="gpt-4o-mini"
export AZURE_SEARCH_ENDPOINT="<endpoint>"  # REQUIRED
export AZURE_SEARCH_API_KEY="<key>"  # REQUIRED
export AZURE_SEARCH_INDEX_NAME="rag-lab-search"

# Azure Document Intelligence (required)
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="<endpoint>"  # REQUIRED
export AZURE_DOCUMENT_INTELLIGENCE_API_KEY="<key>"  # REQUIRED

# Application Insights (required)
export APPLICATIONINSIGHTS_CONNECTION_STRING="<connection-string>"  # REQUIRED

# Validation script
validate_env() {
  REQUIRED_VARS=(
    "AzureWebJobsStorage"
    "AZURE_STORAGE_QUEUES_CONNECTION_STRING"
    "DATABASE_URL"
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
  )
  
  MISSING=()
  for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
      MISSING+=("$var")
    fi
  done
  
  if [ ${#MISSING[@]} -gt 0 ]; then
    echo "❌ Missing required environment variables:"
    printf '  - %s\n' "${MISSING[@]}"
    return 1
  fi
  
  echo "✅ All required environment variables are set"
  return 0
}

# Run validation
validate_env
```

**Validation**:
- [ ] Create env_template.sh
- [ ] Document all variables with descriptions
- [ ] Create validation script
- [ ] Test setup on fresh environment
- [ ] Update README with environment setup

**Estimated Effort**: 2 hours  
**Owner**: Documentation Team

---

### GAP-007: No Load Testing / Concurrency Testing
**Priority**: 🟠 HIGH  
**Status**: ❌ NOT TESTED  
**Impact**: Unknown behavior under load, possible bottlenecks

**Problem**:
- Only tested single document processing
- Unknown: How does it handle 10 concurrent documents?
- Unknown: Queue throughput limits
- Unknown: Database connection pooling needs

**Solution**:
```python
#!/usr/bin/env python3
# File: backend/azure_functions/load_test.py

import concurrent.futures
import time
from azure.storage.queue import QueueServiceClient
import json
import base64

def enqueue_test_documents(count: int):
    """Enqueue multiple test documents"""
    conn_str = os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING")
    client = QueueServiceClient.from_connection_string(conn_str)
    queue = client.get_queue_client("ingestion-uploads")
    
    for i in range(count):
        message = {
            "document_id": f"load-test-{i}",
            "source_storage": "azure_blob",
            "filename": f"test-doc-{i}.pdf",
            "attempt": 1,
            "stage": "uploaded",
            "metadata": {"load_test": True}
        }
        queue.send_message(base64.b64encode(json.dumps(message).encode()).decode())
    
    print(f"✅ Enqueued {count} test documents")

def monitor_processing(duration_minutes: int):
    """Monitor processing throughput"""
    # Query Application Insights for processing metrics
    # Track: documents/minute, avg execution time, error rate
    pass

if __name__ == "__main__":
    # Test with 10 concurrent documents
    print("Starting load test with 10 documents...")
    enqueue_test_documents(10)
    monitor_processing(duration_minutes=30)
```

**Test Scenarios**:
- [ ] 10 concurrent small documents (10 pages each)
- [ ] 10 concurrent large documents (200+ pages each)
- [ ] 100 documents queued rapidly
- [ ] Sustained load (1 document/minute for 1 hour)

**Metrics to Track**:
- Throughput (documents/minute)
- Latency (time from enqueue to completion)
- Error rate
- Database connection errors
- Queue backlog size

**Estimated Effort**: 4 hours  
**Owner**: QA / Performance Team

---

## 🟡 MEDIUM PRIORITY GAPS (Quality of Life)

### GAP-008: No Pre-Deployment Checklist
**Priority**: 🟡 MEDIUM  
**Status**: ❌ MISSING

**Solution**:
```markdown
# Pre-Deployment Checklist

## Code Quality
- [ ] All tests passing locally
- [ ] No linter errors
- [ ] Code reviewed
- [ ] Changes documented

## Build & Package
- [ ] build.sh runs successfully
- [ ] backend/ directory created in azure_functions/
- [ ] requirements.txt up to date
- [ ] No local development files in package

## Configuration
- [ ] Environment variables validated
- [ ] Secrets not hardcoded
- [ ] host.json reviewed
- [ ] function.json bindings correct

## Pre-Deployment Tests
- [ ] Local function app tested
- [ ] Integration tests passed
- [ ] Database migrations applied
- [ ] Smoke test script ready

## Deployment
- [ ] Deploy to staging first
- [ ] Run smoke tests on staging
- [ ] Monitor logs for 5 minutes
- [ ] Swap to production if healthy

## Post-Deployment
- [ ] Verify all functions registered
- [ ] Check for startup errors
- [ ] Test queue processing
- [ ] Monitor for 30 minutes
- [ ] Document any issues
```

**Estimated Effort**: 1 hour  
**Owner**: Team Lead

---

### GAP-009: No Monitoring Dashboard
**Priority**: 🟡 MEDIUM  
**Status**: ❌ MISSING

**Solution**:
Create Application Insights dashboard with:
- Queue depth over time
- Function execution duration (p50, p95, p99)
- Error rate
- Throughput (docs/hour)
- Database connection errors
- Time-limiting trigger rate (re-enqueues)

**Estimated Effort**: 3 hours  
**Owner**: DevOps Team

---

### GAP-010: No Alerting Rules
**Priority**: 🟡 MEDIUM  
**Status**: ❌ MISSING

**Solution**:
```bash
# Create alert rules
az monitor metrics alert create \
  --name "function-errors-high" \
  --resource-group rag-lab \
  --scopes /subscriptions/.../resourceGroups/rag-lab/providers/Microsoft.Web/sites/func-raglab-uploadworkers \
  --condition "count ErrorCount > 10" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action <action-group-id>

# Alerts needed:
# - Error rate > 10% for 5 minutes
# - Queue depth > 100 for 10 minutes
# - Execution time > 280s (approaching timeout)
# - No executions for 30 minutes (dead functions)
```

**Estimated Effort**: 2 hours  
**Owner**: DevOps Team

---

### GAP-011: Queue Cleanup Strategy Missing
**Priority**: 🟡 MEDIUM  
**Status**: ⚠️ MANUAL

**Problem**:
- Poison queue messages accumulate
- No automated cleanup
- Manual intervention required

**Solution**:
```python
# Automated poison queue handler
# Runs daily, logs failed messages, archives, and clears queue
```

**Estimated Effort**: 2 hours  
**Owner**: Backend Team

---

### GAP-012: No Disaster Recovery Plan
**Priority**: 🟡 MEDIUM  
**Status**: ❌ MISSING

**Solution**:
Document:
- How to recover from total Azure outage
- How to replay failed documents
- How to export/import queue messages
- Backup strategy for database
- RTO/RPO targets

**Estimated Effort**: 4 hours  
**Owner**: SRE Team

---

### GAP-013: Requirements.txt Not Comprehensive
**Priority**: 🟡 MEDIUM  
**Status**: ⚠️ INCOMPLETE

**Problem**:
```txt
# backend/azure_functions/requirements.txt
azure-functions
azure-storage-queue
azure-identity
# Missing: All dependencies from backend/requirements.txt!
```

**Solution**:
```bash
# Merge requirements
cat ../requirements.txt >> requirements.txt
# Remove duplicates
sort -u requirements.txt -o requirements.txt
```

**Validation**:
- [ ] Test deployment with merged requirements
- [ ] Verify all imports work in deployed function
- [ ] Document dependency management process

**Estimated Effort**: 1 hour  
**Owner**: Backend Team

---

### GAP-014: No CI/CD Pipeline
**Priority**: 🟡 MEDIUM  
**Status**: ❌ MISSING

**Solution**:
```yaml
# .github/workflows/deploy-functions.yml
name: Deploy Azure Functions

on:
  push:
    branches: [main]
    paths:
      - 'backend/src/**'
      - 'backend/azure_functions/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Build deployment package
        run: |
          cd backend/azure_functions
          ./build.sh
      
      - name: Deploy to staging
        run: |
          func azure functionapp publish func-raglab-uploadworkers --slot staging --python
      
      - name: Run smoke tests
        run: ./smoke_test.sh --slot staging
      
      - name: Swap to production
        if: success()
        run: |
          az functionapp deployment slot swap \
            --name func-raglab-uploadworkers \
            --resource-group rag-lab \
            --slot staging
```

**Estimated Effort**: 6 hours  
**Owner**: DevOps Team

---

## 📚 DOCUMENTATION GAPS

### GAP-015: Deployment Guide Missing
**Priority**: 🟡 MEDIUM  
**Status**: ⚠️ INCOMPLETE

Create: `backend/azure_functions/DEPLOYMENT.md`

**Estimated Effort**: 2 hours

---

### GAP-016: Troubleshooting Guide Missing
**Priority**: 🟡 MEDIUM  
**Status**: ⚠️ INCOMPLETE

Create: `backend/azure_functions/TROUBLESHOOTING.md`
- Common errors and solutions
- How to check logs
- How to debug failed functions
- How to inspect queue messages

**Estimated Effort**: 3 hours

---

### GAP-017: Architecture Diagram Missing
**Priority**: 🟡 MEDIUM  
**Status**: ❌ MISSING

Create diagram showing:
- Azure Functions workers
- Queue flow
- Database connections
- Blob storage
- Azure services integration

**Estimated Effort**: 2 hours

---

## 🎯 Recommended Prioritization

### Phase 1: Unblock E2E Testing (DAY 1)
1. **GAP-001**: Fix deployment package (30 min) - **CRITICAL**
2. **GAP-003**: Reduce visibility timeout (15 min) - Faster iteration
3. **GAP-013**: Merge requirements.txt (1 hour) - Prevent import errors
4. **GAP-004**: Create smoke test (2 hours) - Catch issues early

**Total Effort**: ~4 hours  
**Outcome**: Can run full e2e tests on Azure

### Phase 2: Stabilize Testing (DAY 2-3)
5. **GAP-002**: Setup Azure PostgreSQL (4 hours) - Stable database
6. **GAP-006**: Document environment (2 hours) - Reproducible setup
7. **GAP-008**: Pre-deployment checklist (1 hour) - Prevent mistakes

**Total Effort**: ~7 hours  
**Outcome**: Reliable, repeatable testing

### Phase 3: Production Readiness (WEEK 1)
8. **GAP-005**: Deployment slots & rollback (3 hours)
9. **GAP-007**: Load testing (4 hours)
10. **GAP-009**: Monitoring dashboard (3 hours)
11. **GAP-010**: Alerting rules (2 hours)

**Total Effort**: ~12 hours  
**Outcome**: Production-ready deployment

### Phase 4: Long-term Maintenance (WEEK 2+)
12-17. Remaining gaps (documentation, CI/CD, disaster recovery)

**Total Effort**: ~20 hours  
**Outcome**: Sustainable operations

---

## ✅ Success Criteria

### E2E Testing Unblocked
- [ ] Functions deploy without errors
- [ ] Queue messages processed successfully
- [ ] Time-based limiting observable in logs
- [ ] Document completes all 116 batches
- [ ] No ModuleNotFoundError
- [ ] Processing completes in < 30 minutes

### Production Ready
- [ ] All Phase 1-3 gaps addressed
- [ ] Load tested with 10+ concurrent documents
- [ ] Monitoring and alerting operational
- [ ] Deployment process documented
- [ ] Rollback tested and verified
- [ ] On-call runbook created

---

**Gap Analysis Date**: 2026-01-14  
**Total Gaps**: 17 (2 critical, 6 high, 9 medium)  
**Estimated Total Effort**: ~60 hours  
**Time to Unblock Testing**: ~4 hours (Phase 1)  
**Time to Production Ready**: ~23 hours (Phases 1-3)
