# Azure Functions Cloud Testing - Documentation Index

**Last Updated**: 2026-01-14  
**Status**: ⚠️ Queue Trigger Issue - See Handoff Document

---

## 🚨 Start Here

### For New Agents
👉 **Read First**: [`HANDOFF_NEXT_AGENT_QUEUE_TRIGGER.md`](./HANDOFF_NEXT_AGENT_QUEUE_TRIGGER.md)

This document contains:
- Complete problem description
- Step-by-step debugging guide
- All testing commands
- Common issues and solutions
- Success criteria

---

## 📚 Documentation Structure

### Setup & Configuration
- **[Environment Variables](../setup/ENVIRONMENT_VARIABLES.md)** - Environment variable documentation
- **[Setup Complete Summary](./SETUP_COMPLETE_SUMMARY.md)** - Infrastructure setup status
- **[Cloud Deployment Status](./CLOUD_DEPLOYMENT_STATUS.md)** - Deployment details

### Current Issues
- **[Queue Trigger Investigation](./QUEUE_TRIGGER_INVESTIGATION.md)** - Detailed investigation report
- **[Handoff Document](./HANDOFF_NEXT_AGENT_QUEUE_TRIGGER.md)** - **START HERE** for debugging

### Implementation Details
- **[Time-Based Limiting](./TIME_BASED_LIMITING_IMPLEMENTATION.md)** - Time-based batch limiting implementation
- **[Bug Fix Summary](./BUG_FIX_SUMMARY.md)** - Initial page range bug fix
- **[E2E Test Results](./E2E_TEST_RESULTS.md)** - Test results and findings

### Historical Context
- **[Azure Functions Test Results](./AZURE_FUNCTIONS_TEST_RESULTS.md)** - Initial test results
- **[Validation Results](./VALIDATION_RESULTS.md)** - Local validation results
- **[Cloud Test Summary](./CLOUD_TEST_SUMMARY.md)** - Cloud test summary

---

## 🎯 Quick Status

### ✅ Complete
- Cloud Supabase database setup
- Azure Functions deployment
- Build script fixes
- Environment variable configuration
- ModuleNotFoundError resolution
- Test data preparation

### ⚠️ In Progress
- Queue trigger debugging (BLOCKING)

### 📋 Pending
- E2E cloud testing
- Time-based limiting validation in cloud
- Full document processing validation

---

## 🔧 Quick Commands

### Check Queue Status
```bash
cd /Users/aq_home/1Projects/rag_evaluator
source backend/venv/bin/activate
export AZURE_STORAGE_QUEUES_CONNECTION_STRING="..."
python scripts/monitor_queue_health.py
```

### Check Function App Status
```bash
az functionapp show \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "{state:state, runtime:siteConfig.linuxFxVersion}" \
  --output table
```

### Check Database
```bash
cd /Users/aq_home/1Projects/rag_evaluator
export $(grep -v '^#' .env.prod | xargs)
psql "$DATABASE_URL" -c "SELECT id, filename, status, chunks_created FROM documents WHERE id = 'f7df526a-97df-4e85-836d-b3478a72ae6d';"
```

---

## 📝 Key Resources

### Azure Resources
- **Function App**: `func-raglab-uploadworkers` (Resource Group: `rag-lab`)
- **Storage Account**: `raglabqueues`
- **Supabase Cloud**: `oeyivkusvlgyuorcjime.supabase.co`
- **Application Insights**: `ai-rag-evaluator`

### Test Data
- **Document ID**: `f7df526a-97df-4e85-836d-b3478a72ae6d`
- **Filename**: `scan_classic_hmo.pdf`
- **Location**: Azure Blob Storage container `documents`

### Key Files
- **Function Entry**: `backend/azure_functions/ingestion-worker/__init__.py`
- **Function Binding**: `backend/azure_functions/ingestion-worker/function.json`
- **Host Config**: `backend/azure_functions/host.json`
- **Build Script**: `backend/azure_functions/build.sh`

---

## 🆘 Troubleshooting

### Queue Trigger Not Firing
👉 See: [`HANDOFF_NEXT_AGENT_QUEUE_TRIGGER.md`](./HANDOFF_NEXT_AGENT_QUEUE_TRIGGER.md)

### ModuleNotFoundError
👉 See: [`CLOUD_DEPLOYMENT_STATUS.md`](./CLOUD_DEPLOYMENT_STATUS.md) - Section "Build Script Fix"

### Environment Variables
👉 See: [`../setup/ENVIRONMENT_VARIABLES.md`](../setup/ENVIRONMENT_VARIABLES.md)

### Database Connection Issues
👉 See: [`SETUP_COMPLETE_SUMMARY.md`](./SETUP_COMPLETE_SUMMARY.md) - Section "Cloud Supabase Setup"

---

## 📞 Next Steps

1. **Read the handoff document**: [`HANDOFF_NEXT_AGENT_QUEUE_TRIGGER.md`](./HANDOFF_NEXT_AGENT_QUEUE_TRIGGER.md)
2. **Access Azure Portal Log Stream** (highest priority)
3. **Follow the debugging steps** in the handoff document
4. **Update documentation** with findings and solution

---

**For detailed instructions, see the handoff document!** 🚀
