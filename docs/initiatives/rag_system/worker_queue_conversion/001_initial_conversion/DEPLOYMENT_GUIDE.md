# Azure Functions Deployment Guide

This guide provides step-by-step instructions for deploying Azure Functions and configuring environment variables.

## Prerequisites

1. **Azure CLI** installed and authenticated (`az login`)
2. **Azure Functions Core Tools** installed (`func --version`)
3. **Access to Azure resources**:
   - Function App: `func-raglab-uploadworkers`
   - Resource Group: `rag-lab`
   - Storage Account: `raglabqueues`

## Quick Start

### Step 1: Set Up Environment Variables

Run the setup script to create `.env.local` with Azure resource values pre-filled:

```bash
./scripts/setup_env_local.sh
```

This script will:
- Retrieve Azure Storage connection string
- Retrieve Application Insights connection string
- Create `.env.local` template with TODO items for manual values

**Then edit `.env.local`** and fill in:
- Supabase credentials (SUPABASE_URL, SUPABASE_KEY, DATABASE_URL)
- Azure AI Foundry credentials (AZURE_AI_FOUNDRY_ENDPOINT, AZURE_AI_FOUNDRY_API_KEY)
- Azure AI Search credentials (AZURE_SEARCH_*, AZURE_SEARCH_INDEX_NAME)
- Azure Document Intelligence credentials (AZURE_DOCUMENT_INTELLIGENCE_*)
- Azure Blob Storage (if different from queue storage)

### Step 2: Set Up Git-Based Deployment (Recommended)

Configure automatic deployment from your Git repository:

```bash
./scripts/setup_git_deployment.sh
```

Or manually specify repository URL:

```bash
./scripts/setup_git_deployment.sh https://github.com/your-org/rag_evaluator.git main
```

This will:
- Connect Function App to your Git repository
- Configure automatic deployments on push to main branch
- Set up build script to include backend code

**After setup, deployments happen automatically when you push to Git!**

#### Alternative: Manual Deployment

If you prefer manual deployment or need to deploy without Git:

```bash
./scripts/deploy_azure_functions.sh
```

This script will:
- Create deployment package with backend code
- Deploy all 4 worker functions (ingestion, chunking, embedding, indexing)
- Verify deployment

### Step 3: Configure Function App Environment Variables

Configure all environment variables in the Function App:

```bash
./scripts/configure_function_app_env.sh
```

This script reads from `.env.local` and sets all required variables in the Function App.

### Step 4: Verify Deployment

Check that all functions are deployed:

```bash
az functionapp function list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[].{Name:name, Status:status}" -o table
```

Expected output:
```
Name                Status
------------------  --------
ingestion-worker    Enabled
chunking-worker     Enabled
embedding-worker    Enabled
indexing-worker     Enabled
```

Verify environment variables:

```bash
az functionapp config appsettings list \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --query "[?contains(name, 'DATABASE') || contains(name, 'SUPABASE') || contains(name, 'AZURE_AI') || contains(name, 'AZURE_SEARCH') || contains(name, 'AZURE_DOCUMENT')].{Name:name}" -o table
```

## Manual Steps

### Get Azure Resource Values

To get values for your `.env.local` file:

```bash
./scripts/get_azure_env_values.sh
```

This outputs:
- Azure Storage connection string
- Application Insights connection string
- List of currently configured Function App variables
- Template for required variables

### Manual Deployment (Alternative)

If the deployment script fails, you can deploy manually:

```bash
cd infra/azure/azure_functions
func azure functionapp publish func-raglab-uploadworkers --python
```

### Manual Environment Variable Configuration

If the configuration script fails, set variables manually:

```bash
az functionapp config appsettings set \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --settings \
    DATABASE_URL="your-database-url" \
    SUPABASE_URL="your-supabase-url" \
    SUPABASE_KEY="your-supabase-key" \
    # ... (add all other variables)
```

## Troubleshooting

### Functions Not Deploying

1. **Check Azure Functions Core Tools version**:
   ```bash
   func --version
   ```
   Should be 4.x or later.

2. **Check Python version compatibility**:
   - Local Python version should match Function App runtime (3.12)
   - Warning about version mismatch is OK, Azure will use 3.12

3. **Check deployment logs**:
   - Review the deployment output for errors
   - Check Azure Portal → Function App → Deployment Center for detailed logs

### Environment Variables Not Set

1. **Verify .env.local exists and has values**:
   ```bash
   cat .env.local
   ```

2. **Check script execution**:
   ```bash
   ./scripts/configure_function_app_env.sh
   ```

3. **Verify in Azure Portal**:
   - Navigate to Function App → Configuration → Application settings
   - Check that all required variables are present

### Functions Not Triggering

1. **Check queue connection**:
   - Verify `AzureWebJobsStorage` is set correctly
   - Verify `AZURE_STORAGE_QUEUES_CONNECTION_STRING` is set

2. **Check queue names**:
   - Verify queue names match function.json configurations
   - Queues: `ingestion-uploads`, `ingestion-chunking`, `ingestion-embeddings`, `ingestion-indexing`

3. **Check function status**:
   ```bash
   az functionapp function show \
     --name func-raglab-uploadworkers \
     --resource-group rag-lab \
     --function-name ingestion-worker
   ```

## Scripts Reference

### `scripts/setup_env_local.sh`
Creates `.env.local` file with Azure resource values pre-filled.

### `scripts/get_azure_env_values.sh`
Outputs Azure resource values for manual `.env.local` configuration.

### `scripts/setup_git_deployment.sh` ⭐ **Recommended**
Configures Git-based deployment for automatic deployments on push.

### `scripts/deploy_azure_functions.sh`
Manual deployment script (fallback if Git deployment not used).

### `scripts/configure_function_app_env.sh`
Configures Function App environment variables from `.env.local`.

## Next Steps

After successful deployment:

1. **Run Phase 2 Validation**:
   - Re-run Phase 2 tasks to verify deployment
   - Test queue triggers
   - Verify function configurations

2. **Run Integration Tests**:
   - Execute end-to-end pipeline tests
   - Test queue message processing
   - Verify status transitions

3. **Monitor Application Insights**:
   - Check function execution logs
   - Monitor queue processing
   - Track errors and performance

## Related Documentation

- `GIT_DEPLOYMENT.md` - **Git-based deployment guide (recommended)**
- `phase_5_azure_functions_deployment.md` - Detailed deployment guide
- `phase_5_testing.md` - Testing and validation
- `../../setup/environment_variables.md` - Environment variable reference

