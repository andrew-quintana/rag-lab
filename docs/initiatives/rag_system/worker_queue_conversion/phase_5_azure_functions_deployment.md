# Phase 5: Azure Functions Deployment Guide

## Overview

This guide documents the process for deploying workers as Azure Functions with queue triggers. All four workers (ingestion, chunking, embedding, indexing) are deployed as separate functions within a single Function App.

## Prerequisites

1. **Azure Subscription** with appropriate permissions
2. **Azure CLI** installed and configured (`az login`)
3. **Azure Functions Core Tools** installed (`func --version`)
4. **Python 3.9+** installed locally
5. **Database Migrations Applied** (see `phase_5_migration_guide.md`)
6. **Azure Storage Account** with Queue Storage enabled
7. **Azure Application Insights** resource (recommended)

## Architecture

```
Azure Function App (Consumption Plan)
├── ingestion-worker (queue trigger: ingestion-uploads)
├── chunking-worker (queue trigger: ingestion-chunking)
├── embedding-worker (queue trigger: ingestion-embeddings)
└── indexing-worker (queue trigger: ingestion-indexing)
```

## Step 0: Set Resource Names

Before creating resources, set these environment variables with your actual resource names. You can either set them manually or load from your `.env.local` file:

```bash
# Option 1: Set manually
export RESOURCE_GROUP="rag-lab"
export STORAGE_ACCOUNT_NAME="<your-storage-account-name>"  # e.g., raglabqueues
export APP_INSIGHTS_NAME="<your-app-insights-name>"       # e.g., ai-rag-evaluator
export FUNCTION_APP_NAME="<your-function-app-name>"       # e.g., func-rag-workers
export LOCATION="eastus"  # or your preferred Azure region

# Option 2: Load from .env.local (if you store resource names there)
# Note: This is optional - resource names are Azure resource identifiers, not app config
if [ -f .env.local ]; then
  # Load app config variables (these will be used in Step 2)
  export $(grep -v '^#' .env.local | grep -E '^(DATABASE_URL|SUPABASE_|AZURE_)' | xargs)
fi
```

**Important**: Resource names (storage account, function app, etc.) are Azure resource identifiers and are typically set once during initial setup. Application configuration variables (from `.env.local`) are set separately in Step 2.

## Step 1: Create Azure Resources

### 1.1 Create Resource Group

```bash
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION
```

### 1.2 Create Storage Account

```bash
az storage account create \
  --name $STORAGE_ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS \
  --kind StorageV2
```

### 1.3 Create Application Insights

```bash
az monitor app-insights component create \
  --app $APP_INSIGHTS_NAME \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP \
  --application-type web
```

### 1.4 Create Function App

```bash
# Get storage account connection string (for AzureWebJobsStorage)
AZURE_STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
  --name $STORAGE_ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv)

# Get Application Insights instrumentation key
AZURE_APPINSIGHTS_INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app $APP_INSIGHTS_NAME \
  --resource-group $RESOURCE_GROUP \
  --query instrumentationKey -o tsv)

# Create Function App
az functionapp create \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --storage-account $STORAGE_ACCOUNT_NAME \
  --consumption-plan-location $LOCATION \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --app-insights $APP_INSIGHTS_NAME \
  --os-type Linux
```

## Step 2: Configure Environment Variables

### 2.1 Set Application Settings

**Load environment variables from `.env.local`** (recommended):

```bash
# Load all variables from .env.local
if [ -f .env.local ]; then
  export $(grep -v '^#' .env.local | xargs)
else
  echo "Warning: .env.local not found. Set variables manually or create .env.local file."
fi

# Determine storage connection string to use
# Priority: 1) AZURE_STORAGE_CONNECTION_STRING from .env.local (matches .env.example)
#           2) AZURE_STORAGE_QUEUES_CONNECTION_STRING from .env.local (legacy)
#           3) AZURE_BLOB_CONNECTION_STRING from .env.local
#           4) AZURE_STORAGE_CONNECTION_STRING from Step 1.4 (Azure CLI)
STORAGE_CONN_STRING="${AZURE_STORAGE_CONNECTION_STRING:-${AZURE_STORAGE_QUEUES_CONNECTION_STRING:-${AZURE_BLOB_CONNECTION_STRING:-$AZURE_STORAGE_CONNECTION_STRING}}}"

# Set required environment variables in Function App
# These match the variable names in your .env.local file
az functionapp config appsettings set \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    DATABASE_URL="${DATABASE_URL}" \
    SUPABASE_URL="${SUPABASE_URL}" \
    SUPABASE_KEY="${SUPABASE_KEY}" \
    AZURE_AI_FOUNDRY_ENDPOINT="${AZURE_AI_FOUNDRY_ENDPOINT}" \
    AZURE_AI_FOUNDRY_API_KEY="${AZURE_AI_FOUNDRY_API_KEY}" \
    AZURE_AI_FOUNDRY_EMBEDDING_MODEL="${AZURE_AI_FOUNDRY_EMBEDDING_MODEL:-text-embedding-3-small}" \
    AZURE_AI_FOUNDRY_GENERATION_MODEL="${AZURE_AI_FOUNDRY_GENERATION_MODEL:-gpt-4o}" \
    AZURE_SEARCH_ENDPOINT="${AZURE_SEARCH_ENDPOINT}" \
    AZURE_SEARCH_API_KEY="${AZURE_SEARCH_API_KEY}" \
    AZURE_SEARCH_INDEX_NAME="${AZURE_SEARCH_INDEX_NAME}" \
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="${AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT}" \
    AZURE_DOCUMENT_INTELLIGENCE_API_KEY="${AZURE_DOCUMENT_INTELLIGENCE_API_KEY}" \
    AZURE_BLOB_CONNECTION_STRING="${AZURE_BLOB_CONNECTION_STRING}" \
    AZURE_BLOB_CONTAINER_NAME="${AZURE_BLOB_CONTAINER_NAME}" \
    AZURE_STORAGE_QUEUES_CONNECTION_STRING="${AZURE_STORAGE_CONNECTION_STRING:-${AZURE_STORAGE_QUEUES_CONNECTION_STRING:-$STORAGE_CONN_STRING}}" \
    APPLICATIONINSIGHTS_CONNECTION_STRING="${AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING:-${APPLICATIONINSIGHTS_CONNECTION_STRING:-}}" \
    AzureWebJobsStorage="$STORAGE_CONN_STRING"
```

**Note**: 
- `AzureWebJobsStorage` is required by Azure Functions runtime for queue triggers. It uses the same connection string as `AZURE_STORAGE_QUEUES_CONNECTION_STRING` (or falls back to `AZURE_BLOB_CONNECTION_STRING` or the value from Step 1.4).
- In `.env.local`, use `AZURE_STORAGE_CONNECTION_STRING` (matches `.env.example`). This maps to `AZURE_STORAGE_QUEUES_CONNECTION_STRING` in Azure Functions.
- In `.env.local`, use `AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING` (matches `.env.example`). This maps to `APPLICATIONINSIGHTS_CONNECTION_STRING` in Azure Functions.
- `APPINSIGHTS_INSTRUMENTATIONKEY` is automatically set by Azure Functions when Application Insights is configured in Step 1.4, so it doesn't need to be set manually unless you want to override it.
```

**Alternative: Set values directly** (if not using .env.local):

**Note**: `$AZURE_STORAGE_CONNECTION_STRING` and `$APPINSIGHTS_INSTRUMENTATIONKEY` are set in Step 1.4 above. If running this step separately, set them first:

```bash
# Get storage account connection string (if not already set in Step 1.4)
AZURE_STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
  --name $STORAGE_ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv)

# Get Application Insights instrumentation key (if not already set in Step 1.4)
APPINSIGHTS_INSTRUMENTATIONKEY=$(az monitor app-insights component show \
  --app $APP_INSIGHTS_NAME \
  --resource-group $RESOURCE_GROUP \
  --query instrumentationKey -o tsv)

# Then set Function App settings
az functionapp config appsettings set \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    DATABASE_URL="postgresql://user:pass@host:5432/db" \
    SUPABASE_URL="https://your-project.supabase.co" \
    SUPABASE_KEY="your-supabase-key" \
    AZURE_AI_FOUNDRY_ENDPOINT="https://your-endpoint.openai.azure.com/" \
    AZURE_AI_FOUNDRY_API_KEY="your-api-key" \
    AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net" \
    AZURE_SEARCH_API_KEY="your-search-key" \
    AZURE_SEARCH_INDEX_NAME="your-index-name" \
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://your-endpoint.cognitiveservices.azure.com/" \
    AZURE_DOCUMENT_INTELLIGENCE_API_KEY="your-doc-intelligence-key" \
    AZURE_BLOB_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=..." \
    AZURE_BLOB_CONTAINER_NAME="your-container-name" \
    AZURE_STORAGE_QUEUES_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
    AzureWebJobsStorage="$AZURE_STORAGE_CONNECTION_STRING"
```

### 2.2 Configure Azure Key Vault (Optional but Recommended)

For production, use Azure Key Vault to store secrets:

```bash
# Set Key Vault name (optional)
export KEY_VAULT_NAME="<your-key-vault-name>"  # e.g., kv-rag-evaluator

# Create Key Vault
az keyvault create \
  --name $KEY_VAULT_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Store secrets in Key Vault
az keyvault secret set --vault-name $KEY_VAULT_NAME --name DatabaseUrl --value "${DATABASE_URL}"
az keyvault secret set --vault-name $KEY_VAULT_NAME --name SupabaseKey --value "${SUPABASE_KEY}"
az keyvault secret set --vault-name $KEY_VAULT_NAME --name AzureAIFoundryApiKey --value "${AZURE_AI_FOUNDRY_API_KEY}"
az keyvault secret set --vault-name $KEY_VAULT_NAME --name AzureSearchApiKey --value "${AZURE_SEARCH_API_KEY}"
az keyvault secret set --vault-name $KEY_VAULT_NAME --name AzureDocumentIntelligenceApiKey --value "${AZURE_DOCUMENT_INTELLIGENCE_API_KEY}"
# ... (store other secrets as needed)

# Grant Function App access to Key Vault
az functionapp identity assign \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP

# Get principal ID
PRINCIPAL_ID=$(az functionapp identity show \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query principalId -o tsv)

# Grant access
az keyvault set-policy \
  --name $KEY_VAULT_NAME \
  --object-id $PRINCIPAL_ID \
  --secret-permissions get list

# Reference secrets in app settings
az functionapp config appsettings set \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    "@Microsoft.KeyVault(SecretUri=https://${KEY_VAULT_NAME}.vault.azure.net/secrets/DatabaseUrl/)" \
    "@Microsoft.KeyVault(SecretUri=https://${KEY_VAULT_NAME}.vault.azure.net/secrets/SupabaseKey/)"
```

## Step 3: Deploy Functions

### 3.1 Git-Based Deployment (Recommended) ⭐

**Best Practice**: Use Git-based deployment for automatic deployments synced with your repository.

#### 3.1.1 Set Up Git Deployment

```bash
# Run setup script (auto-detects git remote)
./scripts/setup_git_deployment.sh

# Or specify repository URL manually
./scripts/setup_git_deployment.sh https://github.com/your-org/rag_evaluator.git main
```

This configures:
- Automatic deployment on push to main branch
- Build script that includes backend code
- Native Azure Functions integration

#### 3.1.2 How It Works

1. **Build Script**: `infra/azure/azure_functions/build.sh` runs automatically during deployment
   - Copies `backend/rag_eval/` to deployment package
   - Updates function import paths
   - Prepares complete deployment package

2. **Deployment Trigger**: Push to configured branch (default: `main`)
   ```bash
   git push origin main
   ```

3. **Azure Functions**: Automatically pulls, builds, and deploys

#### 3.1.3 Verify Git Deployment

Check deployment status in Azure Portal:
- Navigate to Function App → Deployment Center
- View deployment history and logs
- See build script output

Or via CLI:
```bash
az functionapp deployment source show \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP
```

### 3.2 Manual Deployment (Alternative)

If you prefer manual deployment or need to deploy without Git:

#### 3.2.1 Deploy Using Script

```bash
./scripts/deploy_azure_functions.sh
```

#### 3.2.2 Deploy Using Azure Functions Core Tools

```bash
# Login to Azure
az login

# Navigate to Azure Functions directory
cd infra/azure/azure_functions

# Run build script first
./build.sh

# Deploy Function App
func azure functionapp publish $FUNCTION_APP_NAME --python
```

#### 3.2.3 Deploy Using Azure CLI

```bash
# Create deployment package
cd infra/azure/azure_functions
./build.sh  # Prepare package
zip -r functionapp.zip .

# Deploy using Azure CLI
az functionapp deployment source config-zip \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --src functionapp.zip
```

## Step 4: Verify Deployment

### 4.1 Check Function App Status

```bash
az functionapp list \
  --resource-group $RESOURCE_GROUP \
  --output table
```

### 4.2 List Functions

```bash
az functionapp function list \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --output table
```

### 4.3 Test Queue Triggers

Verify that queue triggers are configured correctly:

```bash
# Check function configuration
az functionapp function show \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --function-name ingestion-worker
```

### 4.4 Monitor Logs

```bash
# Stream logs
az functionapp log tail \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP
```

## Step 5: Configure Queues

Ensure Azure Storage Queues are created (they auto-create on first message, but you can create them manually):

```bash
# Get storage account key
STORAGE_KEY=$(az storage account keys list \
  --account-name $STORAGE_ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --query [0].value -o tsv)

# Create queues
az storage queue create \
  --name ingestion-uploads \
  --account-name $STORAGE_ACCOUNT_NAME \
  --account-key $STORAGE_KEY

az storage queue create \
  --name ingestion-chunking \
  --account-name $STORAGE_ACCOUNT_NAME \
  --account-key $STORAGE_KEY

az storage queue create \
  --name ingestion-embeddings \
  --account-name $STORAGE_ACCOUNT_NAME \
  --account-key $STORAGE_KEY

az storage queue create \
  --name ingestion-indexing \
  --account-name $STORAGE_ACCOUNT_NAME \
  --account-key $STORAGE_KEY

az storage queue create \
  --name ingestion-dead-letter \
  --account-name $STORAGE_ACCOUNT_NAME \
  --account-key $STORAGE_KEY
```

## Step 6: Application Insights Configuration

### 6.1 Verify Application Insights Integration

Application Insights should be automatically configured when creating the Function App. Verify:

```bash
az functionapp config appsettings list \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "[?name=='APPINSIGHTS_INSTRUMENTATIONKEY']"
```

### 6.2 Set Up Custom Metrics (Optional)

You can configure custom metrics in Application Insights for:
- Queue depth monitoring
- Worker processing time
- Failure rates
- Document processing throughput

## Step 7: Testing Deployment

### 7.1 Test Queue Trigger

Send a test message to the queue:

```bash
# Using Azure CLI
az storage message put \
  --queue-name ingestion-uploads \
  --content '{"document_id":"test-123","source_storage":"supabase","filename":"test.pdf","attempt":1,"stage":"uploaded"}' \
  --account-name $STORAGE_ACCOUNT_NAME \
  --account-key $STORAGE_KEY
```

### 7.2 Monitor Function Execution

Check Application Insights or function logs to verify the function executed:

```bash
az functionapp log tail \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP
```

## Troubleshooting

### Functions Not Triggering

1. **Check Queue Connection**: Verify `AzureWebJobsStorage` setting points to correct storage account
2. **Check Queue Name**: Ensure queue name matches function.json configuration
3. **Check Function Status**: Verify functions are enabled in Azure Portal

### Import Errors

1. **Check Requirements**: Ensure all dependencies in `requirements.txt` are installed
2. **Check Python Path**: Verify backend code is accessible from function directory
3. **Check Logs**: Review function logs for specific import errors

### Timeout Issues

1. **Increase Timeout**: Update `functionTimeout` in `host.json` (max 10 minutes for Consumption Plan)
2. **Optimize Processing**: Review worker code for performance bottlenecks
3. **Check Cold Starts**: Monitor cold start latency in Application Insights

### Connection Errors

1. **Verify Environment Variables**: Check all required settings are configured
2. **Check Network**: Ensure Function App can reach Supabase and Azure services
3. **Verify Credentials**: Test connection strings and API keys

## Monitoring and Maintenance

### Application Insights Queries

Monitor worker performance:

```kusto
// Function execution count
traces
| where operation_Name startswith "ingestion-worker"
| summarize count() by bin(timestamp, 1h)

// Error rate
traces
| where severityLevel >= 3
| summarize count() by operation_Name, bin(timestamp, 1h)

// Processing time
traces
| where operation_Name contains "worker"
| summarize avg(duration), max(duration) by operation_Name
```

### Scaling Configuration

Consumption Plan automatically scales based on queue depth. Monitor:
- Queue depth per queue
- Function execution count
- Concurrent executions
- Cold start frequency

## Next Steps

After deployment:
1. Run integration tests with real Azure resources
2. Monitor Application Insights for errors
3. Test end-to-end pipeline flow
4. Proceed with performance testing

