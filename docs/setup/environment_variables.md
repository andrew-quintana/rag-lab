# Environment Variables Reference

This document lists all environment variables needed for the RAG Evaluator platform, including Azure Functions and Application Insights configuration.

## Quick Start

Create a `.env.local` file in the project root with your actual values. Copy the template below and fill in your credentials.

## Environment Variables Template

```bash
# =============================================================================
# Database (Supabase Postgres)
# =============================================================================
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres

# =============================================================================
# Azure AI Foundry
# =============================================================================
AZURE_AI_FOUNDRY_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_AI_FOUNDRY_API_KEY=your-ai-foundry-api-key
AZURE_AI_FOUNDRY_EMBEDDING_MODEL=text-embedding-3-small
AZURE_AI_FOUNDRY_GENERATION_MODEL=gpt-4o

# =============================================================================
# Azure AI Search
# =============================================================================
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your-search-api-key
AZURE_SEARCH_INDEX_NAME=your-index-name

# =============================================================================
# Azure Document Intelligence
# =============================================================================
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=your-document-intelligence-api-key

# =============================================================================
# Azure Blob Storage
# =============================================================================
AZURE_BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net
AZURE_BLOB_CONTAINER_NAME=your-container-name

# =============================================================================
# Azure Storage Queues (for Worker-Queue Architecture)
# =============================================================================
# Connection string for Azure Storage Account with Queue Storage enabled
# Used by workers for queue operations (enqueue, dequeue, etc.)
# Can use the same connection string as AZURE_BLOB_CONNECTION_STRING if using same storage account
AZURE_STORAGE_QUEUES_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net

# =============================================================================
# Azure Functions (for deployed workers)
# =============================================================================
# AzureWebJobsStorage is automatically set by Azure Functions when deployed
# It uses the storage account connection string for the Function App
# For local development/testing, you can set it manually:
# AZUREWEBJOBSSTORAGE=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net

# =============================================================================
# Azure Application Insights
# =============================================================================
# APPINSIGHTS_INSTRUMENTATIONKEY is automatically set by Azure Functions when Application Insights is configured
# For local development/testing, you can set it manually:
# APPINSIGHTS_INSTRUMENTATIONKEY=your-instrumentation-key

# =============================================================================
# API Configuration (with defaults)
# =============================================================================
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
```

## Variable Descriptions

### Database (Supabase)

- **SUPABASE_URL**: Your Supabase project URL (e.g., `https://xxxxx.supabase.co`)
- **SUPABASE_KEY**: Your Supabase anon/public key
- **DATABASE_URL**: PostgreSQL connection string for direct database access

### Azure AI Foundry

- **AZURE_AI_FOUNDRY_ENDPOINT**: Your Azure AI Foundry endpoint URL
- **AZURE_AI_FOUNDRY_API_KEY**: API key for Azure AI Foundry
- **AZURE_AI_FOUNDRY_EMBEDDING_MODEL**: Embedding model name (default: `text-embedding-3-small`)
- **AZURE_AI_FOUNDRY_GENERATION_MODEL**: Generation model name (default: `gpt-4o`)

### Azure AI Search

- **AZURE_SEARCH_ENDPOINT**: Your Azure AI Search service endpoint
- **AZURE_SEARCH_API_KEY**: Admin API key for Azure AI Search
- **AZURE_SEARCH_INDEX_NAME**: Name of your search index

### Azure Document Intelligence

- **AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT**: Your Document Intelligence endpoint URL
- **AZURE_DOCUMENT_INTELLIGENCE_API_KEY**: API key for Document Intelligence

### Azure Blob Storage

- **AZURE_BLOB_CONNECTION_STRING**: Connection string for Azure Blob Storage account
- **AZURE_BLOB_CONTAINER_NAME**: Name of the blob container for document storage

### Azure Storage Queues (Worker-Queue Architecture)

- **AZURE_STORAGE_QUEUES_CONNECTION_STRING**: Connection string for Azure Storage Account with Queue Storage enabled
  - Used by workers for queue operations (enqueue, dequeue, etc.)
  - Can use the same connection string as `AZURE_BLOB_CONNECTION_STRING` if using the same storage account
  - The queue client also checks `AZURE_BLOB_CONNECTION_STRING` as a fallback

### Azure Functions (Deployed Workers)

- **AZUREWEBJOBSSTORAGE**: Storage account connection string for Azure Functions
  - **Automatically set** by Azure Functions when deployed
  - Used by Azure Functions runtime for queue triggers and internal operations
  - Only needed for local development/testing of Azure Functions

### Azure Application Insights

- **APPINSIGHTS_INSTRUMENTATIONKEY**: Application Insights instrumentation key
  - **Automatically set** by Azure Functions when Application Insights is configured
  - Used for logging, metrics, and telemetry
  - Only needed for local development/testing

### API Configuration

- **API_HOST**: Host address for the API server (default: `0.0.0.0`)
- **API_PORT**: Port number for the API server (default: `8000`)
- **API_RELOAD**: Enable auto-reload for development (default: `true`)

## Azure Functions Deployment

When deploying workers as Azure Functions, these environment variables are set via Azure Function App configuration:

1. **Via Azure CLI** (see `phase_5_azure_functions_deployment.md`):
   ```bash
   az functionapp config appsettings set \
     --name func-rag-workers \
     --resource-group rag-lab \
     --settings \
       DATABASE_URL="..." \
       SUPABASE_URL="..." \
       # ... (all other variables)
   ```

2. **Via Azure Portal**:
   - Navigate to Function App → Configuration → Application settings
   - Add each variable as a new application setting

3. **Via Azure Key Vault** (recommended for production):
   - Store secrets in Key Vault
   - Reference them in Function App settings using `@Microsoft.KeyVault(SecretUri=...)`

## Local Development

For local development:

1. Create `.env.local` in the project root
2. Copy the template above
3. Fill in your actual values
4. The application will automatically load from `.env.local`

## Notes

- **No quotes needed**: Connection strings should not be wrapped in quotes
- **Same storage account**: You can use the same connection string for both Blob Storage and Queue Storage if they're in the same storage account
- **Auto-set variables**: `AZUREWEBJOBSSTORAGE` and `APPINSIGHTS_INSTRUMENTATIONKEY` are automatically configured by Azure Functions - you only need to set them manually for local testing
- **Priority**: The queue client checks `AZURE_BLOB_CONNECTION_STRING` first, then falls back to `AZURE_STORAGE_QUEUES_CONNECTION_STRING`

## Related Documentation

- **Azure Queue Setup**: `docs/setup/azure_queues.md`
- **Azure Functions Deployment**: `docs/initiatives/rag_system/worker_queue_conversion/phase_5_azure_functions_deployment.md`
- **Configuration Loading**: `backend/rag_eval/core/config.py`

