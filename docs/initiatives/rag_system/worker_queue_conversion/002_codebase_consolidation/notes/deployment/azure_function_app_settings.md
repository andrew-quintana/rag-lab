# Azure Function App Settings

This document lists all required and optional environment variables for Azure Functions deployment.

## Required Settings

All of these variables must be set in Azure Function App configuration for production deployment.

### Database (Supabase Postgres)

- **`DATABASE_URL`** (Required)
  - PostgreSQL connection string
  - Format: `postgresql://user:password@host:port/database`
  - Example: `postgresql://postgres:password@db.project.supabase.co:5432/postgres`

- **`SUPABASE_URL`** (Required)
  - Supabase project URL
  - Format: `https://your-project.supabase.co`
  - Example: `https://abcdefghijklmnop.supabase.co`

- **`SUPABASE_KEY`** (Required)
  - Supabase anon/service role key
  - Format: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
  - Can use anon key or service role key (service role has more permissions)

### Azure AI Foundry

- **`AZURE_AI_FOUNDRY_ENDPOINT`** (Required)
  - Azure OpenAI endpoint
  - Format: `https://your-endpoint.openai.azure.com/`
  - Example: `https://my-foundry.openai.azure.com/`

- **`AZURE_AI_FOUNDRY_API_KEY`** (Required)
  - Azure OpenAI API key
  - Format: `32-character-hex-string`
  - Example: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`

- **`AZURE_AI_FOUNDRY_EMBEDDING_MODEL`** (Optional, Default: `text-embedding-3-small`)
  - Embedding model name
  - Example: `text-embedding-3-small`

- **`AZURE_AI_FOUNDRY_GENERATION_MODEL`** (Optional, Default: `gpt-4o`)
  - Generation model name
  - Example: `gpt-4o`

### Azure AI Search

- **`AZURE_SEARCH_ENDPOINT`** (Required)
  - Azure AI Search service endpoint
  - Format: `https://your-service.search.windows.net`
  - Example: `https://my-search.search.windows.net`

- **`AZURE_SEARCH_API_KEY`** (Required)
  - Azure AI Search API key
  - Format: `admin-key-or-query-key`
  - Example: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`

- **`AZURE_SEARCH_INDEX_NAME`** (Required)
  - Azure AI Search index name
  - Format: `lowercase-alphanumeric-with-hyphens`
  - Example: `document-index`

### Azure Document Intelligence

- **`AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`** (Required)
  - Azure Document Intelligence endpoint
  - Format: `https://your-endpoint.cognitiveservices.azure.com/`
  - Example: `https://my-doc-intel.cognitiveservices.azure.com/`

- **`AZURE_DOCUMENT_INTELLIGENCE_API_KEY`** (Required)
  - Azure Document Intelligence API key
  - Format: `32-character-hex-string`
  - Example: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`

### Azure Storage

- **`AzureWebJobsStorage`** (Required)
  - Azure Storage connection string for Azure Functions runtime
  - Format: `DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net`
  - Used by Azure Functions for queue triggers and internal operations
  - Automatically set by Azure Functions, but can be overridden

- **`AZURE_STORAGE_QUEUES_CONNECTION_STRING`** (Required)
  - Azure Storage connection string for queue operations
  - Format: `DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net`
  - Can be same as `AzureWebJobsStorage` if using same storage account
  - Used by workers for queue operations (enqueue, dequeue, etc.)

### Optional Settings

- **`AZURE_BLOB_CONNECTION_STRING`** (Optional)
  - Azure Blob Storage connection string
  - Format: `DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net`
  - Only needed if using Azure Blob Storage for document storage
  - If not set, workers will use Supabase Storage

- **`AZURE_BLOB_CONTAINER_NAME`** (Optional)
  - Azure Blob Storage container name
  - Format: `lowercase-alphanumeric-with-hyphens`
  - Only needed if using Azure Blob Storage
  - Example: `documents`

- **`APPLICATIONINSIGHTS_CONNECTION_STRING`** (Optional)
  - Application Insights connection string
  - Format: `InstrumentationKey=...;IngestionEndpoint=https://...`
  - Automatically set by Azure Functions if Application Insights is configured
  - Used for logging, metrics, and telemetry

## Setting Variables

### Via Azure CLI

```bash
az functionapp config appsettings set \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --settings \
    DATABASE_URL="postgresql://..." \
    SUPABASE_URL="https://..." \
    SUPABASE_KEY="..." \
    AZURE_AI_FOUNDRY_ENDPOINT="https://..." \
    AZURE_AI_FOUNDRY_API_KEY="..." \
    AZURE_AI_FOUNDRY_EMBEDDING_MODEL="text-embedding-3-small" \
    AZURE_AI_FOUNDRY_GENERATION_MODEL="gpt-4o" \
    AZURE_SEARCH_ENDPOINT="https://..." \
    AZURE_SEARCH_API_KEY="..." \
    AZURE_SEARCH_INDEX_NAME="..." \
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://..." \
    AZURE_DOCUMENT_INTELLIGENCE_API_KEY="..." \
    AzureWebJobsStorage="DefaultEndpointsProtocol=https;..." \
    AZURE_STORAGE_QUEUES_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
```

### Via Azure Portal

1. Navigate to Function App → Configuration → Application settings
2. Click "+ New application setting" for each variable
3. Enter variable name and value
4. Click "Save" to apply changes

### Via Script

Use the provided script to set variables from `.env.local`:

```bash
./scripts/configure_function_app_env.sh
```

Or:

```bash
./backend/scripts/set_function_app_env_vars.sh
```

## Variable Validation

Run the validation script to check all required variables are set:

```bash
python scripts/validate_config.py --cloud
```

## Security Best Practices

1. **Use Azure Key Vault** (Recommended for production):
   - Store secrets in Key Vault
   - Reference them in Function App settings using `@Microsoft.KeyVault(SecretUri=...)`
   - Example: `DATABASE_URL=@Microsoft.KeyVault(SecretUri=https://vault.vault.azure.net/secrets/database-url/)`

2. **Use Service Principal** (For CI/CD):
   - Use service principal with appropriate permissions
   - Store credentials securely in CI/CD secrets

3. **Rotate Keys Regularly**:
   - Rotate API keys and connection strings regularly
   - Update Function App settings when rotating

4. **Monitor Access**:
   - Enable Application Insights for monitoring
   - Review access logs regularly

## Troubleshooting

### Variables Not Available

**Problem**: Variables not accessible in functions.

**Solutions**:
1. Verify variables are set in Function App settings (not just `.env.local`)
2. Check variable names match exactly (case-sensitive)
3. Restart Function App after setting variables
4. Check Application Insights logs for configuration errors

### Connection String Issues

**Problem**: Connection strings not working.

**Solutions**:
1. Verify connection strings are complete (not truncated)
2. Check storage account exists and is accessible
3. Verify account keys are correct
4. Check network restrictions (firewall rules)

### Missing Variables

**Problem**: Required variables not set.

**Solutions**:
1. Run validation script: `python scripts/validate_config.py --cloud`
2. Check this document for complete list
3. Verify all variables are set in Function App settings

## Related Documentation

- [Configuration Guide](./configuration_guide.md) - Configuration loading strategy
- [Environment Variables Reference](../../../../setup/environment_variables.md) - Complete variable reference
- [Local Development Guide](../../001_initial_conversion/LOCAL_DEVELOPMENT.md) - Local setup instructions

---

**Last Updated**: 2025-12-09  
**Related Initiative**: 002 (Codebase Consolidation)  
**Status**: Current

