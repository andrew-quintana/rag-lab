# Environment Variable Name Mapping

This document explains the mapping between `.env.local` variable names (matching `.env.example`) and Azure Functions variable names.

## Variable Name Mapping

### Storage Connection String

| .env.local / .env.example | Azure Functions | Notes |
|---------------------------|-----------------|-------|
| `AZURE_STORAGE_CONNECTION_STRING` | `AZURE_STORAGE_QUEUES_CONNECTION_STRING` | Primary variable name in `.env.local` |
| `AZURE_STORAGE_QUEUES_CONNECTION_STRING` | `AZURE_STORAGE_QUEUES_CONNECTION_STRING` | Legacy name (still supported) |
| `AZURE_BLOB_CONNECTION_STRING` | `AZURE_BLOB_CONNECTION_STRING` | Used as fallback |

**Usage**: The deployment scripts automatically map `AZURE_STORAGE_CONNECTION_STRING` from `.env.local` to `AZURE_STORAGE_QUEUES_CONNECTION_STRING` in Azure Functions.

### Application Insights

| .env.local / .env.example | Azure Functions | Notes |
|---------------------------|-----------------|-------|
| `AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING` | `APPLICATIONINSIGHTS_CONNECTION_STRING` | Primary variable name in `.env.local` |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | `APPLICATIONINSIGHTS_CONNECTION_STRING` | Legacy name (still supported) |
| `AZURE_APPINSIGHTS_INSTRUMENTATION_KEY` | `APPINSIGHTS_INSTRUMENTATIONKEY` | For local development only |

**Usage**: The deployment scripts automatically map `AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING` from `.env.local` to `APPLICATIONINSIGHTS_CONNECTION_STRING` in Azure Functions.

### Other Variables

All other variables use the same name in both `.env.local` and Azure Functions:
- `DATABASE_URL`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `AZURE_AI_FOUNDRY_*`
- `AZURE_SEARCH_*`
- `AZURE_DOCUMENT_INTELLIGENCE_*`
- `AZURE_BLOB_*`

## Default Values

### Azure AI Foundry Models

| Variable | .env.example Default | Azure Functions Default | Notes |
|----------|---------------------|------------------------|-------|
| `AZURE_AI_FOUNDRY_EMBEDDING_MODEL` | `text-embedding-ada-002` | `text-embedding-3-small` | Different defaults |
| `AZURE_AI_FOUNDRY_GENERATION_MODEL` | `gpt-4o` | `gpt-4o` | Same |
| `AZURE_AI_FOUNDRY_REASONING_MODEL` | `gpt-4o` | (not set) | Only in .env.example |

**Note**: The `.env.example` uses `text-embedding-ada-002` as the default, while the code defaults to `text-embedding-3-small`. Use the value that matches your Azure AI Foundry configuration.

## Script Behavior

The deployment scripts (`configure_function_app_env.sh`) handle the mapping automatically:

1. **Reads from `.env.local`** using the variable names from `.env.example`
2. **Maps to Azure Functions** variable names when needed
3. **Supports both** old and new variable names for backward compatibility

## Example

In your `.env.local` file:
```bash
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=...;..."
```

When deployed to Azure Functions, these become:
```bash
AZURE_STORAGE_QUEUES_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=...;..."
```

The mapping is handled automatically by the deployment scripts.

