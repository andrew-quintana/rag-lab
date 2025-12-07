#!/bin/bash
# Script to configure environment variables in Azure Function App
# This script reads from .env.local and sets all required variables in the Function App

set -e

RESOURCE_GROUP="rag-lab"
FUNCTION_APP_NAME="func-raglab-uploadworkers"
ENV_FILE=".env.local"

if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE not found"
  echo "Please create $ENV_FILE with your environment variables"
  echo "You can use ./scripts/get_azure_env_values.sh to get Azure resource values"
  exit 1
fi

echo "=========================================="
echo "Configuring Function App Environment Variables"
echo "=========================================="
echo "Function App: $FUNCTION_APP_NAME"
echo "Resource Group: $RESOURCE_GROUP"
echo ""

# Load variables from .env.local
# This will export all variables that don't start with #
export $(grep -v '^#' $ENV_FILE | grep -v '^$' | xargs)

# Get storage connection string (support both AZURE_STORAGE_CONNECTION_STRING from .env.example and AZURE_STORAGE_QUEUES_CONNECTION_STRING)
STORAGE_CONN_STRING="${AZURE_STORAGE_CONNECTION_STRING:-${AZURE_STORAGE_QUEUES_CONNECTION_STRING:-${AZURE_BLOB_CONNECTION_STRING}}}"

if [ -z "$STORAGE_CONN_STRING" ]; then
  echo "Error: AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_QUEUES_CONNECTION_STRING, or AZURE_BLOB_CONNECTION_STRING not found in $ENV_FILE"
  exit 1
fi

echo "Setting environment variables..."

# Build the settings string
SETTINGS=""
SETTINGS="${SETTINGS} DATABASE_URL=\"${DATABASE_URL}\""
SETTINGS="${SETTINGS} SUPABASE_URL=\"${SUPABASE_URL}\""
SETTINGS="${SETTINGS} SUPABASE_KEY=\"${SUPABASE_KEY}\""
SETTINGS="${SETTINGS} AZURE_AI_FOUNDRY_ENDPOINT=\"${AZURE_AI_FOUNDRY_ENDPOINT}\""
SETTINGS="${SETTINGS} AZURE_AI_FOUNDRY_API_KEY=\"${AZURE_AI_FOUNDRY_API_KEY}\""
SETTINGS="${SETTINGS} AZURE_AI_FOUNDRY_EMBEDDING_MODEL=\"${AZURE_AI_FOUNDRY_EMBEDDING_MODEL:-text-embedding-3-small}\""
SETTINGS="${SETTINGS} AZURE_AI_FOUNDRY_GENERATION_MODEL=\"${AZURE_AI_FOUNDRY_GENERATION_MODEL:-gpt-4o}\""
SETTINGS="${SETTINGS} AZURE_SEARCH_ENDPOINT=\"${AZURE_SEARCH_ENDPOINT}\""
SETTINGS="${SETTINGS} AZURE_SEARCH_API_KEY=\"${AZURE_SEARCH_API_KEY}\""
SETTINGS="${SETTINGS} AZURE_SEARCH_INDEX_NAME=\"${AZURE_SEARCH_INDEX_NAME}\""
SETTINGS="${SETTINGS} AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=\"${AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT}\""
SETTINGS="${SETTINGS} AZURE_DOCUMENT_INTELLIGENCE_API_KEY=\"${AZURE_DOCUMENT_INTELLIGENCE_API_KEY}\""

# Optional variables
if [ -n "$AZURE_BLOB_CONNECTION_STRING" ]; then
  SETTINGS="${SETTINGS} AZURE_BLOB_CONNECTION_STRING=\"${AZURE_BLOB_CONNECTION_STRING}\""
fi

if [ -n "$AZURE_BLOB_CONTAINER_NAME" ]; then
  SETTINGS="${SETTINGS} AZURE_BLOB_CONTAINER_NAME=\"${AZURE_BLOB_CONTAINER_NAME}\""
fi

# Storage queues connection string (map AZURE_STORAGE_CONNECTION_STRING to AZURE_STORAGE_QUEUES_CONNECTION_STRING for Azure Functions)
STORAGE_QUEUES_CONN="${AZURE_STORAGE_CONNECTION_STRING:-${AZURE_STORAGE_QUEUES_CONNECTION_STRING}}"
if [ -n "$STORAGE_QUEUES_CONN" ]; then
  SETTINGS="${SETTINGS} AZURE_STORAGE_QUEUES_CONNECTION_STRING=\"${STORAGE_QUEUES_CONN}\""
fi

# AzureWebJobsStorage (required by Azure Functions runtime)
SETTINGS="${SETTINGS} AzureWebJobsStorage=\"${STORAGE_CONN_STRING}\""

# Application Insights (support both AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING from .env.example and APPLICATIONINSIGHTS_CONNECTION_STRING)
APP_INSIGHTS_CONN="${AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING:-${APPLICATIONINSIGHTS_CONNECTION_STRING}}"
if [ -n "$APP_INSIGHTS_CONN" ]; then
  SETTINGS="${SETTINGS} APPLICATIONINSIGHTS_CONNECTION_STRING=\"${APP_INSIGHTS_CONN}\""
fi

# Execute the Azure CLI command
echo "Executing: az functionapp config appsettings set ..."
eval "az functionapp config appsettings set \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings $SETTINGS"

echo ""
echo "=========================================="
echo "Environment variables configured successfully!"
echo "=========================================="
echo ""
echo "Verifying configuration..."
az functionapp config appsettings list \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "[?contains(name, 'DATABASE') || contains(name, 'SUPABASE') || contains(name, 'AZURE_AI') || contains(name, 'AZURE_SEARCH') || contains(name, 'AZURE_DOCUMENT') || contains(name, 'AZURE_BLOB') || contains(name, 'AZURE_STORAGE_QUEUES')].{Name:name}" -o table

