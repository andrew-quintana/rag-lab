#!/bin/bash
# Script to set environment variables for Azure Function App
# Usage: ./set_function_app_env_vars.sh [path-to-.env.local]

set -e

RESOURCE_GROUP="rag-lab"
FUNCTION_APP_NAME="func-raglab-uploadworkers"

# Load environment variables from .env.local if provided
if [ -n "$1" ] && [ -f "$1" ]; then
    echo "Loading environment variables from $1"
    export $(grep -v '^#' "$1" | grep -v '^$' | xargs)
elif [ -f ".env.local" ]; then
    echo "Loading environment variables from .env.local"
    export $(grep -v '^#' .env.local | grep -v '^$' | xargs)
else
    echo "Warning: .env.local not found. Please provide path to .env.local file or create one."
    echo "Usage: $0 [path-to-.env.local]"
    echo ""
    echo "You can also set variables manually by exporting them before running this script."
    exit 1
fi

# Get storage connection string
AZURE_STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
  --name raglabqueues \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv)

# Determine storage connection string to use
STORAGE_CONN_STRING="${AZURE_STORAGE_QUEUES_CONNECTION_STRING:-${AZURE_BLOB_CONNECTION_STRING:-$AZURE_STORAGE_CONNECTION_STRING}}"

echo "Setting environment variables for Function App: $FUNCTION_APP_NAME"
echo "Resource Group: $RESOURCE_GROUP"
echo ""

# Set all required environment variables
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
    AZURE_BLOB_CONNECTION_STRING="${AZURE_BLOB_CONNECTION_STRING:-$STORAGE_CONN_STRING}" \
    AZURE_BLOB_CONTAINER_NAME="${AZURE_BLOB_CONTAINER_NAME}" \
    AZURE_STORAGE_QUEUES_CONNECTION_STRING="${AZURE_STORAGE_QUEUES_CONNECTION_STRING:-$STORAGE_CONN_STRING}" \
    AzureWebJobsStorage="$STORAGE_CONN_STRING" \
  --output table

echo ""
echo "Environment variables configured successfully!"
echo ""
echo "To verify, run:"
echo "  az functionapp config appsettings list --name $FUNCTION_APP_NAME --resource-group $RESOURCE_GROUP"

