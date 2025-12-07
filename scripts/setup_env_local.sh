#!/bin/bash
# Script to help set up .env.local file with Azure resource values
# This script generates a template with Azure values pre-filled where possible

set -e

RESOURCE_GROUP="rag-lab"
FUNCTION_APP_NAME="func-raglab-uploadworkers"
STORAGE_ACCOUNT_NAME="raglabqueues"
ENV_FILE=".env.local"
ENV_EXAMPLE=".env.example"

echo "=========================================="
echo "Setting up .env.local file"
echo "=========================================="
echo ""

# Check if .env.local already exists
if [ -f "$ENV_FILE" ]; then
  echo "Warning: $ENV_FILE already exists."
  read -p "Do you want to overwrite it? (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Exiting."
    exit 1
  fi
  echo "Backing up existing $ENV_FILE to ${ENV_FILE}.backup"
  cp "$ENV_FILE" "${ENV_FILE}.backup"
fi

# Get Storage Account connection string
echo "Retrieving Azure Storage connection string..."
STORAGE_CONN_STRING=$(az storage account show-connection-string \
  --name $STORAGE_ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv 2>/dev/null || echo "")

# Get Application Insights connection string
echo "Retrieving Application Insights connection string..."
APP_INSIGHTS_CONN_STRING=$(az functionapp config appsettings list \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "[?name=='APPLICATIONINSIGHTS_CONNECTION_STRING'].value" -o tsv 2>/dev/null || echo "")

# Create .env.local file
echo "Creating $ENV_FILE..."
cat > "$ENV_FILE" << EOF
# =============================================================================
# Environment Variables for RAG Evaluator
# =============================================================================
# Generated on: $(date)
# 
# This file contains your actual environment variable values.
# DO NOT commit this file to version control (it's in .gitignore)
# =============================================================================

# =============================================================================
# Database (Supabase Postgres)
# =============================================================================
# TODO: Fill in your Supabase credentials
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres

# =============================================================================
# Azure AI Foundry
# =============================================================================
# TODO: Fill in your Azure AI Foundry credentials
AZURE_AI_FOUNDRY_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_AI_FOUNDRY_API_KEY=your-ai-foundry-api-key
AZURE_AI_FOUNDRY_EMBEDDING_MODEL=text-embedding-ada-002
AZURE_AI_FOUNDRY_GENERATION_MODEL=gpt-4o
AZURE_AI_FOUNDRY_REASONING_MODEL=gpt-4o

# =============================================================================
# Azure AI Search
# =============================================================================
# TODO: Fill in your Azure AI Search credentials
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your-search-api-key
AZURE_SEARCH_INDEX_NAME=your-index-name

# =============================================================================
# Azure Document Intelligence
# =============================================================================
# TODO: Fill in your Azure Document Intelligence credentials
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=your-document-intelligence-api-key

# =============================================================================
# Azure Blob Storage
# =============================================================================
# TODO: Fill in if using different storage account for blobs
AZURE_BLOB_CONNECTION_STRING=
AZURE_BLOB_CONTAINER_NAME=your-container-name

# =============================================================================
# Azure Storage Queues (for Worker-Queue Architecture)
# =============================================================================
# Auto-filled from Azure Storage Account: $STORAGE_ACCOUNT_NAME
# Note: This maps to AZURE_STORAGE_QUEUES_CONNECTION_STRING in Azure Functions
EOF

if [ -n "$STORAGE_CONN_STRING" ]; then
  echo "AZURE_STORAGE_CONNECTION_STRING=\"$STORAGE_CONN_STRING\"" >> "$ENV_FILE"
else
  echo "AZURE_STORAGE_CONNECTION_STRING=" >> "$ENV_FILE"
  echo "# Warning: Could not retrieve storage connection string" >> "$ENV_FILE"
fi

cat >> "$ENV_FILE" << EOF

# =============================================================================
# Azure Application Insights
# =============================================================================
# Auto-filled from Function App: $FUNCTION_APP_NAME
EOF

if [ -n "$APP_INSIGHTS_CONN_STRING" ]; then
  echo "AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING=\"$APP_INSIGHTS_CONN_STRING\"" >> "$ENV_FILE"
else
  echo "AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING=" >> "$ENV_FILE"
  echo "# Warning: Could not retrieve Application Insights connection string" >> "$ENV_FILE"
fi

cat >> "$ENV_FILE" << EOF

# =============================================================================
# API Configuration (with defaults)
# =============================================================================
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
EOF

echo ""
echo "=========================================="
echo "Created $ENV_FILE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit $ENV_FILE and fill in the TODO items:"
echo "   - Supabase credentials (SUPABASE_URL, SUPABASE_KEY, DATABASE_URL)"
echo "   - Azure AI Foundry credentials (AZURE_AI_FOUNDRY_ENDPOINT, AZURE_AI_FOUNDRY_API_KEY)"
echo "   - Azure AI Search credentials (AZURE_SEARCH_*, AZURE_SEARCH_INDEX_NAME)"
echo "   - Azure Document Intelligence credentials (AZURE_DOCUMENT_INTELLIGENCE_*)"
echo "   - Azure Blob Storage (if different from queue storage)"
echo ""
echo "2. After filling in all values, configure the Function App:"
echo "   ./scripts/configure_function_app_env.sh"
echo ""
echo "3. Verify the configuration:"
echo "   az functionapp config appsettings list --name $FUNCTION_APP_NAME --resource-group $RESOURCE_GROUP"
echo ""

