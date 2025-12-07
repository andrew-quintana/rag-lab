#!/bin/bash
# Script to get Azure resource values for .env.local configuration
# This script queries Azure resources and outputs the values needed for .env.local

set -e

RESOURCE_GROUP="rag-lab"
FUNCTION_APP_NAME="func-raglab-uploadworkers"
STORAGE_ACCOUNT_NAME="raglabqueues"

echo "=========================================="
echo "Azure Environment Variables for .env.local"
echo "=========================================="
echo ""
echo "# Copy the values below to your .env.local file"
echo "# Generated on: $(date)"
echo ""

# Get Storage Account connection string
echo "# ============================================================================="
echo "# Azure Storage Queues"
echo "# ============================================================================="
STORAGE_CONN_STRING=$(az storage account show-connection-string \
  --name $STORAGE_ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv 2>/dev/null || echo "ERROR: Could not get storage connection string")

if [[ $STORAGE_CONN_STRING == ERROR* ]]; then
  echo "# $STORAGE_CONN_STRING"
  echo "AZURE_STORAGE_CONNECTION_STRING="
else
  echo "AZURE_STORAGE_CONNECTION_STRING=\"$STORAGE_CONN_STRING\""
fi

echo ""

# Get Application Insights connection string
echo "# ============================================================================="
echo "# Azure Application Insights"
echo "# ============================================================================="
APP_INSIGHTS_NAME=$(az functionapp show \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "siteConfig.appSettings[?name=='APPLICATIONINSIGHTS_CONNECTION_STRING'].value" -o tsv 2>/dev/null | head -1)

if [ -z "$APP_INSIGHTS_NAME" ]; then
  # Try to get from Application Insights resource directly
  APP_INSIGHTS_RESOURCE=$(az monitor app-insights component list \
    --resource-group $RESOURCE_GROUP \
    --query "[0].name" -o tsv 2>/dev/null || echo "")
  
  if [ -n "$APP_INSIGHTS_RESOURCE" ]; then
    APP_INSIGHTS_CONN_STRING=$(az monitor app-insights component show \
      --app $APP_INSIGHTS_RESOURCE \
      --resource-group $RESOURCE_GROUP \
      --query "connectionString" -o tsv 2>/dev/null || echo "")
    
    if [ -n "$APP_INSIGHTS_CONN_STRING" ]; then
      echo "AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING=\"$APP_INSIGHTS_CONN_STRING\""
    else
      echo "# Could not retrieve Application Insights connection string"
      echo "AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING="
    fi
  else
    echo "# Application Insights not found"
    echo "AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING="
  fi
else
  echo "AZURE_APPLICATIONINSIGHTS_CONNECTION_STRING=\"$APP_INSIGHTS_NAME\""
fi

echo ""

# Get current Function App settings
echo "# ============================================================================="
echo "# Current Function App Environment Variables"
echo "# ============================================================================="
echo "# The following variables are currently set in the Function App:"
az functionapp config appsettings list \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "[].{Name:name}" -o table 2>/dev/null || echo "# Could not retrieve Function App settings"

echo ""
echo "# ============================================================================="
echo "# Required Variables (fill these in manually)"
echo "# ============================================================================="
echo "# These variables need to be set manually or retrieved from your existing configuration:"
echo ""
echo "# Database (Supabase Postgres)"
echo "SUPABASE_URL="
echo "SUPABASE_KEY="
echo "DATABASE_URL="
echo ""
echo "# Azure AI Foundry"
echo "AZURE_AI_FOUNDRY_ENDPOINT="
echo "AZURE_AI_FOUNDRY_API_KEY="
echo "AZURE_AI_FOUNDRY_EMBEDDING_MODEL=text-embedding-ada-002"
echo "AZURE_AI_FOUNDRY_GENERATION_MODEL=gpt-4o"
echo "AZURE_AI_FOUNDRY_REASONING_MODEL=gpt-4o"
echo ""
echo "# Azure AI Search"
echo "AZURE_SEARCH_ENDPOINT="
echo "AZURE_SEARCH_API_KEY="
echo "AZURE_SEARCH_INDEX_NAME="
echo ""
echo "# Azure Document Intelligence"
echo "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="
echo "AZURE_DOCUMENT_INTELLIGENCE_API_KEY="
echo ""
echo "# Azure Blob Storage (if different from queue storage)"
echo "AZURE_BLOB_CONNECTION_STRING="
echo "AZURE_BLOB_CONTAINER_NAME="
echo ""
echo "=========================================="
echo "Done! Copy the values above to your .env.local file"
echo "=========================================="

