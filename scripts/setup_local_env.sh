#!/bin/bash
# Script to setup/update .env.local with local Supabase credentials and required variables
# This script extracts local Supabase credentials from `supabase status` and prompts for Azure AI credentials

set -e

ENV_FILE=".env.local"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE_PATH="$PROJECT_ROOT/$ENV_FILE"

echo "=========================================="
echo "Setting up .env.local for local development"
echo "=========================================="
echo ""

# Check if Supabase is running
if ! command -v supabase &> /dev/null; then
    echo "Error: Supabase CLI not found. Please install it first:"
    echo "  brew install supabase/tap/supabase"
    exit 1
fi

# Check if Supabase is running locally
if ! (cd "$PROJECT_ROOT/infra/supabase" && supabase status >/dev/null 2>&1); then
    echo "Warning: Supabase is not running locally."
    echo "Please start Supabase first:"
    echo "  cd infra/supabase && supabase start"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Extract Supabase credentials from status
echo "Extracting local Supabase credentials..."
SUPABASE_STATUS=$(cd "$PROJECT_ROOT/infra/supabase" 2>/dev/null && supabase status --output json 2>/dev/null || echo "{}")

if [ "$SUPABASE_STATUS" = "{}" ]; then
    echo "Warning: Could not get Supabase status. Using defaults for local development."
    DATABASE_URL="postgresql://postgres:postgres@localhost:54322/postgres"
    SUPABASE_URL="http://localhost:54321"
    SUPABASE_KEY=""
else
    # Parse JSON output (requires jq or python)
    if command -v jq &> /dev/null; then
        DATABASE_URL=$(echo "$SUPABASE_STATUS" | jq -r '.DB_URL // "postgresql://postgres:postgres@localhost:54322/postgres"')
        SUPABASE_URL=$(echo "$SUPABASE_STATUS" | jq -r '.API_URL // "http://localhost:54321"')
        SUPABASE_KEY=$(echo "$SUPABASE_STATUS" | jq -r '.ANON_KEY // ""')
    elif command -v python3 &> /dev/null; then
        DATABASE_URL=$(echo "$SUPABASE_STATUS" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('DB_URL', 'postgresql://postgres:postgres@localhost:54322/postgres'))" 2>/dev/null || echo "postgresql://postgres:postgres@localhost:54322/postgres")
        SUPABASE_URL=$(echo "$SUPABASE_STATUS" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('API_URL', 'http://localhost:54321'))" 2>/dev/null || echo "http://localhost:54321")
        SUPABASE_KEY=$(echo "$SUPABASE_STATUS" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('ANON_KEY', ''))" 2>/dev/null || echo "")
    else
        # Fallback: try to parse from text output
        DATABASE_URL="postgresql://postgres:postgres@localhost:54322/postgres"
        SUPABASE_URL="http://localhost:54321"
        SUPABASE_KEY=""
        echo "Warning: jq or python3 not found. Using default local Supabase values."
        echo "Please manually update SUPABASE_KEY in .env.local after running 'supabase status'"
    fi
fi

# Load existing .env.local if it exists
declare -A EXISTING_VARS
if [ -f "$ENV_FILE_PATH" ]; then
    echo "Loading existing variables from $ENV_FILE..."
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        # Remove quotes from value
        value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
        EXISTING_VARS["$key"]="$value"
    done < <(grep -v '^#' "$ENV_FILE_PATH" | grep '=' || true)
fi

# Function to get or prompt for variable
get_var() {
    local var_name=$1
    local prompt_text=$2
    local default_value=$3
    local is_secret=${4:-false}
    
    # Check if variable already exists
    if [ -n "${EXISTING_VARS[$var_name]}" ]; then
        echo "${EXISTING_VARS[$var_name]}"
        return
    fi
    
    # Use default if provided
    if [ -n "$default_value" ]; then
        echo "$default_value"
        return
    fi
    
    # Prompt user
    if [ "$is_secret" = "true" ]; then
        read -sp "$prompt_text: " value
        echo ""
    else
        read -p "$prompt_text: " value
    fi
    
    echo "$value"
}

# Prompt for Azure AI credentials (preserve existing or use defaults)
echo ""
echo "=========================================="
echo "Azure AI Service Credentials"
echo "=========================================="
echo "Press Enter to keep existing values or use defaults, or enter new values."
echo ""

AZURE_AI_FOUNDRY_ENDPOINT=$(get_var "AZURE_AI_FOUNDRY_ENDPOINT" "Azure AI Foundry Endpoint" "${EXISTING_VARS[AZURE_AI_FOUNDRY_ENDPOINT]}")
AZURE_AI_FOUNDRY_API_KEY=$(get_var "AZURE_AI_FOUNDRY_API_KEY" "Azure AI Foundry API Key" "${EXISTING_VARS[AZURE_AI_FOUNDRY_API_KEY]}" "true")
AZURE_AI_FOUNDRY_EMBEDDING_MODEL=$(get_var "AZURE_AI_FOUNDRY_EMBEDDING_MODEL" "Azure AI Foundry Embedding Model" "${EXISTING_VARS[AZURE_AI_FOUNDRY_EMBEDDING_MODEL]:-text-embedding-3-small}")
AZURE_AI_FOUNDRY_GENERATION_MODEL=$(get_var "AZURE_AI_FOUNDRY_GENERATION_MODEL" "Azure AI Foundry Generation Model" "${EXISTING_VARS[AZURE_AI_FOUNDRY_GENERATION_MODEL]:-gpt-4o}")

AZURE_SEARCH_ENDPOINT=$(get_var "AZURE_SEARCH_ENDPOINT" "Azure AI Search Endpoint" "${EXISTING_VARS[AZURE_SEARCH_ENDPOINT]}")
AZURE_SEARCH_API_KEY=$(get_var "AZURE_SEARCH_API_KEY" "Azure AI Search API Key" "${EXISTING_VARS[AZURE_SEARCH_API_KEY]}" "true")
AZURE_SEARCH_INDEX_NAME=$(get_var "AZURE_SEARCH_INDEX_NAME" "Azure AI Search Index Name" "${EXISTING_VARS[AZURE_SEARCH_INDEX_NAME]}")

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=$(get_var "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT" "Azure Document Intelligence Endpoint" "${EXISTING_VARS[AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT]}")
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=$(get_var "AZURE_DOCUMENT_INTELLIGENCE_API_KEY" "Azure Document Intelligence API Key" "${EXISTING_VARS[AZURE_DOCUMENT_INTELLIGENCE_API_KEY]}" "true")

AZURE_BLOB_CONNECTION_STRING=$(get_var "AZURE_BLOB_CONNECTION_STRING" "Azure Blob Storage Connection String (optional)" "${EXISTING_VARS[AZURE_BLOB_CONNECTION_STRING]}")
AZURE_BLOB_CONTAINER_NAME=$(get_var "AZURE_BLOB_CONTAINER_NAME" "Azure Blob Container Name (optional)" "${EXISTING_VARS[AZURE_BLOB_CONTAINER_NAME]}")

# Create/update .env.local
echo ""
echo "Creating/updating $ENV_FILE..."

cat > "$ENV_FILE_PATH" << EOF
# =============================================================================
# Environment Variables for Local Development
# =============================================================================
# Generated on: $(date)
# 
# This file contains your local development environment variables.
# DO NOT commit this file to version control (it's in .gitignore)
# =============================================================================

# =============================================================================
# Database (Local Supabase Postgres)
# =============================================================================
SUPABASE_URL=${SUPABASE_URL}
SUPABASE_KEY=${SUPABASE_KEY}
DATABASE_URL=${DATABASE_URL}

# =============================================================================
# Azure AI Foundry
# =============================================================================
AZURE_AI_FOUNDRY_ENDPOINT=${AZURE_AI_FOUNDRY_ENDPOINT}
AZURE_AI_FOUNDRY_API_KEY=${AZURE_AI_FOUNDRY_API_KEY}
AZURE_AI_FOUNDRY_EMBEDDING_MODEL=${AZURE_AI_FOUNDRY_EMBEDDING_MODEL}
AZURE_AI_FOUNDRY_GENERATION_MODEL=${AZURE_AI_FOUNDRY_GENERATION_MODEL}

# =============================================================================
# Azure AI Search
# =============================================================================
AZURE_SEARCH_ENDPOINT=${AZURE_SEARCH_ENDPOINT}
AZURE_SEARCH_API_KEY=${AZURE_SEARCH_API_KEY}
AZURE_SEARCH_INDEX_NAME=${AZURE_SEARCH_INDEX_NAME}

# =============================================================================
# Azure Document Intelligence
# =============================================================================
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=${AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT}
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=${AZURE_DOCUMENT_INTELLIGENCE_API_KEY}

# =============================================================================
# Azure Blob Storage (optional)
# =============================================================================
AZURE_BLOB_CONNECTION_STRING=${AZURE_BLOB_CONNECTION_STRING}
AZURE_BLOB_CONTAINER_NAME=${AZURE_BLOB_CONTAINER_NAME}

# =============================================================================
# Azure Storage Queues (for Worker-Queue Architecture)
# =============================================================================
# For local development with Azurite, use UseDevelopmentStorage=true
# This is also set in local.settings.json for Azure Functions runtime
AZURE_STORAGE_QUEUES_CONNECTION_STRING=UseDevelopmentStorage=true
AZURE_STORAGE_CONNECTION_STRING=UseDevelopmentStorage=true

# =============================================================================
# API Configuration (with defaults)
# =============================================================================
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
EOF

echo ""
echo "=========================================="
echo "Created/updated $ENV_FILE"
echo "=========================================="
echo ""
echo "Summary of configured variables:"
echo "  ✓ SUPABASE_URL: $SUPABASE_URL"
echo "  ✓ DATABASE_URL: ${DATABASE_URL:0:50}..."
if [ -n "$SUPABASE_KEY" ]; then
    echo "  ✓ SUPABASE_KEY: ${SUPABASE_KEY:0:20}..."
else
    echo "  ⚠ SUPABASE_KEY: (empty - please set manually)"
fi
echo "  ✓ Azure AI Foundry: ${AZURE_AI_FOUNDRY_ENDPOINT:+configured}"
echo "  ✓ Azure AI Search: ${AZURE_SEARCH_ENDPOINT:+configured}"
echo "  ✓ Azure Document Intelligence: ${AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT:+configured}"
echo ""
echo "Next steps:"
echo "1. Verify all required variables are set (especially SUPABASE_KEY if empty)"
echo "2. Start Azurite: ./scripts/start_azurite.sh"
echo "3. Start local functions: ./scripts/dev_functions_local.sh"
echo ""

