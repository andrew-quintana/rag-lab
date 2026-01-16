#!/bin/bash
# Check Azure Function Status
# Quick helper to check if functions are healthy

set -e

FUNCTION_APP="func-raglab-uploadworkers"
RESOURCE_GROUP="rag-lab"

echo "========================================"
echo "Azure Function Status Check"
echo "========================================"
echo ""

# Check if Azure CLI is available
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI not found. Please install it first."
    exit 1
fi

# Check if logged in
if ! az account show &> /dev/null; then
    echo "❌ Not logged into Azure CLI. Run: az login"
    exit 1
fi

echo "📊 Function App: $FUNCTION_APP"
echo "📊 Resource Group: $RESOURCE_GROUP"
echo ""

# Check function app status
echo "🔍 Checking Function App status..."
STATUS=$(az functionapp show \
    --name "$FUNCTION_APP" \
    --resource-group "$RESOURCE_GROUP" \
    --query "state" -o tsv 2>/dev/null || echo "ERROR")

if [ "$STATUS" = "ERROR" ]; then
    echo "❌ Failed to get function app status"
    echo "   Check that the function app exists and you have access"
    exit 1
fi

echo "   Status: $STATUS"
echo ""

# List functions
echo "📋 Functions:"
az functionapp function list \
    --name "$FUNCTION_APP" \
    --resource-group "$RESOURCE_GROUP" \
    --query "[].{Name:name}" -o table

echo ""

# Check recent invocations (via app insights if available)
echo "💡 To view detailed logs, use one of:"
echo "   1. Azure Portal: https://portal.azure.com → $FUNCTION_APP → Monitor"
echo "   2. Stream logs: az webapp log tail --name $FUNCTION_APP --resource-group $RESOURCE_GROUP"
echo "   3. Check queue health: python scripts/monitor_queue_health.py"
echo ""

echo "✅ Status check complete"

