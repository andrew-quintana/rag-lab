#!/bin/bash
# Script to set up ngrok TCP tunnel for local Supabase PostgreSQL
# This creates a TCP tunnel that Azure Functions can use to connect to local Supabase

set -e

RESOURCE_GROUP="rag-lab"
FUNCTION_APP_NAME="func-raglab-uploadworkers"
NGROK_DOMAIN="db.raglab.ngrok.io"
LOCAL_POSTGRES_PORT="54322"

echo "=========================================="
echo "Setting up ngrok TCP tunnel for PostgreSQL"
echo "=========================================="
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "Error: ngrok not found. Please install it:"
    echo "  brew install ngrok/ngrok/ngrok"
    exit 1
fi

# Check if Supabase is running
if ! nc -z localhost $LOCAL_POSTGRES_PORT 2>/dev/null; then
    echo "Warning: PostgreSQL not running on port $LOCAL_POSTGRES_PORT"
    echo "Please start Supabase first:"
    echo "  cd infra/supabase && supabase start"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPO =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if ngrok app key is set in Azure Function App
echo "Checking for ngrok app key in Azure Function App..."
NGROK_APP_KEY=$(az functionapp config appsettings list \
    --name $FUNCTION_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "[?name=='NGROK_APP_KEY' || name=='NGROK_API_KEY'].value" -o tsv 2>/dev/null | head -1)

if [ -z "$NGROK_APP_KEY" ]; then
    echo "⚠️  Warning: NGROK_APP_KEY not found in Azure Function App settings"
    echo "   You may need to add it manually in Azure Portal"
    echo ""
else
    echo "✓ Found ngrok app key in Azure Function App"
fi

echo ""
echo "=========================================="
echo "Starting ngrok TCP tunnel"
echo "=========================================="
echo ""
echo "For PostgreSQL, you need a TCP tunnel (not HTTP)."
echo "If you have a reserved TCP address, use:"
echo "  ngrok tcp $LOCAL_POSTGRES_PORT --remote-addr=YOUR_RESERVED_ADDRESS"
echo ""
echo "Or use ngrok edge with your custom domain:"
echo "  ngrok tcp $LOCAL_POSTGRES_PORT --domain=$NGROK_DOMAIN"
echo ""
echo "Note: Custom domains for TCP require a reserved TCP address."
echo "      Reserve one at: https://dashboard.ngrok.com/tcp-addresses"
echo ""

# Try to get the ngrok API to check if tunnel is running
NGROK_API="http://localhost:4040"
if curl -s "$NGROK_API/api/tunnels" > /dev/null 2>&1; then
    echo "✓ ngrok is already running"
    echo ""
    echo "Current tunnels:"
    curl -s "$NGROK_API/api/tunnels" | python3 -m json.tool 2>/dev/null | grep -A 5 '"public_url"' || echo "  (No active tunnels)"
    echo ""
    echo "If you need to restart with TCP tunnel, stop ngrok first and run:"
    echo "  ngrok tcp $LOCAL_POSTGRES_PORT --remote-addr=YOUR_RESERVED_TCP_ADDRESS"
else
    echo "ngrok is not running. To start a TCP tunnel:"
    echo ""
    echo "1. If you have a reserved TCP address:"
    echo "   ngrok tcp $LOCAL_POSTGRES_PORT --remote-addr=YOUR_RESERVED_ADDRESS"
    echo ""
    echo "2. Or use a regular TCP tunnel (URL will change on restart):"
    echo "   ngrok tcp $LOCAL_POSTGRES_PORT"
    echo ""
    echo "After starting, note the public URL (e.g., tcp://0.tcp.ngrok.io:12345)"
    echo "Then update Azure Function App DATABASE_URL with:"
    echo "   postgresql://postgres:postgres@[ngrok-host]:[ngrok-port]/postgres"
fi

echo ""
echo "=========================================="
echo "To update Azure Function App DATABASE_URL:"
echo "=========================================="
echo ""
echo "Once you have the ngrok TCP URL, run:"
echo ""
echo "az functionapp config appsettings set \\"
echo "  --name $FUNCTION_APP_NAME \\"
echo "  --resource-group $RESOURCE_GROUP \\"
echo "  --settings DATABASE_URL=\"postgresql://postgres:postgres@[ngrok-host]:[ngrok-port]/postgres\""
echo ""

