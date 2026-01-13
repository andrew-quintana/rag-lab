#!/bin/bash
# Script to update Azure Function App DATABASE_URL with ngrok tunnel URL

set -e

FUNCTION_APP_NAME="func-raglab-uploadworkers"
RESOURCE_GROUP="rag-lab"

echo "=========================================="
echo "Updating Azure Function App DATABASE_URL"
echo "=========================================="
echo ""

# Get tunnel URL from ngrok API
TUNNEL_INFO=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null || echo "")

if [ -z "$TUNNEL_INFO" ]; then
    echo "Error: Could not connect to ngrok API"
    echo "Make sure ngrok tunnel is running: ./scripts/start_ngrok_tunnel.sh"
    exit 1
fi

# Extract TCP URL
TCP_URL=$(echo "$TUNNEL_INFO" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for tunnel in data.get('tunnels', []):
        if tunnel.get('proto') == 'tcp':
            print(tunnel.get('public_url', ''))
            break
except Exception as e:
    sys.exit(1)
" 2>/dev/null)

if [ -z "$TCP_URL" ]; then
    echo "Error: No TCP tunnel found"
    echo "Make sure ngrok TCP tunnel is running (not HTTP)"
    exit 1
fi

# Extract host and port
HOST_PORT=$(echo "$TCP_URL" | sed 's|tcp://||')
HOST=$(echo "$HOST_PORT" | cut -d':' -f1)
PORT=$(echo "$HOST_PORT" | cut -d':' -f2)

DATABASE_URL="postgresql://postgres:postgres@$HOST:$PORT/postgres"

echo "Tunnel URL: $TCP_URL"
echo "DATABASE_URL: $DATABASE_URL"
echo ""
echo "Updating Azure Function App..."
echo ""

az functionapp config appsettings set \
  --name "$FUNCTION_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings DATABASE_URL="$DATABASE_URL" \
  --output table

echo ""
echo "=========================================="
echo "✓ DATABASE_URL updated successfully!"
echo "=========================================="
echo ""
echo "Verifying update..."
az functionapp config appsettings list \
  --name "$FUNCTION_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "[?name=='DATABASE_URL'].{Name:name, Value:value}" -o table




