#!/bin/bash
# Script to extract ngrok TCP tunnel URL and update Azure Function App DATABASE_URL
# This reads the active ngrok tunnel and updates the Azure Function App setting

set -e

RESOURCE_GROUP="rag-lab"
FUNCTION_APP_NAME="func-raglab-uploadworkers"
NGROK_API="http://localhost:4040"

echo "=========================================="
echo "Updating Azure Function App DATABASE_URL from ngrok"
echo "=========================================="
echo ""

# Check if ngrok is running
if ! curl -s "$NGROK_API/api/tunnels" > /dev/null 2>&1; then
    echo "Error: ngrok is not running or API is not accessible"
    echo "Please start ngrok TCP tunnel first:"
    echo "  ngrok tcp 54322 --remote-addr=YOUR_RESERVED_ADDRESS"
    exit 1
fi

# Get active TCP tunnels
echo "Fetching active ngrok tunnels..."
TUNNELS_JSON=$(curl -s "$NGROK_API/api/tunnels")

# Extract TCP tunnel URL
TCP_TUNNEL=$(echo "$TUNNELS_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for tunnel in data.get('tunnels', []):
        if tunnel.get('proto') == 'tcp':
            public_url = tunnel.get('public_url', '')
            if public_url.startswith('tcp://'):
                # Extract host:port from tcp://host:port
                host_port = public_url.replace('tcp://', '')
                print(host_port)
                sys.exit(0)
    print('', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

if [ -z "$TCP_TUNNEL" ]; then
    echo "Error: No active TCP tunnel found"
    echo ""
    echo "Active tunnels:"
    echo "$TUNNELS_JSON" | python3 -m json.tool 2>/dev/null | grep -A 3 '"public_url"' || echo "  (No tunnels)"
    echo ""
    echo "Please start a TCP tunnel:"
    echo "  ngrok tcp 54322 --remote-addr=YOUR_RESERVED_ADDRESS"
    exit 1
fi

echo "✓ Found TCP tunnel: tcp://$TCP_TUNNEL"
echo ""

# Construct DATABASE_URL
DATABASE_URL="postgresql://postgres:postgres@$TCP_TUNNEL/postgres"

echo "Updating Azure Function App DATABASE_URL..."
echo "New DATABASE_URL: $DATABASE_URL"
echo ""

az functionapp config appsettings set \
    --name $FUNCTION_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings DATABASE_URL="$DATABASE_URL"

echo ""
echo "=========================================="
echo "✓ DATABASE_URL updated successfully!"
echo "=========================================="
echo ""
echo "Verifying..."
az functionapp config appsettings list \
    --name $FUNCTION_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "[?name=='DATABASE_URL'].{Name:name, Value:value}" -o table

echo ""
echo "You can now run cloud tests:"
echo "  ./scripts/test_functions_cloud.sh"

