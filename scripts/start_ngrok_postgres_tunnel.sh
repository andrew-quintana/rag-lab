#!/bin/bash
# Script to start ngrok TCP tunnel for PostgreSQL and update Azure Function App DATABASE_URL
# This script starts the tunnel, extracts the connection info, and updates Azure Function App settings

set -e

RESOURCE_GROUP="rag-lab"
FUNCTION_APP_NAME="func-raglab-uploadworkers"
LOCAL_POSTGRES_PORT="54322"

echo "=========================================="
echo "Starting ngrok TCP Tunnel for PostgreSQL"
echo "=========================================="
echo ""

# Check if ngrok is already running
if pgrep -f "ngrok.*tcp.*54322" > /dev/null; then
    echo "⚠️  ngrok TCP tunnel already running"
    echo "Getting tunnel info..."
    
    # Wait a moment for ngrok API to be ready
    sleep 2
    
    # Get tunnel info from ngrok API
    TUNNEL_INFO=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null || echo "")
    
    if [ -n "$TUNNEL_INFO" ]; then
        # Extract TCP tunnel URL
        TCP_URL=$(echo "$TUNNEL_INFO" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for tunnel in data.get('tunnels', []):
        if tunnel.get('proto') == 'tcp':
            print(tunnel.get('public_url', ''))
            break
except:
    pass
" 2>/dev/null || echo "")
        
        if [ -n "$TCP_URL" ]; then
            echo "✓ Found TCP tunnel: $TCP_URL"
            echo ""
            echo "Updating Azure Function App DATABASE_URL..."
            
            # Extract host and port from ngrok URL (format: tcp://0.tcp.ngrok.io:12345)
            HOST_PORT=$(echo "$TCP_URL" | sed 's|tcp://||')
            
            # Update Azure Function App
            az functionapp config appsettings set \
                --name "$FUNCTION_APP_NAME" \
                --resource-group "$RESOURCE_GROUP" \
                --settings "DATABASE_URL=postgresql://postgres:postgres@${HOST_PORT}/postgres" \
                --output none
            
            echo "✓ Updated DATABASE_URL to: postgresql://postgres:postgres@${HOST_PORT}/postgres"
            echo ""
            echo "=========================================="
            echo "✓ Tunnel is running and Azure Function App is configured"
            echo "=========================================="
            echo ""
            echo "Keep this ngrok process running while testing."
            echo "To stop: Press Ctrl+C or kill the ngrok process"
            exit 0
        fi
    fi
    
    echo "⚠️  Could not get TCP tunnel info. Please check ngrok is running correctly."
    exit 1
fi

# Check if HTTP tunnel is running (wrong type)
if pgrep -f "ngrok.*http.*54322" > /dev/null; then
    echo "⚠️  WARNING: HTTP tunnel detected, but PostgreSQL needs TCP tunnel"
    echo "Please stop the HTTP tunnel and run this script again"
    echo "Or run: ngrok tcp 54322"
    exit 1
fi

echo "Starting ngrok TCP tunnel..."
echo "This will run in the background and update Azure Function App automatically"
echo ""

# Start ngrok TCP tunnel in background
ngrok tcp "$LOCAL_POSTGRES_PORT" > /tmp/ngrok_tcp.log 2>&1 &
NGROK_PID=$!

echo "Waiting for ngrok to start..."
sleep 5

# Check if ngrok started successfully
if ! kill -0 $NGROK_PID 2>/dev/null; then
    echo "❌ Failed to start ngrok. Check /tmp/ngrok_tcp.log for errors"
    exit 1
fi

# Wait a bit more for API to be ready
sleep 3

# Get tunnel info
TUNNEL_INFO=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null || echo "")

if [ -z "$TUNNEL_INFO" ]; then
    echo "❌ Could not connect to ngrok API. Check if ngrok started correctly."
    echo "Logs: /tmp/ngrok_tcp.log"
    kill $NGROK_PID 2>/dev/null || true
    exit 1
fi

# Extract TCP tunnel URL
TCP_URL=$(echo "$TUNNEL_INFO" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for tunnel in data.get('tunnels', []):
        if tunnel.get('proto') == 'tcp':
            print(tunnel.get('public_url', ''))
            break
except Exception as e:
    sys.stderr.write(f'Error: {e}\n')
    sys.exit(1)
" 2>/dev/null)

if [ -z "$TCP_URL" ]; then
    echo "❌ Could not find TCP tunnel in ngrok response"
    echo "Response: $TUNNEL_INFO"
    kill $NGROK_PID 2>/dev/null || true
    exit 1
fi

# Extract host and port
HOST_PORT=$(echo "$TCP_URL" | sed 's|tcp://||')

echo "✓ TCP tunnel started: $TCP_URL"
echo ""

# Update Azure Function App
echo "Updating Azure Function App DATABASE_URL..."
az functionapp config appsettings set \
    --name "$FUNCTION_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --settings "DATABASE_URL=postgresql://postgres:postgres@${HOST_PORT}/postgres" \
    --output none

echo "✓ Updated DATABASE_URL to: postgresql://postgres:postgres@${HOST_PORT}/postgres"
echo ""

# Save PID for later cleanup
echo $NGROK_PID > /tmp/ngrok_postgres_tunnel.pid

echo "=========================================="
echo "✓ Tunnel is running (PID: $NGROK_PID)"
echo "✓ Azure Function App DATABASE_URL updated"
echo "=========================================="
echo ""
echo "Tunnel URL: $TCP_URL"
echo "DATABASE_URL: postgresql://postgres:postgres@${HOST_PORT}/postgres"
echo ""
echo "To stop the tunnel:"
echo "  kill \$(cat /tmp/ngrok_postgres_tunnel.pid)"
echo "  or: pkill -f 'ngrok.*tcp.*54322'"
echo ""
echo "Keep this tunnel running while testing cloud functions!"

