#!/bin/bash
# Script to start ngrok TCP tunnel for local Supabase PostgreSQL
# This exposes local PostgreSQL (port 54322) to Azure Functions via ngrok

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NGROK_PID_FILE="$PROJECT_ROOT/.ngrok.pid"
NGROK_LOG_FILE="$PROJECT_ROOT/.ngrok.log"

echo "=========================================="
echo "Starting ngrok TCP Tunnel for PostgreSQL"
echo "=========================================="
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "Error: ngrok is not installed."
    echo "Please install it first:"
    echo "  brew install ngrok/ngrok/ngrok"
    exit 1
fi

# Check if ngrok is already running
if [ -f "$NGROK_PID_FILE" ]; then
    OLD_PID=$(cat "$NGROK_PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠ ngrok is already running (PID: $OLD_PID)"
        echo "Stopping existing tunnel..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$NGROK_PID_FILE"
fi

# Check if there's an HTTP tunnel running (wrong type)
if pgrep -f "ngrok http" > /dev/null; then
    echo "⚠ Found HTTP tunnel running. Stopping it (PostgreSQL needs TCP)..."
    pkill -f "ngrok http" || true
    sleep 2
fi

# Check if Supabase is running
if ! nc -z 127.0.0.1 54322 2>/dev/null; then
    echo "⚠ Warning: PostgreSQL is not running on port 54322"
    echo "Please start Supabase first:"
    echo "  cd infra/supabase && supabase start"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Try to use custom domain if available
# First check if they have a reserved TCP address
echo "Starting ngrok TCP tunnel..."
echo ""

# Check for custom domain in .env.local
if [ -f "$PROJECT_ROOT/.env.local" ]; then
    NGROK_URL_LINE=$(grep "^NGROK_URL=" "$PROJECT_ROOT/.env.local" 2>/dev/null || echo "")
    if [ -n "$NGROK_URL_LINE" ]; then
        NGROK_URL_VALUE=$(echo "$NGROK_URL_LINE" | cut -d'=' -f2- | tr -d ' ' | tr -d '"' | tr -d "'")
        NGROK_DOMAIN=$(echo "$NGROK_URL_VALUE" | sed 's|^https\?://||' | sed 's|/.*$||')
        if [ -n "$NGROK_DOMAIN" ] && [ "$NGROK_DOMAIN" != "$NGROK_URL_VALUE" ]; then
            echo "Found NGROK_URL in .env.local: $NGROK_DOMAIN"
        else
            NGROK_DOMAIN=""
        fi
    fi
fi

# Try TCP tunnel with custom domain (requires reserved TCP address)
if [ -n "$NGROK_DOMAIN" ]; then
    echo "Attempting to use custom domain: $NGROK_DOMAIN"
    echo "Note: This requires a reserved TCP address in ngrok dashboard"
    echo ""
    
    # Start ngrok TCP tunnel with custom domain
    # Note: This will fail if TCP address is not reserved
    ngrok tcp 54322 --remote-addr="$NGROK_DOMAIN:0" > "$NGROK_LOG_FILE" 2>&1 &
    NGROK_PID=$!
    
    sleep 3
    
    # Check if it started successfully
    if ps -p "$NGROK_PID" > /dev/null; then
        echo "$NGROK_PID" > "$NGROK_PID_FILE"
        echo "✓ ngrok TCP tunnel started (PID: $NGROK_PID)"
        echo "  Custom domain: $NGROK_DOMAIN"
    else
        echo "⚠ Failed to start with custom domain (TCP address may not be reserved)"
        echo "Falling back to regular TCP tunnel..."
        # Fall through to regular tunnel
        NGROK_PID=""
    fi
fi

# If custom domain didn't work, use regular TCP tunnel
if [ -z "$NGROK_PID" ] || [ ! -f "$NGROK_PID_FILE" ]; then
    echo "Starting regular ngrok TCP tunnel..."
    ngrok tcp 54322 > "$NGROK_LOG_FILE" 2>&1 &
    NGROK_PID=$!
    sleep 3
    
    if ps -p "$NGROK_PID" > /dev/null; then
        echo "$NGROK_PID" > "$NGROK_PID_FILE"
        echo "✓ ngrok TCP tunnel started (PID: $NGROK_PID)"
    else
        echo "Error: Failed to start ngrok tunnel"
        cat "$NGROK_LOG_FILE" 2>/dev/null || true
        exit 1
    fi
fi

# Wait a moment for ngrok to initialize
sleep 2

# Get the tunnel URL from ngrok API
echo ""
echo "Fetching tunnel information..."
TUNNEL_INFO=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null || echo "")

if [ -n "$TUNNEL_INFO" ]; then
    # Extract TCP URL (format: tcp://0.tcp.ngrok.io:12345)
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
        # Extract host and port
        # Format: tcp://0.tcp.ngrok.io:12345
        HOST_PORT=$(echo "$TCP_URL" | sed 's|tcp://||')
        HOST=$(echo "$HOST_PORT" | cut -d':' -f1)
        PORT=$(echo "$HOST_PORT" | cut -d':' -f2)
        
        echo "✓ Tunnel URL: $TCP_URL"
        echo ""
        echo "=========================================="
        echo "Tunnel Information"
        echo "=========================================="
        echo "Public URL: $TCP_URL"
        echo "Host: $HOST"
        echo "Port: $PORT"
        echo ""
        echo "DATABASE_URL for Azure Function App:"
        echo "postgresql://postgres:postgres@$HOST:$PORT/postgres"
        echo ""
        echo "To update Azure Function App, run:"
        echo "  az functionapp config appsettings set \\"
        echo "    --name func-raglab-uploadworkers \\"
        echo "    --resource-group rag-lab \\"
        echo "    --settings DATABASE_URL=\"postgresql://postgres:postgres@$HOST:$PORT/postgres\""
        echo ""
        echo "Or use the update script:"
        echo "  ./scripts/update_azure_db_url.sh"
        echo ""
        echo "Tunnel is running in background (PID: $NGROK_PID)"
        echo "Logs: $NGROK_LOG_FILE"
        echo "To stop: ./scripts/stop_ngrok_tunnel.sh"
    else
        echo "⚠ Could not extract tunnel URL from ngrok API"
        echo "Check ngrok dashboard: http://localhost:4040"
    fi
else
    echo "⚠ Could not connect to ngrok API"
    echo "Tunnel may still be starting. Check: http://localhost:4040"
fi

echo ""
echo "=========================================="
echo "ngrok tunnel started!"
echo "=========================================="




