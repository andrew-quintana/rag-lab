#!/bin/bash
# Start Azurite (local Azure Storage emulator)

set -e

echo "Starting Azurite..."

# Check if Azurite is installed
if ! command -v azurite &> /dev/null; then
    echo "Error: Azurite is not installed."
    echo "Please install it first:"
    echo "  npm install -g azurite"
    exit 1
fi

# Check if Azurite is already running
if lsof -Pi :10000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Azurite is already running on port 10000"
    echo "To stop it, run: ./scripts/stop_azurite.sh"
    exit 0
fi

# Create data directory if it doesn't exist
AZURITE_DATA_DIR="$HOME/.azurite"
mkdir -p "$AZURITE_DATA_DIR"

# Start Azurite in background
echo "Starting Azurite on ports 10000 (blob), 10001 (queue), 10002 (table)..."
azurite \
    --silent \
    --location "$AZURITE_DATA_DIR" \
    --debug "$AZURITE_DATA_DIR/debug.log" \
    > "$AZURITE_DATA_DIR/azurite.log" 2>&1 &

AZURITE_PID=$!
echo $AZURITE_PID > "$AZURITE_DATA_DIR/azurite.pid"

# Wait a moment for Azurite to start
sleep 2

# Check if Azurite started successfully
if ps -p $AZURITE_PID > /dev/null; then
    echo "✓ Azurite started successfully (PID: $AZURITE_PID)"
    echo "  Blob service: http://127.0.0.1:10000"
    echo "  Queue service: http://127.0.0.1:10001"
    echo "  Table service: http://127.0.0.1:10002"
    echo "  Logs: $AZURITE_DATA_DIR/azurite.log"
    echo ""
    echo "To stop Azurite, run: ./scripts/stop_azurite.sh"
else
    echo "Error: Failed to start Azurite"
    exit 1
fi

