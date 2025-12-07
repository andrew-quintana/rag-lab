#!/bin/bash
# Stop Azurite (local Azure Storage emulator)

set -e

AZURITE_DATA_DIR="$HOME/.azurite"
PID_FILE="$AZURITE_DATA_DIR/azurite.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Azurite PID file not found. Checking for running processes..."
    # Try to find and kill Azurite processes
    PIDS=$(pgrep -f "azurite" || true)
    if [ -z "$PIDS" ]; then
        echo "Azurite is not running"
        exit 0
    else
        echo "Found Azurite processes: $PIDS"
        kill $PIDS
        echo "✓ Stopped Azurite processes"
        exit 0
    fi
fi

AZURITE_PID=$(cat "$PID_FILE")

if ! ps -p $AZURITE_PID > /dev/null 2>&1; then
    echo "Azurite process (PID: $AZURITE_PID) is not running"
    rm -f "$PID_FILE"
    exit 0
fi

echo "Stopping Azurite (PID: $AZURITE_PID)..."
kill $AZURITE_PID

# Wait for process to stop
for i in {1..10}; do
    if ! ps -p $AZURITE_PID > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Force kill if still running
if ps -p $AZURITE_PID > /dev/null 2>&1; then
    echo "Force killing Azurite..."
    kill -9 $AZURITE_PID
fi

rm -f "$PID_FILE"
echo "✓ Azurite stopped"

