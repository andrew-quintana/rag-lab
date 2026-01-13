#!/bin/bash
# Script to stop ngrok tunnel

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NGROK_PID_FILE="$PROJECT_ROOT/.ngrok.pid"

echo "Stopping ngrok tunnel..."

if [ -f "$NGROK_PID_FILE" ]; then
    NGROK_PID=$(cat "$NGROK_PID_FILE")
    if ps -p "$NGROK_PID" > /dev/null 2>&1; then
        kill "$NGROK_PID" 2>/dev/null || true
        echo "✓ Stopped ngrok tunnel (PID: $NGROK_PID)"
    else
        echo "ngrok tunnel was not running"
    fi
    rm -f "$NGROK_PID_FILE"
else
    # Try to find and kill any ngrok processes
    if pgrep -f "ngrok" > /dev/null; then
        pkill -f "ngrok" || true
        echo "✓ Stopped ngrok processes"
    else
        echo "No ngrok processes found"
    fi
fi




