#!/bin/bash
# Start API server with cloud environment variables

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Starting API Server (Cloud Configuration)"
echo "=========================================="
echo ""

# Check if .env.prod exists
if [ ! -f ".env.prod" ]; then
    echo "❌ Error: .env.prod not found"
    exit 1
fi

# Load cloud environment variables
echo "Loading cloud environment variables from .env.prod..."
export $(grep -v '^#' .env.prod | grep -v '^$' | xargs)

# Verify DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ Error: DATABASE_URL not set in .env.prod"
    exit 1
fi

echo "✅ Environment variables loaded"
echo "   DATABASE_URL: $(echo $DATABASE_URL | sed 's/:[^:]*@/:***@/')"
echo ""

# Activate virtual environment
if [ ! -d "backend/venv" ]; then
    echo "❌ Error: Virtual environment not found at backend/venv"
    exit 1
fi

source backend/venv/bin/activate

# Start API server
echo "Starting API server on http://127.0.0.1:8000"
echo "Press CTRL+C to stop"
echo ""

cd backend
uvicorn src.api.main:app --reload --port 8000
