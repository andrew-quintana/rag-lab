#!/bin/bash
# Run Phase 5 integration tests against local Azure Functions
# This script runs tests marked with @pytest.mark.local against local resources (Azurite, local Supabase)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "=========================================="
echo "Running Phase 5 Tests (Local)"
echo "=========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check if Supabase is running
if ! (cd "$PROJECT_ROOT/infra/supabase" && supabase status >/dev/null 2>&1); then
    echo "Error: Supabase is not running."
    echo "Please start it first:"
    echo "  cd infra/supabase && supabase start"
    exit 1
fi
echo "✓ Supabase is running"

# Check if Azurite is running, start if not
if ! lsof -Pi :10000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠ Azurite is not running. Starting Azurite..."
    "$PROJECT_ROOT/scripts/start_azurite.sh"
    # Wait a moment for Azurite to start
    sleep 2
fi
echo "✓ Azurite is running"

# Check if .env.local exists
if [ ! -f "$PROJECT_ROOT/.env.local" ]; then
    echo "Error: .env.local not found."
    echo "Please run: ./scripts/setup_local_env.sh"
    exit 1
fi
echo "✓ .env.local exists"

# Check if local.settings.json exists
if [ ! -f "$PROJECT_ROOT/backend/azure_functions/local.settings.json" ]; then
    echo "Error: local.settings.json not found."
    exit 1
fi
echo "✓ local.settings.json exists"

# Check if functions are running (optional - tests can start them)
if lsof -Pi :7071 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "✓ Azure Functions are running"
else
    echo "⚠ Azure Functions are not running"
    echo "  Tests will need to start functions or you can run: ./scripts/dev_functions_local.sh"
fi

# Verify queues exist
echo ""
echo "Verifying queues in Azurite..."
python3 << PYTHON_SCRIPT
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path("$PROJECT_ROOT/backend")
if backend_path.exists():
    sys.path.insert(0, str(backend_path))

from azure.storage.queue import QueueServiceClient

# Azurite connection string format for Python SDK
connection_string = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"
queue_service = QueueServiceClient.from_connection_string(connection_string)

queues = ["ingestion-uploads", "ingestion-chunking", "ingestion-embeddings", "ingestion-indexing", "ingestion-dead-letter"]

for queue_name in queues:
    try:
        queue_client = queue_service.get_queue_client(queue_name)
        queue_client.get_queue_properties()
        print(f"  ✓ Queue exists: {queue_name}")
    except Exception as e:
        print(f"  ⚠ Queue not found: {queue_name} (will be created on first use)")
PYTHON_SCRIPT

# Run tests
echo ""
echo "=========================================="
echo "Running Phase 5 Integration Tests"
echo "=========================================="
echo ""

cd "$BACKEND_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set Azurite connection string for local testing
export AZURE_STORAGE_QUEUES_CONNECTION_STRING="UseDevelopmentStorage=true"
export AZURE_STORAGE_CONNECTION_STRING="UseDevelopmentStorage=true"

# Run tests with local marker
echo "Running tests marked with @pytest.mark.local and @pytest.mark.integration..."
pytest tests/integration/ \
    -v \
    -m "local and integration" \
    --tb=short \
    "$@"

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✓ All local tests passed!"
    echo "=========================================="
else
    echo "=========================================="
    echo "⚠ Some tests failed (exit code: $TEST_EXIT_CODE)"
    echo "=========================================="
fi

exit $TEST_EXIT_CODE

