#!/bin/bash
# Start all local services for Azure Functions development
# - Starts Azurite (if not running)
# - Creates required queues in Azurite
# - Starts Azure Functions locally

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUNCTIONS_DIR="$PROJECT_ROOT/infra/azure/azure_functions"

echo "=========================================="
echo "Starting Local Azure Functions Development"
echo "=========================================="
echo ""

# Check prerequisites
if ! command -v func &> /dev/null; then
    echo "Error: Azure Functions Core Tools not found."
    echo "Please install it first:"
    echo "  brew tap azure/functions"
    echo "  brew install azure-functions-core-tools@4"
    exit 1
fi

# Start Azurite if not running
if ! lsof -Pi :10000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Starting Azurite..."
    "$PROJECT_ROOT/scripts/start_azurite.sh"
    sleep 3  # Give Azurite time to start
else
    echo "✓ Azurite is already running"
fi

# Create required queues in Azurite
echo ""
echo "Creating required queues in Azurite..."
QUEUES=("ingestion-uploads" "ingestion-chunking" "ingestion-embeddings" "ingestion-indexing" "ingestion-dead-letter")

# Use Azure Storage CLI or Python script to create queues
# For Azurite, we can use the Azure Storage Python SDK
cd "$PROJECT_ROOT"

# Create a temporary Python script to create queues
python3 << PYTHON_SCRIPT
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path("$PROJECT_ROOT/backend")
if backend_path.exists():
    sys.path.insert(0, str(backend_path))

from azure.storage.queue import QueueServiceClient

# Azurite connection string format for Python SDK
# Default Azurite account: devstoreaccount1
# Default Azurite key: Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==
connection_string = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"
queue_service = QueueServiceClient.from_connection_string(connection_string)

queues = ["ingestion-uploads", "ingestion-chunking", "ingestion-embeddings", "ingestion-indexing", "ingestion-dead-letter"]

for queue_name in queues:
    try:
        queue_client = queue_service.get_queue_client(queue_name)
        queue_client.create_queue()
        print(f"  ✓ Created queue: {queue_name}")
    except Exception as e:
        if "QueueAlreadyExists" in str(e) or "409" in str(e):
            print(f"  ✓ Queue already exists: {queue_name}")
        else:
            print(f"  ⚠ Error creating queue {queue_name}: {e}")
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo "Warning: Could not create queues. They will be created automatically on first use."
fi

# Check if .env.local exists
if [ ! -f "$PROJECT_ROOT/.env.local" ]; then
    echo ""
    echo "Warning: .env.local not found."
    echo "Run ./scripts/setup_local_env.sh to create it."
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if local.settings.json exists
if [ ! -f "$FUNCTIONS_DIR/local.settings.json" ]; then
    echo "Error: local.settings.json not found in $FUNCTIONS_DIR"
    echo "This file should have been created as part of the setup."
    exit 1
fi

# Start Azure Functions
echo ""
echo "=========================================="
echo "Starting Azure Functions..."
echo "=========================================="
echo ""
echo "Functions will be available at:"
echo "  - ingestion-worker: triggered by ingestion-uploads queue"
echo "  - chunking-worker: triggered by ingestion-chunking queue"
echo "  - embedding-worker: triggered by ingestion-embeddings queue"
echo "  - indexing-worker: triggered by ingestion-indexing queue"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

cd "$FUNCTIONS_DIR"

# Trap Ctrl+C to cleanup
cleanup() {
    echo ""
    echo "Stopping services..."
    "$PROJECT_ROOT/scripts/stop_azurite.sh" || true
    exit 0
}

trap cleanup INT TERM

# Start functions
func start

