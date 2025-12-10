#!/bin/bash
# Run Phase 5 integration tests against cloud Azure Functions
# This script runs tests marked with @pytest.mark.cloud against cloud resources

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "=========================================="
echo "Running Phase 5 Tests (Cloud)"
echo "=========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check if Azure Functions are deployed
echo "Verifying Azure Functions deployment..."
if ! command -v az &> /dev/null; then
    echo "⚠ Azure CLI not found. Cannot verify Function App deployment."
    echo "  Continuing with tests assuming Function App is deployed..."
else
    FUNCTION_APP_NAME="func-raglab-uploadworkers"
    if az functionapp show --name "$FUNCTION_APP_NAME" --resource-group "rg-raglab" &>/dev/null; then
        echo "✓ Azure Function App '$FUNCTION_APP_NAME' is deployed"
    else
        echo "⚠ Azure Function App '$FUNCTION_APP_NAME' not found or not accessible"
        echo "  Continuing with tests assuming resources are configured..."
    fi
fi

# Check if cloud resources are accessible
echo ""
echo "Verifying cloud resources are accessible..."
echo "  - Azure Storage Account (queues)"
echo "  - Cloud Supabase instance"
echo "  - Azure Functions endpoints"
echo ""
echo "⚠ Make sure you have:"
echo "  - Cloud connection strings configured in environment"
echo "  - Cloud Supabase credentials in environment"
echo "  - Azure Functions deployed and accessible"
echo ""

# Check if required environment variables are set
if [ -z "$AZURE_STORAGE_QUEUES_CONNECTION_STRING" ] && [ -z "$AZURE_BLOB_CONNECTION_STRING" ]; then
    echo "⚠ Warning: AZURE_STORAGE_QUEUES_CONNECTION_STRING or AZURE_BLOB_CONNECTION_STRING not set"
    echo "  Tests may fail if connection strings are not available"
fi

if [ -z "$DATABASE_URL" ]; then
    echo "⚠ Warning: DATABASE_URL not set"
    echo "  Tests may fail if database connection is not available"
fi

cd "$BACKEND_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run tests with cloud marker
echo ""
echo "=========================================="
echo "Running Phase 5 Cloud Tests"
echo "=========================================="
echo ""
echo "Running tests marked with @pytest.mark.cloud and @pytest.mark.integration..."
pytest tests/integration/ \
    -v \
    -m "cloud and integration" \
    --tb=short \
    "$@"

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✓ All cloud tests passed!"
    echo "=========================================="
else
    echo "=========================================="
    echo "⚠ Some tests failed (exit code: $TEST_EXIT_CODE)"
    echo "=========================================="
fi

exit $TEST_EXIT_CODE

