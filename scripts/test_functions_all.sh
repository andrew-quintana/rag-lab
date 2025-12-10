#!/bin/bash
# Run all Phase 5 tests (local and cloud)
# This script runs both local and cloud tests, providing a comprehensive test summary

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "=========================================="
echo "Running All Phase 5 Tests"
echo "=========================================="
echo ""
echo "This script runs:"
echo "  1. Local tests (Azurite, local Supabase)"
echo "  2. Cloud tests (Azure Functions, cloud Supabase) - if environment configured"
echo ""

# Track overall results
LOCAL_TESTS_PASSED=false
CLOUD_TESTS_PASSED=false
LOCAL_EXIT_CODE=0
CLOUD_EXIT_CODE=0

# Run local tests
echo "=========================================="
echo "Step 1: Running Local Tests"
echo "=========================================="
echo ""

if "$PROJECT_ROOT/scripts/test_functions_local.sh"; then
    LOCAL_TESTS_PASSED=true
    echo ""
    echo "✓ Local tests completed successfully"
else
    LOCAL_EXIT_CODE=$?
    echo ""
    echo "⚠ Local tests had failures (exit code: $LOCAL_EXIT_CODE)"
fi

echo ""
echo "=========================================="
echo "Step 2: Running Cloud Tests"
echo "=========================================="
echo ""

# Check if cloud environment is configured
CLOUD_AVAILABLE=true
if [ -z "$AZURE_STORAGE_QUEUES_CONNECTION_STRING" ] && [ -z "$AZURE_BLOB_CONNECTION_STRING" ]; then
    if [ -z "$DATABASE_URL" ]; then
        echo "⚠ Cloud environment not configured (missing connection strings)"
        echo "  Skipping cloud tests"
        CLOUD_AVAILABLE=false
    fi
fi

if [ "$CLOUD_AVAILABLE" = true ]; then
    if "$PROJECT_ROOT/scripts/test_functions_cloud.sh"; then
        CLOUD_TESTS_PASSED=true
        echo ""
        echo "✓ Cloud tests completed successfully"
    else
        CLOUD_EXIT_CODE=$?
        echo ""
        echo "⚠ Cloud tests had failures (exit code: $CLOUD_EXIT_CODE)"
    fi
else
    echo "⊘ Cloud tests skipped (environment not configured)"
    CLOUD_TESTS_PASSED=true  # Treat skipped as passed for summary
fi

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""

if [ "$LOCAL_TESTS_PASSED" = true ]; then
    echo "✓ Local tests: PASSED"
else
    echo "✗ Local tests: FAILED (exit code: $LOCAL_EXIT_CODE)"
fi

if [ "$CLOUD_AVAILABLE" = false ]; then
    echo "⊘ Cloud tests: SKIPPED (environment not configured)"
elif [ "$CLOUD_TESTS_PASSED" = true ]; then
    echo "✓ Cloud tests: PASSED"
else
    echo "✗ Cloud tests: FAILED (exit code: $CLOUD_EXIT_CODE)"
fi

echo ""

# Determine overall exit code
if [ "$LOCAL_TESTS_PASSED" = true ] && [ "$CLOUD_TESTS_PASSED" = true ]; then
    echo "=========================================="
    echo "✓ All tests completed successfully!"
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "⚠ Some tests failed"
    echo "=========================================="
    # Return non-zero if any tests failed
    if [ "$LOCAL_TESTS_PASSED" = false ]; then
        exit $LOCAL_EXIT_CODE
    elif [ "$CLOUD_TESTS_PASSED" = false ]; then
        exit $CLOUD_EXIT_CODE
    else
        exit 1
    fi
fi

