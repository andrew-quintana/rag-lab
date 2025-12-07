#!/bin/bash
# Run all Phase 5 tests in local environment
# This script runs all Phase 5 integration tests against local Supabase and Azurite

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "=========================================="
echo "Phase 5 Complete Test Suite (Local)"
echo "=========================================="
echo ""
echo "Running all Phase 5 tests against:"
echo "  - Local Supabase (PostgreSQL)"
echo "  - Azurite (Azure Storage Emulator)"
echo "  - Local Azure Functions (if running)"
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

# Check if Azurite is running
if ! lsof -Pi :10000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Error: Azurite is not running."
    echo "Please start it first:"
    echo "  ./scripts/start_azurite.sh"
    exit 1
fi
echo "✓ Azurite is running"

# Check if .env.local exists
if [ ! -f "$PROJECT_ROOT/.env.local" ]; then
    echo "Error: .env.local not found."
    echo "Please run: ./scripts/setup_local_env.sh"
    exit 1
fi
echo "✓ .env.local exists"

# Set Azurite connection string
export AZURE_STORAGE_QUEUES_CONNECTION_STRING="UseDevelopmentStorage=true"
export AZURE_STORAGE_CONNECTION_STRING="UseDevelopmentStorage=true"

cd "$BACKEND_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Test results summary
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

echo ""
echo "=========================================="
echo "Test 1: Database Migration Verification"
echo "=========================================="
echo ""
if python scripts/verify_phase5_migrations.py 2>&1; then
    echo "✓ Migration verification PASSED"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo "✗ Migration verification FAILED"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

echo ""
echo "=========================================="
echo "Test 2: Supabase Integration Tests"
echo "=========================================="
echo ""
SUPABASE_RESULT=$(pytest tests/integration/test_supabase_phase5.py -v -m integration --tb=short 2>&1 | tee /tmp/supabase_tests.log)
SUPABASE_PASSED=$(echo "$SUPABASE_RESULT" | grep -oP '\d+(?= passed)' | head -1 || echo "0")
SUPABASE_FAILED=$(echo "$SUPABASE_RESULT" | grep -oP '\d+(?= failed)' | head -1 || echo "0")
SUPABASE_SKIPPED=$(echo "$SUPABASE_RESULT" | grep -oP '\d+(?= skipped)' | head -1 || echo "0")
echo "Supabase tests: $SUPABASE_PASSED passed, $SUPABASE_FAILED failed, $SUPABASE_SKIPPED skipped"
PASSED_TESTS=$((PASSED_TESTS + SUPABASE_PASSED))
FAILED_TESTS=$((FAILED_TESTS + SUPABASE_FAILED))
SKIPPED_TESTS=$((SKIPPED_TESTS + SUPABASE_SKIPPED))
TOTAL_TESTS=$((TOTAL_TESTS + SUPABASE_PASSED + SUPABASE_FAILED + SUPABASE_SKIPPED))

echo ""
echo "=========================================="
echo "Test 3: End-to-End Pipeline Tests"
echo "=========================================="
echo ""
E2E_RESULT=$(pytest tests/integration/test_phase5_e2e_pipeline.py -v -m integration --tb=short 2>&1 | tee /tmp/e2e_tests.log)
E2E_PASSED=$(echo "$E2E_RESULT" | grep -oP '\d+(?= passed)' | head -1 || echo "0")
E2E_FAILED=$(echo "$E2E_RESULT" | grep -oP '\d+(?= failed)' | head -1 || echo "0")
E2E_SKIPPED=$(echo "$E2E_RESULT" | grep -oP '\d+(?= skipped)' | head -1 || echo "0")
echo "E2E tests: $E2E_PASSED passed, $E2E_FAILED failed, $E2E_SKIPPED skipped"
PASSED_TESTS=$((PASSED_TESTS + E2E_PASSED))
FAILED_TESTS=$((FAILED_TESTS + E2E_FAILED))
SKIPPED_TESTS=$((SKIPPED_TESTS + E2E_SKIPPED))
TOTAL_TESTS=$((TOTAL_TESTS + E2E_PASSED + E2E_FAILED + E2E_SKIPPED))

echo ""
echo "=========================================="
echo "Test 4: Performance Tests"
echo "=========================================="
echo ""
PERF_RESULT=$(pytest tests/integration/test_phase5_performance.py -v -m integration -m performance --tb=short 2>&1 | tee /tmp/perf_tests.log)
PERF_PASSED=$(echo "$PERF_RESULT" | grep -oP '\d+(?= passed)' | head -1 || echo "0")
PERF_FAILED=$(echo "$PERF_RESULT" | grep -oP '\d+(?= failed)' | head -1 || echo "0")
PERF_SKIPPED=$(echo "$PERF_RESULT" | grep -oP '\d+(?= skipped)' | head -1 || echo "0")
echo "Performance tests: $PERF_PASSED passed, $PERF_FAILED failed, $PERF_SKIPPED skipped"
PASSED_TESTS=$((PASSED_TESTS + PERF_PASSED))
FAILED_TESTS=$((FAILED_TESTS + PERF_FAILED))
SKIPPED_TESTS=$((SKIPPED_TESTS + PERF_SKIPPED))
TOTAL_TESTS=$((TOTAL_TESTS + PERF_PASSED + PERF_FAILED + PERF_SKIPPED))

echo ""
echo "=========================================="
echo "Phase 5 Test Summary"
echo "=========================================="
echo ""
echo "Total Tests: $TOTAL_TESTS"
echo "  ✓ Passed:  $PASSED_TESTS"
echo "  ✗ Failed:  $FAILED_TESTS"
echo "  ⊘ Skipped: $SKIPPED_TESTS"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo "=========================================="
    echo "✓ All Phase 5 tests PASSED!"
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "⚠ Some tests FAILED"
    echo "=========================================="
    echo ""
    echo "Check test logs for details:"
    echo "  - Supabase tests: /tmp/supabase_tests.log"
    echo "  - E2E tests: /tmp/e2e_tests.log"
    echo "  - Performance tests: /tmp/perf_tests.log"
    exit 1
fi

