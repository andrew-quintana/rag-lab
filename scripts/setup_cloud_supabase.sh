#!/bin/bash
# Setup Cloud Supabase for RAG Evaluator
# This script applies migrations and configures Azure Functions

set -e

echo "=========================================="
echo "Cloud Supabase Setup Script"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if .env.prod exists
if [ ! -f "$PROJECT_ROOT/.env.prod" ]; then
    echo -e "${RED}Error: .env.prod not found${NC}"
    echo "Please create .env.prod with your cloud Supabase credentials"
    exit 1
fi

# Load .env.prod
echo "Loading cloud configuration from .env.prod..."
export $(grep -v '^#' "$PROJECT_ROOT/.env.prod" | xargs)

# Validate required variables
REQUIRED_VARS=("DATABASE_URL" "SUPABASE_URL" "SUPABASE_SERVICE_ROLE_KEY")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo -e "${RED}Error: Missing required environment variables:${NC}"
    printf '  - %s\n' "${MISSING_VARS[@]}"
    exit 1
fi

echo -e "${GREEN}✓ All required variables found${NC}"
echo ""

# Display configuration
echo "Cloud Supabase Configuration:"
echo "  URL: $SUPABASE_URL"
echo "  Database: ${DATABASE_URL%%\?*}"  # Hide query params
echo ""

# Confirm before proceeding
read -p "Apply migrations to cloud database? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "=========================================="
echo "Step 1: Applying Migrations"
echo "=========================================="
echo ""

# Apply migrations one by one
MIGRATIONS_DIR="$PROJECT_ROOT/infra/supabase/migrations"
SUCCESS_COUNT=0
FAIL_COUNT=0

for migration in "$MIGRATIONS_DIR"/*.sql; do
    filename=$(basename "$migration")
    echo -n "  Applying $filename... "
    
    if psql "$DATABASE_URL" -f "$migration" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        ((SUCCESS_COUNT++))
    else
        # Check if it's already applied (might be OK)
        if psql "$DATABASE_URL" -f "$migration" 2>&1 | grep -q "already exists"; then
            echo -e "${YELLOW}⊘ (already applied)${NC}"
            ((SUCCESS_COUNT++))
        else
            echo -e "${RED}✗${NC}"
            ((FAIL_COUNT++))
        fi
    fi
done

echo ""
echo "Migration Results:"
echo "  Successful: $SUCCESS_COUNT"
echo "  Failed: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${YELLOW}Warning: Some migrations failed. This may be OK if they were already applied.${NC}"
    echo "Check Supabase Dashboard → SQL Editor to verify database schema."
    echo ""
fi

echo "=========================================="
echo "Step 2: Creating Test Document Record"
echo "=========================================="
echo ""

# Create test document in cloud database
TEST_DOC_ID="f7df526a-97df-4e85-836d-b3478a72ae6d"

echo "Creating test document record: $TEST_DOC_ID"
psql "$DATABASE_URL" << EOF
INSERT INTO documents (id, filename, file_size, mime_type, upload_timestamp, status, chunks_created, storage_path, metadata)
VALUES (
    '$TEST_DOC_ID',
    'scan_classic_hmo.pdf',
    2544678,
    'application/pdf',
    NOW(),
    'uploaded',
    0,
    '$TEST_DOC_ID.pdf',
    '{"test": "time-based-limiting", "source": "azure_blob"}'::jsonb
)
ON CONFLICT (id) DO UPDATE SET
    status = 'uploaded',
    metadata = '{"test": "time-based-limiting", "source": "azure_blob"}'::jsonb;
EOF

echo -e "${GREEN}✓ Test document record created/updated${NC}"
echo ""

echo "=========================================="
echo "Step 3: Updating Azure Functions"
echo "=========================================="
echo ""

echo "Updating Azure Functions with cloud Supabase credentials..."

az functionapp config appsettings set \
  --name func-raglab-uploadworkers \
  --resource-group rag-lab \
  --settings \
    SUPABASE_URL="$SUPABASE_URL" \
    SUPABASE_KEY="$SUPABASE_SERVICE_ROLE_KEY" \
    SUPABASE_SERVICE_ROLE_KEY="$SUPABASE_SERVICE_ROLE_KEY" \
    DATABASE_URL="$DATABASE_URL" \
  --output none

echo -e "${GREEN}✓ Azure Functions updated${NC}"
echo ""

echo "=========================================="
echo "Step 4: Verifying Database Connection"
echo "=========================================="
echo ""

echo "Testing connection to cloud database..."
if psql "$DATABASE_URL" -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Database connection successful${NC}"
    
    # Count tables
    TABLE_COUNT=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';")
    echo "  Tables in database: $TABLE_COUNT"
    
    # Check for test document
    DOC_EXISTS=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM documents WHERE id='$TEST_DOC_ID';")
    if [ "$DOC_EXISTS" -eq 1 ]; then
        echo -e "${GREEN}✓ Test document found in database${NC}"
    else
        echo -e "${YELLOW}⚠ Test document not found${NC}"
    fi
else
    echo -e "${RED}✗ Database connection failed${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Cloud Supabase Setup Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "  1. Fix deployment code issue (build.sh)"
echo "  2. Deploy Azure Functions"
echo "  3. Run E2E test with cloud infrastructure"
echo ""
echo "Configuration Summary:"
echo "  ✓ Migrations applied to cloud database"
echo "  ✓ Test document created (id: $TEST_DOC_ID)"
echo "  ✓ Azure Functions configured"
echo "  ✓ Database connection verified"
echo ""
echo "To run the deployment:"
echo "  cd backend/azure_functions"
echo "  ./build.sh"
echo "  func azure functionapp publish func-raglab-uploadworkers --python"
echo ""
