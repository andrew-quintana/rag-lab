# Phase 5 Database Migration Guide

## Overview

This guide documents the process for applying database migrations required for Phase 5: Integration Testing & Migration. These migrations add persistence infrastructure for the worker-queue architecture.

## Migrations to Apply

### Migration 0019: Worker Queue Persistence
**File**: `infra/supabase/migrations/0019_add_worker_queue_persistence.sql`

**Changes**:
- Adds `status` column to `documents` table (if not exists)
- Adds timestamp columns: `parsed_at`, `chunked_at`, `embedded_at`, `indexed_at`
- Creates `chunks` table for storing chunks and embeddings
- Adds `extracted_text TEXT` column to `documents` table
- Creates indexes for performance

**Idempotency**: Uses `IF NOT EXISTS` clauses - safe to run multiple times

### Migration 0020: Ingestion Batch Metadata Documentation
**File**: `infra/supabase/migrations/0020_add_ingestion_batch_metadata.sql`

**Changes**:
- Documentation only - no schema changes
- Adds comments documenting batch processing metadata structure
- Documents use of `chunks` table for temporary batch storage

**Idempotency**: Safe to run multiple times (only adds comments)

## Application Methods

### Option 1: Using Supabase CLI (Recommended)

#### For Production Supabase:
```bash
# Navigate to Supabase directory
cd infra/supabase

# Link to your Supabase project (if not already linked)
supabase link --project-ref your-project-ref

# Push migrations to production
supabase db push
```

#### For Local Development:
```bash
# Navigate to Supabase directory
cd infra/supabase

# Reset database (applies all migrations)
supabase db reset

# Or just start (applies pending migrations)
supabase start
```

### Option 2: Manual Application via psql

```bash
# Set DATABASE_URL environment variable
export DATABASE_URL="postgresql://user:password@host:port/database"

# Apply migration 0019
psql $DATABASE_URL -f infra/supabase/migrations/0019_add_worker_queue_persistence.sql

# Apply migration 0020
psql $DATABASE_URL -f infra/supabase/migrations/0020_add_ingestion_batch_metadata.sql
```

### Option 3: Using Supabase Studio

1. Open Supabase Studio (https://app.supabase.com)
2. Navigate to SQL Editor
3. Copy and paste contents of `0019_add_worker_queue_persistence.sql`
4. Run the query
5. Repeat for `0020_add_ingestion_batch_metadata.sql`

## Verification

### 1. Run Verification Script

```bash
cd backend
source venv/bin/activate
python scripts/verify_phase5_migrations.py
```

### 2. Manual Database Verification

Run these SQL queries to verify migrations:

```sql
-- Check status column exists
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'documents' AND column_name = 'status';

-- Check timestamp columns exist
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'documents' 
AND column_name IN ('parsed_at', 'chunked_at', 'embedded_at', 'indexed_at');

-- Check extracted_text column exists
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'documents' AND column_name = 'extracted_text';

-- Check chunks table exists
SELECT table_name
FROM information_schema.tables
WHERE table_name = 'chunks';

-- Check chunks table structure
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'chunks'
ORDER BY ordinal_position;

-- Check indexes exist
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename IN ('documents', 'chunks')
AND indexname IN ('idx_documents_status', 'idx_chunks_document_id');
```

### 3. Expected Results

**documents table** should have:
- `status VARCHAR(50) DEFAULT 'uploaded'`
- `parsed_at TIMESTAMP`
- `chunked_at TIMESTAMP`
- `embedded_at TIMESTAMP`
- `indexed_at TIMESTAMP`
- `extracted_text TEXT`
- Index: `idx_documents_status`

**chunks table** should have:
- `chunk_id VARCHAR(255) PRIMARY KEY`
- `document_id UUID NOT NULL REFERENCES documents(id)`
- `text TEXT NOT NULL`
- `metadata JSONB`
- `embedding JSONB`
- `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`
- Index: `idx_chunks_document_id`

## Pre-Migration Checklist

- [ ] Backup production database
- [ ] Verify database connection
- [ ] Review migration files for correctness
- [ ] Ensure sufficient database permissions
- [ ] Plan maintenance window if needed (migrations are non-blocking but good practice)

## Post-Migration Checklist

- [ ] Verify all columns and tables created
- [ ] Verify indexes created correctly
- [ ] Run verification script
- [ ] Test sample queries
- [ ] Run Supabase integration tests
- [ ] Monitor database performance

## Rollback Plan

These migrations use `IF NOT EXISTS` and are additive only. If rollback is needed:

1. **Remove chunks table** (if empty or data can be regenerated):
   ```sql
   DROP TABLE IF EXISTS chunks;
   ```

2. **Remove columns from documents table** (if no data depends on them):
   ```sql
   ALTER TABLE documents DROP COLUMN IF EXISTS extracted_text;
   ALTER TABLE documents DROP COLUMN IF EXISTS status;
   ALTER TABLE documents DROP COLUMN IF EXISTS parsed_at;
   ALTER TABLE documents DROP COLUMN IF EXISTS chunked_at;
   ALTER TABLE documents DROP COLUMN IF EXISTS embedded_at;
   ALTER TABLE documents DROP COLUMN IF EXISTS indexed_at;
   ```

3. **Remove indexes**:
   ```sql
   DROP INDEX IF EXISTS idx_documents_status;
   DROP INDEX IF EXISTS idx_chunks_document_id;
   ```

**Warning**: Only rollback if no production data depends on these schema changes.

## Troubleshooting

### Migration Already Applied
If migrations show "already exists" errors, this is expected - migrations are idempotent.

### Permission Errors
Ensure database user has `ALTER TABLE`, `CREATE TABLE`, and `CREATE INDEX` permissions.

### Connection Errors
Verify `DATABASE_URL` is correct and database is accessible.

### Index Creation Fails
If index creation fails, check if index already exists. Use `IF NOT EXISTS` in manual rollback if needed.

## Next Steps

After migrations are applied:
1. Run verification script
2. Run Supabase integration tests
3. Proceed with Azure Functions deployment
4. Run end-to-end integration tests

