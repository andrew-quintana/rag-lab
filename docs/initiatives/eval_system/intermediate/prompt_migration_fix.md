# Prompt Migration Fix Summary

**Date**: 2024-12-19  
**Issue**: Evaluation prompts not properly inserted into Supabase database  
**Status**: ✅ Fixed

## Problem Identified

1. **Migration SQL Issues**:
   - Migration file used `ON CONFLICT (version_id)` but should use `(prompt_type, evaluator_type, version_name)`
   - Unique constraint was not properly updated in migration 0005
   - Dollar-quoting was used correctly, but conflict resolution was wrong

2. **Testing Gaps**:
   - All tests used mocked `QueryExecutor` - never tested against real database
   - No integration tests to verify prompts actually exist in database
   - No validation that migration SQL is syntactically correct

## Fixes Applied

### 1. Updated Migration 0005 (`0005_add_evaluator_type.sql`)
- Added proper unique constraint: `(prompt_type, evaluator_type, version_name)`
- Drops old constraint `(prompt_type, version_name)` if it exists
- Creates new constraint that includes `evaluator_type`

### 2. Updated Migration 0006 (`0006_add_evaluation_prompts.sql`)
- Changed `ON CONFLICT (version_id) DO NOTHING` to `ON CONFLICT (prompt_type, evaluator_type, version_name) DO NOTHING`
- This matches the unique constraint defined in migration 0005

### 3. Created Validation Script
- `backend/scripts/validate_prompt_migrations.py`
- Validates all three prompts exist in database
- Compares database content with markdown files
- Tests `load_prompt_template()` function
- Can be run manually: `python backend/scripts/validate_prompt_migrations.py`

### 4. Created Integration Tests
- `backend/tests/components/evaluator/test_prompt_database_integration.py`
- Tests that prompts exist in database
- Tests loading via `load_prompt_template()`
- Tests loading via evaluator classes
- Marked with `@pytest.mark.integration` for selective running

## Migration Files Status

### Migration 0005: Add evaluator_type Column
- ✅ Adds `evaluator_type VARCHAR(50)` column
- ✅ Creates index on `evaluator_type`
- ✅ Drops old unique constraint `(prompt_type, version_name)`
- ✅ Creates new unique constraint `(prompt_type, evaluator_type, version_name)`

### Migration 0006: Insert Evaluation Prompts
- ✅ Uses dollar-quoting (`$$prompt$$`) for safe string insertion
- ✅ Inserts all three prompts:
  - `correctness_evaluator` (v1)
  - `hallucination_evaluator` (v1)
  - `risk_direction_evaluator` (v1)
- ✅ Uses correct `ON CONFLICT` clause matching unique constraint
- ✅ Includes `version_id` for each prompt

## How to Apply Migrations

**⚠️ Important**: These migrations preserve existing data. They use `IF NOT EXISTS` and `ON CONFLICT DO NOTHING` to be safe.

### Option 1: Using Supabase CLI (Recommended)
```bash
cd infra/supabase
# Push new migrations to remote database (preserves existing data)
supabase db push

# For local development, link to your project first if needed:
# supabase link --project-ref your-project-ref
# Then push: supabase db push
```

### Option 2: Manual Application via psql
```bash
# Apply migration 0005 (adds column and constraint)
psql $DATABASE_URL -f infra/supabase/migrations/0005_add_evaluator_type.sql

# Apply migration 0006 (inserts prompts)
psql $DATABASE_URL -f infra/supabase/migrations/0006_add_evaluation_prompts.sql
```

### Option 3: Using Supabase Studio
1. Open Supabase Studio
2. Navigate to SQL Editor
3. Copy and paste the contents of `0005_add_evaluator_type.sql`
4. Run the query
5. Repeat for `0006_add_evaluation_prompts.sql`

**Note**: Both migrations are idempotent - safe to run multiple times.

## How to Validate

### 1. Run Validation Script
```bash
cd backend
source venv/bin/activate
python scripts/validate_prompt_migrations.py
```

### 2. Run Integration Tests
```bash
cd backend
source venv/bin/activate
pytest tests/components/evaluator/test_prompt_database_integration.py -v -m integration
```

### 3. Manual Database Check
```sql
-- Check all evaluation prompts
SELECT version_id, version_name, prompt_type, evaluator_type, 
       LENGTH(prompt_text) as prompt_length
FROM prompt_versions
WHERE prompt_type = 'evaluation'
ORDER BY evaluator_type;

-- Should return 3 rows:
-- eval_correctness_v1 | v1 | evaluation | correctness_evaluator | ~2377
-- eval_hallucination_v1 | v1 | evaluation | hallucination_evaluator | ~4415
-- eval_risk_direction_v1 | v1 | evaluation | risk_direction_evaluator | ~7475
```

## Testing Improvements

### Before
- ❌ All tests used mocked `QueryExecutor`
- ❌ No verification that prompts exist in database
- ❌ No integration tests

### After
- ✅ Integration tests verify database storage
- ✅ Validation script for manual verification
- ✅ Tests verify both database queries and function loading
- ✅ Unit tests still use mocks (for speed)
- ✅ Integration tests use real database (for validation)

## Next Steps

1. **Apply Migrations**: Run migrations on your Supabase instance
2. **Run Validation**: Execute validation script to verify prompts are inserted
3. **Run Integration Tests**: Verify all integration tests pass
4. **Update CI/CD**: Consider adding integration tests to CI pipeline (optional)

## Files Changed

1. `infra/supabase/migrations/0005_add_evaluator_type.sql` - Added unique constraint
2. `infra/supabase/migrations/0006_add_evaluation_prompts.sql` - Fixed ON CONFLICT clause
3. `backend/scripts/validate_prompt_migrations.py` - New validation script
4. `backend/tests/components/evaluator/test_prompt_database_integration.py` - New integration tests

## Verification Checklist

- [ ] Migration 0005 applied successfully
- [ ] Migration 0006 applied successfully
- [ ] Validation script runs without errors
- [ ] All three prompts exist in database
- [ ] Integration tests pass
- [ ] `load_prompt_template()` works for all three evaluator types
- [ ] Evaluator classes can load prompts from database

