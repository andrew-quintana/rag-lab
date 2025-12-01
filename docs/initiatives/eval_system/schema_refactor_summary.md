# Schema Refactor Summary

**Date**: 2024-12-19  
**Purpose**: Refactor prompt storage schema to use new table structure and naming conventions

## Schema Changes

### Database Table Changes

1. **Table Renamed**: `prompt_versions` → `prompts`
2. **Column Changes**:
   - `version_id` (VARCHAR) → `id` (UUID, auto-generated)
   - `version_name` (VARCHAR) → `version` (VARCHAR, semantic versioning e.g., "v0.1")
   - `evaluator_type` (VARCHAR) → `name` (VARCHAR, nullable)
   - Added: `live` (BOOLEAN, default false)

### Versioning Strategy

- **Semantic Versioning**: Uses format `v0.1`, `v0.2`, etc.
- **First Digit**: Captures larger refactors (e.g., v0.x for initial versions, v1.x for major changes)
- **Live Flag**: Boolean field marks the active/live version for a given `prompt_type` and `name` combination

### Migration Files

1. **0007_rename_prompts_table.sql**: 
   - Creates new `prompts` table
   - Migrates data from `prompt_versions`
   - Updates foreign key references
   - Adds indexes and constraints

2. **0008_insert_evaluation_prompts_v0_1.sql**:
   - Inserts evaluation prompts with version `v0.1`
   - Marks all as `live=true`

## Code Changes

### Function Signatures Updated

#### `load_prompt_template()`
- **Old**: `load_prompt_template(version: str, query_executor: QueryExecutor, prompt_type: str = "rag", evaluator_type: Optional[str] = None)`
- **New**: `load_prompt_template(version: Optional[str] = None, query_executor: Optional[QueryExecutor] = None, prompt_type: str = "rag", name: Optional[str] = None, live: bool = True)`
- **Changes**:
  - `version` is now optional (defaults to live version if None)
  - `evaluator_type` → `name`
  - Added `live` parameter (defaults to True)

#### `BaseEvaluatorNode.__init__()`
- **Old**: `__init__(prompt_version: str, prompt_type: str = "evaluation", evaluator_type: Optional[str] = None, ...)`
- **New**: `__init__(prompt_version: Optional[str] = None, prompt_type: str = "evaluation", name: Optional[str] = None, live: bool = True, ...)`
- **Changes**:
  - `prompt_version` is now optional
  - `evaluator_type` → `name`
  - Added `live` parameter

#### Evaluator Classes
- `CorrectnessEvaluator`, `HallucinationEvaluator`, `RiskDirectionEvaluator`
- All updated to use new parameter names and optional version

#### Module-Level Functions
- `classify_correctness()`, `classify_hallucination()`, `classify_risk_direction()`
- All updated to accept `prompt_version` and `live` parameters

### Query Logic Updates

#### Live Version Queries
When `live=True` and `version=None`, queries use:
```sql
SELECT prompt_text
FROM prompts
WHERE prompt_type = %s AND name = %s AND live = true
ORDER BY version DESC
LIMIT 1
```

#### Specific Version Queries
When `version` is provided:
```sql
SELECT prompt_text
FROM prompts
WHERE prompt_type = %s AND name = %s AND version = %s
```

### Cache Key Format
- **Old**: `{prompt_type}:{evaluator_type}` or `{prompt_type}:{version_name}`
- **New**: `{prompt_type}:{name}:live` or `{prompt_type}:{name}:{version}` or `{prompt_type}:live`

## Files Modified

### Migrations
- `infra/supabase/migrations/0007_rename_prompts_table.sql` (new)
- `infra/supabase/migrations/0008_insert_evaluation_prompts_v0_1.sql` (new)

### Core Services
- `backend/rag_eval/services/rag/generation.py`
- `backend/rag_eval/services/evaluator/base_evaluator.py`
- `backend/rag_eval/services/evaluator/correctness.py`
- `backend/rag_eval/services/evaluator/hallucination.py`
- `backend/rag_eval/services/evaluator/risk_direction.py`

### Database Models
- `backend/rag_eval/db/models.py` (PromptVersion dataclass updated)

## Testing Status

### Migration Applied
✅ Migration 0007 applied successfully
✅ Migration 0008 applied successfully
✅ All 3 evaluation prompts inserted with version `v0.1` and `live=true`

### Code Status
✅ All linter errors resolved
✅ Imports working correctly
⚠️ Tests need updating to match new schema

## Next Steps

1. Update test files to use new parameter names (`name` instead of `evaluator_type`)
2. Update test queries to match new SQL structure
3. Run comprehensive test suite
4. Update documentation


