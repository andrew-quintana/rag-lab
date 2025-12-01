# Prompt Type Enhancement

## Overview
Enhanced the `prompt_versions` table to support different types of prompts, allowing the same version name to exist for different prompt types (e.g., "rag", "evaluation", "summarization").

## Changes Made

### Database Migration
**File**: `infra/supabase/migrations/0002_add_prompt_type.sql`

- Added `prompt_type` column (VARCHAR(50), default: 'rag')
- Changed unique constraint from `version_name` to `(prompt_type, version_name)`
- Added index on `prompt_type` for faster lookups
- Backward compatible: existing records default to 'rag'

### Code Changes

#### `backend/rag_eval/services/rag/generation.py`
- Updated `load_prompt_template()` to accept optional `prompt_type` parameter (default: "rag")
- Updated `construct_prompt()` to accept optional `prompt_type` parameter (default: "rag")
- Updated cache key format from `version` to `{prompt_type}:{version}`
- Updated SQL query to filter by both `prompt_type` and `version_name`

#### `backend/rag_eval/db/models.py`
- Updated `PromptVersion` dataclass to include `prompt_type` field (default: "rag")

### Test Updates
- Updated all existing tests to work with new cache key format
- Added tests for different prompt types
- Added test for backward compatibility (default prompt_type="rag")
- All 23 tests passing

## Usage

### Basic Usage (Backward Compatible)
```python
from rag_eval.services.rag.generation import load_prompt_template, construct_prompt
from rag_eval.db.queries import QueryExecutor

# Default prompt_type is "rag" - works exactly as before
template = load_prompt_template("v1", query_executor)
prompt = construct_prompt(query, chunks, "v1", query_executor)
```

### Using Different Prompt Types
```python
# Load RAG prompt
rag_template = load_prompt_template("v1", query_executor, prompt_type="rag")
rag_prompt = construct_prompt(query, chunks, "v1", query_executor, prompt_type="rag")

# Load evaluation prompt (same version name, different type)
eval_template = load_prompt_template("v1", query_executor, prompt_type="evaluation")
eval_prompt = construct_prompt(query, chunks, "v1", query_executor, prompt_type="evaluation")

# Load summarization prompt
summarization_template = load_prompt_template("v1", query_executor, prompt_type="summarization")
```

## Database Schema

### Before
```sql
CREATE TABLE prompt_versions (
    version_id VARCHAR(255) PRIMARY KEY,
    version_name VARCHAR(100) NOT NULL UNIQUE,  -- Only one "v1" allowed
    prompt_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### After
```sql
CREATE TABLE prompt_versions (
    version_id VARCHAR(255) PRIMARY KEY,
    version_name VARCHAR(100) NOT NULL,
    prompt_text TEXT NOT NULL,
    prompt_type VARCHAR(50) NOT NULL DEFAULT 'rag',  -- New field
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (prompt_type, version_name)  -- Can have "v1" for each type
);
```

## Example Data

```sql
-- RAG prompts
INSERT INTO prompt_versions (version_id, version_name, prompt_type, prompt_text) VALUES
('pv1', 'v1', 'rag', 'You are a helpful assistant. Query: {query}. Context: {context}.');

-- Evaluation prompts (same version name, different type)
INSERT INTO prompt_versions (version_id, version_name, prompt_type, prompt_text) VALUES
('pv2', 'v1', 'evaluation', 'Evaluate this answer. Query: {query}. Answer: {answer}.');

-- Summarization prompts
INSERT INTO prompt_versions (version_id, version_name, prompt_type, prompt_text) VALUES
('pv3', 'v1', 'summarization', 'Summarize the following: {context}');
```

## Migration Instructions

1. **Run the migration**:
   ```bash
   # Apply migration to your Supabase database
   psql $DATABASE_URL -f infra/supabase/migrations/0002_add_prompt_type.sql
   ```

2. **Update existing data** (if needed):
   ```sql
   -- All existing records will have prompt_type='rag' by default
   -- If you want to explicitly set it:
   UPDATE prompt_versions SET prompt_type = 'rag' WHERE prompt_type IS NULL;
   ```

3. **No code changes required** for existing code - backward compatible!

## Benefits

1. **Flexibility**: Support multiple prompt types (rag, evaluation, summarization, etc.)
2. **Organization**: Better organization of prompts by type
3. **Versioning**: Same version name can exist for different types
4. **Backward Compatible**: Existing code works without changes
5. **Caching**: Cache keys include prompt_type, so different types are cached separately

## Common Prompt Types

Suggested prompt types:
- `rag` - RAG (Retrieval-Augmented Generation) prompts (default)
- `evaluation` - Evaluation/judgment prompts
- `summarization` - Summarization prompts
- `classification` - Classification prompts
- `extraction` - Information extraction prompts
- `custom` - Custom prompt types

## Notes

- Cache keys are now `{prompt_type}:{version_name}` format
- All existing code continues to work (defaults to "rag")
- Migration is backward compatible (existing records get prompt_type='rag')
- Unique constraint ensures no duplicate (prompt_type, version_name) combinations

---

**Date**: 2025-01-27  
**Related**: [Phase 5 Implementation](./phase_5_handoff.md)


