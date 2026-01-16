-- Migration: Refactor meta_eval_summaries table
-- Remove old columns (delta_grounding, delta_relevance, delta_hallucination, judge_consistency, version_1, version_2, timestamp)
-- Add new columns (prompt_version_id, dataset_id)
-- Keep meta_eval JSONB column

-- Step 1: Drop old columns if they exist
ALTER TABLE meta_eval_summaries
DROP COLUMN IF EXISTS delta_grounding,
DROP COLUMN IF EXISTS delta_relevance,
DROP COLUMN IF EXISTS delta_hallucination,
DROP COLUMN IF EXISTS judge_consistency,
DROP COLUMN IF EXISTS version_1,
DROP COLUMN IF EXISTS version_2,
DROP COLUMN IF EXISTS timestamp,
DROP COLUMN IF EXISTS created_at;

-- Step 2: Drop old foreign key constraints if they exist
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'meta_eval_summaries_version_1_fkey'
    ) THEN
        ALTER TABLE meta_eval_summaries DROP CONSTRAINT meta_eval_summaries_version_1_fkey;
    END IF;
    
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'meta_eval_summaries_version_2_fkey'
    ) THEN
        ALTER TABLE meta_eval_summaries DROP CONSTRAINT meta_eval_summaries_version_2_fkey;
    END IF;
END $$;

-- Step 3: Drop old index if it exists
DROP INDEX IF EXISTS idx_meta_eval_summaries_versions;

-- Step 4: Add new columns
ALTER TABLE meta_eval_summaries
ADD COLUMN IF NOT EXISTS prompt_version_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS dataset_id UUID;

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Step 5: Add foreign key constraints
ALTER TABLE meta_eval_summaries
ADD CONSTRAINT meta_eval_summaries_dataset_id_fkey
FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE;

-- Note: prompt_version_id stores version strings (VARCHAR), not UUIDs
-- Since prompts.id is UUID and prompts.version is not unique, we cannot create a foreign key
-- The application code will handle validation of prompt_version_id references
COMMENT ON COLUMN meta_eval_summaries.prompt_version_id IS 
'References prompt version string. Application code validates this reference since version is not unique in prompts table.';

-- Step 6: Make new columns NOT NULL (after data migration if needed)
-- We'll make them NOT NULL in a separate step to allow for existing data
-- For new tables, you can add NOT NULL directly

-- Step 7: Create new indexes
CREATE INDEX IF NOT EXISTS idx_meta_eval_summaries_prompt_version_id 
ON meta_eval_summaries(prompt_version_id);

CREATE INDEX IF NOT EXISTS idx_meta_eval_summaries_dataset_id 
ON meta_eval_summaries(dataset_id);

-- Step 8: Update comments
COMMENT ON TABLE meta_eval_summaries IS 
'Stores meta-evaluation summaries with full MetaEvaluationResult JSON. Tracks prompt version and dataset used.';

COMMENT ON COLUMN meta_eval_summaries.id IS 'Unique identifier (UUID) for the meta-evaluation summary';
COMMENT ON COLUMN meta_eval_summaries.prompt_version_id IS 'Reference to the prompt version used in evaluation';
COMMENT ON COLUMN meta_eval_summaries.dataset_id IS 'Reference to the evaluation dataset used (UUID)';
COMMENT ON COLUMN meta_eval_summaries.meta_eval IS 
'JSONB storage of MetaEvaluationResult (judge_correct, explanation, ground_truth_correctness, ground_truth_hallucination, ground_truth_risk_direction, ground_truth_risk_impact)';

