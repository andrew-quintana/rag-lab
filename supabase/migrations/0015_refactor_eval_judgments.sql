-- Migration: Refactor eval_judgments table
-- Remove old score columns (grounding_score, relevance_score, hallucination_risk)
-- Add judge_output JSONB column to store full JudgeEvaluationResult
-- Rename prompt_version to prompt_version_id and timestamp to created_at

-- Step 1: Drop old columns if they exist
ALTER TABLE eval_judgments
DROP COLUMN IF EXISTS grounding_score,
DROP COLUMN IF EXISTS relevance_score,
DROP COLUMN IF EXISTS hallucination_risk,
DROP COLUMN IF EXISTS judge_reasoning;

-- Step 2: Rename columns
ALTER TABLE eval_judgments
RENAME COLUMN prompt_version TO prompt_version_id;

ALTER TABLE eval_judgments
RENAME COLUMN timestamp TO created_at;

-- Step 3: Add judge_output JSONB column
ALTER TABLE eval_judgments
ADD COLUMN IF NOT EXISTS judge_output JSONB;

-- Step 4: Create GIN index on JSONB column for efficient JSON queries
CREATE INDEX IF NOT EXISTS idx_eval_judgments_judge_output_gin 
ON eval_judgments USING GIN (judge_output);

-- Step 5: Update indexes (rename if needed, or create new ones)
-- The existing index on prompt_version will need to be recreated for prompt_version_id
DROP INDEX IF EXISTS idx_eval_judgments_prompt_version;

CREATE INDEX IF NOT EXISTS idx_eval_judgments_prompt_version_id 
ON eval_judgments(prompt_version_id);

-- Update timestamp index to created_at
DROP INDEX IF EXISTS idx_eval_judgments_timestamp;

CREATE INDEX IF NOT EXISTS idx_eval_judgments_created_at 
ON eval_judgments(created_at);

-- Step 6: Update foreign key constraint for prompt_version_id
-- Note: prompt_version_id stores version strings (VARCHAR), not UUIDs
-- Since prompts.id is UUID and prompts.version is not unique, we cannot create a foreign key
-- The application code will handle validation of prompt_version_id references
DO $$
BEGIN
    -- Drop old foreign key if it exists
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'eval_judgments_prompt_version_fkey'
    ) THEN
        ALTER TABLE eval_judgments DROP CONSTRAINT eval_judgments_prompt_version_fkey;
    END IF;
END $$;

-- Step 8: Change judgment_id to id UUID (if not already done by migration 0018)
-- This is handled by migration 0018_standardize_ids_to_uuid.sql
-- Keeping this migration focused on removing old columns and adding judge_output

-- Step 9: Update comments
COMMENT ON TABLE eval_judgments IS 
'Stores evaluation judgments with full JudgeEvaluationResult JSON. Tracks prompt version and query.';

COMMENT ON COLUMN eval_judgments.id IS 'Unique identifier (UUID) for the evaluation judgment';
COMMENT ON COLUMN eval_judgments.query_id IS 'Reference to the query that was evaluated (UUID)';
COMMENT ON COLUMN eval_judgments.prompt_version_id IS 'Reference to the prompt version used in evaluation';
COMMENT ON COLUMN eval_judgments.created_at IS 'Timestamp when the judgment was created';
COMMENT ON COLUMN eval_judgments.judge_output IS 
'JSONB storage of JudgeEvaluationResult (correctness_binary, hallucination_binary, risk_direction, risk_impact, reasoning, failure_mode)';

