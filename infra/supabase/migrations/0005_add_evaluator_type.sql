-- Migration: Add evaluator_type to prompt_versions table
-- This allows specific evaluator nodes to be identified within the "evaluation" prompt_type
-- e.g., "correctness_evaluator", "hallucination_evaluator", "risk_direction_evaluator"

-- Add evaluator_type column (nullable, only used for evaluation prompts)
ALTER TABLE prompt_versions 
ADD COLUMN IF NOT EXISTS evaluator_type VARCHAR(50);

-- Drop the old unique constraint on (prompt_type, version_name) if it exists
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'prompt_versions_prompt_type_version_name_unique'
    ) THEN
        ALTER TABLE prompt_versions
        DROP CONSTRAINT prompt_versions_prompt_type_version_name_unique;
    END IF;
END $$;

-- Add new unique constraint on (prompt_type, evaluator_type, version_name)
-- This allows same version_name for different prompt types AND different evaluator types
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'prompt_versions_prompt_type_evaluator_type_version_name_unique'
    ) THEN
        ALTER TABLE prompt_versions
        ADD CONSTRAINT prompt_versions_prompt_type_evaluator_type_version_name_unique
        UNIQUE (prompt_type, evaluator_type, version_name);
    END IF;
END $$;

-- Create index for faster lookups by evaluator_type
CREATE INDEX IF NOT EXISTS idx_prompt_versions_evaluator_type 
ON prompt_versions(evaluator_type);

-- Add comment to document the field
COMMENT ON COLUMN prompt_versions.evaluator_type IS 
'Type of evaluator node for evaluation prompts (e.g., "correctness_evaluator", "hallucination_evaluator", "risk_direction_evaluator"). Only used when prompt_type="evaluation". NULL for other prompt types.';

