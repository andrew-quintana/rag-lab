-- Migration: Add prompt_type to prompt_versions table
-- This allows different types of prompts to be tracked separately
-- e.g., "rag", "evaluation", "summarization", etc.

-- Add prompt_type column with default value 'rag' for backward compatibility
ALTER TABLE prompt_versions 
ADD COLUMN IF NOT EXISTS prompt_type VARCHAR(50) NOT NULL DEFAULT 'rag';

-- NOTE: We keep the unique constraint on version_name for foreign key compatibility.
-- This means version_name must still be unique across all types.
-- For full multi-type support with same version names, you would need to:
-- 1. Drop foreign key constraints on model_answers, eval_judgments, meta_eval_summaries
-- 2. Drop the unique constraint on version_name
-- 3. Rely only on the composite unique constraint (prompt_type, version_name)
-- 4. Update foreign keys to reference (prompt_type, version_name) if needed

-- Add new unique constraint on (prompt_type, version_name)
-- This allows same version_name for different prompt types (if version_name constraint is removed)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'prompt_versions_prompt_type_version_name_unique'
    ) THEN
        ALTER TABLE prompt_versions 
        ADD CONSTRAINT prompt_versions_prompt_type_version_name_unique 
        UNIQUE (prompt_type, version_name);
    END IF;
END $$;

-- Create index for faster lookups by prompt_type
CREATE INDEX IF NOT EXISTS idx_prompt_versions_prompt_type 
ON prompt_versions(prompt_type);

-- Add comment to document the field
COMMENT ON COLUMN prompt_versions.prompt_type IS 
'Type of prompt (e.g., "rag", "evaluation", "summarization"). Allows different prompt types to have the same version_name if the version_name unique constraint is removed.';

