-- Migration: Rename prompt_versions to prompts and update schema
-- This migration:
-- 1. Renames table from prompt_versions to prompts
-- 2. Changes version_id to id (UUID)
-- 3. Changes version_name to version (semantic versioning)
-- 4. Changes evaluator_type to name
-- 5. Adds live boolean field

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Step 1: Create new prompts table with updated schema
CREATE TABLE IF NOT EXISTS prompts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(50) NOT NULL,
    prompt_type VARCHAR(50) NOT NULL DEFAULT 'rag',
    name VARCHAR(50),  -- Renamed from evaluator_type, nullable for non-evaluation prompts
    prompt_text TEXT NOT NULL,
    live BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Unique constraint: (prompt_type, name, version) for evaluation prompts
    -- For non-evaluation prompts, name is NULL, so constraint is (prompt_type, version)
    CONSTRAINT prompts_prompt_type_name_version_unique UNIQUE (prompt_type, name, version)
);

-- Step 2: Migrate data from old table to new table
-- Generate UUIDs for existing records and map old columns to new
INSERT INTO prompts (id, version, prompt_type, name, prompt_text, live, created_at)
SELECT 
    uuid_generate_v4() as id,
    CASE 
        WHEN version_name LIKE 'eval_%' THEN '0.1'  -- Convert eval_*_v1 to 0.1
        WHEN version_name = 'v1' THEN '0.1'  -- Convert v1 to 0.1
        WHEN version_name IS NULL THEN '0.1'
        ELSE version_name
    END as version,
    COALESCE(prompt_type, 'rag') as prompt_type,
    evaluator_type as name,  -- evaluator_type becomes name
    prompt_text,
    CASE 
        WHEN evaluator_type IS NOT NULL THEN true  -- Mark evaluation prompts as live
        ELSE false
    END as live,  -- Evaluation prompts are live, others are not
    COALESCE(created_at, CURRENT_TIMESTAMP) as created_at
FROM prompt_versions
ON CONFLICT (prompt_type, name, version) DO NOTHING;

-- Step 3: Update foreign key references in other tables
-- Note: These tables reference prompt_versions(version_name), which maps to prompts(version)
-- We need to update the foreign keys to reference the new table

-- Update model_answers table
DO $$
BEGIN
    -- Drop old foreign key if it exists
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'model_answers_prompt_version_fkey'
    ) THEN
        ALTER TABLE model_answers DROP CONSTRAINT model_answers_prompt_version_fkey;
    END IF;
    
    -- Add new foreign key to prompts table
    -- Note: This references (prompt_type, version) since name can be NULL
    -- For now, we'll reference version only and assume prompt_type='rag' for existing data
    ALTER TABLE model_answers
    ADD CONSTRAINT model_answers_prompt_version_fkey
    FOREIGN KEY (prompt_version) REFERENCES prompts(version);
END $$;

-- Update eval_judgments table
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'eval_judgments_prompt_version_fkey'
    ) THEN
        ALTER TABLE eval_judgments DROP CONSTRAINT eval_judgments_prompt_version_fkey;
    END IF;
    
    ALTER TABLE eval_judgments
    ADD CONSTRAINT eval_judgments_prompt_version_fkey
    FOREIGN KEY (prompt_version) REFERENCES prompts(version);
END $$;

-- Update meta_eval_summaries table
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
    
    ALTER TABLE meta_eval_summaries
    ADD CONSTRAINT meta_eval_summaries_version_1_fkey
    FOREIGN KEY (version_1) REFERENCES prompts(version);
    
    ALTER TABLE meta_eval_summaries
    ADD CONSTRAINT meta_eval_summaries_version_2_fkey
    FOREIGN KEY (version_2) REFERENCES prompts(version);
END $$;

-- Step 4: Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_prompts_prompt_type ON prompts(prompt_type);
CREATE INDEX IF NOT EXISTS idx_prompts_name ON prompts(name);
CREATE INDEX IF NOT EXISTS idx_prompts_version ON prompts(version);
CREATE INDEX IF NOT EXISTS idx_prompts_live ON prompts(live);
CREATE INDEX IF NOT EXISTS idx_prompts_prompt_type_name_live ON prompts(prompt_type, name, live) WHERE live = true;

-- Step 5: Add comments
COMMENT ON TABLE prompts IS 'Stores prompt templates with semantic versioning. The live field marks the active version for a given prompt_type and name.';
COMMENT ON COLUMN prompts.id IS 'Unique identifier (UUID)';
COMMENT ON COLUMN prompts.version IS 'Semantic version (e.g., 0.1, 0.2, 1.0). First digit captures major refactors. No "v" prefix.';
COMMENT ON COLUMN prompts.prompt_type IS 'Type of prompt (e.g., "rag", "evaluation", "summarization")';
COMMENT ON COLUMN prompts.name IS 'Name/identifier for evaluation prompts (e.g., "correctness_evaluator", "hallucination_evaluator"). NULL for non-evaluation prompts.';
COMMENT ON COLUMN prompts.live IS 'Boolean flag indicating if this is the live/active version for the prompt_type and name combination';

-- Step 6: Drop old table (after data migration and FK updates)
-- Note: We keep the old table for now in case of rollback needs
-- Uncomment the following line after verifying the migration:
-- DROP TABLE IF EXISTS prompt_versions CASCADE;

