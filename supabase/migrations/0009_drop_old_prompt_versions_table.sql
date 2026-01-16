-- Migration: Drop old prompt_versions table
-- This migration removes the old prompt_versions table after data has been migrated to prompts table
-- 
-- WARNING: This migration permanently deletes the old table. Only run this after:
-- 1. Migration 0007 has successfully migrated all data
-- 2. Migration 0008 has inserted new prompts
-- 3. All foreign key references have been updated
-- 4. System has been tested and verified working with new schema

-- Drop the old prompt_versions table
-- Note: CASCADE will drop dependent objects if any remain
DROP TABLE IF EXISTS prompt_versions CASCADE;

-- Verify the table is gone
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'prompt_versions'
    ) THEN
        RAISE EXCEPTION 'prompt_versions table still exists after DROP';
    END IF;
END $$;


