-- Migration: Add meta_eval JSONB column to meta_eval_summaries table
-- This migration adds a JSONB column to store the full MetaEvaluationResult structure
-- as scoped in Phase 11, including judge_correct, explanation, and ground truth fields

-- Add meta_eval JSONB column
ALTER TABLE meta_eval_summaries
ADD COLUMN IF NOT EXISTS meta_eval JSONB;

-- Create GIN index on JSONB column for efficient JSON queries
CREATE INDEX IF NOT EXISTS idx_meta_eval_summaries_meta_eval_gin 
ON meta_eval_summaries USING GIN (meta_eval);

-- Add comment to document the column
COMMENT ON COLUMN meta_eval_summaries.meta_eval IS 
'JSONB storage of MetaEvaluationResult (judge_correct, explanation, ground_truth_correctness, ground_truth_hallucination, ground_truth_risk_direction, ground_truth_risk_impact)';

