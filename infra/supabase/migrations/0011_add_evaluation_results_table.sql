-- Migration: Add evaluation_results table for logging evaluation results
-- This table stores evaluation results from the RAG evaluation pipeline
-- All metrics are stored as JSONB for flexibility and extensibility

-- Evaluation results table
CREATE TABLE IF NOT EXISTS evaluation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    example_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    judge_output JSONB NOT NULL,
    meta_eval_output JSONB NOT NULL,
    beir_metrics JSONB NOT NULL,
    judge_performance_metrics JSONB,
    metadata JSONB
);

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_evaluation_results_example_id ON evaluation_results(example_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_timestamp ON evaluation_results(timestamp);

-- Create GIN indexes on JSONB columns for efficient JSON queries
CREATE INDEX IF NOT EXISTS idx_evaluation_results_judge_output_gin ON evaluation_results USING GIN (judge_output);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_meta_eval_output_gin ON evaluation_results USING GIN (meta_eval_output);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_beir_metrics_gin ON evaluation_results USING GIN (beir_metrics);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_judge_performance_metrics_gin ON evaluation_results USING GIN (judge_performance_metrics);

-- Add comment to document the table
COMMENT ON TABLE evaluation_results IS 
'Stores evaluation results from the RAG evaluation pipeline. All metrics are stored as JSONB for flexibility and extensibility. Supports both single result and batch logging.';

COMMENT ON COLUMN evaluation_results.id IS 'Unique identifier (UUID) for the evaluation result';
COMMENT ON COLUMN evaluation_results.example_id IS 'Reference to the evaluation example that was evaluated';
COMMENT ON COLUMN evaluation_results.timestamp IS 'Timestamp when the evaluation was performed';
COMMENT ON COLUMN evaluation_results.judge_output IS 'JSONB storage of JudgeEvaluationResult (correctness, hallucination, risk_direction, risk_impact, reasoning, failure_mode)';
COMMENT ON COLUMN evaluation_results.meta_eval_output IS 'JSONB storage of MetaEvaluationResult (judge_correct, explanation, ground_truth_* fields)';
COMMENT ON COLUMN evaluation_results.beir_metrics IS 'JSONB storage of BEIRMetricsResult (recall_at_k, precision_at_k, ndcg_at_k)';
COMMENT ON COLUMN evaluation_results.judge_performance_metrics IS 'JSONB storage of JudgePerformanceMetrics (optional, calculated from batch results)';
COMMENT ON COLUMN evaluation_results.metadata IS 'Additional flexible metadata for extensibility';


