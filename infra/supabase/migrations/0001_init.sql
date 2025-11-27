-- Initial database schema for RAG Evaluation Platform

-- Prompt versions table
CREATE TABLE IF NOT EXISTS prompt_versions (
    version_id VARCHAR(255) PRIMARY KEY,
    version_name VARCHAR(100) NOT NULL UNIQUE,
    prompt_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Queries table
CREATE TABLE IF NOT EXISTS queries (
    query_id VARCHAR(255) PRIMARY KEY,
    query_text TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Retrieval logs table
CREATE TABLE IF NOT EXISTS retrieval_logs (
    log_id VARCHAR(255) PRIMARY KEY,
    query_id VARCHAR(255) NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
    chunk_id VARCHAR(255) NOT NULL,
    similarity_score FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model answers table
CREATE TABLE IF NOT EXISTS model_answers (
    answer_id VARCHAR(255) PRIMARY KEY,
    query_id VARCHAR(255) NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
    answer_text TEXT NOT NULL,
    prompt_version VARCHAR(100) NOT NULL REFERENCES prompt_versions(version_name),
    retrieved_chunk_ids TEXT[] NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation judgments table
CREATE TABLE IF NOT EXISTS eval_judgments (
    judgment_id VARCHAR(255) PRIMARY KEY,
    query_id VARCHAR(255) NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
    prompt_version VARCHAR(100) NOT NULL REFERENCES prompt_versions(version_name),
    grounding_score FLOAT NOT NULL CHECK (grounding_score >= 0 AND grounding_score <= 1),
    relevance_score FLOAT NOT NULL CHECK (relevance_score >= 0 AND relevance_score <= 1),
    hallucination_risk FLOAT NOT NULL CHECK (hallucination_risk >= 0 AND hallucination_risk <= 1),
    judge_reasoning TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Meta-evaluation summaries table
CREATE TABLE IF NOT EXISTS meta_eval_summaries (
    summary_id VARCHAR(255) PRIMARY KEY,
    version_1 VARCHAR(100) NOT NULL REFERENCES prompt_versions(version_name),
    version_2 VARCHAR(100) NOT NULL REFERENCES prompt_versions(version_name),
    delta_grounding FLOAT NOT NULL,
    delta_relevance FLOAT NOT NULL,
    delta_hallucination FLOAT NOT NULL,
    judge_consistency FLOAT NOT NULL CHECK (judge_consistency >= 0 AND judge_consistency <= 1),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp);
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_query_id ON retrieval_logs(query_id);
CREATE INDEX IF NOT EXISTS idx_model_answers_query_id ON model_answers(query_id);
CREATE INDEX IF NOT EXISTS idx_model_answers_prompt_version ON model_answers(prompt_version);
CREATE INDEX IF NOT EXISTS idx_eval_judgments_query_id ON eval_judgments(query_id);
CREATE INDEX IF NOT EXISTS idx_eval_judgments_prompt_version ON eval_judgments(prompt_version);
CREATE INDEX IF NOT EXISTS idx_meta_eval_summaries_versions ON meta_eval_summaries(version_1, version_2);

