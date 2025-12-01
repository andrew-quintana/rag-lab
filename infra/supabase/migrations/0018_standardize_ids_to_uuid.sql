-- Migration: Standardize all tables to use 'id' UUID primary key
-- Remove judge_reasoning from eval_judgments (already in judge_output JSONB)
-- This migration updates all tables to use consistent 'id UUID' primary keys

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- 0. PROMPT_VERSIONS: Change version_id to id UUID (if table still exists)
-- ============================================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'prompt_versions'
    ) AND EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'prompt_versions' AND column_name = 'version_id'
    ) THEN
        ALTER TABLE prompt_versions ADD COLUMN id UUID DEFAULT uuid_generate_v4();
        UPDATE prompt_versions SET id = version_id::uuid WHERE version_id IS NOT NULL;
        ALTER TABLE prompt_versions DROP CONSTRAINT IF EXISTS prompt_versions_pkey;
        ALTER TABLE prompt_versions ADD PRIMARY KEY (id);
        ALTER TABLE prompt_versions DROP COLUMN version_id;
    END IF;
END $$;

-- ============================================================================
-- 1. EVAL_JUDGMENTS: Remove judge_reasoning, change judgment_id to id UUID
-- ============================================================================
DO $$
BEGIN
    -- Drop judge_reasoning column
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'eval_judgments' AND column_name = 'judge_reasoning'
    ) THEN
        ALTER TABLE eval_judgments DROP COLUMN judge_reasoning;
    END IF;
    
    -- Change judgment_id to id UUID
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'eval_judgments' AND column_name = 'judgment_id'
    ) THEN
        -- Add new id column
        ALTER TABLE eval_judgments ADD COLUMN id UUID DEFAULT uuid_generate_v4();
        
        -- Copy data from judgment_id to id
        UPDATE eval_judgments SET id = judgment_id::uuid WHERE judgment_id IS NOT NULL;
        
        -- Drop old primary key constraint
        ALTER TABLE eval_judgments DROP CONSTRAINT IF EXISTS eval_judgments_pkey;
        
        -- Set id as primary key
        ALTER TABLE eval_judgments ADD PRIMARY KEY (id);
        
        -- Drop old judgment_id column
        ALTER TABLE eval_judgments DROP COLUMN judgment_id;
    END IF;
END $$;

-- ============================================================================
-- 2. QUERIES: Change query_id to id UUID
-- ============================================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'queries' AND column_name = 'query_id'
    ) THEN
        -- Add new id column
        ALTER TABLE queries ADD COLUMN id UUID DEFAULT uuid_generate_v4();
        
        -- Copy data from query_id to id
        UPDATE queries SET id = query_id::uuid WHERE query_id IS NOT NULL;
        
        -- Update foreign key references in other tables
        -- retrieval_logs
        IF EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'retrieval_logs' AND column_name = 'query_id'
        ) THEN
            ALTER TABLE retrieval_logs ADD COLUMN query_id_new UUID;
            UPDATE retrieval_logs rl SET query_id_new = q.id 
            FROM queries q WHERE q.query_id::text = rl.query_id::text;
            ALTER TABLE retrieval_logs DROP COLUMN query_id;
            ALTER TABLE retrieval_logs RENAME COLUMN query_id_new TO query_id;
        END IF;
        
        -- model_answers
        IF EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'model_answers' AND column_name = 'query_id'
        ) THEN
            ALTER TABLE model_answers ADD COLUMN query_id_new UUID;
            UPDATE model_answers ma SET query_id_new = q.id 
            FROM queries q WHERE q.query_id::text = ma.query_id::text;
            ALTER TABLE model_answers DROP COLUMN query_id;
            ALTER TABLE model_answers RENAME COLUMN query_id_new TO query_id;
        END IF;
        
        -- eval_judgments
        IF EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'eval_judgments' AND column_name = 'query_id'
        ) THEN
            ALTER TABLE eval_judgments ADD COLUMN query_id_new UUID;
            UPDATE eval_judgments ej SET query_id_new = q.id 
            FROM queries q WHERE q.query_id::text = ej.query_id::text;
            ALTER TABLE eval_judgments DROP COLUMN query_id;
            ALTER TABLE eval_judgments RENAME COLUMN query_id_new TO query_id;
        END IF;
        
        -- Drop old primary key and foreign keys
        ALTER TABLE queries DROP CONSTRAINT IF EXISTS queries_pkey;
        ALTER TABLE queries DROP CONSTRAINT IF EXISTS retrieval_logs_query_id_fkey;
        ALTER TABLE queries DROP CONSTRAINT IF EXISTS model_answers_query_id_fkey;
        ALTER TABLE queries DROP CONSTRAINT IF EXISTS eval_judgments_query_id_fkey;
        
        -- Set id as primary key
        ALTER TABLE queries ADD PRIMARY KEY (id);
        
        -- Recreate foreign keys
        ALTER TABLE retrieval_logs 
        ADD CONSTRAINT retrieval_logs_query_id_fkey 
        FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE;
        
        ALTER TABLE model_answers 
        ADD CONSTRAINT model_answers_query_id_fkey 
        FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE;
        
        ALTER TABLE eval_judgments 
        ADD CONSTRAINT eval_judgments_query_id_fkey 
        FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE;
        
        -- Drop old query_id column
        ALTER TABLE queries DROP COLUMN query_id;
    END IF;
END $$;

-- ============================================================================
-- 3. RETRIEVAL_LOGS: Change log_id to id UUID
-- ============================================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'retrieval_logs' AND column_name = 'log_id'
    ) THEN
        ALTER TABLE retrieval_logs ADD COLUMN id UUID DEFAULT uuid_generate_v4();
        UPDATE retrieval_logs SET id = log_id::uuid WHERE log_id IS NOT NULL;
        ALTER TABLE retrieval_logs DROP CONSTRAINT IF EXISTS retrieval_logs_pkey;
        ALTER TABLE retrieval_logs ADD PRIMARY KEY (id);
        ALTER TABLE retrieval_logs DROP COLUMN log_id;
    END IF;
END $$;

-- ============================================================================
-- 4. MODEL_ANSWERS: Change answer_id to id UUID
-- ============================================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'model_answers' AND column_name = 'answer_id'
    ) THEN
        ALTER TABLE model_answers ADD COLUMN id UUID DEFAULT uuid_generate_v4();
        UPDATE model_answers SET id = answer_id::uuid WHERE answer_id IS NOT NULL;
        ALTER TABLE model_answers DROP CONSTRAINT IF EXISTS model_answers_pkey;
        ALTER TABLE model_answers ADD PRIMARY KEY (id);
        ALTER TABLE model_answers DROP COLUMN answer_id;
    END IF;
END $$;

-- ============================================================================
-- 5. DOCUMENTS: Change document_id to id UUID
-- ============================================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'document_id'
    ) THEN
        ALTER TABLE documents ADD COLUMN id UUID DEFAULT uuid_generate_v4();
        UPDATE documents SET id = document_id::uuid WHERE document_id IS NOT NULL;
        ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_pkey;
        ALTER TABLE documents ADD PRIMARY KEY (id);
        ALTER TABLE documents DROP COLUMN document_id;
    END IF;
END $$;

-- ============================================================================
-- 6. DATASETS: Change dataset_id to id UUID
-- ============================================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'datasets' AND column_name = 'dataset_id'
    ) THEN
        -- Update foreign key references in meta_eval_summaries
        IF EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'meta_eval_summaries' AND column_name = 'dataset_id'
        ) THEN
            ALTER TABLE meta_eval_summaries ADD COLUMN dataset_id_new UUID;
            UPDATE meta_eval_summaries mes SET dataset_id_new = d.dataset_id::uuid 
            FROM datasets d WHERE d.dataset_id::text = mes.dataset_id::text;
            ALTER TABLE meta_eval_summaries DROP CONSTRAINT IF EXISTS meta_eval_summaries_dataset_id_fkey;
            ALTER TABLE meta_eval_summaries DROP COLUMN dataset_id;
            ALTER TABLE meta_eval_summaries RENAME COLUMN dataset_id_new TO dataset_id;
        END IF;
        
        ALTER TABLE datasets ADD COLUMN id UUID DEFAULT uuid_generate_v4();
        UPDATE datasets SET id = dataset_id::uuid WHERE dataset_id IS NOT NULL;
        ALTER TABLE datasets DROP CONSTRAINT IF EXISTS datasets_pkey;
        ALTER TABLE datasets ADD PRIMARY KEY (id);
        
        -- Recreate foreign key
        ALTER TABLE meta_eval_summaries 
        ADD CONSTRAINT meta_eval_summaries_dataset_id_fkey 
        FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE;
        
        ALTER TABLE datasets DROP COLUMN dataset_id;
    END IF;
END $$;

-- ============================================================================
-- 7. EVALUATION_RESULTS: Change result_id to id UUID
-- ============================================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'evaluation_results' AND column_name = 'result_id'
    ) THEN
        ALTER TABLE evaluation_results ADD COLUMN id UUID DEFAULT uuid_generate_v4();
        UPDATE evaluation_results SET id = result_id::uuid WHERE result_id IS NOT NULL;
        ALTER TABLE evaluation_results DROP CONSTRAINT IF EXISTS evaluation_results_pkey;
        ALTER TABLE evaluation_results ADD PRIMARY KEY (id);
        ALTER TABLE evaluation_results DROP COLUMN result_id;
    END IF;
END $$;

-- ============================================================================
-- 8. META_EVAL_SUMMARIES: Change summary_id to id UUID, remove timestamp/created_at
-- ============================================================================
DO $$
BEGIN
    -- Drop timestamp/created_at columns if they exist
    ALTER TABLE meta_eval_summaries
    DROP COLUMN IF EXISTS timestamp,
    DROP COLUMN IF EXISTS created_at;
    
    -- Drop index on created_at if it exists
    DROP INDEX IF EXISTS idx_meta_eval_summaries_created_at;
    
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'meta_eval_summaries' AND column_name = 'summary_id'
    ) THEN
        ALTER TABLE meta_eval_summaries ADD COLUMN id UUID DEFAULT uuid_generate_v4();
        UPDATE meta_eval_summaries SET id = summary_id::uuid WHERE summary_id IS NOT NULL;
        ALTER TABLE meta_eval_summaries DROP CONSTRAINT IF EXISTS meta_eval_summaries_pkey;
        ALTER TABLE meta_eval_summaries ADD PRIMARY KEY (id);
        ALTER TABLE meta_eval_summaries DROP COLUMN summary_id;
    END IF;
END $$;

-- Update comments
COMMENT ON COLUMN eval_judgments.id IS 'Unique identifier (UUID) for the evaluation judgment';
COMMENT ON COLUMN queries.id IS 'Unique identifier (UUID) for the query';
COMMENT ON COLUMN retrieval_logs.id IS 'Unique identifier (UUID) for the retrieval log';
COMMENT ON COLUMN model_answers.id IS 'Unique identifier (UUID) for the model answer';
COMMENT ON COLUMN documents.id IS 'Unique identifier (UUID) for the document';
COMMENT ON COLUMN datasets.id IS 'Unique identifier (UUID) for the dataset';
COMMENT ON COLUMN evaluation_results.id IS 'Unique identifier (UUID) for the evaluation result';
COMMENT ON COLUMN meta_eval_summaries.id IS 'Unique identifier (UUID) for the meta-evaluation summary';

