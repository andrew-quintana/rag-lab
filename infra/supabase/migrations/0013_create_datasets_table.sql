-- Migration: Create datasets table for tracking evaluation datasets
-- This table stores metadata about evaluation datasets stored in blob storage
-- The actual dataset file (JSON) is stored in Supabase Storage, this table tracks metadata

-- Datasets table
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name VARCHAR(255) NOT NULL,
    filename VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100) DEFAULT 'application/json',
    storage_path VARCHAR(500) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'uploaded',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(dataset_name);
CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at);
CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status);
CREATE INDEX IF NOT EXISTS idx_datasets_filename ON datasets(filename);

-- Add comment to document the table
COMMENT ON TABLE datasets IS 
'Stores metadata about evaluation datasets stored in Supabase Storage. The actual dataset file (JSON) is stored in blob storage, this table tracks metadata and storage path.';

COMMENT ON COLUMN datasets.id IS 'Unique identifier (UUID) for the dataset';
COMMENT ON COLUMN datasets.dataset_name IS 'Human-readable name for the dataset';
COMMENT ON COLUMN datasets.filename IS 'Original filename of the dataset file';
COMMENT ON COLUMN datasets.file_size IS 'Size of the dataset file in bytes';
COMMENT ON COLUMN datasets.mime_type IS 'MIME type of the dataset file (typically application/json)';
COMMENT ON COLUMN datasets.storage_path IS 'Path to the dataset file in Supabase Storage';
COMMENT ON COLUMN datasets.description IS 'Optional description of the dataset';
COMMENT ON COLUMN datasets.status IS 'Status of the dataset (e.g., uploaded, processing, ready, error)';
COMMENT ON COLUMN datasets.created_at IS 'Timestamp when the dataset was created/uploaded';
COMMENT ON COLUMN datasets.metadata IS 'Additional flexible metadata for the dataset';

