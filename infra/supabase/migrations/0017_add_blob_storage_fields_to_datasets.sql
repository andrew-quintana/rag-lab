-- Migration: Add blob storage fields to datasets table
-- This migration adds fields needed to track datasets stored in Supabase Storage

-- Add new columns for blob storage
ALTER TABLE datasets
ADD COLUMN IF NOT EXISTS filename VARCHAR(500),
ADD COLUMN IF NOT EXISTS file_size BIGINT,
ADD COLUMN IF NOT EXISTS mime_type VARCHAR(100) DEFAULT 'application/json',
ADD COLUMN IF NOT EXISTS storage_path VARCHAR(500),
ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'uploaded';

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status);
CREATE INDEX IF NOT EXISTS idx_datasets_filename ON datasets(filename);

-- Update comments
COMMENT ON COLUMN datasets.filename IS 'Original filename of the dataset file';
COMMENT ON COLUMN datasets.file_size IS 'Size of the dataset file in bytes';
COMMENT ON COLUMN datasets.mime_type IS 'MIME type of the dataset file (typically application/json)';
COMMENT ON COLUMN datasets.storage_path IS 'Path to the dataset file in Supabase Storage';
COMMENT ON COLUMN datasets.status IS 'Status of the dataset (e.g., uploaded, processing, ready, error)';

-- Update table comment
COMMENT ON TABLE datasets IS 
'Stores metadata about evaluation datasets stored in Supabase Storage. The actual dataset file (JSON) is stored in blob storage, this table tracks metadata and storage path.';

