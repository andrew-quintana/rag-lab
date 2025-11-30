-- Migration: Add documents table for tracking uploaded documents
-- This table stores metadata about documents uploaded to Supabase Storage

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR(255) PRIMARY KEY,
    filename VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100),
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'uploaded',
    chunks_created INTEGER,
    storage_path VARCHAR(500) NOT NULL,
    preview_image_path VARCHAR(500),
    metadata JSONB
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_documents_timestamp ON documents(upload_timestamp);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_mime_type ON documents(mime_type);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);

-- Add comment to document the table
COMMENT ON TABLE documents IS 
'Stores metadata about documents uploaded to Supabase Storage. Tracks file information, processing status, and storage paths.';

