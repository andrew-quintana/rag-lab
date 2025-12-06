-- Migration: Add persistence infrastructure for worker-queue architecture
-- This migration adds columns and tables needed for storing intermediate data
-- between pipeline stages (extracted text, chunks, embeddings, status tracking)

-- Add status column to documents table if not exists (may already exist from 0003)
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'uploaded';

-- Add timestamp columns for each pipeline stage
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parsed_at TIMESTAMP;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunked_at TIMESTAMP;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedded_at TIMESTAMP;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMP;

-- Add extracted_text column to documents table for storing extracted text
-- RFC recommends starting with database column; can migrate to storage if size becomes issue
ALTER TABLE documents ADD COLUMN IF NOT EXISTS extracted_text TEXT;

-- Create chunks table for storing chunks and embeddings
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    metadata JSONB,
    embedding JSONB,  -- Store embeddings as JSONB array of floats
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- Add comments to document the schema
COMMENT ON COLUMN documents.status IS 'Document processing status: uploaded, parsed, chunked, embedded, indexed, or failed_*';
COMMENT ON COLUMN documents.extracted_text IS 'Extracted text from document (stored in database; can migrate to storage if size becomes issue)';
COMMENT ON COLUMN documents.parsed_at IS 'Timestamp when text extraction completed';
COMMENT ON COLUMN documents.chunked_at IS 'Timestamp when chunking completed';
COMMENT ON COLUMN documents.embedded_at IS 'Timestamp when embedding generation completed';
COMMENT ON COLUMN documents.indexed_at IS 'Timestamp when indexing completed';
COMMENT ON TABLE chunks IS 'Stores chunks and embeddings for documents. Each chunk belongs to a document and can have an associated embedding vector.';
COMMENT ON COLUMN chunks.embedding IS 'Embedding vector stored as JSONB array of floats';

