-- Migration: Document ingestion batch processing metadata structure
-- This migration documents the metadata structure used for batch processing
-- No schema changes are needed - we use existing chunks table and documents.metadata JSONB

-- The ingestion metadata is stored in documents.metadata->'ingestion' JSONB field
-- Structure:
-- {
--   "ingestion": {
--     "num_pages": 0,
--     "num_batches_total": 0,
--     "last_successful_page": 0,
--     "next_unparsed_batch_index": 0,
--     "parsing_status": "pending" | "in_progress" | "completed" | "failed",
--     "batch_size": 2,
--     "batches_completed": {},
--     "parsing_started_at": null,
--     "parsing_completed_at": null,
--     "errors": []
--   }
-- }

-- Batch results are temporarily stored in chunks table with chunk_id = "batch_XXX"
-- These are cleaned up immediately after being merged into extracted_text

-- Add comment to document the metadata structure
COMMENT ON COLUMN documents.metadata IS 
'JSONB metadata field. For ingestion batch processing, stores progress in metadata->ingestion with structure: {num_pages, num_batches_total, last_successful_page, next_unparsed_batch_index, parsing_status, batch_size, batches_completed, parsing_started_at, parsing_completed_at, errors}';

COMMENT ON TABLE chunks IS 
'Stores chunks and embeddings for documents. Also temporarily stores batch extraction results with chunk_id LIKE "batch_%" for resumable processing.';

