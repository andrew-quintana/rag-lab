-- Setup storage bucket and RLS policies for datasets

-- Create the storage bucket if it doesn't exist
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'datasets',
    'datasets',
    true,  -- Public bucket for easy access
    10485760,  -- 10MB limit (datasets are typically smaller than documents)
    ARRAY['application/json', 'text/json']  -- Allow JSON files
)
ON CONFLICT (id) DO UPDATE
SET public = true,
    file_size_limit = 10485760,
    allowed_mime_types = ARRAY['application/json', 'text/json'];

-- Drop existing policies if they exist (for idempotency)
DROP POLICY IF EXISTS "Public read access for datasets" ON storage.objects;
DROP POLICY IF EXISTS "Service role upload for datasets" ON storage.objects;
DROP POLICY IF EXISTS "Service role delete for datasets" ON storage.objects;
DROP POLICY IF EXISTS "Service role update for datasets" ON storage.objects;

-- Allow public read access to datasets
CREATE POLICY "Public read access for datasets"
ON storage.objects FOR SELECT
USING (bucket_id = 'datasets');

-- Allow service role (and anon) to upload datasets
-- This allows the backend service to upload files using either service_role or anon key
CREATE POLICY "Service role upload for datasets"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'datasets');

-- Allow service role to delete datasets
CREATE POLICY "Service role delete for datasets"
ON storage.objects FOR DELETE
USING (bucket_id = 'datasets');

-- Allow service role to update datasets
CREATE POLICY "Service role update for datasets"
ON storage.objects FOR UPDATE
USING (bucket_id = 'datasets')
WITH CHECK (bucket_id = 'datasets');

