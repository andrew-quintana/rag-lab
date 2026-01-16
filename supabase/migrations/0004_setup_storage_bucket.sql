-- Setup storage bucket and RLS policies for documents

-- Create the storage bucket if it doesn't exist
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'documents',
    'documents',
    true,  -- Public bucket for easy access
    52428800,  -- 50MB limit
    NULL  -- Allow all MIME types
)
ON CONFLICT (id) DO UPDATE
SET public = true,
    file_size_limit = 52428800;

-- Note: RLS is already enabled on storage.objects by default in Supabase
-- We only need to create policies, not enable RLS

-- Drop existing policies if they exist (for idempotency)
DROP POLICY IF EXISTS "Public read access for documents" ON storage.objects;
DROP POLICY IF EXISTS "Service role upload for documents" ON storage.objects;
DROP POLICY IF EXISTS "Service role delete for documents" ON storage.objects;
DROP POLICY IF EXISTS "Service role update for documents" ON storage.objects;

-- Allow public read access to documents
CREATE POLICY "Public read access for documents"
ON storage.objects FOR SELECT
USING (bucket_id = 'documents');

-- Allow service role (and anon) to upload documents
-- This allows the backend service to upload files using either service_role or anon key
CREATE POLICY "Service role upload for documents"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'documents');

-- Allow service role to delete documents
CREATE POLICY "Service role delete for documents"
ON storage.objects FOR DELETE
USING (bucket_id = 'documents');

-- Allow service role to update documents
CREATE POLICY "Service role update for documents"
ON storage.objects FOR UPDATE
USING (bucket_id = 'documents')
WITH CHECK (bucket_id = 'documents');

