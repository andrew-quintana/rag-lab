# Supabase Local Development Setup

This directory contains Supabase configuration and migrations for local development.

## Prerequisites

- Docker and Docker Compose installed
- Supabase CLI installed: `brew install supabase/tap/supabase` (macOS) or see [Supabase CLI docs](https://supabase.com/docs/guides/cli)

## Initial Setup

1. Start Supabase locally:
   ```bash
   supabase start
   ```

2. Run migrations:
   ```bash
   supabase db reset
   ```

## Database Connection

After starting Supabase, you'll get connection details. Use these in your `.env` file:

- Database URL: `postgresql://postgres:postgres@localhost:54322/postgres`
- Supabase URL: `http://localhost:54321`
- Supabase Key: Check output of `supabase start`

## Useful Commands

- `supabase start` - Start local Supabase instance
- `supabase stop` - Stop local Supabase instance
- `supabase db reset` - Reset database and run migrations
- `supabase db push` - Push local migrations to remote (if linked)
- `supabase status` - Check status of local instance

## Schema Migrations

Migrations are stored in `migrations/` directory. They are automatically applied when you run `supabase db reset` or `supabase start`.

## Seed Data

Demo data is in `seed/demo_data.sql` and is automatically loaded when resetting the database.

## Storage Bucket Setup

The application uses Supabase Storage to store uploaded documents. You need to create a storage bucket named "documents" for the document management system to work.

### Creating the Storage Bucket

1. **Using Supabase Studio (Recommended for local development)**:
   - Start Supabase: `supabase start`
   - Open Supabase Studio: `http://localhost:54323`
   - Navigate to Storage section
   - Click "New bucket"
   - Name: `documents`
   - Public bucket: **Yes** (for preview images and downloads)
   - Click "Create bucket"

2. **Using Supabase CLI**:
   ```bash
   # Create the bucket
   supabase storage create documents --public
   ```

3. **Using SQL (via Supabase Studio SQL Editor or psql)**:
   ```sql
   -- Create the storage bucket
   INSERT INTO storage.buckets (id, name, public)
   VALUES ('documents', 'documents', true)
   ON CONFLICT (id) DO NOTHING;
   ```

### Storage Policies

For the documents bucket to work properly, you may need to set up storage policies. By default, if the bucket is public, files can be accessed via public URLs. For more control, you can set up Row Level Security (RLS) policies:

```sql
-- Allow public read access to documents
CREATE POLICY "Public read access"
ON storage.objects FOR SELECT
USING (bucket_id = 'documents');

-- Allow authenticated users to upload
CREATE POLICY "Authenticated upload"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'documents');

-- Allow authenticated users to delete their own files
CREATE POLICY "Authenticated delete"
ON storage.objects FOR DELETE
USING (bucket_id = 'documents');
```

**Note**: For local development, public buckets are typically sufficient. For production, implement proper authentication and RLS policies.

### Verifying Storage Setup

After creating the bucket, verify it exists:

```bash
# List all buckets
supabase storage list
```

You should see the `documents` bucket in the list.

