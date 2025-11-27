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

