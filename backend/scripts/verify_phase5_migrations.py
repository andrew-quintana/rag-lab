#!/usr/bin/env python3
"""Verification script for Phase 5 database migrations

This script verifies that migrations 0019 and 0020 have been applied correctly
to the Supabase database. It checks for:
- Status and timestamp columns in documents table
- Extracted_text column in documents table
- Chunks table existence and structure
- Indexes on documents.status and chunks.document_id
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from src.core.config import Config
from src.db.connection import DatabaseConnection
from typing import List, Tuple


def check_column_exists(
    cursor, table_name: str, column_name: str
) -> Tuple[bool, dict]:
    """Check if a column exists in a table"""
    cursor.execute("""
        SELECT column_name, data_type, column_default, is_nullable
        FROM information_schema.columns
        WHERE table_name = %s AND column_name = %s
    """, (table_name, column_name))
    
    result = cursor.fetchone()
    if result:
        return True, {
            'column_name': result[0],
            'data_type': result[1],
            'column_default': result[2],
            'is_nullable': result[3]
        }
    return False, {}


def check_table_exists(cursor, table_name: str) -> bool:
    """Check if a table exists"""
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = %s
    """, (table_name,))
    return cursor.fetchone() is not None


def check_index_exists(cursor, index_name: str) -> bool:
    """Check if an index exists"""
    cursor.execute("""
        SELECT indexname
        FROM pg_indexes
        WHERE indexname = %s
    """, (index_name,))
    return cursor.fetchone() is not None


def verify_migration_0019(cursor) -> Tuple[bool, List[str]]:
    """Verify migration 0019: Worker Queue Persistence"""
    errors = []
    all_passed = True
    
    print("\n📋 Verifying Migration 0019: Worker Queue Persistence")
    print("=" * 60)
    
    # Check status column
    print("\n1. Checking status column in documents table...")
    exists, info = check_column_exists(cursor, 'documents', 'status')
    if exists:
        print(f"   ✅ status column exists: {info['data_type']} (default: {info['column_default']})")
    else:
        print("   ❌ status column missing")
        errors.append("documents.status column missing")
        all_passed = False
    
    # Check timestamp columns
    timestamp_columns = ['parsed_at', 'chunked_at', 'embedded_at', 'indexed_at']
    print("\n2. Checking timestamp columns in documents table...")
    for col in timestamp_columns:
        exists, info = check_column_exists(cursor, 'documents', col)
        if exists:
            print(f"   ✅ {col} column exists: {info['data_type']}")
        else:
            print(f"   ❌ {col} column missing")
            errors.append(f"documents.{col} column missing")
            all_passed = False
    
    # Check extracted_text column
    print("\n3. Checking extracted_text column in documents table...")
    exists, info = check_column_exists(cursor, 'documents', 'extracted_text')
    if exists:
        print(f"   ✅ extracted_text column exists: {info['data_type']}")
    else:
        print("   ❌ extracted_text column missing")
        errors.append("documents.extracted_text column missing")
        all_passed = False
    
    # Check chunks table
    print("\n4. Checking chunks table...")
    if check_table_exists(cursor, 'chunks'):
        print("   ✅ chunks table exists")
        
        # Check chunks table columns
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'chunks'
            ORDER BY ordinal_position
        """)
        columns = cursor.fetchall()
        expected_columns = {
            'chunk_id': 'varchar',
            'document_id': 'uuid',
            'text': 'text',
            'metadata': 'jsonb',
            'embedding': 'jsonb',
            'created_at': 'timestamp without time zone'
        }
        
        found_columns = {col[0]: col[1] for col in columns}
        print(f"   Found {len(columns)} columns:")
        for col_name, col_type, nullable in columns:
            marker = "✅" if col_name in expected_columns else "⚠️"
            print(f"     {marker} {col_name}: {col_type} (nullable: {nullable})")
        
        # Verify expected columns
        for expected_col, expected_type in expected_columns.items():
            if expected_col not in found_columns:
                errors.append(f"chunks.{expected_col} column missing")
                all_passed = False
    else:
        print("   ❌ chunks table missing")
        errors.append("chunks table missing")
        all_passed = False
    
    # Check indexes
    print("\n5. Checking indexes...")
    indexes_to_check = [
        ('idx_documents_status', 'documents.status'),
        ('idx_chunks_document_id', 'chunks.document_id')
    ]
    
    for index_name, description in indexes_to_check:
        if check_index_exists(cursor, index_name):
            print(f"   ✅ {index_name} exists ({description})")
        else:
            print(f"   ❌ {index_name} missing ({description})")
            errors.append(f"Index {index_name} missing")
            all_passed = False
    
    return all_passed, errors


def verify_migration_0020(cursor) -> Tuple[bool, List[str]]:
    """Verify migration 0020: Ingestion Batch Metadata Documentation"""
    errors = []
    all_passed = True
    
    print("\n📋 Verifying Migration 0020: Ingestion Batch Metadata Documentation")
    print("=" * 60)
    
    # Migration 0020 only adds comments, so we just verify the structure it documents exists
    print("\n1. Verifying metadata column exists (for batch processing)...")
    exists, info = check_column_exists(cursor, 'documents', 'metadata')
    if exists:
        print(f"   ✅ metadata column exists: {info['data_type']}")
    else:
        print("   ⚠️  metadata column missing (may exist from earlier migration)")
        # Not a critical error - metadata may have been added earlier
    
    print("\n2. Migration 0020 is documentation-only (no schema changes)")
    print("   ✅ No schema verification needed")
    
    return all_passed, errors


def main():
    """Main verification function"""
    print("🔍 Phase 5 Database Migration Verification")
    print("=" * 60)
    
    try:
        # Load configuration
        config = Config.from_env()
        
        if not config.database_url:
            print("\n❌ ERROR: DATABASE_URL not set in environment")
            print("   Please set DATABASE_URL in .env.local or environment variables")
            sys.exit(1)
        
        # Connect to database
        print("\n📡 Connecting to database...")
        db_conn = DatabaseConnection(config)
        conn = db_conn.get_connection()
        cursor = conn.cursor()
        
        print("   ✅ Connected successfully")
        
        # Verify migrations
        migration_0019_passed, migration_0019_errors = verify_migration_0019(cursor)
        migration_0020_passed, migration_0020_errors = verify_migration_0020(cursor)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 60)
        
        if migration_0019_passed:
            print("\n✅ Migration 0019: PASSED")
        else:
            print("\n❌ Migration 0019: FAILED")
            print("   Errors:")
            for error in migration_0019_errors:
                print(f"     - {error}")
        
        if migration_0020_passed:
            print("\n✅ Migration 0020: PASSED")
        else:
            print("\n❌ Migration 0020: FAILED")
            print("   Errors:")
            for error in migration_0020_errors:
                print(f"     - {error}")
        
        all_passed = migration_0019_passed and migration_0020_passed
        
        if all_passed:
            print("\n🎉 All migrations verified successfully!")
            print("\n✅ Ready to proceed with Phase 5 integration testing")
            sys.exit(0)
        else:
            print("\n⚠️  Some migrations are incomplete")
            print("   Please apply missing migrations before proceeding")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    main()

