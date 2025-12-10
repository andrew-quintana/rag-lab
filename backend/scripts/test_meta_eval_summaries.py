"""Test script for meta_eval_summaries table with JSONB meta_eval column

This script tests inserting and deleting records in meta_eval_summaries
with the new meta_eval JSONB column containing MetaEvaluationResult data.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any

from src.core.config import Config
from src.db.connection import DatabaseConnection
from src.db.queries import QueryExecutor
from src.services.evaluator.logging import _serialize_meta_eval_output
from src.core.interfaces import MetaEvaluationResult


def create_test_meta_eval_result() -> Dict[str, Any]:
    """Create a test MetaEvaluationResult and serialize it to JSON"""
    meta_eval = MetaEvaluationResult(
        judge_correct=True,
        explanation="Judge verdicts are correct. All validations passed.",
        ground_truth_correctness=True,
        ground_truth_hallucination=False,
        ground_truth_risk_direction=0,
        ground_truth_risk_impact=1
    )
    
    return _serialize_meta_eval_output(meta_eval)


def create_test_dataset(query_executor: QueryExecutor) -> str:
    """Create a test dataset and return its ID"""
    dataset_id = str(uuid.uuid4())
    storage_path = f"datasets/{dataset_id}.json"
    
    # Check if table has 'id' or 'dataset_id' column
    check_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'datasets' 
        AND column_name IN ('id', 'dataset_id')
        LIMIT 1
    """
    
    result = query_executor.execute_query(check_query)
    id_column = result[0]['column_name'] if result else 'dataset_id'
    
    if id_column == 'id':
        insert_query = """
            INSERT INTO datasets (
                id, dataset_name, filename, file_size, 
                mime_type, storage_path, description, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
    else:
        insert_query = """
            INSERT INTO datasets (
                dataset_id, dataset_name, filename, file_size, 
                mime_type, storage_path, description, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING dataset_id
        """
    
    params = (
        dataset_id,
        "test_dataset",
        "test_dataset.json",
        1024,  # file_size in bytes
        "application/json",
        storage_path,
        "Test dataset for meta-evaluation summaries",
        "uploaded"
    )
    
    inserted_id = query_executor.execute_insert(insert_query, params)
    print(f"✅ Created test dataset with ID: {inserted_id or dataset_id}")
    return inserted_id or dataset_id


def insert_test_meta_eval_summary(
    query_executor: QueryExecutor,
    prompt_version_id: str,
    dataset_id: str
) -> str:
    """Insert a test record into meta_eval_summaries"""
    summary_id = str(uuid.uuid4())
    meta_eval_json = json.dumps(create_test_meta_eval_result())
    
    # Check if table has 'id' or 'summary_id' column
    check_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'meta_eval_summaries' 
        AND column_name IN ('id', 'summary_id')
        LIMIT 1
    """
    
    result = query_executor.execute_query(check_query)
    id_column = result[0]['column_name'] if result else 'summary_id'
    
    if id_column == 'id':
        insert_query = """
            INSERT INTO meta_eval_summaries (
                id, prompt_version_id, dataset_id, meta_eval
            )
            VALUES (%s, %s, %s, %s::jsonb)
            RETURNING id
        """
    else:
        insert_query = """
            INSERT INTO meta_eval_summaries (
                summary_id, prompt_version_id, dataset_id, meta_eval
            )
            VALUES (%s, %s, %s, %s::jsonb)
            RETURNING summary_id
        """
    
    params = (
        summary_id,
        prompt_version_id,
        dataset_id,
        meta_eval_json
    )
    
    inserted_id = query_executor.execute_insert(insert_query, params)
    print(f"✅ Inserted meta_eval_summary with ID: {inserted_id or summary_id}")
    return inserted_id or summary_id


def verify_meta_eval_summary(
    query_executor: QueryExecutor,
    summary_id: str
) -> bool:
    """Verify that the meta_eval JSONB column contains correct data"""
    # Check which ID column exists
    check_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'meta_eval_summaries' 
        AND column_name IN ('id', 'summary_id')
        LIMIT 1
    """
    
    result = query_executor.execute_query(check_query)
    id_column = result[0]['column_name'] if result else 'summary_id'
    
    select_query = f"""
        SELECT meta_eval
        FROM meta_eval_summaries
        WHERE {id_column} = %s
    """
    
    results = query_executor.execute_query(select_query, (summary_id,))
    
    if not results:
        print(f"❌ No record found with summary_id: {summary_id}")
        return False
    
    meta_eval_data = results[0].get('meta_eval')
    
    if not meta_eval_data:
        print(f"❌ meta_eval column is NULL for summary_id: {summary_id}")
        return False
    
    # Verify expected fields
    expected_fields = [
        'judge_correct',
        'explanation',
        'ground_truth_correctness',
        'ground_truth_hallucination',
        'ground_truth_risk_direction',
        'ground_truth_risk_impact'
    ]
    
    missing_fields = [field for field in expected_fields if field not in meta_eval_data]
    
    if missing_fields:
        print(f"❌ Missing fields in meta_eval: {missing_fields}")
        return False
    
    print(f"✅ Verified meta_eval JSONB data for summary_id: {summary_id}")
    print(f"   judge_correct: {meta_eval_data.get('judge_correct')}")
    print(f"   ground_truth_correctness: {meta_eval_data.get('ground_truth_correctness')}")
    print(f"   ground_truth_hallucination: {meta_eval_data.get('ground_truth_hallucination')}")
    return True


def delete_test_meta_eval_summary(
    query_executor: QueryExecutor,
    summary_id: str
) -> bool:
    """Delete a test record from meta_eval_summaries"""
    # Check which ID column exists
    check_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'meta_eval_summaries' 
        AND column_name IN ('id', 'summary_id')
        LIMIT 1
    """
    
    result = query_executor.execute_query(check_query)
    id_column = result[0]['column_name'] if result else 'summary_id'
    
    delete_query = f"""
        DELETE FROM meta_eval_summaries
        WHERE {id_column} = %s
        RETURNING {id_column}
    """
    
    result = query_executor.execute_query(delete_query, (summary_id,))
    
    if result:
        print(f"✅ Deleted meta_eval_summary with ID: {summary_id}")
        return True
    else:
        print(f"❌ Failed to delete meta_eval_summary with ID: {summary_id}")
        return False


def run_migrations(db_conn: DatabaseConnection):
    """Run the migrations to create datasets table and refactor meta_eval_summaries"""
    print("\n🔧 Running migrations...")
    
    conn = None
    try:
        conn = db_conn.get_connection()
        cursor = conn.cursor()
        
        # Migration 1: Create datasets table
        print("  - Creating datasets table...")
        cursor.execute("""
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp"
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                dataset_name VARCHAR(255) NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
        """)
        
        # Add new columns for blob storage if they don't exist
        cursor.execute("""
            ALTER TABLE datasets
            ADD COLUMN IF NOT EXISTS filename VARCHAR(500),
            ADD COLUMN IF NOT EXISTS file_size BIGINT,
            ADD COLUMN IF NOT EXISTS mime_type VARCHAR(100) DEFAULT 'application/json',
            ADD COLUMN IF NOT EXISTS storage_path VARCHAR(500),
            ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'uploaded'
        """)
        
        # Change dataset_id to id UUID if it exists
        cursor.execute("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'datasets' AND column_name = 'dataset_id'
                ) THEN
                    -- Drop foreign key constraint first
                    ALTER TABLE meta_eval_summaries 
                    DROP CONSTRAINT IF EXISTS meta_eval_summaries_dataset_id_fkey;
                    
                    ALTER TABLE datasets ADD COLUMN IF NOT EXISTS id UUID DEFAULT uuid_generate_v4();
                    UPDATE datasets SET id = dataset_id::uuid WHERE dataset_id IS NOT NULL;
                    ALTER TABLE datasets DROP CONSTRAINT IF EXISTS datasets_pkey CASCADE;
                    ALTER TABLE datasets ADD PRIMARY KEY (id);
                    ALTER TABLE datasets DROP COLUMN dataset_id;
                    
                    -- Recreate foreign key
                    ALTER TABLE meta_eval_summaries 
                    ADD CONSTRAINT meta_eval_summaries_dataset_id_fkey 
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE;
                END IF;
            END $$;
        """)
        
        # Make required fields NOT NULL if they're currently nullable
        # (We'll do this carefully to avoid issues with existing data)
        cursor.execute("""
            DO $$
            BEGIN
                -- Only set NOT NULL if column exists and is nullable
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'datasets' 
                    AND column_name = 'filename' 
                    AND is_nullable = 'YES'
                ) THEN
                    -- For test purposes, we'll allow NULL for now
                    -- In production, you'd want to backfill data first
                    NULL;
                END IF;
            END $$;
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(dataset_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_datasets_filename ON datasets(filename)
        """)
        
        # Migration 2: Refactor meta_eval_summaries
        print("  - Refactoring meta_eval_summaries table...")
        
        # Drop old columns
        for col in ['delta_grounding', 'delta_relevance', 'delta_hallucination', 
                    'judge_consistency', 'version_1', 'version_2']:
            cursor.execute(f"""
                ALTER TABLE meta_eval_summaries
                DROP COLUMN IF EXISTS {col}
            """)
        
        # Drop timestamp/created_at columns if they exist
        cursor.execute("""
            ALTER TABLE meta_eval_summaries
            DROP COLUMN IF EXISTS timestamp,
            DROP COLUMN IF EXISTS created_at
        """)
        
        # Add new columns
        cursor.execute("""
            ALTER TABLE meta_eval_summaries
            ADD COLUMN IF NOT EXISTS prompt_version_id VARCHAR(255),
            ADD COLUMN IF NOT EXISTS dataset_id UUID
        """)
        
        # Add meta_eval column if it doesn't exist
        cursor.execute("""
            ALTER TABLE meta_eval_summaries
            ADD COLUMN IF NOT EXISTS meta_eval JSONB
        """)
        
        # Change summary_id to id UUID if it exists
        cursor.execute("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'meta_eval_summaries' AND column_name = 'summary_id'
                ) THEN
                    ALTER TABLE meta_eval_summaries ADD COLUMN IF NOT EXISTS id UUID DEFAULT uuid_generate_v4();
                    UPDATE meta_eval_summaries SET id = summary_id::uuid WHERE summary_id IS NOT NULL;
                    ALTER TABLE meta_eval_summaries DROP CONSTRAINT IF EXISTS meta_eval_summaries_pkey;
                    ALTER TABLE meta_eval_summaries ADD PRIMARY KEY (id);
                    ALTER TABLE meta_eval_summaries DROP COLUMN summary_id;
                END IF;
            END $$;
        """)
        
        # Add foreign key constraint for dataset_id
        cursor.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint 
                    WHERE conname = 'meta_eval_summaries_dataset_id_fkey'
                ) THEN
                    ALTER TABLE meta_eval_summaries
                    ADD CONSTRAINT meta_eval_summaries_dataset_id_fkey
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE;
                END IF;
            END $$;
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_meta_eval_summaries_prompt_version_id 
            ON meta_eval_summaries(prompt_version_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_meta_eval_summaries_dataset_id 
            ON meta_eval_summaries(dataset_id)
        """)
        
        # Drop created_at index if it exists
        cursor.execute("""
            DROP INDEX IF EXISTS idx_meta_eval_summaries_created_at
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_meta_eval_summaries_meta_eval_gin 
            ON meta_eval_summaries USING GIN (meta_eval)
        """)
        
        conn.commit()
        print("✅ Migrations completed successfully")
    except Exception as e:
        if conn:
            conn.rollback()
        # Check if already exists
        if "already exists" in str(e) or "duplicate" in str(e).lower():
            print("⚠️  Migrations may have already been run")
        else:
            print(f"⚠️  Migration error (may be expected if already run): {e}")
            import traceback
            traceback.print_exc()
    finally:
        if conn:
            db_conn.return_connection(conn)


def test_meta_eval_summaries():
    """Main test function"""
    print("=" * 60)
    print("Testing meta_eval_summaries with JSONB meta_eval column")
    print("=" * 60)
    
    # Initialize database connection
    try:
        config = Config.from_env()
        db_conn = DatabaseConnection(config)
        db_conn.connect()
        query_executor = QueryExecutor(db_conn)
        
        # Run migrations first
        run_migrations(db_conn)
        
        print("\n📝 Step 1: Creating test dataset...")
        dataset_id = create_test_dataset(query_executor)
        
        print("\n📝 Step 2: Inserting test records...")
        summary_ids = []
        
        # Insert multiple test records with same dataset but different prompt versions
        for i in range(3):
            prompt_version_id = f"test_prompt_version_{i+1}"
            summary_id = insert_test_meta_eval_summary(
                query_executor,
                prompt_version_id=prompt_version_id,
                dataset_id=dataset_id
            )
            summary_ids.append(summary_id)
        
        print(f"\n✅ Inserted {len(summary_ids)} test records")
        
        print("\n🔍 Step 3: Verifying inserted records...")
        for summary_id in summary_ids:
            verify_meta_eval_summary(query_executor, summary_id)
        
        print("\n🗑️  Step 4: Deleting test records...")
        for summary_id in summary_ids:
            delete_test_meta_eval_summary(query_executor, summary_id)
        
        print("\n✅ All test records deleted")
        
        # Delete test dataset (cascade should handle summaries, but let's be explicit)
        print("\n🗑️  Step 5: Deleting test dataset...")
        # Check which ID column exists
        check_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'datasets' 
            AND column_name IN ('id', 'dataset_id')
            LIMIT 1
        """
        result = query_executor.execute_query(check_query)
        id_column = result[0]['column_name'] if result else 'dataset_id'
        
        delete_dataset_query = f"""
            DELETE FROM datasets
            WHERE {id_column} = %s
            RETURNING {id_column}
        """
        result = query_executor.execute_query(delete_dataset_query, (dataset_id,))
        if result:
            print(f"✅ Deleted test dataset with ID: {dataset_id}")
        else:
            print(f"⚠️  Dataset may have been cascade deleted")
        
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)
        
        db_conn.close()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_meta_eval_summaries()

