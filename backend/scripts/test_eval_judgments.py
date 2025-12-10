"""Test script for eval_judgments table with JSONB judge_output column

This script tests inserting and deleting records in eval_judgments
with the new judge_output JSONB column containing JudgeEvaluationResult data.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any

from src.core.config import Config
from src.db.connection import DatabaseConnection
from src.db.queries import QueryExecutor
from src.services.evaluator.logging import _serialize_judge_output
from src.core.interfaces import JudgeEvaluationResult


def create_test_judge_output() -> Dict[str, Any]:
    """Create a test JudgeEvaluationResult and serialize it to JSON"""
    judge_output = JudgeEvaluationResult(
        correctness_binary=True,
        hallucination_binary=False,
        risk_direction=0,
        risk_impact=1,
        reasoning="The answer is correct and grounded in the retrieved context.",
        failure_mode=None
    )
    
    return _serialize_judge_output(judge_output)


def run_migration(db_conn: DatabaseConnection):
    """Run the migration to refactor eval_judgments table and standardize IDs"""
    print("\n🔧 Running migration to refactor tables...")
    
    conn = None
    try:
        conn = db_conn.get_connection()
        cursor = conn.cursor()
        
        # Enable UUID extension
        cursor.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        
        # Migrate queries table: query_id -> id UUID
        print("  - Migrating queries table...")
        # Drop foreign key constraints first
        cursor.execute("""
            ALTER TABLE retrieval_logs DROP CONSTRAINT IF EXISTS retrieval_logs_query_id_fkey;
            ALTER TABLE model_answers DROP CONSTRAINT IF EXISTS model_answers_query_id_fkey;
            ALTER TABLE eval_judgments DROP CONSTRAINT IF EXISTS eval_judgments_query_id_fkey;
        """)
        
        cursor.execute("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'queries' AND column_name = 'query_id'
                ) THEN
                    -- Add id column
                    ALTER TABLE queries ADD COLUMN IF NOT EXISTS id UUID DEFAULT uuid_generate_v4();
                    -- Try to convert existing query_id to UUID, generate new UUID if it fails
                    UPDATE queries SET id = CASE 
                        WHEN query_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$' 
                        THEN query_id::uuid
                        ELSE uuid_generate_v4()
                    END WHERE id IS NULL;
                    
                    -- Update dependent tables' query_id columns to UUID
                    -- retrieval_logs
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = 'retrieval_logs' AND column_name = 'query_id'
                        AND data_type != 'uuid'
                    ) THEN
                        ALTER TABLE retrieval_logs ADD COLUMN query_id_new UUID;
                        UPDATE retrieval_logs rl SET query_id_new = q.id 
                        FROM queries q 
                        WHERE q.query_id::text = rl.query_id::text 
                           OR rl.query_id LIKE '%' || q.id::text || '%';
                        ALTER TABLE retrieval_logs DROP COLUMN query_id;
                        ALTER TABLE retrieval_logs RENAME COLUMN query_id_new TO query_id;
                    END IF;
                    
                    -- model_answers
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = 'model_answers' AND column_name = 'query_id'
                        AND data_type != 'uuid'
                    ) THEN
                        ALTER TABLE model_answers ADD COLUMN query_id_new UUID;
                        UPDATE model_answers ma SET query_id_new = q.id 
                        FROM queries q 
                        WHERE q.query_id::text = ma.query_id::text 
                           OR ma.query_id LIKE '%' || q.id::text || '%';
                        ALTER TABLE model_answers DROP COLUMN query_id;
                        ALTER TABLE model_answers RENAME COLUMN query_id_new TO query_id;
                    END IF;
                    
                    -- eval_judgments (will be handled separately below)
                    
                    -- Now drop and recreate primary key
                    ALTER TABLE queries DROP CONSTRAINT IF EXISTS queries_pkey CASCADE;
                    ALTER TABLE queries ADD PRIMARY KEY (id);
                    ALTER TABLE queries DROP COLUMN query_id;
                END IF;
            END $$;
        """)
        
        # Drop old columns from eval_judgments
        print("  - Dropping old score columns from eval_judgments...")
        for col in ['grounding_score', 'relevance_score', 'hallucination_risk']:
            cursor.execute(f"""
                ALTER TABLE eval_judgments
                DROP COLUMN IF EXISTS {col}
            """)
        
        # Drop judge_reasoning
        cursor.execute("""
            ALTER TABLE eval_judgments
            DROP COLUMN IF EXISTS judge_reasoning
        """)
        
        # Rename columns in eval_judgments
        print("  - Renaming columns in eval_judgments...")
        cursor.execute("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'eval_judgments' AND column_name = 'prompt_version'
                ) THEN
                    ALTER TABLE eval_judgments RENAME COLUMN prompt_version TO prompt_version_id;
                END IF;
                
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'eval_judgments' AND column_name = 'timestamp'
                ) THEN
                    ALTER TABLE eval_judgments RENAME COLUMN timestamp TO created_at;
                END IF;
            END $$;
        """)
        
        # Update query_id references in eval_judgments (skip if already UUID)
        cursor.execute("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'eval_judgments' AND column_name = 'query_id'
                ) AND (
                    SELECT data_type FROM information_schema.columns 
                    WHERE table_name = 'eval_judgments' AND column_name = 'query_id'
                ) != 'uuid' THEN
                    ALTER TABLE eval_judgments ADD COLUMN query_id_new UUID;
                    -- Try to match by extracting UUID from query_id or find matching query
                    UPDATE eval_judgments ej SET query_id_new = q.id 
                    FROM queries q 
                    WHERE q.id::text = ej.query_id::text 
                       OR ej.query_id LIKE '%' || q.id::text || '%'
                       OR (ej.query_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$' 
                           AND ej.query_id::uuid = q.id);
                    ALTER TABLE eval_judgments DROP COLUMN query_id;
                    ALTER TABLE eval_judgments RENAME COLUMN query_id_new TO query_id;
                END IF;
            END $$;
        """)
        
        # Recreate foreign key constraints
        cursor.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint 
                    WHERE conname = 'retrieval_logs_query_id_fkey'
                ) THEN
                    ALTER TABLE retrieval_logs 
                    ADD CONSTRAINT retrieval_logs_query_id_fkey 
                    FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE;
                END IF;
                
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint 
                    WHERE conname = 'model_answers_query_id_fkey'
                ) THEN
                    ALTER TABLE model_answers 
                    ADD CONSTRAINT model_answers_query_id_fkey 
                    FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE;
                END IF;
            END $$;
        """)
        
        # Add judge_output column
        print("  - Adding judge_output JSONB column...")
        cursor.execute("""
            ALTER TABLE eval_judgments
            ADD COLUMN IF NOT EXISTS judge_output JSONB
        """)
        
        # Change judgment_id to id UUID
        print("  - Changing judgment_id to id UUID...")
        cursor.execute("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'eval_judgments' AND column_name = 'judgment_id'
                ) THEN
                    ALTER TABLE eval_judgments ADD COLUMN IF NOT EXISTS id UUID DEFAULT uuid_generate_v4();
                    UPDATE eval_judgments SET id = judgment_id::uuid WHERE judgment_id IS NOT NULL;
                    ALTER TABLE eval_judgments DROP CONSTRAINT IF EXISTS eval_judgments_pkey;
                    ALTER TABLE eval_judgments ADD PRIMARY KEY (id);
                    ALTER TABLE eval_judgments DROP COLUMN judgment_id;
                END IF;
            END $$;
        """)
        
        # Create GIN index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_eval_judgments_judge_output_gin 
            ON eval_judgments USING GIN (judge_output)
        """)
        
        # Recreate indexes
        cursor.execute("""
            DROP INDEX IF EXISTS idx_eval_judgments_prompt_version
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_eval_judgments_prompt_version_id 
            ON eval_judgments(prompt_version_id)
        """)
        
        cursor.execute("""
            DROP INDEX IF EXISTS idx_eval_judgments_timestamp
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_eval_judgments_created_at 
            ON eval_judgments(created_at)
        """)
        
        # Recreate foreign key for query_id in eval_judgments
        cursor.execute("""
            ALTER TABLE eval_judgments 
            DROP CONSTRAINT IF EXISTS eval_judgments_query_id_fkey
        """)
        
        cursor.execute("""
            ALTER TABLE eval_judgments 
            ADD CONSTRAINT IF NOT EXISTS eval_judgments_query_id_fkey 
            FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE
        """)
        
        conn.commit()
        print("✅ Migration completed successfully")
    except Exception as e:
        if conn:
            conn.rollback()
        if "already exists" in str(e) or "duplicate" in str(e).lower():
            print("⚠️  Migration may have already been run")
        else:
            print(f"⚠️  Migration error (may be expected if already run): {e}")
            import traceback
            traceback.print_exc()
    finally:
        if conn:
            db_conn.return_connection(conn)


def create_test_query(query_executor: QueryExecutor) -> str:
    """Create a test query and return its ID"""
    query_id = str(uuid.uuid4())
    
    # Check if table has 'id' or 'query_id' column
    check_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'queries' 
        AND column_name IN ('id', 'query_id')
        LIMIT 1
    """
    
    result = query_executor.execute_query(check_query)
    column_name = result[0]['column_name'] if result else 'query_id'
    
    if column_name == 'id':
        insert_query = """
            INSERT INTO queries (id, query_text)
            VALUES (%s, %s)
            RETURNING id
        """
    else:
        insert_query = """
            INSERT INTO queries (query_id, query_text)
            VALUES (%s, %s)
            RETURNING query_id
        """
    
    params = (query_id, "What is the copay for specialist visits?")
    
    inserted_id = query_executor.execute_insert(insert_query, params)
    print(f"✅ Created test query with ID: {inserted_id or query_id}")
    return inserted_id or query_id


def insert_test_eval_judgment(
    query_executor: QueryExecutor,
    query_id: str,
    prompt_version_id: str = "test_prompt_version_1"
) -> str:
    """Insert a test record into eval_judgments"""
    judgment_id = str(uuid.uuid4())
    judge_output_json = json.dumps(create_test_judge_output())
    
    # Check if table has 'id' or 'judgment_id' column
    check_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'eval_judgments' 
        AND column_name IN ('id', 'judgment_id')
        LIMIT 1
    """
    
    result = query_executor.execute_query(check_query)
    id_column = result[0]['column_name'] if result else 'judgment_id'
    
    if id_column == 'id':
        insert_query = """
            INSERT INTO eval_judgments (
                id, query_id, prompt_version_id, judge_output, created_at
            )
            VALUES (%s, %s, %s, %s::jsonb, %s)
            RETURNING id
        """
    else:
        insert_query = """
            INSERT INTO eval_judgments (
                judgment_id, query_id, prompt_version_id, judge_output, created_at
            )
            VALUES (%s, %s, %s, %s::jsonb, %s)
            RETURNING judgment_id
        """
    
    params = (
        judgment_id,
        query_id,
        prompt_version_id,
        judge_output_json,
        datetime.now()
    )
    
    inserted_id = query_executor.execute_insert(insert_query, params)
    print(f"✅ Inserted eval_judgment with ID: {inserted_id or judgment_id}")
    return inserted_id or judgment_id


def verify_eval_judgment(
    query_executor: QueryExecutor,
    judgment_id: str
) -> bool:
    """Verify that the judge_output JSONB column contains correct data"""
    # Check which ID column exists
    check_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'eval_judgments' 
        AND column_name IN ('id', 'judgment_id')
        LIMIT 1
    """
    
    result = query_executor.execute_query(check_query)
    id_column = result[0]['column_name'] if result else 'judgment_id'
    
    select_query = f"""
        SELECT judge_output
        FROM eval_judgments
        WHERE {id_column} = %s
    """
    
    results = query_executor.execute_query(select_query, (judgment_id,))
    
    if not results:
        print(f"❌ No record found with judgment_id: {judgment_id}")
        return False
    
    judge_output_data = results[0].get('judge_output')
    
    if not judge_output_data:
        print(f"❌ judge_output column is NULL for judgment_id: {judgment_id}")
        return False
    
    # Verify expected fields
    expected_fields = [
        'correctness_binary',
        'hallucination_binary',
        'risk_direction',
        'risk_impact',
        'reasoning',
        'failure_mode'
    ]
    
    missing_fields = [field for field in expected_fields if field not in judge_output_data]
    
    if missing_fields:
        print(f"❌ Missing fields in judge_output: {missing_fields}")
        return False
    
    print(f"✅ Verified judge_output JSONB data for judgment_id: {judgment_id}")
    print(f"   correctness_binary: {judge_output_data.get('correctness_binary')}")
    print(f"   hallucination_binary: {judge_output_data.get('hallucination_binary')}")
    print(f"   risk_direction: {judge_output_data.get('risk_direction')}")
    print(f"   risk_impact: {judge_output_data.get('risk_impact')}")
    return True


def delete_test_eval_judgment(
    query_executor: QueryExecutor,
    judgment_id: str
) -> bool:
    """Delete a test record from eval_judgments"""
    # Check which ID column exists
    check_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'eval_judgments' 
        AND column_name IN ('id', 'judgment_id')
        LIMIT 1
    """
    
    result = query_executor.execute_query(check_query)
    id_column = result[0]['column_name'] if result else 'judgment_id'
    
    delete_query = f"""
        DELETE FROM eval_judgments
        WHERE {id_column} = %s
        RETURNING {id_column}
    """
    
    result = query_executor.execute_query(delete_query, (judgment_id,))
    
    if result:
        print(f"✅ Deleted eval_judgment with ID: {judgment_id}")
        return True
    else:
        print(f"❌ Failed to delete eval_judgment with ID: {judgment_id}")
        return False


def test_eval_judgments():
    """Main test function"""
    print("=" * 60)
    print("Testing eval_judgments with JSONB judge_output column")
    print("=" * 60)
    
    # Initialize database connection
    try:
        config = Config.from_env()
        db_conn = DatabaseConnection(config)
        db_conn.connect()
        query_executor = QueryExecutor(db_conn)
        
        # Run migration first
        run_migration(db_conn)
        
        print("\n📝 Step 1: Creating test query...")
        query_id = create_test_query(query_executor)
        
        print("\n📝 Step 2: Inserting test records...")
        judgment_ids = []
        
        # Insert multiple test records
        for i in range(3):
            prompt_version_id = f"test_prompt_version_{i+1}"
            judgment_id = insert_test_eval_judgment(
                query_executor,
                query_id=query_id,
                prompt_version_id=prompt_version_id
            )
            judgment_ids.append(judgment_id)
        
        print(f"\n✅ Inserted {len(judgment_ids)} test records")
        
        print("\n🔍 Step 3: Verifying inserted records...")
        for judgment_id in judgment_ids:
            verify_eval_judgment(query_executor, judgment_id)
        
        print("\n🗑️  Step 4: Deleting test records...")
        for judgment_id in judgment_ids:
            delete_test_eval_judgment(query_executor, judgment_id)
        
        print("\n✅ All test records deleted")
        
        # Delete test query
        print("\n🗑️  Step 5: Deleting test query...")
        # Check which ID column exists
        check_query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'queries' 
            AND column_name IN ('id', 'query_id')
            LIMIT 1
        """
        result = query_executor.execute_query(check_query)
        id_column = result[0]['column_name'] if result else 'query_id'
        
        delete_query = f"""
            DELETE FROM queries
            WHERE {id_column} = %s
            RETURNING {id_column}
        """
        result = query_executor.execute_query(delete_query, (query_id,))
        if result:
            print(f"✅ Deleted test query with ID: {query_id}")
        
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
    test_eval_judgments()

