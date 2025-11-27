"""SQL wrappers for inserts/selects"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from rag_eval.db.connection import DatabaseConnection
from rag_eval.core.exceptions import DatabaseError
from rag_eval.core.logging import get_logger

logger = get_logger("db.queries")


class QueryExecutor:
    """Executes SQL queries against the database"""
    
    def __init__(self, db_conn: DatabaseConnection):
        self.db_conn = db_conn
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results"""
        conn = None
        try:
            conn = self.db_conn.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.commit()
            return results
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"Query failed: {e}")
        finally:
            if conn:
                self.db_conn.return_connection(conn)
    
    def execute_insert(self, query: str, params: Optional[tuple] = None) -> Optional[str]:
        """Execute an INSERT query and return the inserted ID"""
        conn = None
        try:
            conn = self.db_conn.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.fetchone()[0] if cursor.rowcount > 0 else None
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Insert execution failed: {e}")
            raise DatabaseError(f"Insert failed: {e}")
        finally:
            if conn:
                self.db_conn.return_connection(conn)

