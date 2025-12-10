"""SQL wrappers for inserts/selects"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from src.db.connection import DatabaseConnection
from src.core.exceptions import DatabaseError
from src.core.logging import get_logger

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
        """Execute an INSERT query and return the inserted ID if RETURNING clause is used"""
        conn = None
        try:
            conn = self.db_conn.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            # Only try to fetch if query has RETURNING clause
            if "RETURNING" in query.upper():
                result = cursor.fetchone()
                return result[0] if result else None
            # Otherwise, just return None (insert succeeded if no exception)
            return None
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Insert execution failed: {e}")
            raise DatabaseError(f"Insert failed: {e}")
        finally:
            if conn:
                self.db_conn.return_connection(conn)

