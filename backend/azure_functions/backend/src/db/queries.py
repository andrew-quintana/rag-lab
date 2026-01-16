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
        cursor = None
        try:
            conn = self.db_conn.get_connection()
            
            # Ensure connection is in a clean state
            if conn.closed:
                # If connection is closed, get a new one
                conn = self.db_conn.get_connection()
            
            cursor = conn.cursor()
            
            # Ensure params is a tuple (psycopg2 requirement)
            if params is None:
                params = ()
            elif not isinstance(params, tuple):
                params = tuple(params) if params else ()
            
            # Execute query - handle potential psycopg2 version differences
            try:
                cursor.execute(query, params)
            except IndexError as e:
                # If we get an IndexError during execute, it might be a psycopg2 internal issue
                # Try with explicit parameter formatting
                logger.warning(f"IndexError during cursor.execute, retrying with explicit params: {e}")
                cursor.close()
                cursor = conn.cursor()
                # Ensure params are properly formatted
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
            
            # Safely extract column names from cursor description
            # cursor.description returns Column objects (psycopg2 2.8+), use .name attribute
            columns = []
            if cursor.description:
                try:
                    for desc in cursor.description:
                        if hasattr(desc, 'name'):
                            # psycopg2 2.8+ returns Column objects
                            columns.append(desc.name)
                        elif isinstance(desc, (tuple, list)) and len(desc) > 0:
                            # Fallback for older psycopg2 versions that return tuples
                            columns.append(desc[0])
                        else:
                            logger.warning(f"Unexpected cursor description format: {desc}")
                except (IndexError, TypeError, AttributeError) as e:
                    logger.warning(f"Error extracting column names from cursor description: {e}")
                    columns = []
            
            # Fetch all results and convert to dictionaries
            rows = cursor.fetchall()
            if not columns and rows:
                # If we couldn't get column names but have rows, use generic names
                num_cols = len(rows[0]) if rows else 0
                columns = [f"column_{i}" for i in range(num_cols)]
            
            results = [dict(zip(columns, row)) for row in rows]
            conn.commit()
            return results
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise DatabaseError(f"Query failed: {e}")
        finally:
            if cursor:
                cursor.close()
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

