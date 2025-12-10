"""DB connection + pool"""

import psycopg2
from psycopg2 import pool
from typing import Optional
from src.core.config import Config
from src.core.exceptions import DatabaseError
from src.core.logging import get_logger

logger = get_logger("db.connection")


class DatabaseConnection:
    """Manages database connection pool"""
    
    def __init__(self, config: Config):
        self.config = config
        self._pool: Optional[pool.ThreadedConnectionPool] = None
    
    def connect(self):
        """Initialize connection pool"""
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.config.database_url
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise DatabaseError(f"Database connection failed: {e}")
    
    def get_connection(self):
        """Get a connection from the pool"""
        if self._pool is None:
            self.connect()
        return self._pool.getconn()
    
    def return_connection(self, conn):
        """Return a connection to the pool"""
        if self._pool:
            self._pool.putconn(conn)
    
    def close(self):
        """Close all connections in the pool"""
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed")

