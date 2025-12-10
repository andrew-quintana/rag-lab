"""Shared pytest fixtures for integration tests

This module consolidates common fixtures used across integration tests to eliminate
duplication and ensure consistency. Unit test fixtures (like mock_config) remain in
individual test files since they may have test-specific configurations.
"""

import pytest
import os
from src.core.config import Config
# DatabaseConnection imported lazily in db_conn fixture to avoid import errors
# when psycopg2 is not available


def _is_local_development(config) -> bool:
    """Check if running in local development mode (Azurite)"""
    connection_string = config.azure_blob_connection_string or os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING", "")
    return connection_string == "UseDevelopmentStorage=true"


@pytest.fixture(scope="module")
def config():
    """Load configuration from environment
    
    This fixture loads configuration from environment variables using Config.from_env().
    It's module-scoped to avoid reloading configuration for each test in a module.
    
    Used by: Integration tests that need real configuration
    """
    return Config.from_env()


@pytest.fixture(scope="module")
def db_conn(config):
    """Create database connection for integration tests
    
    This fixture creates a DatabaseConnection using the config fixture.
    It skips tests if DATABASE_URL is not set.
    
    Used by: Integration tests that need real database connections
    """
    if not config.database_url:
        pytest.skip("DATABASE_URL not set - skipping integration tests")
    # Import here to avoid import errors when psycopg2 is not available
    from src.db.connection import DatabaseConnection
    return DatabaseConnection(config)


@pytest.fixture(scope="module")
def is_local(config):
    """Check if running in local development mode (Azurite)
    
    This fixture determines if tests are running against local resources
    (Azurite, local Supabase) vs cloud resources.
    
    Used by: Tests that need to differentiate between local and cloud environments
    """
    return _is_local_development(config)

