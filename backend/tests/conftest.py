"""Shared pytest fixtures for integration tests

This module consolidates common fixtures used across integration tests to eliminate
duplication and ensure consistency. Unit test fixtures (like mock_config) remain in
individual test files since they may have test-specific configurations.
"""

import pytest
import os
from pathlib import Path
from src.core.config import Config
# DatabaseConnection imported lazily in db_conn fixture to avoid import errors
# when psycopg2 is not available


def _is_local_development(config) -> bool:
    """Check if running in local development mode (Azurite)"""
    connection_string = config.azure_blob_connection_string or os.getenv("AZURE_STORAGE_QUEUES_CONNECTION_STRING", "")
    return connection_string == "UseDevelopmentStorage=true"


def _get_env_file_path():
    """Get the path to the environment file to use
    
    Priority:
    1. ENV_FILE environment variable (if set)
    2. .env.prod in project root (if exists)
    3. Default behavior (Config.from_env() will look for .env.local)
    """
    # Check for ENV_FILE environment variable first
    env_file_from_env = os.getenv("ENV_FILE")
    if env_file_from_env:
        return Path(env_file_from_env)
    
    # Check for .env.prod in project root
    # conftest.py is in backend/tests/, so:
    # __file__ = backend/tests/conftest.py
    # parent = backend/tests/
    # parent.parent = backend/
    # parent.parent.parent = project root
    backend_dir = Path(__file__).parent.parent
    project_root = backend_dir.parent
    env_prod_path = project_root / ".env.prod"
    
    if env_prod_path.exists():
        return env_prod_path
    
    # Return None to use default behavior
    return None


@pytest.fixture(scope="module")
def config():
    """Load configuration from environment
    
    This fixture loads configuration from environment variables using Config.from_env().
    It prioritizes .env.prod if it exists, otherwise uses default behavior.
    It's module-scoped to avoid reloading configuration for each test in a module.
    
    Used by: Integration tests that need real configuration
    """
    env_file = _get_env_file_path()
    if env_file:
        return Config.from_env(env_file=str(env_file))
    return Config.from_env()


@pytest.fixture(scope="module")
def supabase_service(config):
    """Create Supabase REST API service for integration tests
    
    This fixture creates a SupabaseDatabaseService using the config fixture.
    It skips tests if SUPABASE_URL and SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY) are not set.
    
    Used by: Integration tests that need real database connections via REST API
    """
    if not config.supabase_url or (not config.supabase_anon_key and not config.supabase_service_role_key):
        pytest.skip("SUPABASE_URL and SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY) not set - skipping integration tests")
    # Import here to avoid import errors when supabase client is not available
    from src.db.supabase_db_service import SupabaseDatabaseService
    return SupabaseDatabaseService(config)


@pytest.fixture(scope="module")
def is_local(config):
    """Check if running in local development mode (Azurite)
    
    This fixture determines if tests are running against local resources
    (Azurite, local Supabase) vs cloud resources.
    
    Used by: Tests that need to differentiate between local and cloud environments
    """
    return _is_local_development(config)

