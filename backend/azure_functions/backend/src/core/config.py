"""Environment configuration loader"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration loaded from environment variables
    
    Configuration is loaded from environment variables, with support for loading
    from .env.local file. By default, looks for .env.local in the project root.
    Can also specify a custom env file path.
    
    Environment variables can be set via:
    - .env.local file (recommended for local development)
    - System environment variables
    - Command-line flags (when supported by the application)
    """
    
    # Database (Supabase Postgres) - Required fields
    supabase_url: str
    supabase_anon_key: str  # Anonymous key for REST API operations
    database_url: str
    
    # Azure AI Foundry - Required fields
    azure_ai_foundry_endpoint: str
    azure_ai_foundry_api_key: str
    
    # Azure AI Search - Required fields
    azure_search_endpoint: str
    azure_search_api_key: str
    azure_search_index_name: str
    
    # Azure Document Intelligence - Required fields
    azure_document_intelligence_endpoint: str
    azure_document_intelligence_api_key: str
    
    # Azure Blob Storage - Required fields
    azure_blob_connection_string: str
    azure_blob_container_name: str
    
    # Optional fields (with defaults) - must come after all required fields
    supabase_service_role_key: str = ""  # Service role key (optional, falls back to anon key)
    database_password: str = ""  # Database password (optional, not used with REST API)
    azure_ai_foundry_embedding_model: str = "text-embedding-3-small"
    azure_ai_foundry_generation_model: str = "gpt-4o"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration from environment variables
        
        Args:
            env_file: Optional path to .env file. If None, looks for .env.local
                     in the project root (backend/.env.local or ../.env.local).
                     If specified, loads from that path.
                     Also checks ENV_FILE environment variable if env_file is None.
        
        Returns:
            Config instance with loaded values
        """
        # Determine env file path
        if env_file:
            # Use explicitly provided path
            env_path = Path(env_file)
        else:
            # Check for ENV_FILE environment variable (for cloud/prod mode)
            env_file_from_env = os.getenv("ENV_FILE")
            if env_file_from_env:
                env_path = Path(env_file_from_env)
            else:
                # Default: look for .env.local in project root
                # Try backend/.env.local first, then ../.env.local (project root)
                backend_dir = Path(__file__).parent.parent.parent
                project_root = backend_dir.parent
                env_path = backend_dir / ".env.local"
                if not env_path.exists():
                    env_path = project_root / ".env.local"
        
        # Load environment variables from file if it exists
        if env_path.exists():
            load_dotenv(env_path, override=True)
        elif env_file:
            # If user specified a file that doesn't exist, warn but continue
            import warnings
            warnings.warn(f"Specified env file not found: {env_file}. Using system environment variables.")
        
        return cls(
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
            supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),  # Optional, falls back to anon key
            database_url=os.getenv("SUPABASE_DB_URL", os.getenv("DATABASE_URL", "")),  # Support both for backward compatibility
            database_password=os.getenv("SUPABASE_DB_PASSWORD", ""),  # Optional, not used with REST API
            azure_ai_foundry_endpoint=os.getenv("AZURE_AI_FOUNDRY_ENDPOINT", ""),
            azure_ai_foundry_api_key=os.getenv("AZURE_AI_FOUNDRY_API_KEY", ""),
            azure_ai_foundry_embedding_model=os.getenv("AZURE_AI_FOUNDRY_EMBEDDING_MODEL", "text-embedding-3-small"),
            azure_ai_foundry_generation_model=os.getenv("AZURE_AI_FOUNDRY_GENERATION_MODEL", "gpt-4o"),
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
            azure_search_api_key=os.getenv("AZURE_SEARCH_API_KEY", ""),
            azure_search_index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", ""),
            azure_document_intelligence_endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", ""),
            azure_document_intelligence_api_key=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY", ""),
            azure_blob_connection_string=os.getenv("AZURE_BLOB_CONNECTION_STRING", ""),
            azure_blob_container_name=os.getenv("AZURE_BLOB_CONTAINER_NAME", ""),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            api_reload=os.getenv("API_RELOAD", "true").lower() == "true",
        )

