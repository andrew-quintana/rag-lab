"""Environment configuration loader"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration loaded from environment variables"""
    
    # Database (Supabase Postgres)
    supabase_url: str
    supabase_key: str
    database_url: str
    
    # Azure AI Foundry
    azure_ai_foundry_endpoint: str
    azure_ai_foundry_api_key: str
    
    # Azure AI Search
    azure_search_endpoint: str
    azure_search_api_key: str
    azure_search_index_name: str
    
    # Azure Blob Storage
    azure_blob_connection_string: str
    azure_blob_container_name: str
    
    # Azure AI Foundry (with defaults)
    azure_ai_foundry_embedding_model: str = "text-embedding-ada-002"
    azure_ai_foundry_generation_model: str = "gpt-4"
    
    # API (with defaults)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        return cls(
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_key=os.getenv("SUPABASE_KEY", ""),
            database_url=os.getenv("DATABASE_URL", ""),
            azure_ai_foundry_endpoint=os.getenv("AZURE_AI_FOUNDRY_ENDPOINT", ""),
            azure_ai_foundry_api_key=os.getenv("AZURE_AI_FOUNDRY_API_KEY", ""),
            azure_ai_foundry_embedding_model=os.getenv("AZURE_AI_FOUNDRY_EMBEDDING_MODEL", "text-embedding-ada-002"),
            azure_ai_foundry_generation_model=os.getenv("AZURE_AI_FOUNDRY_GENERATION_MODEL", "gpt-4"),
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
            azure_search_api_key=os.getenv("AZURE_SEARCH_API_KEY", ""),
            azure_search_index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", ""),
            azure_blob_connection_string=os.getenv("AZURE_BLOB_CONNECTION_STRING", ""),
            azure_blob_container_name=os.getenv("AZURE_BLOB_CONTAINER_NAME", ""),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            api_reload=os.getenv("API_RELOAD", "true").lower() == "true",
        )

