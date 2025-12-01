"""
Configuration management for RAG Chatbot.

This module handles all configuration from environment variables,
providing type-safe access to configuration values with sensible defaults.
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Uses Pydantic's BaseSettings for automatic validation and type conversion.
    All values can be overridden via environment variables.
    """
    
    # ============================================================================
    # OpenAI Configuration
    # ============================================================================
    
    openai_api_key: str = Field(
        ...,  # Required field
        description="OpenAI API key for GPT and embeddings"
    )
    
    openai_model: str = Field(
        default="gpt-4-turbo-preview",
        description="Model to use for chat completions"
    )
    
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model to use for generating embeddings"
    )
    
    # ============================================================================
    # Application Configuration
    # ============================================================================
    
    app_name: str = Field(
        default="RAG Chatbot",
        description="Application name"
    )
    
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    # ============================================================================
    # Vector Store Configuration
    # ============================================================================
    
    chroma_persist_directory: str = Field(
        default="./data/chroma",
        description="Directory to persist ChromaDB data"
    )
    
    chroma_collection_name: str = Field(
        default="documents",
        description="Name of the ChromaDB collection"
    )
    
    # ============================================================================
    # RAG Configuration
    # ============================================================================
    
    chunk_size: int = Field(
        default=1000,
        ge=100,  # Greater than or equal to 100
        le=2000,  # Less than or equal to 2000
        description="Size of document chunks in characters"
    )
    
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Overlap between consecutive chunks"
    )
    
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of chunks to retrieve for context"
    )
    
    max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Maximum tokens in generated response"
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation (0=deterministic, 2=creative)"
    )
    
    # ============================================================================
    # API Configuration
    # ============================================================================
    
    max_request_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum request size in bytes"
    )
    
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        description="Maximum requests per minute per client"
    )
    
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    # ============================================================================
    # Monitoring Configuration
    # ============================================================================
    
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Port for Prometheus metrics endpoint"
    )
    
    # ============================================================================
    # Pydantic Settings Configuration
    # ============================================================================
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields in .env
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are loaded only once,
    improving performance and ensuring consistency.
    
    Returns:
        Settings: Application settings instance
        
    Example:
        ```python
        from app.config import get_settings
        
        settings = get_settings()
        print(settings.openai_api_key)
        ```
    """
    return Settings()


# ============================================================================
# Helper Functions
# ============================================================================

def get_openai_config() -> dict:
    """
    Get OpenAI configuration as dictionary.
    
    Useful for passing to OpenAI client initialization.
    
    Returns:
        dict: OpenAI configuration
        
    Example:
        ```python
        config = get_openai_config()
        client = OpenAI(**config)
        ```
    """
    settings = get_settings()
    return {
        "api_key": settings.openai_api_key,
        "model": settings.openai_model,
        "embedding_model": settings.openai_embedding_model,
    }


def get_rag_config() -> dict:
    """
    Get RAG configuration as dictionary.
    
    Returns:
        dict: RAG configuration
        
    Example:
        ```python
        config = get_rag_config()
        service = RAGService(**config)
        ```
    """
    settings = get_settings()
    return {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
    }


def get_vector_store_config() -> dict:
    """
    Get vector store configuration as dictionary.
    
    Returns:
        dict: Vector store configuration
        
    Example:
        ```python
        config = get_vector_store_config()
        store = VectorStore(**config)
        ```
    """
    settings = get_settings()
    return {
        "persist_directory": settings.chroma_persist_directory,
        "collection_name": settings.chroma_collection_name,
    }


# ============================================================================
# Production Considerations
# ============================================================================

"""
Why this configuration approach?

1. **Type Safety**: Pydantic validates all values at startup
   - Catch configuration errors before they cause runtime issues
   - Get type hints in IDE for all config values
   
2. **Environment Variables**: 12-factor app compliance
   - Easy to configure in different environments (dev/staging/prod)
   - No secrets in code
   - Works with Docker, Kubernetes, etc.
   
3. **Validation**: Automatic validation of constraints
   - chunk_size must be between 100-2000
   - temperature must be between 0-2
   - Fails fast on invalid config
   
4. **Caching**: Settings loaded once and cached
   - Better performance
   - Consistent values throughout request lifecycle
   
5. **Documentation**: Every field documented
   - Self-documenting configuration
   - Easy to understand what each setting does
   
Production Tips:
- Set DEBUG=false in production
- Use LOG_LEVEL=INFO or WARNING in production
- Rotate OPENAI_API_KEY regularly
- Monitor rate limits to prevent abuse
- Use environment-specific .env files (.env.dev, .env.prod)
"""
