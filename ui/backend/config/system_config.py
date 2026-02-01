"""
System configuration for the Psychiatry AI Scribe backend.
Uses Gemini API for LLM operations and local sentence-transformers for embeddings.
"""

import os
from typing import List
from dataclasses import dataclass, field


@dataclass
class GeminiConfig:
    """Gemini API configuration settings."""
    api_key: str = ""
    default_model: str = "gemini-2.0-flash-exp"
    reranking_model: str = "gemini-2.0-flash-exp"
    attribute_model: str = "gemini-2.0-flash-exp"
    
    @classmethod
    def from_env(cls) -> 'GeminiConfig':
        """Create Gemini config from environment variables."""
        return cls(
            api_key=os.getenv('GEMINI_API_KEY', ''),
            default_model=os.getenv('GEMINI_DEFAULT_MODEL', cls.default_model),
            reranking_model=os.getenv('GEMINI_RERANKING_MODEL', cls.reranking_model),
            attribute_model=os.getenv('GEMINI_ATTRIBUTE_MODEL', cls.attribute_model),
        )


@dataclass
class EmbeddingConfig:
    """Embedding model configuration (local sentence-transformers)."""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 768
    
    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """Create embedding config from environment variables."""
        return cls(
            model_name=os.getenv('EMBEDDING_MODEL', cls.model_name),
        )


@dataclass
class ModelConfig:
    """Model-specific configuration settings."""
    # Default model parameters
    default_temperature: float = 0.1
    default_max_tokens: int = 4000
    
    # Attribute generation settings
    attribute_generation_temperature: float = 0.1
    attribute_generation_max_tokens: int = 4000
    
    # Attribute evaluation settings
    attribute_evaluation_temperature: float = 0.1
    attribute_evaluation_max_tokens: int = 1000
    
    # Reranking settings
    reranking_temperature: float = 0.1
    reranking_max_tokens: int = 1000
    
    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Create model config from environment variables."""
        return cls(
            default_temperature=float(os.getenv('MODEL_DEFAULT_TEMPERATURE', cls.default_temperature)),
            default_max_tokens=int(os.getenv('MODEL_DEFAULT_MAX_TOKENS', cls.default_max_tokens)),
            attribute_generation_temperature=float(os.getenv('MODEL_ATTR_GEN_TEMPERATURE', cls.attribute_generation_temperature)),
            attribute_generation_max_tokens=int(os.getenv('MODEL_ATTR_GEN_MAX_TOKENS', cls.attribute_generation_max_tokens)),
            attribute_evaluation_temperature=float(os.getenv('MODEL_ATTR_EVAL_TEMPERATURE', cls.attribute_evaluation_temperature)),
            attribute_evaluation_max_tokens=int(os.getenv('MODEL_ATTR_EVAL_MAX_TOKENS', cls.attribute_evaluation_max_tokens)),
            reranking_temperature=float(os.getenv('MODEL_RERANKING_TEMPERATURE', cls.reranking_temperature)),
            reranking_max_tokens=int(os.getenv('MODEL_RERANKING_MAX_TOKENS', cls.reranking_max_tokens)),
        )


@dataclass
class APIConfig:
    """API-related configuration settings."""
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"])
    max_file_size_mb: int = 50
    temp_file_cleanup_hours: int = 24
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create API config from environment variables."""
        cors_origins = os.getenv('API_CORS_ORIGINS', ','.join(cls().cors_origins)).split(',')
        return cls(
            cors_origins=[origin.strip() for origin in cors_origins],
            max_file_size_mb=int(os.getenv('API_MAX_FILE_SIZE_MB', cls().max_file_size_mb)),
            temp_file_cleanup_hours=int(os.getenv('API_TEMP_FILE_CLEANUP_HOURS', cls().temp_file_cleanup_hours)),
        )


@dataclass
class PathConfig:
    """File path configuration."""
    datasets_dir: str = "../data/datasets"
    embeddings_dir: str = "../data/embeddings"
    output_dir: str = "./data/output"
    
    @classmethod
    def from_env(cls) -> 'PathConfig':
        """Create path config from environment variables."""
        return cls(
            datasets_dir=os.getenv('DATASETS_DIR', cls.datasets_dir),
            embeddings_dir=os.getenv('EMBEDDINGS_DIR', cls.embeddings_dir),
            output_dir=os.getenv('OUTPUT_DIR', cls.output_dir),
        )


@dataclass
class SystemConfig:
    """Main system configuration container."""
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Available reranking models for UI
    available_reranking_models: List[dict] = field(default_factory=lambda: [
        {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash", "description": "Fast and efficient (recommended)"},
        {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "description": "Good balance of speed/quality"},
        {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "description": "Highest quality, slower"},
    ])
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Create system config from environment variables."""
        return cls(
            gemini=GeminiConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            models=ModelConfig.from_env(),
            api=APIConfig.from_env(),
            paths=PathConfig.from_env(),
        )


# Global configuration instance
config = SystemConfig.from_env()