"""
Configuration package for Psychiatry AI Scribe backend.
"""

from config.system_config import config, SystemConfig, GeminiConfig, EmbeddingConfig, ModelConfig, APIConfig, PathConfig
from config.prompts import ExampleGenerationPrompts

__all__ = [
    'config',
    'SystemConfig',
    'GeminiConfig', 
    'EmbeddingConfig',
    'ModelConfig',
    'APIConfig',
    'PathConfig',
    'ExampleGenerationPrompts'
]