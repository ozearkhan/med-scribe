"""
Backend services for the Psychiatry AI Scribe API.
"""

# Only import Gemini-based services (no AWS dependencies)
from .gemini_client import GeminiClient, get_gemini_client
from .embedding_classifier import EmbeddingClassifier, get_classifier

__all__ = [
    'GeminiClient',
    'get_gemini_client',
    'EmbeddingClassifier', 
    'get_classifier'
]