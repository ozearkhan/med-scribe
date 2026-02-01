"""
Embedding-based classifier for Psychiatry AI Scribe.
Uses sentence-transformers for embedding similarity and Gemini for reranking.
"""

import os
import json
import gzip
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Run: pip install sentence-transformers")

from services.gemini_client import GeminiClient, RerankResult, get_gemini_client

logger = logging.getLogger(__name__)


@dataclass
class ClassCandidate:
    """A classification candidate with scores."""
    class_id: str
    class_name: str
    description: str
    similarity_score: float
    rerank_score: Optional[float] = None
    attribute_score: Optional[float] = None
    effective_score: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationResult:
    """Full classification result."""
    predicted_class: ClassCandidate
    alternatives: List[ClassCandidate]
    reranked: bool = False
    attribute_validated: bool = False
    attribute_validation: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "predicted_class": {
                "name": self.predicted_class.class_name,
                "description": self.predicted_class.description,
                "metadata": self.predicted_class.metadata
            },
            "effective_score": self.predicted_class.effective_score,
            "similarity_score": self.predicted_class.similarity_score,
            "rerank_score": self.predicted_class.rerank_score,
            "attribute_score": self.predicted_class.attribute_score,
            "reranked": self.reranked,
            "attribute_validated": self.attribute_validated,
            "attribute_validation": self.attribute_validation,
            "alternatives": [
                {
                    "name": alt.class_name,
                    "description": alt.description,
                    "effective_score": alt.effective_score,
                    "similarity_score": alt.similarity_score,
                    "rerank_score": alt.rerank_score,
                    "reasoning": alt.reasoning
                }
                for alt in self.alternatives
            ],
            "processing_time_ms": self.processing_time_ms
        }


class EmbeddingClassifier:
    """
    Embedding-based classifier with optional LLM reranking and attribute validation.
    
    Flow:
    1. Embed input text using sentence-transformers
    2. Find top-K similar classes by cosine similarity  
    3. (Optional) Rerank candidates using Gemini LLM
    4. (Optional) Validate against class attributes
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        gemini_client: Optional[GeminiClient] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            embedding_model_name: Sentence transformer model name
            gemini_client: Optional Gemini client for reranking
        """
        self.embedding_model_name = embedding_model_name
        self.gemini_client = gemini_client or get_gemini_client()
        
        # Lazy load embedding model
        self._embedding_model: Optional[SentenceTransformer] = None
        
        # Loaded datasets and embeddings
        self.loaded_datasets: Dict[str, Dict[str, Any]] = {}
        self.loaded_embeddings: Dict[str, np.ndarray] = {}
        self.loaded_attributes: Dict[str, Dict[str, Any]] = {}
    
    @property
    def embedding_model(self) -> Optional[SentenceTransformer]:
        """Lazy load embedding model."""
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
        return self._embedding_model
    
    def load_dataset(self, dataset_id: str, dataset_path: Path) -> bool:
        """
        Load a dataset from JSON file.
        
        Args:
            dataset_id: Unique identifier for the dataset
            dataset_path: Path to the dataset JSON file
        
        Returns:
            True if loaded successfully
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.loaded_datasets[dataset_id] = data
            logger.info(f"Loaded dataset {dataset_id} with {len(data.get('classes', []))} classes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}: {e}")
            return False
    
    def load_embeddings(self, dataset_id: str, embeddings_path: Path) -> bool:
        """
        Load pre-computed embeddings for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            embeddings_path: Path to the embeddings file (.pkl.gz)
        
        Returns:
            True if loaded successfully
        """
        try:
            with gzip.open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different embedding file formats
            if isinstance(data, dict):
                raw_embeddings = data.get('embeddings')
                class_names = data.get('class_names', [])
                
                if raw_embeddings is None:
                    logger.error(f"No 'embeddings' key found in {embeddings_path}")
                    return False
                
                # If embeddings is a dict {class_name: embedding_vector}
                if isinstance(raw_embeddings, dict):
                    # Use class_names order if available, otherwise use dict keys
                    if class_names:
                        embeddings_list = [raw_embeddings[name] for name in class_names if name in raw_embeddings]
                    else:
                        embeddings_list = list(raw_embeddings.values())
                    embeddings = np.array(embeddings_list)
                elif isinstance(raw_embeddings, np.ndarray):
                    embeddings = raw_embeddings
                elif isinstance(raw_embeddings, list):
                    embeddings = np.array(raw_embeddings)
                else:
                    logger.error(f"Unknown embeddings format: {type(raw_embeddings)}")
                    return False
            elif isinstance(data, np.ndarray):
                embeddings = data
            else:
                logger.error(f"Unknown data format in {embeddings_path}: {type(data)}")
                return False
            
            self.loaded_embeddings[dataset_id] = embeddings
            logger.info(f"Loaded embeddings for {dataset_id}: shape {embeddings.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embeddings for {dataset_id}: {e}")
            return False
    
    def load_attributes(self, dataset_id: str, attributes_path: Path) -> bool:
        """
        Load class attributes for validation.
        
        Args:
            dataset_id: Dataset identifier
            attributes_path: Path to the attributes JSON file
        
        Returns:
            True if loaded successfully
        """
        try:
            with open(attributes_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.loaded_attributes[dataset_id] = data
            logger.info(f"Loaded attributes for {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load attributes for {dataset_id}: {e}")
            return False
    
    def generate_embeddings(self, dataset_id: str, output_path: Path) -> bool:
        """
        Generate embeddings for a loaded dataset.
        
        Args:
            dataset_id: Dataset identifier (must be loaded first)
            output_path: Path to save embeddings
        
        Returns:
            True if generated successfully
        """
        if dataset_id not in self.loaded_datasets:
            logger.error(f"Dataset {dataset_id} not loaded")
            return False
        
        if self.embedding_model is None:
            logger.error("Embedding model not available")
            return False
        
        try:
            dataset = self.loaded_datasets[dataset_id]
            classes = dataset.get('classes', [])
            
            # Generate embeddings for class descriptions
            descriptions = [cls.get('description', '') for cls in classes]
            embeddings = self.embedding_model.encode(descriptions, show_progress_bar=True)
            
            # Store in memory
            self.loaded_embeddings[dataset_id] = np.array(embeddings)
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(output_path, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'class_names': [cls.get('name', '') for cls in classes],
                    'model_name': self.embedding_model_name
                }, f)
            
            logger.info(f"Generated and saved embeddings for {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for {dataset_id}: {e}")
            return False
    
    def classify(
        self,
        text: str,
        dataset_id: str,
        top_k: int = 5,
        use_reranking: bool = False,
        use_attribute_validation: bool = False,
        task_type: str = "classification"
    ) -> ClassificationResult:
        """
        Classify text against a loaded dataset.
        
        Args:
            text: Input text to classify
            dataset_id: Dataset to classify against
            top_k: Number of top candidates to return
            use_reranking: Whether to use LLM reranking
            use_attribute_validation: Whether to validate against attributes
            task_type: Type of classification (section, domain, entity, etc.)
        
        Returns:
            ClassificationResult with predictions and scores
        """
        import time
        start_time = time.time()
        
        # Validate dataset is loaded
        if dataset_id not in self.loaded_datasets:
            raise ValueError(f"Dataset {dataset_id} not loaded")
        if dataset_id not in self.loaded_embeddings:
            raise ValueError(f"Embeddings for {dataset_id} not loaded")
        
        dataset = self.loaded_datasets[dataset_id]
        embeddings = self.loaded_embeddings[dataset_id]
        classes = dataset.get('classes', [])
        
        # Step 1: Compute embedding for input text
        text_embedding = self.embedding_model.encode([text])[0]
        
        # Step 2: Compute cosine similarity
        similarities = self._cosine_similarity(text_embedding, embeddings)
        
        # Get top-K candidates
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        candidates = []
        for idx in top_indices:
            cls = classes[idx]
            candidates.append(ClassCandidate(
                class_id=cls.get('id', str(idx)),
                class_name=cls.get('name', f'Class {idx}'),
                description=cls.get('description', ''),
                similarity_score=float(similarities[idx]),
                effective_score=float(similarities[idx]),
                metadata=cls.get('metadata', {})
            ))
        
        # Step 3: Optional LLM reranking
        reranked = False
        if use_reranking and self.gemini_client and self.gemini_client.is_available():
            candidates = self._rerank_candidates(text, candidates, task_type)
            reranked = True
        
        # Step 4: Optional attribute validation
        attribute_validation = None
        if use_attribute_validation and dataset_id in self.loaded_attributes:
            attribute_validation = self._validate_attributes(
                text,
                candidates[0] if candidates else None,
                dataset_id
            )
            if attribute_validation and candidates:
                candidates[0].attribute_score = attribute_validation.get('score', 1.0)
        
        # Build result
        processing_time = (time.time() - start_time) * 1000
        
        predicted = candidates[0] if candidates else None
        alternatives = candidates[1:] if len(candidates) > 1 else []
        
        return ClassificationResult(
            predicted_class=predicted,
            alternatives=alternatives,
            reranked=reranked,
            attribute_validated=attribute_validation is not None,
            attribute_validation=attribute_validation,
            processing_time_ms=processing_time
        )
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a vector and a matrix of vectors."""
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
        return np.dot(vec2_norm, vec1_norm)
    
    def _rerank_candidates(
        self,
        text: str,
        candidates: List[ClassCandidate],
        task_type: str
    ) -> List[ClassCandidate]:
        """Rerank candidates using LLM."""
        # Prepare candidates for reranking
        candidate_tuples = [
            (c.class_name, c.description, c.similarity_score)
            for c in candidates
        ]
        
        # Call Gemini reranker
        rerank_results = self.gemini_client.rerank_candidates(
            text, candidate_tuples, task_type
        )
        
        # Update candidates with rerank scores
        rerank_map = {r.class_name: r for r in rerank_results}
        
        for candidate in candidates:
            if candidate.class_name in rerank_map:
                result = rerank_map[candidate.class_name]
                candidate.rerank_score = result.score
                candidate.reasoning = result.reasoning
                # Effective score is weighted combination
                candidate.effective_score = (
                    0.3 * candidate.similarity_score + 
                    0.7 * result.score
                )
            else:
                candidate.effective_score = candidate.similarity_score
        
        # Sort by effective score
        candidates.sort(key=lambda c: c.effective_score, reverse=True)
        return candidates
    
    def _validate_attributes(
        self,
        text: str,
        candidate: Optional[ClassCandidate],
        dataset_id: str
    ) -> Optional[Dict[str, Any]]:
        """Validate candidate against class attributes."""
        if not candidate:
            return None
        
        attributes = self.loaded_attributes.get(dataset_id, {})
        class_attributes = None
        
        # Find attributes for this class
        for class_attr in attributes.get('classes', []):
            if class_attr.get('class_name') == candidate.class_name:
                class_attributes = class_attr.get('required_attributes', {})
                break
        
        if not class_attributes:
            return {
                'is_valid': True,
                'score': 1.0,
                'conditions_met': [],
                'conditions_not_met': [],
                'explanation': 'No attributes defined for this class'
            }
        
        # Use Gemini for validation
        result = self.gemini_client.validate_attributes(
            text, candidate.class_name, class_attributes
        )
        
        return {
            'is_valid': result.is_valid,
            'score': result.score,
            'conditions_met': result.conditions_met,
            'conditions_not_met': result.conditions_not_met,
            'explanation': result.explanation
        }


# Global classifier instance
_classifier: Optional[EmbeddingClassifier] = None


def get_classifier() -> EmbeddingClassifier:
    """Get or create global classifier instance."""
    global _classifier
    if _classifier is None:
        from config import config
        _classifier = EmbeddingClassifier(
            embedding_model_name=config.embedding.model_name
        )
    return _classifier
