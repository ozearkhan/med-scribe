"""
Multi-Classifier Pipeline Processor for Psychiatry AI Scribe.
Runs all classification datasets on a clinical note and returns structured output.
"""

import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from services.embedding_classifier import EmbeddingClassifier, get_classifier, ClassificationResult

logger = logging.getLogger(__name__)


# Dataset configuration - Focused on Section Classification (per mentor guidance)
# Removed: psychiatry_entities, psychiatry_relations (NER/RE not in scope)
PIPELINE_DATASETS = [
    {
        "id": "psychiatry_sections",
        "name": "Section Classification",
        "description": "Identify note section type (CC, HPI, MSE, etc.)",
        "category": "sections"
    },
    {
        "id": "psychiatry_domains", 
        "name": "Symptom Domain Classification",
        "description": "Classify symptom domains (mood, anxiety, psychosis, etc.)",
        "category": "domains"
    },
    {
        "id": "psychiatry_safety",
        "name": "Safety Screening",
        "description": "Screen for suicidal/homicidal ideation",
        "category": "safety"
    }
]


@dataclass
class ClassificationItem:
    """Single classification result."""
    name: str
    description: str
    score: float
    reasoning: str = ""


@dataclass 
class CategoryResult:
    """Results for one classification category."""
    category: str
    dataset_id: str
    dataset_name: str
    top_result: ClassificationItem
    alternatives: List[ClassificationItem] = field(default_factory=list)
    processing_time_ms: float = 0.0


@dataclass
class PipelineResult:
    """Full pipeline processing result."""
    note_id: str
    input_text: str
    processing_time_ms: float
    results: Dict[str, CategoryResult]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "note_id": self.note_id,
            "input_text": self.input_text[:500] + "..." if len(self.input_text) > 500 else self.input_text,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "results": {
                category: {
                    "category": result.category,
                    "dataset_id": result.dataset_id,
                    "dataset_name": result.dataset_name,
                    "top_result": {
                        "name": result.top_result.name,
                        "description": result.top_result.description,
                        "score": round(result.top_result.score, 4),
                        "reasoning": result.top_result.reasoning
                    },
                    "alternatives": [
                        {
                            "name": alt.name,
                            "description": alt.description,
                            "score": round(alt.score, 4)
                        }
                        for alt in result.alternatives
                    ],
                    "processing_time_ms": round(result.processing_time_ms, 2)
                }
                for category, result in self.results.items()
            },
            "summary": self.summary
        }


class PipelineProcessor:
    """
    Multi-classifier pipeline processor.
    
    Runs all classification datasets on a clinical note and aggregates results.
    """
    
    def __init__(
        self,
        classifier: Optional[EmbeddingClassifier] = None,
        datasets_dir: Optional[Path] = None,
        embeddings_dir: Optional[Path] = None
    ):
        """Initialize pipeline processor."""
        self.classifier = classifier or get_classifier()
        self.datasets_dir = datasets_dir or Path(__file__).parent.parent.parent / "data" / "datasets"
        self.embeddings_dir = embeddings_dir or Path(__file__).parent.parent.parent / "data" / "embeddings"
        
        # Track loaded datasets
        self.loaded_datasets: List[str] = []
    
    def initialize(self) -> bool:
        """
        Initialize pipeline by loading all datasets and embeddings.
        
        Returns:
            True if all datasets loaded successfully
        """
        logger.info("Initializing pipeline processor...")
        
        for dataset_config in PIPELINE_DATASETS:
            dataset_id = dataset_config["id"]
            dataset_path = self.datasets_dir / f"{dataset_id}.json"
            embeddings_path = self.embeddings_dir / f"{dataset_id}_embeddings.pkl.gz"
            
            # Check files exist
            if not dataset_path.exists():
                logger.warning(f"Dataset file not found: {dataset_path}")
                continue
                
            if not embeddings_path.exists():
                logger.warning(f"Embeddings file not found: {embeddings_path}")
                continue
            
            # Load if not already loaded
            if dataset_id not in self.classifier.loaded_datasets:
                self.classifier.load_dataset(dataset_id, dataset_path)
            
            if dataset_id not in self.classifier.loaded_embeddings:
                self.classifier.load_embeddings(dataset_id, embeddings_path)
            
            if dataset_id in self.classifier.loaded_datasets and dataset_id in self.classifier.loaded_embeddings:
                self.loaded_datasets.append(dataset_id)
                logger.info(f"Loaded dataset: {dataset_id}")
        
        logger.info(f"Pipeline initialized with {len(self.loaded_datasets)} datasets")
        return len(self.loaded_datasets) > 0
    
    def process_note(
        self,
        text: str,
        top_k: int = 3,
        use_reranking: bool = False
    ) -> PipelineResult:
        """
        Process a clinical note through all classifiers.
        
        Args:
            text: Clinical note text
            top_k: Number of top candidates per classifier
            use_reranking: Whether to use LLM reranking
            
        Returns:
            PipelineResult with all classifications
        """
        start_time = time.time()
        note_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Processing note {note_id} ({len(text)} chars)")
        
        results: Dict[str, CategoryResult] = {}
        
        # Run each classifier
        for dataset_config in PIPELINE_DATASETS:
            dataset_id = dataset_config["id"]
            category = dataset_config["category"]
            
            if dataset_id not in self.loaded_datasets:
                logger.warning(f"Skipping unloaded dataset: {dataset_id}")
                continue
            
            try:
                cat_start = time.time()
                
                # Classify
                result = self.classifier.classify(
                    text=text,
                    dataset_id=dataset_id,
                    top_k=top_k,
                    use_reranking=use_reranking,
                    task_type=category
                )
                
                cat_time = (time.time() - cat_start) * 1000
                
                # Convert to CategoryResult
                if result.predicted_class:
                    top_item = ClassificationItem(
                        name=result.predicted_class.class_name,
                        description=result.predicted_class.description,
                        score=result.predicted_class.effective_score,
                        reasoning=result.predicted_class.reasoning
                    )
                    
                    alt_items = [
                        ClassificationItem(
                            name=alt.class_name,
                            description=alt.description,
                            score=alt.effective_score
                        )
                        for alt in result.alternatives
                    ]
                    
                    results[category] = CategoryResult(
                        category=category,
                        dataset_id=dataset_id,
                        dataset_name=dataset_config["name"],
                        top_result=top_item,
                        alternatives=alt_items,
                        processing_time_ms=cat_time
                    )
                    
                    logger.info(f"  {category}: {top_item.name} ({top_item.score:.2%})")
                    
            except Exception as e:
                logger.error(f"Error classifying with {dataset_id}: {e}")
                continue
        
        # Generate summary
        summary = self._generate_summary(results)
        
        total_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            note_id=note_id,
            input_text=text,
            processing_time_ms=total_time,
            results=results,
            summary=summary
        )
    
    def _generate_summary(self, results: Dict[str, CategoryResult]) -> Dict[str, Any]:
        """Generate a summary from classification results."""
        summary = {
            "primary_section": None,
            "primary_domain": None,
            "safety_status": None,
            "key_findings": []
        }
        
        # Extract primary classifications
        if "sections" in results:
            summary["primary_section"] = results["sections"].top_result.name
            
        if "domains" in results:
            summary["primary_domain"] = results["domains"].top_result.name
            
        if "safety" in results:
            safety_result = results["safety"].top_result.name
            if "denied" in safety_result.lower() or "no" in safety_result.lower():
                summary["safety_status"] = "No active SI/HI identified"
            elif "active" in safety_result.lower():
                summary["safety_status"] = "⚠️ ACTIVE SI/HI DETECTED - Requires immediate assessment"
            else:
                summary["safety_status"] = safety_result
        
        # Key findings
        for category, result in results.items():
            if result.top_result.score > 0.5:
                summary["key_findings"].append({
                    "category": category,
                    "finding": result.top_result.name,
                    "confidence": f"{result.top_result.score:.0%}"
                })
        
        return summary


# Global pipeline instance
_pipeline: Optional[PipelineProcessor] = None


def get_pipeline() -> PipelineProcessor:
    """Get or create global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = PipelineProcessor()
        _pipeline.initialize()
    return _pipeline
