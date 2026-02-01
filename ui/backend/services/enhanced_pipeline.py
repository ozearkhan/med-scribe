"""
Enhanced Pipeline Processor with Section Splitting.
Splits notes into sections first, then classifies each section.
"""

import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from services.embedding_classifier import EmbeddingClassifier, get_classifier
from services.section_splitter import SectionSplitter, get_section_splitter, NoteSection

logger = logging.getLogger(__name__)


@dataclass
class SectionResult:
    """Classification results for a single section."""
    section_id: int
    detected_type: str  # From keyword detection
    classified_type: str  # From embedding classification
    classification_score: float
    content: str
    content_preview: str
    domains: List[Dict[str, Any]]  # Top symptom domains
    safety: Dict[str, Any]  # Safety screening result


@dataclass
class EnhancedPipelineResult:
    """Full pipeline result with section-level analysis."""
    note_id: str
    input_text: str
    processing_time_ms: float
    sections: List[SectionResult]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "note_id": self.note_id,
            "input_text": self.input_text[:300] + "..." if len(self.input_text) > 300 else self.input_text,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "total_sections": len(self.sections),
            "sections": [
                {
                    "section_id": s.section_id,
                    "detected_type": s.detected_type,
                    "classified_type": s.classified_type,
                    "classification_score": round(s.classification_score, 4),
                    "content_preview": s.content_preview,
                    "domains": s.domains[:3],  # Top 3 domains
                    "safety": s.safety
                }
                for s in self.sections
            ],
            "summary": self.summary
        }


class EnhancedPipelineProcessor:
    """
    Enhanced pipeline that splits notes into sections before classification.
    
    Flow (Focused on Section Classification per mentor guidance):
    1. Split note into sections using keyword detection
    2. For each section:
       - Confirm/refine section type via embedding classification
       - Classify symptom domains
       - Run safety screening
    3. Aggregate results
    """
    
    def __init__(
        self,
        classifier: Optional[EmbeddingClassifier] = None,
        section_splitter: Optional[SectionSplitter] = None,
        datasets_dir: Optional[Path] = None,
        embeddings_dir: Optional[Path] = None
    ):
        self.classifier = classifier or get_classifier()
        self.section_splitter = section_splitter or get_section_splitter()
        self.datasets_dir = datasets_dir or Path(__file__).parent.parent.parent / "data" / "datasets"
        self.embeddings_dir = embeddings_dir or Path(__file__).parent.parent.parent / "data" / "embeddings"
        
        self.loaded_datasets: List[str] = []
        self._model_preloaded = False
    
    def initialize(self, preload_model: bool = True) -> bool:
        """Initialize pipeline and optionally preload embedding model."""
        logger.info("Initializing enhanced pipeline processor...")
        
        # Load required datasets (focused on section classification per mentor guidance)
        dataset_ids = [
            "psychiatry_sections",
            "psychiatry_domains", 
            "psychiatry_safety"
        ]
        
        for dataset_id in dataset_ids:
            dataset_path = self.datasets_dir / f"{dataset_id}.json"
            embeddings_path = self.embeddings_dir / f"{dataset_id}_embeddings.pkl.gz"
            
            if not dataset_path.exists():
                logger.warning(f"Dataset not found: {dataset_path}")
                continue
                
            if dataset_id not in self.classifier.loaded_datasets:
                self.classifier.load_dataset(dataset_id, dataset_path)
            
            if embeddings_path.exists() and dataset_id not in self.classifier.loaded_embeddings:
                self.classifier.load_embeddings(dataset_id, embeddings_path)
            
            if dataset_id in self.classifier.loaded_datasets:
                self.loaded_datasets.append(dataset_id)
        
        # Preload embedding model to avoid first-request delay
        if preload_model and not self._model_preloaded:
            logger.info("Preloading embedding model...")
            _ = self.classifier.embedding_model  # Triggers lazy load
            self._model_preloaded = True
            logger.info("Embedding model preloaded!")
        
        logger.info(f"Enhanced pipeline initialized with {len(self.loaded_datasets)} datasets")
        return len(self.loaded_datasets) > 0
    
    def process_note(
        self,
        text: str,
        top_k: int = 3,
        use_reranking: bool = False
    ) -> EnhancedPipelineResult:
        """
        Process clinical note with section splitting.
        """
        start_time = time.time()
        note_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Processing note {note_id} ({len(text)} chars) with section splitting")
        
        # Step 1: Split into sections
        raw_sections = self.section_splitter.split_note(text)
        logger.info(f"Split into {len(raw_sections)} sections")
        
        # Step 2: Process each section
        section_results = []
        for idx, section in enumerate(raw_sections):
            section_result = self._process_section(idx, section, top_k, use_reranking)
            section_results.append(section_result)
        
        # Step 3: Generate summary
        summary = self._generate_summary(section_results)
        
        total_time = (time.time() - start_time) * 1000
        
        return EnhancedPipelineResult(
            note_id=note_id,
            input_text=text,
            processing_time_ms=total_time,
            sections=section_results,
            summary=summary
        )
    
    def _process_section(
        self,
        idx: int,
        section: NoteSection,
        top_k: int,
        use_reranking: bool
    ) -> SectionResult:
        """Process a single section through all classifiers."""
        content = section.content
        
        # Use detected type or classify
        if section.confidence > 0.7:
            classified_type = section.section_type
            classification_score = section.confidence
        else:
            # Classify section type via embeddings
            try:
                result = self.classifier.classify(
                    text=content,
                    dataset_id="psychiatry_sections",
                    top_k=1,
                    use_reranking=use_reranking
                )
                classified_type = result.predicted_class.class_name if result.predicted_class else "Unknown"
                classification_score = result.predicted_class.effective_score if result.predicted_class else 0.0
            except Exception as e:
                logger.error(f"Section classification failed: {e}")
                classified_type = section.section_type
                classification_score = 0.0
        
        # Classify symptom domains
        domains = self._classify_category(content, "psychiatry_domains", top_k, use_reranking)
        
        # Safety screening
        safety = self._classify_safety(content, use_reranking)
        
        return SectionResult(
            section_id=idx,
            detected_type=section.section_type,
            classified_type=classified_type,
            classification_score=classification_score,
            content=content,
            content_preview=content[:150] + "..." if len(content) > 150 else content,
            domains=domains,
            safety=safety
        )
    
    def _classify_category(
        self,
        text: str,
        dataset_id: str,
        top_k: int,
        use_reranking: bool
    ) -> List[Dict[str, Any]]:
        """Classify text against a category and return top results."""
        if dataset_id not in self.loaded_datasets:
            return []
        
        try:
            result = self.classifier.classify(
                text=text,
                dataset_id=dataset_id,
                top_k=top_k,
                use_reranking=use_reranking
            )
            
            results = []
            if result.predicted_class:
                results.append({
                    "name": result.predicted_class.class_name,
                    "score": round(result.predicted_class.effective_score, 4)
                })
            for alt in result.alternatives:
                results.append({
                    "name": alt.class_name,
                    "score": round(alt.effective_score, 4)
                })
            return results
            
        except Exception as e:
            logger.error(f"Classification failed for {dataset_id}: {e}")
            return []
    
    def _classify_safety(self, text: str, use_reranking: bool) -> Dict[str, Any]:
        """Run safety screening on text."""
        if "psychiatry_safety" not in self.loaded_datasets:
            return {"status": "unknown", "score": 0}
        
        try:
            result = self.classifier.classify(
                text=text,
                dataset_id="psychiatry_safety",
                top_k=3,
                use_reranking=use_reranking
            )
            
            if result.predicted_class:
                name = result.predicted_class.class_name
                score = result.predicted_class.effective_score
                
                # Determine safety status
                if "denied" in name.lower() or "no" in name.lower():
                    status = "safe"
                elif "active" in name.lower():
                    status = "alert"
                elif "passive" in name.lower():
                    status = "monitor"
                else:
                    status = "review"
                
                return {
                    "status": status,
                    "finding": name,
                    "score": round(score, 4)
                }
            
        except Exception as e:
            logger.error(f"Safety classification failed: {e}")
        
        return {"status": "unknown", "score": 0}
    
    def _generate_summary(self, sections: List[SectionResult]) -> Dict[str, Any]:
        """Generate summary from section results."""
        summary = {
            "total_sections": len(sections),
            "section_types": [],
            "primary_domains": [],
            "safety_alerts": []
        }
        
        # Collect section types
        for s in sections:
            if s.classified_type != "Unknown" and s.classification_score > 0.3:
                summary["section_types"].append({
                    "type": s.classified_type,
                    "score": round(s.classification_score, 2)
                })
        
        # Collect top domains across all sections
        domain_scores = {}
        for s in sections:
            for d in s.domains:
                name = d["name"]
                if name not in domain_scores or d["score"] > domain_scores[name]:
                    domain_scores[name] = d["score"]
        summary["primary_domains"] = sorted(
            [{"name": k, "score": v} for k, v in domain_scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )[:5]
        
        # Collect safety alerts
        for s in sections:
            if s.safety.get("status") in ["alert", "monitor"]:
                summary["safety_alerts"].append({
                    "section": s.detected_type,
                    "finding": s.safety.get("finding"),
                    "status": s.safety.get("status")
                })
        
        # Overall safety status
        if any(a["status"] == "alert" for a in summary["safety_alerts"]):
            summary["overall_safety"] = "⚠️ ALERT - Active SI/HI detected"
        elif any(a["status"] == "monitor" for a in summary["safety_alerts"]):
            summary["overall_safety"] = "⚡ MONITOR - Passive ideation detected"
        else:
            summary["overall_safety"] = "✅ No active SI/HI identified"
        
        return summary


# Global instance
_enhanced_pipeline: Optional[EnhancedPipelineProcessor] = None


def get_enhanced_pipeline() -> EnhancedPipelineProcessor:
    """Get or create enhanced pipeline instance."""
    global _enhanced_pipeline
    if _enhanced_pipeline is None:
        _enhanced_pipeline = EnhancedPipelineProcessor()
        _enhanced_pipeline.initialize(preload_model=True)
    return _enhanced_pipeline
