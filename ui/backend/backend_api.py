"""
FastAPI backend for Psychiatry AI Scribe.
Provides REST API endpoints for classification using embeddings + Gemini reranking.
"""

import os
import sys
import json
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration and services
from config import config
from services.gemini_client import GeminiClient, get_gemini_client
from services.embedding_classifier import EmbeddingClassifier, get_classifier, ClassificationResult
from services.pipeline_processor import PipelineProcessor, get_pipeline

# Initialize FastAPI app
app = FastAPI(
    title="Psychiatry AI Scribe - Classification API",
    description="Embedding-based classification with Gemini LLM reranking for psychiatric notes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory paths
DATASETS_DIR = Path(__file__).parent.parent / "data" / "datasets"
EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "embeddings"
OUTPUT_DIR = Path(__file__).parent / "data" / "output"

# Ensure directories exist
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Pydantic Models
# ============================================================

class ClassDefinitionModel(BaseModel):
    id: str
    name: str
    description: str
    examples: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetModel(BaseModel):
    id: str
    name: str
    description: str
    classes: List[ClassDefinitionModel]
    embeddingsGenerated: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ClassificationConfigModel(BaseModel):
    use_reranking: bool = False
    reranking_model: Optional[str] = None
    use_attribute_validation: bool = False
    top_k_candidates: int = 5


class ClassificationRequestModel(BaseModel):
    text: str
    dataset_id: str
    config: ClassificationConfigModel = Field(default_factory=ClassificationConfigModel)


class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    type: str = "error"


# ============================================================
# Utility Functions
# ============================================================

def get_dataset_path(dataset_id: str) -> Path:
    """Get path to a dataset JSON file."""
    return DATASETS_DIR / f"{dataset_id}.json"


def get_embeddings_path(dataset_id: str) -> Path:
    """Get path to dataset embeddings file."""
    return EMBEDDINGS_DIR / f"{dataset_id}_embeddings.pkl.gz"


def get_attributes_path(dataset_id: str) -> Path:
    """Get path to dataset attributes file."""
    return DATASETS_DIR / f"{dataset_id}_attributes.json"


def load_dataset(dataset_id: str) -> Optional[DatasetModel]:
    """Load a dataset from file."""
    path = get_dataset_path(dataset_id)
    if not path.exists():
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if embeddings exist
        embeddings_path = get_embeddings_path(dataset_id)
        data['embeddingsGenerated'] = embeddings_path.exists()
        
        # Ensure classes have IDs
        for i, cls in enumerate(data.get('classes', [])):
            if 'id' not in cls:
                cls['id'] = str(uuid.uuid4())
        
        return DatasetModel(**data)
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {e}")
        return None


def list_all_datasets() -> List[DatasetModel]:
    """List all available datasets."""
    datasets = []
    
    for path in DATASETS_DIR.glob("*.json"):
        # Skip attribute files
        if path.name.endswith("_attributes.json"):
            continue
        
        dataset_id = path.stem
        dataset = load_dataset(dataset_id)
        if dataset:
            datasets.append(dataset)
    
    return datasets


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    gemini_client = get_gemini_client()
    return {
        "message": "Psychiatry AI Scribe - Classification API",
        "version": "1.0.0",
        "gemini_available": gemini_client.is_available(),
        "endpoints": {
            "datasets": "/datasets",
            "classification": "/classify/text",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    gemini_client = get_gemini_client()
    classifier = get_classifier()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "gemini_available": gemini_client.is_available(),
            "embedding_model": classifier.embedding_model is not None,
            "datasets_loaded": len(classifier.loaded_datasets)
        }
    }


# ============================================================
# Dataset Endpoints
# ============================================================

@app.get("/datasets", response_model=List[DatasetModel])
async def get_datasets():
    """List all available datasets."""
    return list_all_datasets()


@app.get("/datasets/{dataset_id}", response_model=DatasetModel)
async def get_dataset_by_id(dataset_id: str):
    """Get a specific dataset by ID."""
    dataset = load_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return dataset


@app.post("/datasets/{dataset_id}/embeddings")
async def generate_dataset_embeddings(dataset_id: str):
    """Generate embeddings for a dataset."""
    dataset = load_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    classifier = get_classifier()
    dataset_path = get_dataset_path(dataset_id)
    embeddings_path = get_embeddings_path(dataset_id)
    
    # Load dataset into classifier
    if not classifier.load_dataset(dataset_id, dataset_path):
        raise HTTPException(status_code=500, detail="Failed to load dataset")
    
    # Generate embeddings
    if not classifier.generate_embeddings(dataset_id, embeddings_path):
        raise HTTPException(status_code=500, detail="Failed to generate embeddings")
    
    return {"message": f"Embeddings generated for dataset {dataset_id}"}


# ============================================================
# Classification Endpoints
# ============================================================

@app.post("/classify/text")
async def classify_text(request: ClassificationRequestModel):
    """Classify text using the specified dataset and configuration."""
    try:
        # Load dataset if not already loaded
        classifier = get_classifier()
        dataset_path = get_dataset_path(request.dataset_id)
        embeddings_path = get_embeddings_path(request.dataset_id)
        attributes_path = get_attributes_path(request.dataset_id)
        
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")
        
        # Load dataset
        if request.dataset_id not in classifier.loaded_datasets:
            classifier.load_dataset(request.dataset_id, dataset_path)
        
        # Load embeddings (generate if needed)
        if request.dataset_id not in classifier.loaded_embeddings:
            if embeddings_path.exists():
                classifier.load_embeddings(request.dataset_id, embeddings_path)
            else:
                # Generate embeddings on-the-fly
                classifier.generate_embeddings(request.dataset_id, embeddings_path)
        
        # Load attributes if validation requested
        if request.config.use_attribute_validation and attributes_path.exists():
            if request.dataset_id not in classifier.loaded_attributes:
                classifier.load_attributes(request.dataset_id, attributes_path)
        
        # Perform classification
        result = classifier.classify(
            text=request.text,
            dataset_id=request.dataset_id,
            top_k=request.config.top_k_candidates,
            use_reranking=request.config.use_reranking,
            use_attribute_validation=request.config.use_attribute_validation
        )
        
        return result.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Classification failed",
                details=str(e),
                type="processing_error"
            ).model_dump()
        )


# ============================================================
# Reranking Models Endpoint
# ============================================================

@app.get("/models/reranking")
async def get_reranking_models():
    """Get available reranking models."""
    return {
        "models": config.available_reranking_models
    }


# ============================================================
# Attributes Endpoints
# ============================================================

@app.get("/attributes/{dataset_id}/exists")
async def check_attributes_exist(dataset_id: str):
    """Check if attributes exist for a dataset."""
    attributes_path = get_attributes_path(dataset_id)
    return {"exists": attributes_path.exists()}


@app.get("/attributes/{dataset_id}")
async def get_dataset_attributes(dataset_id: str):
    """Get attributes for a dataset."""
    attributes_path = get_attributes_path(dataset_id)
    
    if not attributes_path.exists():
        raise HTTPException(status_code=404, detail=f"Attributes for {dataset_id} not found")
    
    with open(attributes_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# Pipeline Processing Endpoints
# ============================================================

class PipelineRequestModel(BaseModel):
    text: str
    top_k: int = 3
    use_reranking: bool = False


@app.post("/process/note")
async def process_note(request: PipelineRequestModel):
    """
    Process a clinical note through all classifiers.
    
    Runs all 5 datasets (sections, domains, entities, safety, relations)
    and returns aggregated structured output.
    """
    try:
        pipeline = get_pipeline()
        
        result = pipeline.process_note(
            text=request.text,
            top_k=request.top_k,
            use_reranking=request.use_reranking
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Pipeline processing error: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Pipeline processing failed",
                details=str(e),
                type="processing_error"
            ).model_dump()
        )


@app.get("/pipeline/status")
async def get_pipeline_status():
    """Get pipeline status and loaded datasets."""
    pipeline = get_pipeline()
    return {
        "status": "ready" if len(pipeline.loaded_datasets) > 0 else "not_ready",
        "loaded_datasets": pipeline.loaded_datasets,
        "total_datasets": 5
    }


# ============================================================
# Enhanced Pipeline (Section Splitting)
# ============================================================

@app.post("/process/note/enhanced")
async def process_note_enhanced(request: PipelineRequestModel):
    """
    Process a clinical note with section splitting.
    
    Splits the note into sections first, then classifies each section
    through all datasets (domains, entities, safety, relations).
    """
    try:
        from services.enhanced_pipeline import get_enhanced_pipeline
        
        pipeline = get_enhanced_pipeline()
        
        result = pipeline.process_note(
            text=request.text,
            top_k=request.top_k,
            use_reranking=request.use_reranking
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Enhanced pipeline error: {e}\\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Enhanced pipeline processing failed",
                details=str(e),
                type="processing_error"
            ).model_dump()
        )


# ============================================================
# Startup Event
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Psychiatry AI Scribe backend...")
    
    # Initialize classifier (lazy load embedding model)
    classifier = get_classifier()
    
    # Pre-load all psychiatry datasets
    datasets = list_all_datasets()
    for dataset in datasets:
        dataset_path = get_dataset_path(dataset.id)
        embeddings_path = get_embeddings_path(dataset.id)
        
        # Load dataset
        classifier.load_dataset(dataset.id, dataset_path)
        
        # Load embeddings if available
        if embeddings_path.exists():
            classifier.load_embeddings(dataset.id, embeddings_path)
    
    logger.info(f"Loaded {len(datasets)} datasets")
    logger.info("Backend ready!")
