"""
Psychiatry Embedding Generator

Generates embeddings for psychiatry classification datasets using
sentence-transformers (all-mpnet-base-v2).

Following the AWS Extreme Text Classifier approach:
- Load class definitions from JSON datasets
- Generate embeddings for class descriptions
- Save embeddings for similarity-based classification

Usage:
    python generate_embeddings.py
"""

import json
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    exit(1)


# Configuration
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DATASETS_DIR = Path(__file__).parent.parent / "data" / "datasets"
EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "embeddings"

DATASET_FILES = [
    "psychiatry_sections_dataset.json",
    "psychiatry_domains_dataset.json",
    "psychiatry_entities_dataset.json",
    "psychiatry_relations_dataset.json",
    "psychiatry_safety_dataset.json",
]


def load_dataset(filepath: Path) -> Dict[str, Any]:
    """Load a classification dataset from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_class_embeddings(
    model: SentenceTransformer,
    dataset: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for each class in the dataset.
    
    Returns dict mapping class name to embedding vector.
    """
    embeddings = {}
    
    for class_def in dataset["classes"]:
        class_name = class_def["name"]
        class_description = class_def["description"]
        
        # Generate embedding for the description
        embedding = model.encode(class_description, convert_to_numpy=True)
        embeddings[class_name] = embedding
        
        print(f"  Generated embedding for: {class_name}")
    
    return embeddings


def save_embeddings(
    embeddings: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    output_path: Path
) -> None:
    """Save embeddings to compressed pickle file."""
    data = {
        "metadata": metadata,
        "embeddings": embeddings,
        "class_names": list(embeddings.keys()),
    }
    
    with gzip.open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"  Saved to: {output_path}")


def load_embeddings(filepath: Path) -> Dict[str, Any]:
    """Load embeddings from compressed pickle file."""
    with gzip.open(filepath, "rb") as f:
        return pickle.load(f)


def main():
    """Generate embeddings for all psychiatry datasets."""
    print("=" * 60)
    print("Psychiatry Embedding Generator")
    print("=" * 60)
    
    # Create output directory
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Process each dataset
    for dataset_file in DATASET_FILES:
        dataset_path = DATASETS_DIR / dataset_file
        
        if not dataset_path.exists():
            print(f"\nSkipping (not found): {dataset_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_file}")
        print("=" * 60)
        
        # Load dataset
        dataset = load_dataset(dataset_path)
        print(f"Loaded {len(dataset['classes'])} classes from {dataset['metadata']['domain']}")
        
        # Generate embeddings
        embeddings = generate_class_embeddings(model, dataset)
        
        # Create output filename
        output_file = dataset_file.replace("_dataset.json", "_embeddings.pkl.gz")
        output_path = EMBEDDINGS_DIR / output_file
        
        # Save embeddings
        metadata = {
            "model": MODEL_NAME,
            "domain": dataset["metadata"]["domain"],
            "version": dataset["metadata"]["version"],
            "num_classes": len(embeddings),
            "embedding_dim": model.get_sentence_embedding_dimension(),
        }
        
        save_embeddings(embeddings, metadata, output_path)
    
    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    
    # Summary
    print("\nGenerated embeddings:")
    for f in EMBEDDINGS_DIR.glob("*.pkl.gz"):
        data = load_embeddings(f)
        print(f"  - {f.name}: {data['metadata']['num_classes']} classes")


if __name__ == "__main__":
    main()
