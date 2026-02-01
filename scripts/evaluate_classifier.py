"""
Psychiatry Classification Evaluator

Evaluates the embedding-based classification accuracy using
the collected psychiatry note examples.

Usage:
    python evaluate_classifier.py
"""

import json
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    exit(1)


# Configuration
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "embeddings"
DATASETS_DIR = Path(__file__).parent.parent / "data" / "datasets"


def load_embeddings(filepath: Path) -> Dict[str, Any]:
    """Load embeddings from compressed pickle file."""
    with gzip.open(filepath, "rb") as f:
        return pickle.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def classify_text(
    text: str,
    model: SentenceTransformer,
    class_embeddings: Dict[str, np.ndarray],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Classify text using embedding similarity.
    
    Returns list of (class_name, confidence) tuples sorted by confidence.
    """
    # Generate embedding for input text
    text_embedding = model.encode(text, convert_to_numpy=True)
    
    # Calculate similarity to each class
    similarities = []
    for class_name, class_embedding in class_embeddings.items():
        sim = cosine_similarity(text_embedding, class_embedding)
        similarities.append((class_name, float(sim)))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


# =============================================================================
# TEST CASES FOR SECTION CLASSIFICATION
# =============================================================================
SECTION_TEST_CASES = [
    # (text, expected_class)
    ("I'm constantly worried about everything, even when I know there's no need.", "Chief Complaint"),
    ("Patient reports feeling down and anxious all the time.", "Chief Complaint"),
    ("Main Issue: Anger outbursts and irritability for 6 months.", "Chief Complaint"),
    
    ("Persistent, excessive worry most days, difficulty controlling the worry. Onset approximately 6 months ago. Severity 7/10.", "History of Present Illness"),
    ("Getting angry over little things, hitting wife, irritable on mother. Sleep disturbance - sleep at 2AM, waking up 5AM.", "History of Present Illness"),
    ("Symptoms include persistent low mood, anhedonia, fatigue, interrupted sleep with early morning awakening.", "History of Present Illness"),
    
    ("No prior diagnoses or treatments. No hospitalizations or suicidality history.", "Past Psychiatric History"),
    ("OCD diagnosed 5 years ago. Previous trial of fluoxetine with partial response.", "Past Psychiatric History"),
    
    ("Denies alcohol, tobacco, or illicit drugs. Occasional caffeine 2 cups/day.", "Substance Use History"),
    ("Smoking tobacco, nicotine, alcohol once a month.", "Substance Use History"),
    
    ("Mother has history of depression treated with SSRIs.", "Family Psychiatric History"),
    ("History of psychiatric illness in mother.", "Family Psychiatric History"),
    
    ("Single, lives alone, unemployed, high school graduate. Remote work previously in IT.", "Social History"),
    ("B.Tech software engineer, married, upper middle class, living with wife.", "Social History"),
    
    ("Appearance: Well-groomed. Behavior: Cooperative. Speech: Normal. Mood: Anxious. Affect: Constricted.", "Mental Status Examination"),
    ("General appearance: Unkempt, hyper alert. PMA increased. Talk: QTR increased. Affect perplexed, mood dysphoric.", "Mental Status Examination"),
    
    ("Major Depressive Disorder, single episode, moderate (F32.1). No suicidal ideation.", "Assessment"),
    ("Generalized Anxiety Disorder (GAD), moderate. Risk Assessment: No SI or HI.", "Assessment"),
    
    ("Start sertraline 50mg daily. Refer to CBT therapist. Follow-up 2 weeks.", "Plan"),
    ("Continue fluvoxamine 200mg daily. Continue weekly ERP with therapist. Follow-up 4 weeks.", "Plan"),
]

# =============================================================================
# TEST CASES FOR DOMAIN CLASSIFICATION
# =============================================================================
DOMAIN_TEST_CASES = [
    ("Sleep at 2AM, waking up 5AM, interrupted sleep.", "Sleep"),
    ("Patient reports insomnia with early morning awakening.", "Sleep"),
    ("Sleep improved to 6-7 hours with medication.", "Sleep"),
    
    ("Persistent low mood, feeling down, hopeless.", "Mood"),
    ("Mood dysphoric, feeling sad and empty.", "Mood"),
    ("Mood more stable on current medication regimen.", "Mood"),
    
    ("Excessive worry about everything, can't stop worrying.", "Anxiety"),
    ("Panic attacks 2-3 times per week with palpitations.", "Anxiety"),
    ("Feels overwhelmed and worried about future attacks.", "Anxiety"),
    
    ("Getting angry over little things, hitting wife.", "Anger"),
    ("Irritability, anger outbursts, temper issues.", "Anger"),
    ("Decreased irritability and better impulse control.", "Anger"),
    
    ("Decreased appetite with 10 lb weight loss.", "Appetite"),
    ("Appetite remains decreased, not eating well.", "Appetite"),
    
    ("Fatigue, tiredness, decreased energy.", "Energy"),
    ("Energy levels slightly better with treatment.", "Energy"),
    
    ("Difficulty concentrating at work, can't focus.", "Concentration"),
    ("Attention slightly impaired, brain fog.", "Concentration"),
    
    ("Anhedonia, loss of interest in activities.", "Interest"),
    ("Continued anhedonia despite medication.", "Interest"),
    
    ("Not bathing, wearing same clothes 3-4 days.", "Self-care"),
    ("Unkempt appearance, poor personal hygiene.", "Self-care"),
    
    ("Auditory hallucinations minimal, paranoid thoughts decreased.", "Psychosis"),
    ("No delusions or hallucinations reported.", "Psychosis"),
    
    ("Denies suicidal ideation, no SI or HI.", "Safety"),
    ("No suicidal or homicidal ideation reported.", "Safety"),
    ("Columbia Suicide Scale: Negative.", "Safety"),
    
    ("Strong cravings but maintaining abstinence.", "Substance"),
    ("60 days of sobriety, attending AA meetings.", "Substance"),
    ("Alcohol once a month, smoking tobacco.", "Substance"),
]

# =============================================================================
# TEST CASES FOR SAFETY CLASSIFICATION
# =============================================================================
SAFETY_TEST_CASES = [
    ("Denies suicidal ideation.", "SI_Denied"),
    ("No SI or HI.", "SI_Denied"),
    ("Columbia Suicide Scale: Negative.", "SI_Denied"),
    ("No suicidal thoughts reported.", "SI_Denied"),
    
    ("Patient reports thoughts of wanting to die.", "SI_Present"),
    ("Active suicidal ideation with plan.", "SI_Present"),
    ("Passive SI, wishes they were dead.", "SI_Present"),
    
    ("Denies homicidal ideation.", "HI_Denied"),
    ("No thoughts of harming others.", "HI_Denied"),
    ("No HI or violent ideation.", "HI_Denied"),
    
    ("Thoughts of harming others, HI present.", "HI_Present"),
    ("Homicidal ideation toward ex-partner.", "HI_Present"),
    
    ("Denies self-harm behavior.", "SelfHarm_Denied"),
    ("No history of cutting or self-injury.", "SelfHarm_Denied"),
    
    ("History of cutting, recent self-harm.", "SelfHarm_Present"),
    ("Engaged in self-injurious behavior.", "SelfHarm_Present"),
]


def evaluate_classifier(
    model: SentenceTransformer,
    embeddings_data: Dict[str, Any],
    test_cases: List[Tuple[str, str]],
    dataset_name: str
) -> Dict[str, Any]:
    """
    Evaluate classification accuracy on test cases.
    
    Returns metrics dict.
    """
    class_embeddings = embeddings_data["embeddings"]
    
    top1_correct = 0
    top3_correct = 0
    results = []
    
    for text, expected_class in test_cases:
        predictions = classify_text(text, model, class_embeddings, top_k=3)
        
        top1_pred = predictions[0][0]
        top3_preds = [p[0] for p in predictions]
        
        top1_match = top1_pred == expected_class
        top3_match = expected_class in top3_preds
        
        if top1_match:
            top1_correct += 1
        if top3_match:
            top3_correct += 1
        
        results.append({
            "text": text[:50] + "..." if len(text) > 50 else text,
            "expected": expected_class,
            "predicted": top1_pred,
            "confidence": predictions[0][1],
            "top1_correct": top1_match,
            "top3_correct": top3_match,
        })
    
    total = len(test_cases)
    metrics = {
        "dataset": dataset_name,
        "total_tests": total,
        "top1_accuracy": top1_correct / total,
        "top3_accuracy": top3_correct / total,
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "results": results,
    }
    
    return metrics


def print_evaluation_report(metrics: Dict[str, Any]) -> None:
    """Print formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {metrics['dataset']}")
    print("=" * 60)
    print(f"Total test cases: {metrics['total_tests']}")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.1%} ({metrics['top1_correct']}/{metrics['total_tests']})")
    print(f"Top-3 Accuracy: {metrics['top3_accuracy']:.1%} ({metrics['top3_correct']}/{metrics['total_tests']})")
    
    # Show incorrect predictions
    incorrect = [r for r in metrics["results"] if not r["top1_correct"]]
    if incorrect:
        print(f"\nIncorrect predictions ({len(incorrect)}):")
        for r in incorrect:
            print(f"  Text: {r['text']}")
            print(f"    Expected: {r['expected']}, Got: {r['predicted']} ({r['confidence']:.2f})")


def main():
    """Run evaluation on all datasets."""
    print("=" * 60)
    print("Psychiatry Classification Evaluator")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    all_metrics = []
    
    # Evaluate Sections
    sections_path = EMBEDDINGS_DIR / "psychiatry_sections_embeddings.pkl.gz"
    if sections_path.exists():
        embeddings_data = load_embeddings(sections_path)
        metrics = evaluate_classifier(model, embeddings_data, SECTION_TEST_CASES, "Sections")
        print_evaluation_report(metrics)
        all_metrics.append(metrics)
    else:
        print("\nSections embeddings not found. Run generate_embeddings.py first.")
    
    # Evaluate Domains
    domains_path = EMBEDDINGS_DIR / "psychiatry_domains_embeddings.pkl.gz"
    if domains_path.exists():
        embeddings_data = load_embeddings(domains_path)
        metrics = evaluate_classifier(model, embeddings_data, DOMAIN_TEST_CASES, "Domains")
        print_evaluation_report(metrics)
        all_metrics.append(metrics)
    else:
        print("\nDomains embeddings not found. Run generate_embeddings.py first.")
    
    # Evaluate Safety
    safety_path = EMBEDDINGS_DIR / "psychiatry_safety_embeddings.pkl.gz"
    if safety_path.exists():
        embeddings_data = load_embeddings(safety_path)
        metrics = evaluate_classifier(model, embeddings_data, SAFETY_TEST_CASES, "Safety")
        print_evaluation_report(metrics)
        all_metrics.append(metrics)
    else:
        print("\nSafety embeddings not found. Run generate_embeddings.py first.")
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for m in all_metrics:
        print(f"  {m['dataset']}: Top-1={m['top1_accuracy']:.1%}, Top-3={m['top3_accuracy']:.1%}")


if __name__ == "__main__":
    main()
