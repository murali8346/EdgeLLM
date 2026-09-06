"""
Complete EdgeLLM experiment pipeline.

Pipeline:
    1. Load combined dataset
    2. Split into train/validation/test sets
    3. Train TF-IDF + Logistic Regression baseline
    4. Evaluate validation and test performance
    5. Save model and experiment results
    6. Run sample predictions
"""

from __future__ import annotations

import json
import pickle
import time
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from src.dataset import load_dataset, split_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "restaurant_intents_combined.jsonl"
)

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def save_json(data: dict, file_path: Path) -> None:
    """Save a dictionary as formatted JSON."""
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def print_distribution(name: str, rows: list[dict]) -> None:
    """Print class distribution."""
    counts = Counter(row["intent"] for row in rows)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Total examples: {len(rows)}")

    for intent, count in sorted(counts.items()):
        print(f"{intent}: {count}")


def build_baseline() -> Pipeline:
    """Create the TF-IDF + Logistic Regression baseline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    texts: list[str],
    labels: list[str],
    split_name: str,
) -> dict:
    """Evaluate the model on one dataset split."""
    predictions = model.predict(texts)

    accuracy = accuracy_score(labels, predictions)

    report = classification_report(
        labels,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(labels, predictions)

    print(f"\n{split_name} accuracy: {accuracy:.4f}")
    print(
        classification_report(
            labels,
            predictions,
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def run_sample_predictions(model: Pipeline) -> None:
    """Run representative restaurant-assistant queries."""
    samples = [
        "Bring me a chicken biryani.",
        "What ingredients are in the biryani?",
        "Can you recommend something spicy?",
        "Add one more naan to our order.",
        "Cancel the coffee.",
        "Where is our food?",
        "Can we get the bill?",
        "Please bring extra spoons.",
        "Can you tell me a joke?",
        "Make the curry less spicy.",
    ]

    print("\nSample Predictions")
    print("------------------")

    for text in samples:
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        confidence = float(max(probabilities))

        print(f"\nText: {text}")
        print(f"Intent: {prediction}")
        print(f"Confidence: {confidence:.4f}")


def main() -> None:
    print("=" * 70)
    print("EdgeLLM COMPLETE BASELINE EXPERIMENT")
    print("=" * 70)

    print(f"\nLoading dataset from:\n{DATASET_PATH}")

    rows = load_dataset(DATASET_PATH)

    train_rows, validation_rows, test_rows = split_dataset(
        rows,
        test_size=0.10,
        validation_size=0.10,
        random_state=42,
    )

    print_distribution("Full Dataset", rows)
    print_distribution("Training Dataset", train_rows)
    print_distribution("Validation Dataset", validation_rows)
    print_distribution("Test Dataset", test_rows)

    train_texts = [row["text"] for row in train_rows]
    train_labels = [row["intent"] for row in train_rows]

    validation_texts = [row["text"] for row in validation_rows]
    validation_labels = [row["intent"] for row in validation_rows]

    test_texts = [row["text"] for row in test_rows]
    test_labels = [row["intent"] for row in test_rows]

    print("\nTraining baseline model...")

    model = build_baseline()

    start_time = time.perf_counter()

    model.fit(train_texts, train_labels)

    training_time = time.perf_counter() - start_time

    print(f"Training time: {training_time:.4f} seconds")

    validation_results = evaluate_model(
        model,
        validation_texts,
        validation_labels,
        "Validation",
    )

    test_results = evaluate_model(
        model,
        test_texts,
        test_labels,
        "Test",
    )

    model_path = MODELS_DIR / "tfidf_logistic_regression.pkl"

    with model_path.open("wb") as file:
        pickle.dump(model, file)

    results = {
        "experiment": "tfidf_logistic_regression_baseline",
        "dataset": str(DATASET_PATH),
        "dataset_size": len(rows),
        "train_size": len(train_rows),
        "validation_size": len(validation_rows),
        "test_size": len(test_rows),
        "training_time_seconds": training_time,
        "validation": validation_results,
        "test": test_results,
        "model_path": str(model_path),
    }

    results_path = RESULTS_DIR / "baseline_experiment_results.json"
    save_json(results, results_path)

    print(f"\nModel saved to:\n{model_path}")
    print(f"Results saved to:\n{results_path}")

    run_sample_predictions(model)

    print("\n" + "=" * 70)
    print("BASELINE EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()