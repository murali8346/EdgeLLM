import json
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from src.dataset import load_dataset, split_dataset


DATASET_PATH = "data/raw/restaurant_intents.jsonl"
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")


def save_json(data, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    rows = load_dataset(DATASET_PATH)
    train, validation, test = split_dataset(rows)

    train_texts = [row["text"] for row in train]
    train_labels = [row["intent"] for row in train]

    validation_texts = [row["text"] for row in validation]
    validation_labels = [row["intent"] for row in validation]

    test_texts = [row["text"] for row in test]
    test_labels = [row["intent"] for row in test]

    model = Pipeline(
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
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    start_time = time.perf_counter()
    model.fit(train_texts, train_labels)
    training_time = time.perf_counter() - start_time

    validation_predictions = model.predict(validation_texts)
    test_predictions = model.predict(test_texts)

    validation_accuracy = accuracy_score(
        validation_labels,
        validation_predictions,
    )

    test_accuracy = accuracy_score(
        test_labels,
        test_predictions,
    )

    print(f"Training time: {training_time:.4f} seconds")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print("\nTest classification report:")
    print(
        classification_report(
            test_labels,
            test_predictions,
            zero_division=0,
        )
    )

    results = {
        "model": "TF-IDF + Logistic Regression",
        "dataset_size": len(rows),
        "training_size": len(train),
        "validation_size": len(validation),
        "test_size": len(test),
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "training_time_seconds": training_time,
        "classification_report": classification_report(
            test_labels,
            test_predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    save_json(
        results,
        RESULTS_DIR / "baseline_results.json",
    )

    print("\nResults saved to results/baseline_results.json")


if __name__ == "__main__":
    main()