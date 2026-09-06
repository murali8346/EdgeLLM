import json
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "restaurant_intents_combined.jsonl"
)

INTENTS = [
    "menu_question",
    "menu_recommendation",
    "food_customization",
    "create_order",
    "modify_order",
    "cancel_order",
    "bill_request",
    "order_status",
    "special_request",
    "unknown",
]


def load_dataset(file_path: str):
    """
    Load a JSONL intent-classification dataset.

    Each line must contain:
    {
        "text": "...",
        "intent": "..."
    }
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if set(record.keys()) != {"text", "intent"}:
                raise ValueError(
                    f"Line {line_number} must contain only "
                    "'text' and 'intent'."
                )

            if not isinstance(record["text"], str):
                raise TypeError(
                    f"Line {line_number}: 'text' must be a string."
                )

            if not isinstance(record["intent"], str):
                raise TypeError(
                    f"Line {line_number}: 'intent' must be a string."
                )

            if record["intent"] not in INTENTS:
                raise ValueError(
                    f"Line {line_number}: unknown intent "
                    f"'{record['intent']}'."
                )

            rows.append(record)

    if not rows:
        raise ValueError("Dataset is empty.")

    return rows


def split_dataset(rows, test_size=0.10, validation_size=0.10, random_state=42):
    """
    Split rows into train, validation, and test sets.

    Default split:
    - 80% training
    - 10% validation
    - 10% testing
    """
    texts = [row["text"] for row in rows]
    labels = [row["intent"] for row in rows]

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=test_size + validation_size,
        random_state=random_state,
        stratify=labels,
    )

    relative_test_size = test_size / (test_size + validation_size)

    validation_texts, test_texts, validation_labels, test_labels = (
        train_test_split(
            temp_texts,
            temp_labels,
            test_size=relative_test_size,
            random_state=random_state,
            stratify=temp_labels,
        )
    )

    train = [
        {"text": text, "intent": label}
        for text, label in zip(train_texts, train_labels)
    ]

    validation = [
        {"text": text, "intent": label}
        for text, label in zip(validation_texts, validation_labels)
    ]

    test = [
        {"text": text, "intent": label}
        for text, label in zip(test_texts, test_labels)
    ]

    return train, validation, test


def print_dataset_summary(name, rows):
    counts = Counter(row["intent"] for row in rows)

    print(f"\n{name} set: {len(rows)} examples")

    for intent in INTENTS:
        print(f"  {intent}: {counts[intent]}")


if __name__ == "__main__":
    dataset_path = "data/raw/restaurant_intents.jsonl"

    rows = load_dataset(dataset_path)
    train, validation, test = split_dataset(rows)

    print_dataset_summary("Full", rows)
    print_dataset_summary("Training", train)
    print_dataset_summary("Validation", validation)
    print_dataset_summary("Test", test)