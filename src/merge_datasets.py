"""
Merge and validate multiple restaurant intent datasets.

Expected input files:
    data/raw/restaurant_intents_v0.1.jsonl
    data/raw/restaurant_intents_v0.2.jsonl
    data/raw/restaurant_intents_v0.3.jsonl

Output file:
    data/raw/restaurant_intents_combined.jsonl
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

INPUT_FILES = [
    RAW_DATA_DIR / "restaurant_intents_v0.1.jsonl",
    RAW_DATA_DIR / "restaurant_intents_v0.2.jsonl",
    RAW_DATA_DIR / "restaurant_intents_v0.3.jsonl",
]

OUTPUT_FILE = RAW_DATA_DIR / "restaurant_intents_combined.jsonl"

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

EXPECTED_FIELDS = {"text", "intent"}


def load_jsonl(file_path: Path) -> list[dict]:
    """Load and validate one JSONL dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    rows = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path}, line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{file_path}, line {line_number}: "
                    "record must be a JSON object"
                )

            if set(record.keys()) != EXPECTED_FIELDS:
                raise ValueError(
                    f"{file_path}, line {line_number}: "
                    f"expected fields {EXPECTED_FIELDS}, "
                    f"got {set(record.keys())}"
                )

            text = record["text"]
            intent = record["intent"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{file_path}, line {line_number}: "
                    "'text' must be a non-empty string"
                )

            if not isinstance(intent, str) or intent not in INTENTS:
                raise ValueError(
                    f"{file_path}, line {line_number}: "
                    f"invalid intent: {intent}"
                )

            rows.append(
                {
                    "text": text.strip(),
                    "intent": intent,
                }
            )

    return rows


def validate_dataset(rows: list[dict], dataset_name: str) -> None:
    """Print validation information for one dataset."""
    intent_counts = Counter(row["intent"] for row in rows)

    print(f"\nDataset: {dataset_name}")
    print(f"Total examples: {len(rows)}")

    for intent in INTENTS:
        print(f"{intent}: {intent_counts[intent]}")

    missing_intents = [
        intent for intent in INTENTS if intent_counts[intent] == 0
    ]

    if missing_intents:
        raise ValueError(
            f"{dataset_name} is missing intents: {missing_intents}"
        )


def detect_duplicates(rows: list[dict]) -> tuple[set[str], dict[str, set[str]]]:
    """
    Detect duplicate texts and conflicting labels.

    Returns:
        duplicate_texts:
            Texts appearing more than once.

        conflicting_labels:
            Texts assigned to more than one intent.
    """
    text_to_intents = defaultdict(set)
    text_counts = Counter()

    for row in rows:
        text = row["text"].casefold()
        text_to_intents[text].add(row["intent"])
        text_counts[text] += 1

    duplicate_texts = {
        text for text, count in text_counts.items() if count > 1
    }

    conflicting_labels = {
        text: intents
        for text, intents in text_to_intents.items()
        if len(intents) > 1
    }

    return duplicate_texts, conflicting_labels


def remove_exact_duplicates(rows: list[dict]) -> list[dict]:
    """
    Remove exact duplicate texts while preserving the first occurrence.

    Duplicate texts with conflicting labels are not silently resolved.
    They must be handled separately before training.
    """
    seen_texts = set()
    unique_rows = []

    for row in rows:
        normalized_text = row["text"].casefold()

        if normalized_text in seen_texts:
            continue

        seen_texts.add(normalized_text)
        unique_rows.append(row)

    return unique_rows


def save_jsonl(rows: list[dict], output_file: Path) -> None:
    """Save records as JSONL."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def main() -> None:
    all_rows = []

    print("Loading datasets...")

    for input_file in INPUT_FILES:
        rows = load_jsonl(input_file)
        validate_dataset(rows, input_file.name)
        all_rows.extend(rows)

    print("\n" + "=" * 60)
    print("COMBINED DATASET BEFORE DEDUPLICATION")
    print("=" * 60)

    print(f"Total examples: {len(all_rows)}")

    combined_counts = Counter(row["intent"] for row in all_rows)

    for intent in INTENTS:
        print(f"{intent}: {combined_counts[intent]}")

    duplicate_texts, conflicting_labels = detect_duplicates(all_rows)

    print("\n" + "=" * 60)
    print("DUPLICATE ANALYSIS")
    print("=" * 60)

    print(f"Duplicate text count: {len(duplicate_texts)}")
    print(f"Conflicting-label text count: {len(conflicting_labels)}")

    if conflicting_labels:
        print("\nConflicting labels detected:")

        for text, labels in sorted(conflicting_labels.items()):
            print(f"- {text}")
            print(f"  Labels: {sorted(labels)}")

        raise ValueError(
            "Conflicting labels detected. "
            "Resolve them before creating the combined dataset."
        )

    unique_rows = remove_exact_duplicates(all_rows)

    print("\n" + "=" * 60)
    print("FINAL DATASET AFTER DEDUPLICATION")
    print("=" * 60)

    print(f"Total examples: {len(unique_rows)}")

    final_counts = Counter(row["intent"] for row in unique_rows)

    for intent in INTENTS:
        print(f"{intent}: {final_counts[intent]}")

    save_jsonl(unique_rows, OUTPUT_FILE)

    print("\nCombined dataset saved to:")
    print(OUTPUT_FILE)

    if len(unique_rows) != 600:
        print(
            f"\nWarning: expected 600 examples, "
            f"but found {len(unique_rows)}."
        )

    if any(count != 60 for count in final_counts.values()):
        print("\nWarning: dataset is not perfectly balanced.")

    else:
        print("\nDataset is perfectly balanced: 60 examples per intent.")


if __name__ == "__main__":
    main()