import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "restaurant_intents_combined.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are a restaurant table-assistant model.

Classify the customer's request and return only valid JSON.

Allowed intents:
- menu_question
- menu_recommendation
- food_customization
- create_order
- modify_order
- cancel_order
- bill_request
- order_status
- special_request
- unknown

Use this format:
{
  "intent": "one_allowed_intent",
  "entities": {}
}

Do not add explanations or Markdown.
"""


def convert_record(record):
    text = record["text"]
    intent = record["intent"]

    target = {
        "intent": intent,
        "entities": {}
    }

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": text
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    target,
                    ensure_ascii=False
                )
            }
        ]
    }


def main():
    records = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if "text" not in record or "intent" not in record:
                raise ValueError(
                    f"Invalid record at line {line_number}"
                )

            records.append(record)

    random.Random(42).shuffle(records)

    total = len(records)
    train_end = int(total * 0.8)
    validation_end = int(total * 0.9)

    train_records = records[:train_end]
    validation_records = records[train_end:validation_end]
    test_records = records[validation_end:]

    splits = {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }

    for split_name, split_records in splits.items():
        output_path = OUTPUT_DIR / f"{split_name}.jsonl"

        with output_path.open("w", encoding="utf-8") as file:
            for record in split_records:
                converted = convert_record(record)
                file.write(
                    json.dumps(
                        converted,
                        ensure_ascii=False
                    ) + "\n"
                )

        print(
            f"{split_name}: "
            f"{len(split_records)} examples → {output_path}"
        )

    print("\nDataset preparation completed.")


if __name__ == "__main__":
    main()