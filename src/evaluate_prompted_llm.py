import json
import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import INTENTS, split_dataset
from src.validation import validate_json_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "restaurant_intents_combined.jsonl"
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "prompted_llm_evaluation_results.json"
)


SYSTEM_PROMPT = """You are a restaurant assistant intent classifier.

Classify the customer's message into exactly one of these intents:

menu_question
menu_recommendation
food_customization
create_order
modify_order
cancel_order
bill_request
order_status
special_request
unknown

Return only valid JSON using this format:
{"intent": "<intent_name>", "entities": {}}

Do not add explanations or extra text.
"""


def build_prompt(text: str) -> str:
    return (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT
        + "<|im_end|>\n"
        "<|im_start|>user\n"
        + text
        + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_prediction(
    model,
    tokenizer,
    text: str,
) -> str:
    prompt = build_prompt(text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    raw_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return raw_output


def load_dataset_rows():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    print(f"Loading base model: {BASE_MODEL}")
    print(f"Using dataset: {DATASET_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
        device_map="auto",
    )

    model.eval()

    rows = load_dataset_rows()
    _, _, test_rows = split_dataset(rows)

    correct = 0
    json_valid = 0
    schema_valid = 0

    failures = []
    latencies = []

    y_true = []
    y_pred = []

    total_examples = len(test_rows)

    print(f"Test examples: {total_examples}")
    print("Starting evaluation...\n")

    for index, row in enumerate(test_rows, start=1):
        text = row["text"]
        expected_intent = row["intent"]

        start_time = time.perf_counter()

        raw_output = generate_prediction(
            model,
            tokenizer,
            text,
        )

        latency = time.perf_counter() - start_time
        latencies.append(latency)

        validation = validate_json_text(raw_output)

        is_valid = validation.get("valid", False)
        data = validation.get("data")

        predicted_intent = None

        if is_valid and isinstance(data, dict):
            predicted_intent = data.get("intent")

        if is_valid:
            json_valid += 1
            schema_valid += 1

        if predicted_intent == expected_intent:
            correct += 1

        y_true.append(expected_intent)

        if predicted_intent in INTENTS:
            y_pred.append(predicted_intent)
        else:
            y_pred.append("invalid")

        if predicted_intent != expected_intent:
            failures.append(
                {
                    "text": text,
                    "expected_intent": expected_intent,
                    "predicted_intent": predicted_intent,
                    "raw_output": raw_output,
                    "validation": validation,
                }
            )

        print(
            f"[{index}/{total_examples}] "
            f"Expected: {expected_intent} | "
            f"Predicted: {predicted_intent} | "
            f"Latency: {latency:.2f}s"
        )

    accuracy = correct / total_examples
    json_validity = json_valid / total_examples
    schema_validity = schema_valid / total_examples

    average_latency = sum(latencies) / len(latencies)
    minimum_latency = min(latencies)
    maximum_latency = max(latencies)

    classification_metrics = classification_report(
        y_true,
        y_pred,
        labels=INTENTS,
        output_dict=True,
        zero_division=0,
    )

    macro_f1 = classification_metrics["macro avg"]["f1-score"]

    results = {
        "experiment": "prompted_pretrained_qwen_baseline",
        "base_model": BASE_MODEL,
        "dataset": str(DATASET_PATH),
        "test_size": total_examples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "json_validity": json_validity,
        "schema_validity": schema_validity,
        "average_latency_seconds": average_latency,
        "min_latency_seconds": minimum_latency,
        "max_latency_seconds": maximum_latency,
        "classification_report": classification_metrics,
        "failures": failures,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nEvaluation completed.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"JSON validity: {json_validity:.4f}")
    print(f"Schema validity: {schema_validity:.4f}")
    print(
        f"Average latency: "
        f"{average_latency:.4f} seconds"
    )
    print(
        f"Minimum latency: "
        f"{minimum_latency:.4f} seconds"
    )
    print(
        f"Maximum latency: "
        f"{maximum_latency:.4f} seconds"
    )
    print(f"Failures: {len(failures)}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()