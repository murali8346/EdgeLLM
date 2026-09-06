import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = PROJECT_ROOT / "models" / "qwen_restaurant_lora"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

ALLOWED_INTENTS = {
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
}


def load_test_data():
    records = []

    with TEST_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            messages = record["messages"]

            user_text = next(
                message["content"]
                for message in messages
                if message["role"] == "user"
            )

            assistant_text = next(
                message["content"]
                for message in messages
                if message["role"] == "assistant"
            )

            expected_output = json.loads(assistant_text)

            records.append(
                {
                    "text": user_text,
                    "intent": expected_output["intent"],
                }
            )

    return records


def load_model():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME
    )

    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
        device_map="auto",
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
    )

    model.eval()

    return tokenizer, model


def extract_json(text):
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def predict(text, tokenizer, model):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    latency = time.perf_counter() - start_time

    generated_tokens = outputs[
        0
    ][inputs["input_ids"].shape[1]:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    parsed_output = extract_json(generated_text)

    if isinstance(parsed_output, dict):
        predicted_intent = parsed_output.get("intent")
    else:
        predicted_intent = None

    valid_schema = (
        isinstance(parsed_output, dict)
        and predicted_intent in ALLOWED_INTENTS
        and isinstance(
            parsed_output.get("entities"),
            dict,
        )
    )

    return {
        "predicted_intent": predicted_intent,
        "raw_output": generated_text,
        "valid_json": parsed_output is not None,
        "valid_schema": valid_schema,
        "latency_seconds": latency,
    }


def main():
    records = load_test_data()
    tokenizer, model = load_model()

    y_true = []
    y_pred = []

    valid_json_count = 0
    valid_schema_count = 0
    latencies = []
    failures = []

    print(f"\nEvaluating {len(records)} test examples...\n")

    for index, record in enumerate(records, start=1):
        result = predict(
            record["text"],
            tokenizer,
            model,
        )

        expected_intent = record["intent"]
        predicted_intent = result["predicted_intent"]

        y_true.append(expected_intent)

        if predicted_intent in ALLOWED_INTENTS:
            y_pred.append(predicted_intent)
        else:
            y_pred.append("invalid_output")

        valid_json_count += int(result["valid_json"])
        valid_schema_count += int(result["valid_schema"])
        latencies.append(result["latency_seconds"])

        if predicted_intent != expected_intent:
            failures.append(
                {
                    "text": record["text"],
                    "expected": expected_intent,
                    "predicted": predicted_intent,
                    "raw_output": result["raw_output"],
                }
            )

        print(
            f"[{index}/{len(records)}] "
            f"Expected: {expected_intent:<22} "
            f"Predicted: {str(predicted_intent):<22} "
            f"Time: {result['latency_seconds']:.3f}s"
        )

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=sorted(ALLOWED_INTENTS),
        zero_division=0,
        output_dict=True,
    )

    macro_f1 = report["macro avg"]["f1-score"]

    results = {
        "model": BASE_MODEL_NAME,
        "adapter_path": str(ADAPTER_PATH),
        "test_examples": len(records),
        "intent_accuracy": accuracy,
        "macro_f1": macro_f1,
        "json_validity": valid_json_count / len(records),
        "schema_validity": valid_schema_count / len(records),
        "average_latency_seconds": sum(latencies) / len(latencies),
        "minimum_latency_seconds": min(latencies),
        "maximum_latency_seconds": max(latencies),
        "classification_report": report,
        "failure_count": len(failures),
        "failures": failures,
    }

    output_path = RESULTS_DIR / "llm_evaluation_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("LLM EVALUATION RESULTS")
    print("=" * 60)
    print(f"Intent accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"JSON validity: {results['json_validity']:.4f}")
    print(f"Schema validity: {results['schema_validity']:.4f}")
    print(
        "Average latency: "
        f"{results['average_latency_seconds']:.4f} seconds"
    )
    print(
        "Minimum latency: "
        f"{results['minimum_latency_seconds']:.4f} seconds"
    )
    print(
        "Maximum latency: "
        f"{results['maximum_latency_seconds']:.4f} seconds"
    )
    print(f"Failures: {len(failures)}")
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()