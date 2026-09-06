import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = PROJECT_ROOT / "models" / "qwen_restaurant_lora"

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


def load_model():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME
    )

    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
    """
    Extract the first JSON object from model output.
    """
    text = text.strip()

    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"```", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def validate_output(result):
    if not isinstance(result, dict):
        return False

    if result.get("intent") not in ALLOWED_INTENTS:
        return False

    if not isinstance(result.get("entities"), dict):
        return False

    return True


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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[
        0
    ][inputs["input_ids"].shape[1]:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    parsed_output = extract_json(generated_text)

    return {
        "input": text,
        "raw_output": generated_text,
        "parsed_output": parsed_output,
        "valid_json": parsed_output is not None,
        "valid_schema": validate_output(parsed_output),
    }


def main():
    tokenizer, model = load_model()

    print("\nModel loaded successfully.")
    print("Type 'exit' to stop.\n")

    while True:
        text = input("Customer: ").strip()

        if text.lower() == "exit":
            break

        if not text:
            continue

        result = predict(
            text,
            tokenizer,
            model,
        )

        print("\nRaw output:")
        print(result["raw_output"])

        print("\nParsed output:")
        print(
            json.dumps(
                result["parsed_output"],
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nValid JSON:", result["valid_json"])
        print("Valid schema:", result["valid_schema"])
        print("-" * 60)


if __name__ == "__main__":
    main()