import json
from typing import Any


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


def validate_llm_output(output: Any) -> dict:
    """
    Validate and normalize the structured output produced by the LLM.

    Expected format:
    {
        "intent": "menu_question",
        "entities": {}
    }
    """

    # Convert JSON string into a Python object
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return {
                "valid": False,
                "error": "invalid_json",
                "message": "The model output is not valid JSON.",
                "data": None,
            }

    # Output must be a JSON object
    if not isinstance(output, dict):
        return {
            "valid": False,
            "error": "invalid_structure",
            "message": "The model output must be a JSON object.",
            "data": None,
        }

    # Check required fields
    if "intent" not in output:
        return {
            "valid": False,
            "error": "missing_intent",
            "message": "The model output is missing the intent field.",
            "data": None,
        }

    intent = output["intent"]

    # Intent must be a string
    if not isinstance(intent, str):
        return {
            "valid": False,
            "error": "invalid_intent_type",
            "message": "The intent field must be a string.",
            "data": None,
        }

    intent = intent.strip()

    # Intent must belong to the allowed intent set
    if intent not in ALLOWED_INTENTS:
        return {
            "valid": False,
            "error": "unknown_intent",
            "message": f"Unsupported intent: {intent}",
            "data": None,
        }

    # Entities are optional, but must be an object if present
    entities = output.get("entities", {})

    if entities is None:
        entities = {}

    if not isinstance(entities, dict):
        return {
            "valid": False,
            "error": "invalid_entities",
            "message": "The entities field must be a JSON object.",
            "data": None,
        }

    validated_data = {
        "intent": intent,
        "entities": entities,
    }

    return {
        "valid": True,
        "error": None,
        "message": "Valid LLM output.",
        "data": validated_data,
    }


def validate_json_text(output_text: str) -> dict:
    """
    Extract and validate JSON from raw model output.

    This supports outputs containing extra text around the JSON object.
    """

    output_text = output_text.strip()

    # First, try parsing the complete output
    try:
        parsed = json.loads(output_text)
        return validate_llm_output(parsed)
    except json.JSONDecodeError:
        pass

    # If extra text exists, attempt to extract the first JSON object
    start_index = output_text.find("{")
    end_index = output_text.rfind("}")

    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return {
            "valid": False,
            "error": "json_not_found",
            "message": "No JSON object was found in the model output.",
            "data": None,
        }

    json_candidate = output_text[start_index:end_index + 1]

    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError:
        return {
            "valid": False,
            "error": "invalid_json",
            "message": "A JSON object was found, but it could not be parsed.",
            "data": None,
        }

    return validate_llm_output(parsed)


def main():
    test_outputs = [
        '{"intent": "menu_question", "entities": {}}',
        '{"intent": "create_order", "entities": {"item": "Chicken Biryani"}}',
        '{"intent": "invalid_intent", "entities": {}}',
        '{"intent": "bill_request"}',
        "The answer is: {\"intent\": \"order_status\", \"entities\": {}}",
        "This is not JSON",
    ]

    for output in test_outputs:
        result = validate_json_text(output)

        print("=" * 60)
        print("Input:", output)
        print("Result:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()