import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def load_json(filename):
    """
    Load a JSON result file from the results directory.
    """
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_percentage(value):
    """
    Convert a decimal metric such as 0.8833 to 88.33%.
    """
    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def format_seconds(value):
    """
    Format latency values.
    """
    if value is None:
        return "N/A"

    return f"{value:.4f}s"


def get_baseline_metrics(result):
    """
    Extract metrics from the TF-IDF + Logistic Regression
    baseline result structure.
    """

    test_result = result.get("test", {})

    classification_report = test_result.get(
        "classification_report",
        {},
    )

    macro_average = classification_report.get(
        "macro avg",
        {},
    )

    return {
        "accuracy": test_result.get("accuracy"),
        "macro_f1": macro_average.get("f1-score"),
        "json_validity": None,
        "schema_validity": None,
        "average_latency_seconds": None,
        "failure_count": None,
    }


def get_llm_metrics(result):
    """
    Extract metrics from prompted and fine-tuned LLM
    evaluation result files.

    Supports either 'intent_accuracy' or 'accuracy'
    as the accuracy field name.
    """

    failures = result.get("failures", [])

    if isinstance(failures, list):
        failure_count = len(failures)
    else:
        failure_count = None

    return {
        "accuracy": result.get(
            "intent_accuracy",
            result.get("accuracy"),
        ),
        "macro_f1": result.get(
            "macro_f1",
            result.get("test_macro_f1"),
        ),
        "json_validity": result.get(
            "json_validity"
        ),
        "schema_validity": result.get(
            "schema_validity"
        ),
        "average_latency_seconds": result.get(
            "average_latency_seconds"
        ),
        "failure_count": failure_count,
    }


def format_failure_count(value):
    """
    Display failure count cleanly.
    """
    if value is None:
        return "N/A"

    return str(value)


def main():
    """
    Load all experiment results, display a comparison,
    calculate improvements, and save a consolidated JSON file.
    """

    baseline_result = load_json(
        "baseline_experiment_results.json"
    )

    prompted_result = load_json(
        "prompted_llm_evaluation_results.json"
    )

    finetuned_result = load_json(
        "llm_evaluation_results.json"
    )

    experiments = [
        {
            "version": "Version 1",
            "model": "TF-IDF + Logistic Regression",
            "metrics": get_baseline_metrics(
                baseline_result
            ),
        },
        {
            "version": "Version 2",
            "model": "Prompted Qwen2.5-0.5B-Instruct",
            "metrics": get_llm_metrics(
                prompted_result
            ),
        },
        {
            "version": "Version 3",
            "model": "LoRA Fine-tuned Qwen2.5-0.5B-Instruct",
            "metrics": get_llm_metrics(
                finetuned_result
            ),
        },
    ]

    print()
    print("EdgeLLM Experiment Comparison")
    print("=" * 110)

    header = (
        f"{'Version':<12}"
        f"{'Model':<48}"
        f"{'Accuracy':>12}"
        f"{'Macro F1':>12}"
        f"{'JSON Valid':>14}"
        f"{'Schema Valid':>16}"
        f"{'Latency':>12}"
        f"{'Failures':>10}"
    )

    print(header)
    print("-" * 110)

    for experiment in experiments:
        metrics = experiment["metrics"]

        failure_display = format_failure_count(
            metrics["failure_count"]
        )

        print(
            f"{experiment['version']:<12}"
            f"{experiment['model']:<48}"
            f"{format_percentage(metrics['accuracy']):>12}"
            f"{format_percentage(metrics['macro_f1']):>12}"
            f"{format_percentage(metrics['json_validity']):>14}"
            f"{format_percentage(metrics['schema_validity']):>16}"
            f"{format_seconds(metrics['average_latency_seconds']):>12}"
            f"{failure_display:>10}"
        )

    prompted_accuracy = prompted_result.get(
        "intent_accuracy",
        prompted_result.get("accuracy"),
    )

    finetuned_accuracy = finetuned_result.get(
        "intent_accuracy",
        finetuned_result.get("accuracy"),
    )

    prompted_json_validity = prompted_result.get(
        "json_validity"
    )

    finetuned_json_validity = finetuned_result.get(
        "json_validity"
    )

    if (
        prompted_accuracy is not None
        and finetuned_accuracy is not None
    ):
        accuracy_improvement = (
            finetuned_accuracy - prompted_accuracy
        ) * 100
    else:
        accuracy_improvement = None

    if (
        prompted_json_validity is not None
        and finetuned_json_validity is not None
    ):
        json_improvement = (
            finetuned_json_validity
            - prompted_json_validity
        ) * 100
    else:
        json_improvement = None

    print()
    print("Improvement Analysis")
    print("-" * 40)

    if accuracy_improvement is not None:
        print(
            "Fine-tuned accuracy improvement over prompted model: "
            f"{accuracy_improvement:.2f} percentage points"
        )
    else:
        print(
            "Fine-tuned accuracy improvement over prompted model: "
            "N/A"
        )

    if json_improvement is not None:
        print(
            "Fine-tuned JSON validity improvement over prompted model: "
            f"{json_improvement:.2f} percentage points"
        )
    else:
        print(
            "Fine-tuned JSON validity improvement over prompted model: "
            "N/A"
        )

    comparison = {
        "project": "EdgeLLM",
        "experiments": experiments,
        "improvement_analysis": {
            "finetuned_accuracy_improvement_percentage_points": (
                accuracy_improvement
            ),
            "finetuned_json_validity_improvement_percentage_points": (
                json_improvement
            ),
        },
    }

    output_path = RESULTS_DIR / "experiment_comparison.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            comparison,
            file,
            indent=2,
        )

    print()
    print(f"Saved comparison to: {output_path}")


if __name__ == "__main__":
    main()