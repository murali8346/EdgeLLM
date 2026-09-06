# EdgeLLM

## Fine-Tuning Compact Language Models for Reliable Physical-World Interaction

EdgeLLM investigates whether a compact language model can reliably convert natural restaurant-table conversations into structured requests suitable for deployment on resource-constrained hardware.

The restaurant assistant is the first demonstration environment. The broader objective is to study compact language models for reliable interaction with physical-world systems such as kiosks, hotel reception systems, retail assistants, industrial interfaces, and local service robots.

## Research Question

> Can a compact fine-tuned language model reliably convert natural restaurant-table conversations into structured menu, ordering, assistance, and billing requests suitable for deployment on resource-constrained hardware?

## Project Overview

The system is designed as a language-understanding component for a restaurant assistant. A customer message is converted into a structured intent and entity representation.

The language model does not directly execute restaurant operations. Validation, business rules, database operations, and external actions remain outside the model.

```text
Customer message
       |
       v
Compact language model
       |
       v
Structured intent/entity JSON
       |
       v
Validation layer
       |
       v
Business rules and database
       |
       v
Restaurant action
```

This separation is important for reliability:

* The model interprets natural language.
* The validation layer checks the model output.
* Business logic determines whether an action is permitted.
* The database provides authoritative restaurant and order information.
* The execution layer performs the actual operation.

## Initial Scope

The assistant currently covers the following intent categories:

* `menu_question`
* `menu_recommendation`
* `food_customization`
* `create_order`
* `modify_order`
* `cancel_order`
* `order_status`
* `special_request`
* `bill_request`
* `unknown`

Example structured output:

```json
{
  "intent": "create_order",
  "entities": {}
}
```

The current version focuses on intent classification and JSON formatting. Entity extraction is planned as a future extension.

### Out of Scope

The initial version does not include:

* Speech-to-text
* Text-to-speech
* Web or mobile interfaces
* POS/KDS integration
* Live restaurant database integration
* Physical hardware deployment
* True entity or slot extraction
* Direct database execution
* Multi-turn conversation management

## Dataset

The combined dataset contains:

* 600 examples
* 10 intent categories
* 60 examples per intent
* 480 training examples
* 60 validation examples
* 60 test examples

The dataset is balanced across all intent categories.

The primary dataset is located at:

```text
data/raw/restaurant_intents_combined.jsonl
```

The dataset was developed through multiple versions:

```text
restaurant_intents_v0.1.jsonl
restaurant_intents_v0.2.jsonl
restaurant_intents_v0.3.jsonl
restaurant_intents_combined.jsonl
```

The later dataset versions were created to add examples for categories that were difficult for the baseline and language model to distinguish.

## Experiments

Three approaches were evaluated using the same dataset and test split.

### Version 1: Traditional Baseline

A TF-IDF feature representation combined with Logistic Regression.

This provides a lightweight non-neural baseline for comparison.

### Version 2: Prompted Compact LLM

A pretrained `Qwen/Qwen2.5-0.5B-Instruct` model used without task-specific fine-tuning.

The model was prompted to return the required intent and entity JSON structure.

### Version 3: LoRA Fine-Tuned Compact LLM

The same `Qwen/Qwen2.5-0.5B-Instruct` model fine-tuned using Low-Rank Adaptation (LoRA).

The fine-tuning objective was to improve:

* Restaurant-specific intent classification
* Structured JSON generation
* Compliance with the supported intent schema
* Consistency on the target task

## Initial Benchmark Results

The following results were obtained on the same 60-example test set.

| Version   | Model                                 | Intent Accuracy | Macro F1 | JSON Validity |
| --------- | ------------------------------------- | --------------: | -------: | ------------: |
| Version 1 | TF-IDF + Logistic Regression          |          83.33% |   82.18% |           N/A |
| Version 2 | Prompted Qwen2.5-0.5B-Instruct        |          26.67% |   21.72% |        88.33% |
| Version 3 | LoRA Fine-Tuned Qwen2.5-0.5B-Instruct |          88.33% |   80.23% |       100.00% |

### Benchmark Interpretation

The fine-tuned model improved intent accuracy by **61.67 percentage points** compared with the prompted model.

It also improved JSON validity from **88.33% to 100.00%**.

These results indicate that task-specific fine-tuning substantially improves the performance of a compact pretrained model on this restaurant-intent task.

The traditional baseline remained competitive, particularly in Macro F1. This makes it useful as a reference point when evaluating whether the additional complexity of a language model is justified.

The fine-tuned model achieved higher overall intent accuracy, while some semantically similar categories remain difficult to distinguish.

## Fine-Tuned Model Evaluation

| Metric              |         Result |
| ------------------- | -------------: |
| Intent accuracy     |         88.33% |
| Macro F1            |         80.23% |
| JSON validity       |        100.00% |
| Schema validity     |        100.00% |
| Average latency     | 1.9053 seconds |
| Minimum latency     | 1.5365 seconds |
| Maximum latency     | 3.3843 seconds |
| Evaluation examples |             60 |

### Hardware

The fine-tuned model was evaluated on:

```text
GPU: NVIDIA GeForce GTX 1650
VRAM: 4 GB
```

The latency measurements were obtained on a desktop GPU. They should not be interpreted as edge-device deployment benchmarks.

CPU-only latency, memory usage, quantized model size, power consumption, and performance on embedded hardware have not yet been evaluated.

## Error Analysis

The fine-tuned model achieved strong overall performance but still confused several semantically related categories.

Examples include:

* `special_request` vs. `create_order`
* `food_customization` vs. `modify_order`
* `menu_recommendation` vs. `menu_question`
* `unknown` vs. supported billing or ordering requests

These errors suggest that future dataset improvements should focus on:

* More contrastive examples
* Clearer intent boundaries
* Additional out-of-domain examples
* More realistic restaurant conversations
* Multi-turn context
* Entity and slot annotations

## Reliability Design

The model is not treated as an autonomous decision-maker.

A production-oriented architecture should enforce the following constraints:

1. The model must return a supported intent.
2. The output must be valid JSON.
3. The output must conform to the expected schema.
4. Entities must be validated against authoritative restaurant data.
5. Business rules must be applied outside the model.
6. Unsupported or ambiguous requests must be routed to clarification or human assistance.
7. The model must not invent menu items, prices, order states, or transaction results.

## Repository Structure

```text
EdgeLLM/
├── data/
│   ├── processed/
│   └── raw/
│       ├── restaurant_intents_v0.1.jsonl
│       ├── restaurant_intents_v0.2.jsonl
│       ├── restaurant_intents_v0.3.jsonl
│       └── restaurant_intents_combined.jsonl
├── models/
│   └── qwen_restaurant_lora/
├── results/
├── src/
│   ├── baseline.py
│   ├── compare_experiments.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── evaluate_llm.py
│   ├── evaluate_prompted_llm.py
│   ├── merge_datasets.py
│   ├── predict.py
│   ├── predict_llm.py
│   ├── prepare_llm_dataset.py
│   ├── run_experiment.py
│   ├── train_llm.py
│   └── validation.py
├── tests/
├── test_dataset.py
├── requirements.txt
└── README.md
```

Generated models, processed datasets, experiment results, virtual environments, and Python cache files are excluded from version control.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

For GPU-enabled PyTorch, install the appropriate PyTorch build for the local CUDA environment before running the LLM experiments.

## Running the Experiments

### Validate the Dataset

```powershell
python test_dataset.py
```

### Run the Traditional Baseline

```powershell
python src/run_experiment.py
```

### Prepare the LLM Dataset

```powershell
python src/prepare_llm_dataset.py
```

### Fine-Tune the Compact LLM

```powershell
python src/train_llm.py
```

### Evaluate the Fine-Tuned LLM

```powershell
python src/evaluate_llm.py
```

### Evaluate the Prompted LLM

```powershell
python src/evaluate_prompted_llm.py
```

### Compare All Experiments

```powershell
python src/compare_experiments.py
```

## Generated Outputs

The experiments generate files such as:

```text
models/tfidf_logistic_regression.pkl
models/qwen_restaurant_lora/
results/baseline_experiment_results.json
results/llm_evaluation_results.json
results/prompted_llm_evaluation_results.json
results/experiment_comparison.json
```

These generated artifacts are excluded from version control by default.

## Limitations

* The dataset is relatively small.
* The evaluation set contains only 60 examples.
* Entity extraction is not yet genuinely evaluated.
* The model does not access live restaurant menus or order databases.
* The model does not directly execute restaurant operations.
* Latency was measured on a desktop GPU rather than an edge device.
* CPU-only deployment has not yet been evaluated.
* Quantization has not yet been evaluated.
* Memory usage and power consumption have not yet been measured.
* The model can still confuse semantically similar intents.
* Multi-turn conversation handling is not implemented.
* No physical restaurant hardware integration has been completed.

## Future Work

* Add entity and slot annotations.
* Evaluate entity extraction accuracy.
* Expand the dataset with harder conversational examples.
* Add multi-turn conversation handling.
* Add explicit out-of-domain and unknown-intent evaluation.
* Improve contrastive examples for similar intent categories.
* Compare quantized model variants.
* Measure CPU and edge-device latency.
* Measure RAM usage and model size.
* Integrate menu and order databases.
* Implement business-rule validation.
* Add clarification handling for ambiguous requests.
* Deploy the model on resource-constrained hardware.
* Connect the model to a physical restaurant assistant interface.

## License

This project is intended for research and experimentation. A formal license will be added in a future revision.
