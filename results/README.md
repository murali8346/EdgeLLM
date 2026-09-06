# Experiment Results

## Dataset

- Total examples: 600
- Intent categories: 10
- Training examples: 480
- Validation examples: 60
- Test examples: 60
- Split random state: 42

## Benchmark Results

| Version | Model | Intent Accuracy | Macro F1 | JSON Validity |
|---|---|---:|---:|---:|
| Version 1 | TF-IDF + Logistic Regression | 83.33% | 82.18% | N/A |
| Version 2 | Prompted Qwen2.5-0.5B-Instruct | 26.67% | 21.72% | 88.33% |
| Version 3 | LoRA Fine-Tuned Qwen2.5-0.5B-Instruct | 88.33% | 80.23% | 100.00% |

## Fine-Tuned Model

- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Fine-tuning method: LoRA
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Training epochs: 3
- Learning rate: 2e-4
- Batch size: 1
- Gradient accumulation steps: 8
- Trainable parameters: 1,081,344
- Trainable parameter percentage: 0.2184%

## Hardware and Latency

- GPU: NVIDIA GeForce GTX 1650
- VRAM: 4 GB
- Average latency: 1.9053 seconds
- Minimum latency: 1.5365 seconds
- Maximum latency: 3.3843 seconds

These are initial experimental results measured on a desktop GPU. They are not edge-device deployment benchmarks.
