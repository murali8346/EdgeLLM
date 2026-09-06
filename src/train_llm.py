import json
import torch

from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen_restaurant_lora"

MAX_LENGTH = 256


def build_prompt(messages):
    system_message = messages[0]["content"]
    user_message = messages[1]["content"]
    assistant_message = messages[2]["content"]

    return (
        f"<|im_start|>system\n"
        f"{system_message}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{assistant_message}<|im_end|>"
    )


def tokenize_example(example, tokenizer):
    prompt = build_prompt(example["messages"])

    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check the PyTorch installation first."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "validation": str(DATA_DIR / "validation.jsonl"),
        },
    )

    tokenized_datasets = raw_datasets.map(
        lambda example: tokenize_example(example, tokenizer),
        remove_columns=raw_datasets["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        optim="adamw_torch",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    print("\nStarting LoRA fine-tuning...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nTraining completed.")
    print("Saved model to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()