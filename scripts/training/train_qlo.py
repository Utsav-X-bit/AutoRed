#!/usr/bin/env python3
"""
QLoRA SFT training for AutoRed attack generator.

Trains Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 with 4-bit QLoRA
on verified/positive AutoRed success datasets.

Usage:
    python scripts/training/train_qlo.py \
        --dataset scripts/training/sft_data/variantc_verified_train.jsonl \
        --val_dataset scripts/training/sft_data/variantc_verified_val.jsonl \
        --output_dir experiment/results/qlo_verified_v1 \
        --epochs 10

    # Positive dataset (291 samples)
    python scripts/training/train_qlo.py \
        --dataset scripts/training/sft_data/variantc_positive_train.jsonl \
        --val_dataset scripts/training/sft_data/variantc_positive_val.jsonl \
        --output_dir experiment/results/qlo_positive_v1 \
        --epochs 6
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTTrainer, SFTConfig


def load_dataset_from_jsonl(train_path, val_path=None):
    """Load train/val datasets from JSONL files."""
    print(f"Loading training data from {train_path}...")

    # Load and verify format
    train_data = []
    with open(train_path) as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
                # Verify it has messages format
                if "messages" not in entry:
                    print(f"  Warning: line {i+1} missing 'messages', skipping")
                    continue
                train_data.append(entry)
            except json.JSONDecodeError as e:
                print(f"  Warning: line {i+1} JSON error: {e}")

    print(f"  Loaded {len(train_data)} training samples")

    train_dataset = Dataset.from_list(train_data)

    val_dataset = None
    if val_path and Path(val_path).exists():
        val_data = []
        with open(val_path) as f:
            for i, line in enumerate(f):
                try:
                    entry = json.loads(line)
                    if "messages" in entry:
                        val_data.append(entry)
                except json.JSONDecodeError:
                    pass

        print(f"  Loaded {len(val_data)} validation samples")
        val_dataset = Dataset.from_list(val_data)

    return train_dataset, val_dataset


def format_prompt(entry):
    """Format a conversation entry for the model."""
    messages = entry["messages"]
    # Simple chat template formatting
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"<|user|>\n{content}</s>\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}</s>\n")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT training for AutoRed generator")
    parser.add_argument("--model_name", type=str,
                        default="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
                        help="Base model name or path")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to training JSONL file")
    parser.add_argument("--val_dataset", type=str, default=None,
                        help="Path to validation JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name (set to empty string to disable)")
    parser.add_argument("--run_name", type=str, default="autored_qlo",
                        help="Run name for logging")
    args = parser.parse_args()

    set_seed(args.seed)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {output_path / 'training_config.json'}")

    # Load dataset
    train_dataset, val_dataset = load_dataset_from_jsonl(args.dataset, args.val_dataset)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        fan_in_fan_out=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Ensure tokenizer has chat template for SFTTrainer
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}<|{{ message['role'] }}|>\n{{ message['content'] }}</s>\n{% endfor %}"

    # SFT Trainer
    print(f"\nSetting up SFTTrainer...")
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        fp16=False,
        bf16=True,
        dataloader_pin_memory=False,
        seed=args.seed,
        report_to="wandb" if args.wandb_project else "none",
        run_name=args.run_name,
        max_seq_length=args.max_length,
    )

    if args.wandb_project:
        sft_config.wandb_project = args.wandb_project

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    # Training
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max length: {args.max_length}")
    print(f"{'='*60}\n")

    train_result = trainer.train()

    # Save final model
    print(f"\nSaving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save metrics
    metrics = train_result.metrics
    metrics_dict = {k: round(float(v), 6) for k, v in metrics.items()}
    with open(output_path / "train_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Training metrics saved")
    print(f"  Train loss: {metrics_dict.get('train_loss', 'N/A')}")

    print(f"\nDone. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
