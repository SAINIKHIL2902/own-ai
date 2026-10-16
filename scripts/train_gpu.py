#!/usr/bin/env bash
"""
Standalone Cloud GPU Training Script for own-ai (Phase 5).
Designed to run on Google Colab (Free NVIDIA T4 GPU) or Kaggle.

Usage in Google Colab:
    !pip install -q transformers datasets peft accelerate bitsandbytes trl
    python train_gpu.py --data_dir ./data --output_dir ./output --epochs 3
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter on Cloud GPU (Google Colab / Kaggle)")
    parser.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing train.jsonl and val.jsonl")
    parser.add_argument("--output_dir", type=str, default="./trained_adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    args = parser.parse_args()

    print("==================================================")
    print(" own-ai: Cloud GPU LoRA Training Pipeline")
    print(f" Base Model: {args.base_model}")
    print(f" Output Dir: {args.output_dir}")
    print(f" Epochs:     {args.epochs}")
    print("==================================================")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Target Compute Device: {device}")

    # Load dataset
    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")
    print(f"Loading data from {train_path}...")
    raw_datasets = load_dataset("json", data_files={"train": train_path, "validation": val_path})

    # Tokenizer & Formatting
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_chat(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(text, truncation=True, max_length=1024)

    tokenized_datasets = raw_datasets.map(format_chat, remove_columns=raw_datasets["train"].column_names)

    # Load model
    print(f"Loading base model {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=(device == "cuda"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving trained LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "status": "COMPLETED",
    }
    with open(os.path.join(args.output_dir, "adapter_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("LoRA training complete! Download the trained_adapter folder to your laptop.")


if __name__ == "__main__":
    main()
