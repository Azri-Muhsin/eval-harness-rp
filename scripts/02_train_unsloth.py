#!/usr/bin/env python3

import unsloth  # MUST be before trl / transformers / peft
from unsloth import FastLanguageModel

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import create_repo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_id", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=250)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--push_to_hf", action="store_true")
    p.add_argument("--hf_repo_id", default=None)
    p.add_argument("--private", action="store_true")
    return p.parse_args()


def format_chat(example, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def find_latest_checkpoint(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None

    checkpoints = []
    for p in output_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                step = int(p.name.split("-")[-1])
                checkpoints.append((step, str(p)))
            except ValueError:
                pass

    if not checkpoints:
        return None

    return sorted(checkpoints)[-1][1]


def main():
    args = parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {args.experiment_id}")
    print(f"Model: {args.model_name}")
    print(f"Output dir: {out}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    # ---- Qwen / TRL EOS-token safety fix ----
    print("Before token fix:")
    print("tokenizer.eos_token:", tokenizer.eos_token)
    print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
    print("tokenizer.pad_token:", tokenizer.pad_token)
    print("tokenizer.pad_token_id:", tokenizer.pad_token_id)

    # Qwen chat models usually use <|im_end|> as the chat end token.
    if "<|im_end|>" in tokenizer.get_vocab():
        tokenizer.eos_token = "<|im_end|>"
    elif tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.special_tokens_map.get("eos_token", None)

    # If pad token is missing, use EOS as PAD.
    if tokenizer.pad_token is None or tokenizer.pad_token == "<|PAD_TOKEN|>":
        tokenizer.pad_token = tokenizer.eos_token

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if hasattr(model, "generation_config"):
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("After token fix:")
    print("tokenizer.eos_token:", tokenizer.eos_token)
    print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
    print("tokenizer.pad_token:", tokenizer.pad_token)
    print("tokenizer.pad_token_id:", tokenizer.pad_token_id)

    if tokenizer.eos_token not in tokenizer.get_vocab():
        raise ValueError(f"EOS token still not in vocab: {tokenizer.eos_token}")
# -----------------------------------------

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_r * 2,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.val_file,
        },
    )

    dataset = dataset.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    fp16 = torch.cuda.is_available() and not bf16

    training_args = SFTConfig(
    output_dir=str(out),

    # Batch / optimization
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    weight_decay=args.weight_decay,

    # Logging / eval / save
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=args.eval_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=2,

    # Precision
    bf16=bf16,
    fp16=fp16,

    # SFT-specific settings
    dataset_text_field="text",
    max_length=args.max_seq_length,
    packing=False,

    eos_token=tokenizer.eos_token,
    pad_token=tokenizer.pad_token,

    # Misc
    report_to="none",
    seed=3407,
    )

    trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    )

    latest_checkpoint = find_latest_checkpoint(out)
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found. Starting fresh.")

    trainer.train(resume_from_checkpoint=latest_checkpoint)

    final_dir = out / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"Saved adapter to: {final_dir}")

    if args.push_to_hf:
        if not args.hf_repo_id:
            raise ValueError("--hf_repo_id is required when --push_to_hf is used")

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable is not set")

        create_repo(args.hf_repo_id, private=args.private, exist_ok=True, token=token)
        model.push_to_hub(args.hf_repo_id, token=token)
        tokenizer.push_to_hub(args.hf_repo_id, token=token)

        print(f"Pushed adapter to Hugging Face: {args.hf_repo_id}")


if __name__ == "__main__":
    main()
