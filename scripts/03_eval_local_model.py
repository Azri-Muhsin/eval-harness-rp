import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from unsloth import FastLanguageModel
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_id", required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--results_file", required=True)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--eval_prompt_type", choices=["label", "reasoning"], default="label")
    return p.parse_args()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def normalize_label(x):
    return x.strip().replace(" ", "_").replace("-", "_")


def extract_label(text, allowed_labels):
    text = text.strip()
    allowed = set(allowed_labels)

    m = re.search(r"Final label\s*:\s*([A-Za-z0-9_ -]+)", text, re.IGNORECASE)
    if m:
        candidate = normalize_label(m.group(1))
        if candidate in allowed:
            return candidate

    first_line = normalize_label(text.splitlines()[0]) if text else ""
    if first_line in allowed:
        return first_line

    for label in allowed_labels:
        if label.lower() in text.lower():
            return label

    return "PARSE_FAIL"


def build_prompt(record, tokenizer, eval_prompt_type):
    if eval_prompt_type == "reasoning":
        system = (
            "You are a healthcare intent classifier. "
            "Return the final label first using 'Final label: <label>', then give a short reason."
        )
    else:
        system = (
            "You are a healthcare intent classifier. "
            "Return only one intent label from the allowed label set."
        )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Query: {record['user_query']}"},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def save_breakdowns(pred_df, output_dir, experiment_id):
    output_dir = Path(output_dir)

    per_language = []
    for lang, g in pred_df.groupby("language"):
        per_language.append({
            "language": lang,
            "n": len(g),
            "accuracy": accuracy_score(g["gold_intent"], g["pred_intent"]),
            "macro_f1": f1_score(g["gold_intent"], g["pred_intent"], average="macro", zero_division=0),
        })

    pd.DataFrame(per_language).to_csv(
        output_dir / f"{experiment_id}_per_language.csv",
        index=False,
    )

    per_intent = []
    for intent, g in pred_df.groupby("gold_intent"):
        per_intent.append({
            "intent": intent,
            "n": len(g),
            "accuracy": accuracy_score(g["gold_intent"], g["pred_intent"]),
            "f1": f1_score(g["gold_intent"], g["pred_intent"], average="macro", zero_division=0),
        })

    pd.DataFrame(per_intent).to_csv(
        output_dir / f"{experiment_id}_per_intent.csv",
        index=False,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.test_file)
    allowed_labels = sorted(set(r["intent"] for r in records))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = PeftModel.from_pretrained(model, args.adapter_dir)
    FastLanguageModel.for_inference(model)

    y_true, y_pred = [], []
    pred_records = []

    for r in tqdm(records, desc=f"Evaluating {args.experiment_id}"):
        prompt = build_prompt(r, tokenizer, args.eval_prompt_type)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_length,
        ).to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        pred = extract_label(generated, allowed_labels)

        y_true.append(r["intent"])
        y_pred.append(pred)

        pred_records.append({
            "experiment_id": args.experiment_id,
            "language": r.get("language"),
            "user_query": r.get("user_query"),
            "gold_intent": r["intent"],
            "pred_intent": pred,
            "raw_output": generated,
        })

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    parse_fail_rate = sum(1 for p in y_pred if p == "PARSE_FAIL") / len(y_pred)

    pred_path = output_dir / f"{args.experiment_id}_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in pred_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"{args.experiment_id}_classification_report.csv"
    )

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        output_dir / f"{args.experiment_id}_confusion_matrix.csv"
    )

    pred_df = pd.DataFrame(pred_records)
    save_breakdowns(pred_df, output_dir, args.experiment_id)

    append_csv(args.results_file, {
        "experiment_id": args.experiment_id,
        "eval_kind": "local_sft",
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "test_file": args.test_file,
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "parse_fail_rate": round(parse_fail_rate, 6),
        "n_test": len(records),
    })

    print({
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "parse_fail_rate": parse_fail_rate,
    })


if __name__ == "__main__":
    main()
