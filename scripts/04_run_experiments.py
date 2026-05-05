import argparse
import subprocess
from pathlib import Path

import yaml


def run(cmd):
    print("\n" + "=" * 100)
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--push_to_hf", action="store_true")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    default_train = cfg.get("default_train", {})
    experiments = cfg["experiments"]

    if args.only:
        wanted = set(args.only)
        experiments = [e for e in experiments if e["id"] in wanted]

    for e in experiments:
        exp_id = e["id"]
        run_dir = Path(cfg["base_output_dir"]) / exp_id
        pred_dir = Path(cfg["prediction_output_dir"])
        adapter_dir = run_dir / "final_adapter"

        batch_size = e.get("batch_size", default_train.get("batch_size", 8))
        grad_accum = e.get("grad_accum", default_train.get("grad_accum", 2))
        lr = e.get("learning_rate", default_train.get("learning_rate", 0.0002))
        lora_r = e.get("lora_r", default_train.get("lora_r", 16))
        lora_dropout = e.get("lora_dropout", default_train.get("lora_dropout", 0.05))
        epochs = e.get("epochs", default_train.get("epochs", 2))
        save_steps = e.get("save_steps", default_train.get("save_steps", 500))
        eval_steps = e.get("eval_steps", default_train.get("eval_steps", 250))
        weight_decay = e.get("weight_decay", default_train.get("weight_decay", 0.01))

        hf_repo_id = f"{cfg['hf_namespace']}/{exp_id}"

        if not args.skip_train:
            cmd = [
                "python", "scripts/02_train_unsloth.py",
                "--experiment_id", exp_id,
                "--model_name", e["model_name"],
                "--train_file", e["train_file"],
                "--val_file", e["val_file"],
                "--output_dir", str(run_dir),
                "--max_seq_length", str(e.get("max_seq_length", 512)),
                "--epochs", str(epochs),
                "--learning_rate", str(lr),
                "--lora_r", str(lora_r),
                "--lora_dropout", str(lora_dropout),
                "--batch_size", str(batch_size),
                "--grad_accum", str(grad_accum),
                "--save_steps", str(save_steps),
                "--eval_steps", str(eval_steps),
                "--weight_decay", str(weight_decay),
            ]

            if args.push_to_hf:
                cmd += ["--push_to_hf", "--hf_repo_id", hf_repo_id]
                if args.private:
                    cmd += ["--private"]

            run(cmd)

        if not args.skip_eval:
            eval_prompt_type = "label"
            max_new_tokens = e.get("max_new_tokens", 16)

            run([
                "python", "scripts/03_eval_local_model.py",
                "--experiment_id", exp_id,
                "--base_model", e["model_name"],
                "--adapter_dir", str(adapter_dir),
                "--test_file", e["test_file"],
                "--output_dir", str(pred_dir),
                "--results_file", cfg["results_file"],
                "--max_seq_length", str(e.get("max_seq_length", 512)),
                "--max_new_tokens", str(max_new_tokens),
                "--eval_prompt_type", eval_prompt_type,
            ])


if __name__ == "__main__":
    main()
