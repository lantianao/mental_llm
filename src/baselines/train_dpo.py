"""Train the CounselBench DPO baseline with TRL.

This script expects DPO JSONL files created by `build_dpo_pairs.py`.
It uses scalar-derived preferences only; rationale/evidence fields are excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DPO baseline with TRL.")
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base instruction model or local model path.",
    )
    parser.add_argument(
        "--train-file",
        default="data/dpo/dpo_pairs_train.jsonl",
        help="Training JSONL with prompt/chosen/rejected fields.",
    )
    parser.add_argument(
        "--dev-file",
        default="data/dpo/dpo_pairs_dev.jsonl",
        help="Validation JSONL with prompt/chosen/rejected fields.",
    )
    parser.add_argument("--output-dir", default="model/checkpoints/dpo_overall")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Imports live inside main so `--help` works even before ML dependencies are installed.
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    train_file = Path(args.train_file)
    dev_file = Path(args.dev_file)
    if not train_file.exists():
        raise FileNotFoundError(f"Missing train file: {train_file}")
    if not dev_file.exists():
        raise FileNotFoundError(f"Missing dev file: {dev_file}")

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(dev_file)},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if args.load_in_4bit:
        model_kwargs.update(
            {
                "load_in_4bit": True,
                "device_map": "auto",
            }
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        **model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    config_kwargs = {
        "output_dir": args.output_dir,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "report_to": "none",
        "bf16": args.bf16,
        "remove_unused_columns": False,
    }
    try:
        training_args = DPOConfig(**config_kwargs)
    except TypeError:
        # Older Transformers releases used `evaluation_strategy`.
        config_kwargs["evaluation_strategy"] = config_kwargs.pop("eval_strategy")
        training_args = DPOConfig(**config_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "peft_config": peft_config,
    }
    try:
        trainer = DPOTrainer(
            **trainer_kwargs,
            processing_class=tokenizer,
        )
    except TypeError:
        # Older TRL releases used `tokenizer`.
        trainer = DPOTrainer(
            **trainer_kwargs,
            tokenizer=tokenizer,
        )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
