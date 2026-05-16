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
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints. Use with save strategy.",
    )
    parser.add_argument(
        "--save-only-model",
        action="store_true",
        help="Save only model weights (no optimizer/scheduler states).",
    )
    parser.add_argument(
        "--report-to",
        default="tensorboard",
        help="Logging backend (e.g., tensorboard, wandb, none).",
    )
    parser.add_argument(
        "--logging-dir",
        default=None,
        help="Directory for training logs (defaults to <output-dir>/logs).",
    )
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
        # Newer Transformers expects a BitsAndBytesConfig instead of load_in_4bit.
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs.update(
                {
                    "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
                    "device_map": "auto",
                }
            )
        except Exception:
            # Fallback for older Transformers that still accept load_in_4bit.
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

    logging_dir = args.logging_dir
    if not logging_dir:
        logging_dir = str(Path(args.output_dir) / "logs")

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
        "logging_dir": logging_dir,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "save_only_model": args.save_only_model,
        "eval_strategy": "epoch",
        "report_to": args.report_to,
        "bf16": args.bf16,
        "remove_unused_columns": False,
    }
    import inspect

    config_local = dict(config_kwargs)
    dpo_sig = inspect.signature(DPOConfig.__init__)
    valid_keys = set(dpo_sig.parameters.keys()) - {"self"}
    if "eval_strategy" in config_local and "eval_strategy" not in valid_keys:
        # Older Transformers releases used `evaluation_strategy`.
        if "evaluation_strategy" in valid_keys:
            config_local["evaluation_strategy"] = config_local.pop("eval_strategy")
    # Drop any unsupported keys (e.g., max_prompt_length in some TRL versions).
    for key in list(config_local.keys()):
        if key not in valid_keys:
            config_local.pop(key)

    training_args = DPOConfig(**config_local)

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
