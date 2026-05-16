"""Evaluate DPO preference pairs with pairwise accuracy and log-prob margins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DPO pairs by log-prob.")
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model name or local path.",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Optional LoRA adapter directory to load.",
    )
    parser.add_argument(
        "--test-file",
        default="data/dpo/dpo_pairs_test.jsonl",
        help="JSONL with prompt/chosen/rejected fields.",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (e.g., cuda, cpu). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file to write summary metrics.",
    )
    parser.add_argument(
        "--per-example-output",
        default=None,
        help="Optional JSONL file to write per-example scores.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_input_ids(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> tuple[list[int], int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    bos = tokenizer.bos_token_id
    bos_offset = 1 if bos is not None else 0

    # Truncate from the left if needed, preserving the response tail.
    total_len = bos_offset + len(prompt_ids) + len(response_ids)
    if total_len > max_length:
        overflow = total_len - max_length
        drop_from_prompt = min(len(prompt_ids), overflow)
        prompt_ids = prompt_ids[drop_from_prompt:]
        overflow -= drop_from_prompt
        if overflow > 0:
            response_ids = response_ids[overflow:]

    input_ids = ([] if bos is None else [bos]) + prompt_ids + response_ids
    response_start = len(input_ids) - len(response_ids)
    return input_ids, response_start


def response_logprob(
    model,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    device: torch.device,
) -> tuple[float, int]:
    input_ids, response_start = build_input_ids(
        tokenizer, prompt=prompt, response=response, max_length=max_length
    )
    if response_start >= len(input_ids):
        return float("nan"), 0

    input_tensor = torch.tensor([input_ids], device=device)
    with torch.inference_mode():
        logits = model(input_tensor).logits
        log_probs = torch.log_softmax(logits, dim=-1)

    # Align logits with next-token targets.
    target_ids = input_tensor[:, 1:]
    log_probs = log_probs[:, :-1, :]
    token_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # Response tokens correspond to the tail of the sequence.
    response_len = len(input_ids) - response_start
    start = max(response_start - 1, 0)
    end = start + response_len
    response_logps = token_logps[:, start:end]
    return float(response_logps.sum().item()), response_len


def main() -> None:
    args = parse_args()

    # Lazy imports so help works without heavy deps.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs.update(
                {
                    "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
                    "device_map": "auto",
                }
            )
        except Exception:
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
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    if not args.load_in_4bit:
        model.to(device)
    model.eval()

    rows = load_jsonl(Path(args.test_file))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    correct = 0
    total = 0
    margins: list[float] = []
    chosen_logps: list[float] = []
    rejected_logps: list[float] = []
    total_tokens = 0

    per_example_out = None
    if args.per_example_output:
        per_example_out = Path(args.per_example_output).open("w", encoding="utf-8")

    for row in rows:
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        chosen_logp, chosen_len = response_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            response=chosen,
            max_length=args.max_length,
            device=device,
        )
        rejected_logp, rejected_len = response_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            response=rejected,
            max_length=args.max_length,
            device=device,
        )

        if not torch.isfinite(torch.tensor([chosen_logp, rejected_logp])).all():
            continue

        margin = chosen_logp - rejected_logp
        is_correct = margin > 0
        correct += int(is_correct)
        total += 1
        margins.append(margin)
        chosen_logps.append(chosen_logp)
        rejected_logps.append(rejected_logp)
        total_tokens += chosen_len + rejected_len

        if per_example_out is not None:
            per_example_out.write(
                json.dumps(
                    {
                        "question_id": row.get("question_id"),
                        "topic": row.get("topic"),
                        "chosen_logp": chosen_logp,
                        "rejected_logp": rejected_logp,
                        "margin": margin,
                        "correct": is_correct,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if per_example_out is not None:
        per_example_out.close()

    summary = {
        "num_examples": total,
        "pairwise_accuracy": (correct / total) if total else 0.0,
        "mean_margin": (sum(margins) / total) if total else 0.0,
        "mean_chosen_logp": (sum(chosen_logps) / total) if total else 0.0,
        "mean_rejected_logp": (sum(rejected_logps) / total) if total else 0.0,
        "total_tokens": total_tokens,
        "max_length": args.max_length,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
