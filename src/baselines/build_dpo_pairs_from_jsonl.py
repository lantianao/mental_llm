"""Build score-derived DPO preference pairs from preprocessed JSONL data.

This script consumes the per-annotation JSONL files produced by the
`5_indep_samples_in_test` preparation. It does not modify the source data.
Instead, it aggregates the five annotations for each (question_id, responder)
into a mean overall score, then constructs DPO pairs within each question.

Unlike the CSV pipeline, this script respects the existing train/dev/test split
already encoded by the directory structure (train.jsonl, dev.jsonl, test.jsonl).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


DEFAULT_INPUT_DIR = "data_preparation/5_indep_samples_in_test/data"
DEFAULT_OUTPUT_DIR = "data/dpo"


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"i am not sure", "not sure", "nan"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def format_prompt(question: str | None) -> str:
    question = (question or "").strip()
    if not question:
        return "Please provide a supportive, safe, and clinically cautious response."
    return (
        "Question:\n"
        f"{question}\n\n"
        "Please provide a supportive, safe, and clinically cautious response."
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_overall_score(row: dict[str, Any]) -> float | None:
    scores = row.get("scores") or {}
    overall = scores.get("overall")
    if isinstance(overall, dict):
        value = overall.get("normalized")
        if value is None:
            value = overall.get("raw")
    else:
        value = overall
    return parse_float(value)


def group_responses(
    rows: Iterable[dict[str, Any]],
    split: str,
) -> tuple[list[dict[str, Any]], int]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        question_id = metadata.get("question_id")
        responder = metadata.get("responder")
        if not question_id or not responder:
            continue
        groups[(question_id, responder)].append(row)

    grouped: list[dict[str, Any]] = []
    mismatched_responses = 0
    for (question_id, responder), group_rows in sorted(groups.items()):
        first = group_rows[0]
        scores = [
            score
            for score in (extract_overall_score(row) for row in group_rows)
            if score is not None
        ]
        if not scores:
            continue
        responses = {row.get("response", "") for row in group_rows}
        if len(responses) > 1:
            mismatched_responses += 1
        response = next(iter(responses)) if responses else ""
        metadata = first.get("metadata") or {}
        grouped.append(
            {
                "question_id": question_id,
                "topic": metadata.get("topic", ""),
                "responder": responder,
                "prompt": format_prompt(first.get("question")),
                "response": response,
                "mean_overall_score": mean(scores),
                "std_overall_score": pstdev(scores) if len(scores) > 1 else 0.0,
                "num_annotations": len(scores),
                "annotation_scores": scores,
                "split": split,
            }
        )
    return grouped, mismatched_responses


def build_pairs(
    grouped: list[dict[str, Any]],
    min_score_diff: float,
    split: str,
) -> list[dict[str, Any]]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in grouped:
        by_question[item["question_id"]].append(item)

    pairs: list[dict[str, Any]] = []
    for question_id, responses in sorted(by_question.items()):
        for left, right in combinations(responses, 2):
            diff = left["mean_overall_score"] - right["mean_overall_score"]
            if abs(diff) < min_score_diff:
                continue
            chosen, rejected = (left, right) if diff > 0 else (right, left)
            pairs.append(
                {
                    "question_id": question_id,
                    "topic": chosen.get("topic", ""),
                    "split": split,
                    "prompt": chosen["prompt"],
                    "chosen": chosen["response"],
                    "rejected": rejected["response"],
                    "chosen_responder": chosen["responder"],
                    "rejected_responder": rejected["responder"],
                    "chosen_score": round(chosen["mean_overall_score"], 4),
                    "rejected_score": round(rejected["mean_overall_score"], 4),
                    "score_diff": round(
                        chosen["mean_overall_score"]
                        - rejected["mean_overall_score"],
                        4,
                    ),
                    "preference_source": "mean_overall_score",
                }
            )
    return pairs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_report(
    grouped: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    question_ids_by_split: dict[str, set[str]],
    min_score_diff: float,
    mismatched_responses: int,
) -> dict[str, Any]:
    pair_split_counts = Counter(pair["split"] for pair in pairs)
    responder_wins = Counter(pair["chosen_responder"] for pair in pairs)
    responder_losses = Counter(pair["rejected_responder"] for pair in pairs)
    topic_pairs = Counter(pair["topic"] for pair in pairs)
    diffs = [pair["score_diff"] for pair in pairs]

    return {
        "min_score_diff": min_score_diff,
        "num_grouped_question_responses": len(grouped),
        "num_questions": sum(len(ids) for ids in question_ids_by_split.values()),
        "question_split_counts": {
            split: len(ids) for split, ids in sorted(question_ids_by_split.items())
        },
        "num_pairs": len(pairs),
        "pair_split_counts": dict(sorted(pair_split_counts.items())),
        "pair_question_split_counts": {
            split: len({pair["question_id"] for pair in pairs if pair["split"] == split})
            for split in sorted(question_ids_by_split)
        },
        "chosen_responder_counts": dict(sorted(responder_wins.items())),
        "rejected_responder_counts": dict(sorted(responder_losses.items())),
        "topic_pair_counts": dict(sorted(topic_pairs.items())),
        "score_diff": {
            "min": min(diffs) if diffs else None,
            "max": max(diffs) if diffs else None,
            "mean": mean(diffs) if diffs else None,
        },
        "response_mismatch_groups": mismatched_responses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DPO preference pairs from preprocessed JSONL splits."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-score-diff", type=float, default=1.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    split_paths = {
        "train": input_dir / "train.jsonl",
        "dev": input_dir / "dev.jsonl",
        "test": input_dir / "test.jsonl",
    }
    for split, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {split} file: {path}. Provide --input-dir with train/dev/test."
            )

    all_grouped: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    question_ids_by_split: dict[str, set[str]] = defaultdict(set)
    mismatched_responses = 0

    for split, path in split_paths.items():
        rows = load_jsonl(path)
        grouped, split_mismatches = group_responses(rows, split)
        pairs = build_pairs(grouped, args.min_score_diff, split)
        write_jsonl(output_dir / f"dpo_pairs_{split}.jsonl", pairs)
        all_grouped.extend(grouped)
        all_pairs.extend(pairs)
        mismatched_responses += split_mismatches
        question_ids_by_split[split].update(
            {item["question_id"] for item in grouped}
        )

    write_jsonl(output_dir / "grouped_responses.jsonl", all_grouped)
    question_split_rows = [
        {"question_id": question_id, "split": split}
        for split, ids in sorted(question_ids_by_split.items())
        for question_id in sorted(ids)
    ]
    (output_dir / "question_split.json").write_text(
        json.dumps(question_split_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = build_report(
        grouped=all_grouped,
        pairs=all_pairs,
        question_ids_by_split=question_ids_by_split,
        min_score_diff=args.min_score_diff,
        mismatched_responses=mismatched_responses,
    )
    report_path = output_dir / "dpo_pair_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
