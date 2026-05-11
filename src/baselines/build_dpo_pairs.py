"""Build score-derived DPO preference pairs from CounselBench-Eval.

This script prepares the data for the project's DPO baseline.

CounselBench-Eval is not already in DPO format. Each CSV row is one clinician's
annotation of one `(question, responder)` answer. DPO, however, needs pairwise
preference examples:

    prompt, chosen_response, rejected_response

So the processing pipeline is:

1. Clean the user question and model/human responses.
2. Aggregate the five clinician `overall_score` annotations for each
   `(questionID, responder)` answer.
3. Split by `questionID`, not by CSV row, to avoid question leakage.
4. Compare answers to the same question and keep pairs where the mean overall
   score differs by at least `--min-score-diff`.
5. Write train/dev/test JSONL files with `prompt`, `chosen`, and `rejected`.

Important experimental choice:

The DPO baseline intentionally uses only scalar-derived preferences from
`overall_score`. It does not use expert rationales (`*_reason`) or evidence
spans (`*_copy`). Those fields are reserved for the rationale-critic method.
Keeping them out of DPO makes the comparison fair: DPO represents standard
preference learning, while the main method tests whether expert rationales add
value beyond scalar preferences.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_INPUT = "data_raw/counselbench_eval.csv"
DEFAULT_OUTPUT_DIR = "data/dpo"


def clean_text(value: str | None) -> str:
    """Normalize light HTML and whitespace from CSV text fields.

    CounselChat questions often contain HTML fragments such as `<br>`. Cleaning
    them before training keeps the prompt close to natural user text and avoids
    teaching the model irrelevant web-format artifacts.
    """
    if value is None:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_float(value: str | None) -> float | None:
    """Parse numeric score fields while ignoring uncertainty labels.

    `overall_score` should be numeric, but this helper is shared with possible
    future score variants where fields may contain "I am not sure".
    """
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"i am not sure", "not sure", "nan"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_prompt(row: dict[str, str]) -> str:
    """Create the DPO prompt from the question title and body.

    We include both `questionTitle` and `questionText` because the title can
    contain useful context that is not always repeated in the body. The final
    instruction keeps generation aligned with the mental-health QA setting
    without exposing any expert labels or rationales.
    """
    title = clean_text(row.get("questionTitle"))
    body = clean_text(row.get("questionText"))
    prompt_parts = []
    if title:
        prompt_parts.append(f"Title: {title}")
    if body:
        prompt_parts.append(f"Question:\n{body}")
    prompt_parts.append(
        "Please provide a supportive, safe, and clinically cautious response."
    )
    return "\n\n".join(prompt_parts)


def load_eval_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_responses(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Aggregate five expert annotations per question-response pair.

    Raw unit:
        one clinician annotation for one answer.

    DPO unit before pair construction:
        one unique answer to one question, with a stable mean expert score.

    We group by `(questionID, responder)` because CounselBench-Eval has four
    responders per question (`gpt4`, `gemini`, `llama3`, `human`) and five expert
    annotations for each response. Averaging the five `overall_score` values
    reduces individual annotator noise and gives a scalar quality estimate for
    pairwise comparison.
    """
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["questionID"], row["responder"])].append(row)

    grouped: list[dict[str, Any]] = []
    for (question_id, responder), group_rows in sorted(groups.items()):
        first = group_rows[0]
        overall_scores = [
            score
            for score in (parse_float(row.get("overall_score")) for row in group_rows)
            if score is not None
        ]
        if not overall_scores:
            continue

        grouped.append(
            {
                "question_id": question_id,
                "topic": first.get("topic", ""),
                "responder": responder,
                "prompt": format_prompt(first),
                "response": clean_text(first.get("response")),
                "mean_overall_score": mean(overall_scores),
                "std_overall_score": pstdev(overall_scores)
                if len(overall_scores) > 1
                else 0.0,
                "num_annotations": len(overall_scores),
                "annotation_scores": overall_scores,
            }
        )
    return grouped


def split_question_ids(
    grouped: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Split by question id to prevent question-level leakage.

    If we randomly split CSV rows or DPO pairs, responses to the same mental
    health question could appear in both train and test. That would make the
    evaluation too optimistic because the model has already seen the same user
    situation during training. Splitting whole `questionID`s keeps train/dev/test
    independent at the prompt level.
    """
    by_topic: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for item in grouped:
        question_id = item["question_id"]
        if question_id in seen:
            continue
        seen.add(question_id)
        by_topic[item.get("topic") or "unknown"].append(question_id)

    rng = random.Random(seed)
    question_ids: list[str] = []
    for topic in sorted(by_topic):
        ids = sorted(by_topic[topic])
        rng.shuffle(ids)
        question_ids.extend(ids)

    # Shuffle after topic interleaving so the split is deterministic but not
    # sorted by topic or question id.
    rng.shuffle(question_ids)

    n_total = len(question_ids)
    n_train = round(n_total * train_ratio)
    n_dev = round(n_total * dev_ratio)

    split_map: dict[str, str] = {}
    for idx, question_id in enumerate(question_ids):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_dev:
            split = "dev"
        else:
            split = "test"
        split_map[question_id] = split
    return split_map


def build_pairs(
    grouped: list[dict[str, Any]],
    split_map: dict[str, str],
    min_score_diff: float,
) -> list[dict[str, Any]]:
    """Build DPO preference pairs within each question.

    DPO preferences should compare answers to the same prompt. Therefore, for
    each question we compare the four responder answers pairwise. The higher
    mean `overall_score` answer becomes `chosen`; the lower-scored answer becomes
    `rejected`.

    We filter out small score gaps with `min_score_diff` because mental-health
    quality ratings are subjective. A tiny difference such as 0.2 may reflect
    annotator noise rather than a reliable preference. The default threshold of
    1.0 creates fewer but cleaner training pairs.
    """
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
                    "split": split_map[question_id],
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
    split_map: dict[str, str],
    min_score_diff: float,
) -> dict[str, Any]:
    split_counts = Counter(split_map.values())
    pair_split_counts = Counter(pair["split"] for pair in pairs)
    responder_wins = Counter(pair["chosen_responder"] for pair in pairs)
    responder_losses = Counter(pair["rejected_responder"] for pair in pairs)
    topic_pairs = Counter(pair["topic"] for pair in pairs)
    diffs = [pair["score_diff"] for pair in pairs]

    return {
        "min_score_diff": min_score_diff,
        "num_grouped_question_responses": len(grouped),
        "num_questions": len(split_map),
        "num_questions_with_pairs": len({pair["question_id"] for pair in pairs}),
        "question_split_counts": dict(sorted(split_counts.items())),
        "num_pairs": len(pairs),
        "pair_split_counts": dict(sorted(pair_split_counts.items())),
        "pair_question_split_counts": {
            split: len(
                {pair["question_id"] for pair in pairs if pair["split"] == split}
            )
            for split in ("train", "dev", "test")
        },
        "chosen_responder_counts": dict(sorted(responder_wins.items())),
        "rejected_responder_counts": dict(sorted(responder_losses.items())),
        "topic_pair_counts": dict(sorted(topic_pairs.items())),
        "score_diff": {
            "min": min(diffs) if diffs else None,
            "max": max(diffs) if diffs else None,
            "mean": mean(diffs) if diffs else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DPO preference pairs from CounselBench-Eval."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-score-diff", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    rows = load_eval_rows(input_path)
    grouped = group_responses(rows)
    split_map = split_question_ids(
        grouped=grouped,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    pairs = build_pairs(
        grouped=grouped,
        split_map=split_map,
        min_score_diff=args.min_score_diff,
    )

    write_jsonl(output_dir / "grouped_responses.jsonl", grouped)
    for split in ("train", "dev", "test"):
        split_pairs = [pair for pair in pairs if pair["split"] == split]
        write_jsonl(output_dir / f"dpo_pairs_{split}.jsonl", split_pairs)

    question_split_rows = [
        {"question_id": question_id, "split": split}
        for question_id, split in sorted(split_map.items())
    ]
    (output_dir / "question_split.json").write_text(
        json.dumps(question_split_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = build_report(grouped, pairs, split_map, args.min_score_diff)
    report_path = output_dir / "dpo_pair_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
