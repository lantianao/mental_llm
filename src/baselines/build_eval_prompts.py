"""Build generation/evaluation prompt files for CounselBench baselines."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prompt_for_question(question: str) -> str:
    return (
        "Question:\n"
        f"{clean_text(question)}\n\n"
        "Please provide a supportive, safe, and clinically cautious response."
    )


def build_adv_prompts(input_path: Path, output_path: Path) -> None:
    rows: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            for mode, question in row.items():
                if not question or not question.strip():
                    continue
                rows.append(
                    {
                        "id": f"adv_{mode}_{row_idx:03d}",
                        "source_file": input_path.name,
                        "target_failure_mode": mode,
                        "question": clean_text(question),
                        "prompt": prompt_for_question(question),
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} prompts to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build prompt JSONL files for baseline generation."
    )
    parser.add_argument("--adv-input", default="data_raw/counselbench_adv.csv")
    parser.add_argument(
        "--adv-output",
        default="data/prompts/counselbench_adv_prompts.jsonl",
    )
    args = parser.parse_args()

    build_adv_prompts(Path(args.adv_input), Path(args.adv_output))


if __name__ == "__main__":
    main()
