# DPO Baseline

This folder contains the basic CounselBench DPO baseline pipeline.

## 1. Build Preference Pairs

```bash
python src/baselines/build_dpo_pairs.py \
  --input data_raw/counselbench_eval.csv \
  --output-dir data/dpo \
  --min-score-diff 1.0
```

If Python is unavailable on Windows, the equivalent PowerShell fallback is:

```powershell
powershell -ExecutionPolicy Bypass -File src/baselines/build_dpo_pairs.ps1 `
  -InputPath data_raw/counselbench_eval.csv `
  -OutputDir data/dpo `
  -MinScoreDiff 1.0
```

Outputs:

```text
data/dpo/grouped_responses.jsonl
data/dpo/question_split.json
data/dpo/dpo_pairs_train.jsonl
data/dpo/dpo_pairs_dev.jsonl
data/dpo/dpo_pairs_test.jsonl
data/dpo/dpo_pair_report.json
```

The main baseline uses only `overall_score`. It aggregates expert annotations by
`(questionID, responder)` and creates preference pairs among responses to the
same question when their mean overall scores differ by at least 1 point.

## 2. Build CounselBench-Adv Generation Prompts

```bash
python src/baselines/build_eval_prompts.py \
  --adv-input data_raw/counselbench_adv.csv \
  --adv-output data/prompts/counselbench_adv_prompts.jsonl
```

## 3. Train DPO

Install training dependencies first:

```bash
pip install -r requirements-dpo.txt
```

Then run:

```bash
python src/baselines/train_dpo.py \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train-file data/dpo/dpo_pairs_train.jsonl \
  --dev-file data/dpo/dpo_pairs_dev.jsonl \
  --output-dir model/checkpoints/dpo_overall \
  --bf16
```

On a small GPU, add `--load-in-4bit` if your platform supports bitsandbytes.

## Notes

- Split is by `questionID`, not by CSV row, to avoid question leakage.
- Rationale/evidence fields are intentionally excluded from DPO.
- Generated output files are derived artifacts and can be regenerated from the raw CSV.
