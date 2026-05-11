# Mental LLM DPO Baseline

This repository contains the initial DPO baseline pipeline for the CounselBench-based mental-health LLM project.

## Current Pipeline

1. Build DPO preference pairs from `data_raw/counselbench_eval.csv`.
2. Train a DPO policy model from `meta-llama/Meta-Llama-3.1-8B-Instruct`.
3. Save checkpoints under `model/checkpoints/`.

Raw data and local model artifacts are intentionally excluded from git. Lightweight processed DPO files under `data/dpo/` are tracked so the baseline can be reproduced quickly on USC CARC.

## Data Preparation

Python:

```bash
python src/baselines/build_dpo_pairs.py --input data_raw/counselbench_eval.csv --output-dir data/dpo --min-score-diff 1.0
```

PowerShell fallback:

```powershell
powershell -ExecutionPolicy Bypass -File src/baselines/build_dpo_pairs.ps1 `
  -InputPath data_raw/counselbench_eval.csv `
  -OutputDir data/dpo `
  -MinScoreDiff 1.0
```

## DPO Training

```bash
python src/baselines/train_dpo.py \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train-file data/dpo/dpo_pairs_train.jsonl \
  --dev-file data/dpo/dpo_pairs_dev.jsonl \
  --output-dir model/checkpoints/dpo_overall \
  --bf16
```
