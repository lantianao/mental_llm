# DPO Baseline

This folder contains the basic CounselBench DPO baseline pipeline.

## 1. Build Preference Pairs

```bash
python src/baselines/build_dpo_pairs.py \
  --input data_raw/counselbench_eval.csv \
  --output-dir data/dpo \
  --min-score-diff 1.0
```

If you want to use the preprocessed JSONL splits (e.g., the
`5_indep_samples_in_test` dataset) without modifying the source data, use:

```bash
python src/baselines/build_dpo_pairs_from_jsonl.py \
  --input-dir /project2/ruishanl_1185/rationale_as_supervison/data_preparation/5_indep_samples_in_test/data \
  --output-dir data/dpo \
  --min-score-diff 1.0
```

This version respects the existing train/dev/test split and aggregates the
five annotations for each `(question_id, responder)` to derive preferences.

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

### Visualization (TensorBoard)

Training logs are written to `<output-dir>/logs` by default. Launch TensorBoard:

```bash
tensorboard --logdir model/checkpoints/dpo_overall/logs --bind_all --port 6006
```

If you do not want logging, pass `--report-to none`.

## 4. Evaluate DPO Pairs

Compute pairwise accuracy on the test split by comparing log-probabilities of
the chosen vs rejected responses:

```bash
python src/baselines/eval_dpo_pairs.py \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --adapter-dir model/checkpoints/dpo_overall \
  --test-file data/dpo/dpo_pairs_test.jsonl \
  --load-in-4bit
```

The script prints a JSON summary with pairwise accuracy and mean margins.

## Notes

- Split is by `questionID`, not by CSV row, to avoid question leakage.
- Rationale/evidence fields are intentionally excluded from DPO.
- Generated output files are derived artifacts and can be regenerated from the raw CSV.
