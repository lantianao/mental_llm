# Summer 2026 Mental LLM Project File Inventory

This folder keeps code and documentation at the project root, while raw CSV data lives under `data_raw/`. Raw files are intentionally left unmodified so notebooks, IDE tabs, and future scripts can trace every derived artifact back to its source.

## Project Documents

| File | Role |
| --- | --- |
| `docs/Mental LLM intro.docx` | Project motivation and high-level research framing for rationale-as-supervision. |
| `docs/Mental LLM plan.docx` | Technical execution plan with stages for data prep, scalar RM, DPO, rationale critic, refinement, and evaluation. |
| `docs/2506.08584v3.pdf` | CounselBench paper. Describes CounselBench-Eval and CounselBench-Adv, annotation protocol, six evaluation dimensions, and failure modes. |

## Benchmark Data

| File | Rows | Main Use |
| --- | ---: | --- |
| `data_raw/counselbench_eval.csv` | 2,000 | Main supervised data. Contains 100 questions x 4 responders x 5 expert annotations. This is the primary source for scalar RM, DPO pairs, and rationale/evidence supervision. |
| `data_raw/counselbench_adv.csv` | 20 | Adversarial prompt set. Six columns x 20 prompts = 120 adversarial questions targeting six failure modes. |
| `data_raw/counselbench-adv-human-annotation.csv` | 1,080 | Human labels for nine model responses on 120 adversarial prompts. Useful for judge/critic evaluation. |
| `data_raw/20220401_counsel_chat.csv` | 2,775 | Original CounselChat question-answer data. Useful for background, additional prompt pools, and source tracing, but not the main DPO training data. |

## Important Field Notes

### `counselbench_eval.csv`

- Unit of row: one expert annotation for one `(questionID, responder)` response.
- Main DPO fields: `questionID`, `questionTitle`, `questionText`, `topic`, `responder`, `response`, `survey_id`, `overall_score`.
- Rationale/evidence fields: `overall_reason`, `medical_copy`, `medical_reason`, `factual_copy`, `factual_reason`, `toxicity_copy`, `toxicity_reason`.
- Safety/quality dimensions: `empathy_score`, `specificity_score`, `medical_advice_score`, `factual_consistency_score`, `toxicity_score`.
- For a clean DPO baseline, use scalar-derived preferences only and do not use rationale or evidence fields.

### `counselbench_adv.csv`

Columns are the six targeted adversarial failure modes:

- `apathetic`
- `assumptions`
- `symptoms`
- `judgmental`
- `medication`
- `therapy`

### `counselbench-adv-human-annotation.csv`

- `mode`: target failure mode.
- `new_post`: adversarial prompt.
- `model_name`: model that generated the answer.
- `new_response`: model answer.
- `human_answer`: whether the response contains the targeted issue (`yes`, `no`, `not sure`).

## Suggested Generated Artifacts

Scripts in `src/baselines/` write derived data files under `data/` by default:

```text
data/
  dpo/
    grouped_responses.jsonl
    question_split.json
    dpo_pairs_train.jsonl
    dpo_pairs_dev.jsonl
    dpo_pairs_test.jsonl
    dpo_pair_report.json
  prompts/
    counselbench_adv_prompts.jsonl
```

These outputs are derived from the raw data and can be regenerated.
