#!/bin/bash
#SBATCH --job-name=dpo_smoke
#SBATCH --partition=gpu
#SBATCH --account=ruishanl_1185
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dpo_smoke_%j.out
#SBATCH --error=logs/dpo_smoke_%j.err

set -euo pipefail

module purge
module load python/3.11.9
module load gcc/13.3.0
module load cuda/12.4.0

cd /home1/tianaola/mental_llm/mental_llm
source .venv/bin/activate

export HUGGINGFACE_HUB_CACHE=/scratch1/tianaola/hf
export TRANSFORMERS_CACHE=/scratch1/tianaola/hf

mkdir -p logs model/checkpoints "$HUGGINGFACE_HUB_CACHE"

python src/baselines/train_dpo.py \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train-file data/dpo/dpo_pairs_train.jsonl \
  --dev-file data/dpo/dpo_pairs_dev.jsonl \
  --output-dir model/checkpoints/dpo_overall_smoke \
  --epochs 0.05 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-length 1024 \
  --max-prompt-length 512 \
  --load-in-4bit
