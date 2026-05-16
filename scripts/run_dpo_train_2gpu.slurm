#!/bin/bash
#SBATCH --job-name=dpo_train_2gpu
#SBATCH --partition=gpu
#SBATCH --account=ruishanl_1185
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dpo_train_2gpu_%j.out
#SBATCH --error=logs/dpo_train_2gpu_%j.err

set -euo pipefail

module purge
module load python/3.11.9
module load gcc/13.3.0
module load cuda/12.4.0

cd /home1/tianaola/mental_llm/mental_llm
source .venv/bin/activate

export HUGGINGFACE_HUB_CACHE=/scratch1/tianaola/hf
export TRANSFORMERS_CACHE=/scratch1/tianaola/hf
export OMP_NUM_THREADS=8

mkdir -p logs model/checkpoints "$HUGGINGFACE_HUB_CACHE"

torchrun --nproc_per_node=2 src/baselines/train_dpo.py \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train-file data/dpo/dpo_pairs_train.jsonl \
  --dev-file data/dpo/dpo_pairs_dev.jsonl \
  --output-dir model/checkpoints/dpo_overall_2gpu \
  --epochs 1.0 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --max-length 2048 \
  --bf16
