#!/bin/bash
#SBATCH --job-name=dpo_train_4gpu_50e
#SBATCH --partition=gpu
#SBATCH --account=ruishanl_1185
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/dpo_train_4gpu_50e_%j.out
#SBATCH --error=logs/dpo_train_4gpu_50e_%j.err

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

# Effective batch size = per_device_train_batch_size * num_gpus * grad_accum
# 2 * 4 * 2 = 16 (4 GPUs total across 2 nodes)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MASTER_ADDR MASTER_PORT

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  /usr/bin/bash -lc "cd /home1/tianaola/mental_llm/mental_llm && source .venv/bin/activate && \
  torchrun --nnodes=${SLURM_NNODES} --nproc_per_node=2 --node_rank=\${SLURM_NODEID} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} src/baselines/train_dpo.py \
  --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train-file data/dpo/dpo_pairs_train.jsonl \
  --dev-file data/dpo/dpo_pairs_dev.jsonl \
  --output-dir model/checkpoints/dpo_overall_4gpu_50e \
  --epochs 50.0 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 2 \
  --max-length 2048 \
  --bf16 \
  --save-total-limit 5"
