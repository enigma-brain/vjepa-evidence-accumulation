#!/bin/bash
# Feature extraction as an array of small 1-GPU jobs (each takes 1/4 of the videos).
# Small requests backfill much faster on FarmShare than one 4-GPU job.
# Usage: sbatch extract.sbatch                 (validation split)
#        SPLIT=train sbatch extract.sbatch
#SBATCH --job-name=vjepa-extract
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate ${EVACC_ENV:-evacc}

# log GPU utilization every 10 s alongside the job
nvidia-smi dmon -s u -d 10 > logs/dmon-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log &
DMON=$!

python extract_features.py --split ${SPLIT:-validation} \
    --shard $SLURM_ARRAY_TASK_ID --num-shards ${SLURM_ARRAY_TASK_COUNT} \
    --num-workers 7 --batch-size ${BATCH_SIZE:-64}

kill $DMON
