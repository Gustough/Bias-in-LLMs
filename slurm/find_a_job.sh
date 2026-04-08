#!/usr/bin/env bash
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --nodes=1
#SBATCH -t 0-010:00:0
#SBATCH --output=slurm/logs/test_query-%j.out
#SBATCH -A NAISS2026-4-124

export HF_HOME=/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache

CONTAINER=/cephyr/users/gusenge/Alvis/hf_llm_inference.sif

echo "Starting at"
date

# ---- Run inside container ----
apptainer exec --nv \
    --bind /mimer/NOBACKUP/groups/naiss2026-4-124/gustav:/mimer/NOBACKUP/groups/naiss2026-4-124/gustav \
    --env HF_HOME=$HF_HOME \
    $CONTAINER \
    python scripts/prompt_model.py

echo "Finishing at:"
date
