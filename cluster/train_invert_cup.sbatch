#!/usr/bin/env bash
#SBATCH --job-name=dexlink-invert-cup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -eo pipefail
source ~/.bashrc
conda activate "${DEXLINK_CONDA_ENV:-genesis}"
set -u

: "${DEXLINK_ROOT:?Set DEXLINK_ROOT to the persistent public checkout}"
: "${DEXLINK_RUN_DIRECTORY:?Set a fresh persistent DEXLINK_RUN_DIRECTORY}"
: "${WANDB_ENTITY:?Set your writable W&B entity}"
: "${DEXLINK_SMOKE_REPORT:?Set the qualified local numerical-audit.json path}"
cd "$DEXLINK_ROOT"
export PYTHONPATH="$DEXLINK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYOPENGL_PLATFORM=egl
python -m dexlink.workflows.cluster_preflight "$DEXLINK_SMOKE_REPORT"
python -m dexlink check invert_cup
python -m dexlink train invert_cup \
  --run-directory "$DEXLINK_RUN_DIRECTORY" \
  --control-steps 50000 --num-envs "${DEXLINK_NUM_ENVS:-1024}" \
  --seed 42 --entity "$WANDB_ENTITY" --project "${WANDB_PROJECT:-dexlink}" \
  --max-seconds 13800
