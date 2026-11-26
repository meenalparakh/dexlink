#!/usr/bin/env bash
#SBATCH --job-name=dexlink-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -eo pipefail
source ~/.bashrc
conda activate "${DEXLINK_CONDA_ENV:-genesis}"
set -u
: "${DEXLINK_ROOT:?Set the persistent checkout path}"
: "${DEXLINK_EVAL_DIRECTORY:?Set a fresh persistent evaluation output}"
cd "$DEXLINK_ROOT"
export PYTHONPATH="$DEXLINK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYOPENGL_PLATFORM=egl
python -m dexlink collect "${DEXLINK_TASK:-invert_cup}" \
  --episodes 50 --parallel-envs 10 --seed 42 --output "$DEXLINK_EVAL_DIRECTORY"
