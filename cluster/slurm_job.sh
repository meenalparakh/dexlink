#!/usr/bin/env bash
#SBATCH --job-name=dexlink
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -eo pipefail
source ~/.bashrc
conda activate "${DEXLINK_CONDA_ENV:-genesis}"
set -u

if [[ -z "${DEXLINK_ROOT:-}" ]]; then
    echo "DEXLINK_ROOT must point to the DexLink checkout." >&2
    exit 2
fi
if [[ -z "${DEXLINK_COMMAND:-}" ]]; then
    echo "DEXLINK_COMMAND must contain the command to run." >&2
    exit 2
fi

cd "${DEXLINK_ROOT}"
export PYTHONPATH="${DEXLINK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
bash -lc "${DEXLINK_COMMAND}"
