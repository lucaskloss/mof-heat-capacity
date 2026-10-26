#!/bin/bash -l
#SBATCH --job-name=zif8-fd-ensemble
#SBATCH --array=0-64%8
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
set -euo pipefail
# Arguments: config, saved geometry, LLPR checkpoint, new ensemble directory.
if (($# != 4)); then echo 'Expected CONFIG STRUCTURE LLPR_CHECKPOINT OUTPUT_DIR' >&2; exit 2; fi
config=$1
structure=$2
llpr_checkpoint=$3
ensemble_dir=$4
index=${SLURM_ARRAY_TASK_ID:?Requires --array=0-64}
if ((index < 0 || index > 64)); then exit 2; fi
if ((index == 0)); then
    label=central
    member_args=()
else
    printf -v label 'member-%02d' "$((index - 1))"
    member_args=(--llpr-checkpoint "$llpr_checkpoint" --member "$((index - 1))")
fi
matrix="${ensemble_dir}/${label}/hessian.npy"
export MOF_DISABLE_CUDA_GRAPHS=1
# Reuse only the experimental launcher; existing production scripts are untouched.
source "${SLURM_SUBMIT_DIR}/scripts/slurm/izar_hessian_two_gpu.sh" \
    --config "$config" --structure "$structure" --output "$matrix" \
    --selection-mode fixed --derivative finite-difference --fd-step 0.00005 \
    "${member_args[@]}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
srun --ntasks=1 python -m mof_heat_capacity.analysis.two_gpu_spectrum --matrix "$matrix"
