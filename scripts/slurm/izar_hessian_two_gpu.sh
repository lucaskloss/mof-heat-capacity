#!/bin/bash -l
#SBATCH --job-name=hessian-two-gpu
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
set -euo pipefail
# Submit from the project root; all arguments go to the separate implementation.
if type module >/dev/null 2>&1; then module purge; fi
eval "$(conda shell.bash hook)"
conda activate "${MOF_ENV_PREFIX:-$HOME/.conda/envs/mof}"
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
# Optional workaround for CUDA graph memset-node failures on Izar.
if [[ "${MOF_DISABLE_CUDA_GRAPHS:-0}" == 1 ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_command_buffer="
fi
echo "XLA_FLAGS: ${XLA_FLAGS:-<default>}"
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=index,name,memory.total --format=csv
srun --ntasks=1 python -m mof_heat_capacity.analysis.harmonic_two_gpu "$@"
