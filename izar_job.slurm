#!/bin/bash -l

# Reusable single-GPU Slurm template for the MOF-5 workflow on SCITAS Izar.
# Submit this file from the mof-heat-capacity directory. The default stage is MD.
# After checking `sacctmgr show qos`, optionally add the appropriate QOS directive.
# The current Python programs do not use MPI or multiple GPUs: keep one node,
# one task, and one GPU. Benchmark cpus-per-task and set time from a short run.

#SBATCH --job-name=mof5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail


MOF_STAGE="${MOF_STAGE:-md}"
MOF_CONFIG="${MOF_CONFIG:-configs/mof5_pet_mad.toml}"
MOF_ENV_PREFIX="${MOF_ENV_PREFIX:-${HOME}/.conda/envs/mof-heat-capacity-izar}"
MOF_STEPS="${MOF_STEPS:-}"
MOF_RERUN="${MOF_RERUN:-0}"
MOF_HEAT_FRAMES="${MOF_HEAT_FRAMES:-}"
MOF_HEAT_OUTPUT="${MOF_HEAT_OUTPUT:-}"


if [[ "${MOF_HEAT_FRAMES}" == "array" ]]; then
    if [[ "${MOF_STAGE}" != "heat-capacity" ]]; then
        echo "error: MOF_HEAT_FRAMES=array requires MOF_STAGE=heat-capacity" >&2
        exit 2
    fi

    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "error: MOF_HEAT_FRAMES=array requires sbatch --array" >&2
        exit 2
    fi

    MOF_HEAT_FRAMES="${SLURM_ARRAY_TASK_ID}"
    MOF_HEAT_OUTPUT="${MOF_HEAT_OUTPUT:-output/heat-capacity-frame-${SLURM_ARRAY_TASK_ID}.npz}"
fi


if type module >/dev/null 2>&1; then
    module purge
fi

if [[ -n "${MOF_CONDA_SH:-}" ]]; then
    if [[ ! -f "${MOF_CONDA_SH}" ]]; then
        echo "error: MOF_CONDA_SH is not a file: ${MOF_CONDA_SH}" >&2
        exit 2
    fi

    source "${MOF_CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "error: conda is unavailable after module purge" >&2
    echo "set MOF_CONDA_SH to <conda-base>/etc/profile.d/conda.sh" >&2
    exit 2
fi

if [[ ! -d "${MOF_ENV_PREFIX}" ]]; then
    echo "error: Conda environment not found: ${MOF_ENV_PREFIX}" >&2
    exit 2
fi

conda activate "${MOF_ENV_PREFIX}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f run.py || ! -f heat_capacity.py || ! -f "${MOF_CONFIG}" ]]; then
    echo "error: submit from mof-heat-capacity and check MOF_CONFIG=${MOF_CONFIG}" >&2
    exit 2
fi

echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Stage:         ${MOF_STAGE}"
echo "Configuration: ${MOF_CONFIG}"
echo "Environment:   ${MOF_ENV_PREFIX}"
nvidia-smi -L
python -c 'import jax; print("JAX devices:", jax.devices())'
python -c 'import torch; print("PyTorch CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))'


run_md() {
    local command=(python run.py --config "${MOF_CONFIG}" --device cuda)

    if [[ -n "${MOF_STEPS}" ]]; then
        command+=(--steps "${MOF_STEPS}")
    fi

    case "${MOF_RERUN,,}" in
        1|true|yes)
            command+=(--rerun)
            ;;
        0|false|no)
            ;;
        *)
            echo "error: MOF_RERUN must be 0/1, false/true, or no/yes" >&2
            exit 2
            ;;
    esac

    srun --ntasks=1 "${command[@]}"
}


run_heat_capacity() {
    local command=(python heat_capacity.py --config "${MOF_CONFIG}")

    if [[ -n "${MOF_HEAT_FRAMES}" ]]; then
        command+=(--frames "${MOF_HEAT_FRAMES}")
    fi

    if [[ -n "${MOF_HEAT_OUTPUT}" ]]; then
        command+=(--output "${MOF_HEAT_OUTPUT}")
    fi

    srun --ntasks=1 "${command[@]}"
}


case "${MOF_STAGE}" in
    md)
        run_md
        ;;
    heat-capacity)
        run_heat_capacity
        ;;
    all)
        run_md
        run_heat_capacity
        ;;
    *)
        echo "error: MOF_STAGE must be md, heat-capacity, or all" >&2
        exit 2
        ;;
esac
