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
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

MOF_STAGE="${MOF_STAGE:-md}"
MOF_CONFIG="${MOF_CONFIG:-configs/mof5_pet_mad.toml}"
MOF_ENV_PREFIX="${MOF_ENV_PREFIX:-${HOME}/.conda/envs/mof-heat-capacity-izar}"
MOF_STEPS="${MOF_STEPS:-}"
MOF_OUTPUT_DIR="${MOF_OUTPUT_DIR:-}"
MOF_PREFIX="${MOF_PREFIX:-}"
MOF_RERUN="${MOF_RERUN:-0}"
MOF_RESUME="${MOF_RESUME:-0}"
MOF_HEAT_FRAMES="${MOF_HEAT_FRAMES:-}"
MOF_HEAT_FRAME_INDICES="${MOF_HEAT_FRAME_INDICES:-}"
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

    MOF_HEAT_FRAMES=""
    MOF_HEAT_FRAME_INDICES="${SLURM_ARRAY_TASK_ID}"
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

if [[ ! -f run.py || ! -f utils/heat_capacity.py || ! -f "${MOF_CONFIG}" ]]; then
    echo "error: submit from mof-heat-capacity and check MOF_CONFIG=${MOF_CONFIG}" >&2
    exit 2
fi

echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Stage:         ${MOF_STAGE}"
echo "Configuration: ${MOF_CONFIG}"
echo "Environment:   ${MOF_ENV_PREFIX}"
nvidia-smi -L


check_pytorch_cuda() {
    python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
if torch.version.cuda is None or torch.version.cuda.split(".", 1)[0] != "12":
    raise RuntimeError(
        "Izar requires a CUDA 12 PyTorch build; recreate the environment from "
        "the repository's pinned environment.yml"
    )
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot initialize the allocated CUDA device")
capability = torch.cuda.get_device_capability(0)
if capability < (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
    raise RuntimeError(
        f"PyTorch {torch.__version__} has no V100 (sm_70) kernels; "
        "recreate the environment from environment.yml"
    )
print("PyTorch device:", torch.cuda.get_device_name(0))
PY
}


check_lammps_cuda() {
    local -a lammps_config=()
    local lammps_command
    local lammps_executable
    local lammps_path
    local cuda_runtime

    mapfile -t lammps_config < <(python - "${MOF_CONFIG}" <<'PY'
from pathlib import Path
import shlex
import sys

from utils.workflow_config import load_run_config

config = load_run_config(Path(sys.argv[1]))
print(config.md_driver)
print(shlex.split(config.lammps_command)[0])
PY
    )

    if ((${#lammps_config[@]} != 2)); then
        echo "error: could not read the MD driver and LAMMPS command from ${MOF_CONFIG}" >&2
        exit 2
    fi

    if [[ "${lammps_config[0]}" == "ase" ]]; then
        return
    fi

    lammps_command="${lammps_config[1]}"
    if ! lammps_executable=$(command -v "${lammps_command}"); then
        echo "error: LAMMPS executable is unavailable: ${lammps_command}" >&2
        exit 2
    fi
    lammps_path=$(readlink -f "${lammps_executable}")
    cuda_runtime=$(ldd "${lammps_path}" 2>/dev/null \
        | awk '$1 ~ /^libcudart\.so/ {print $1; exit}')

    if [[ "${cuda_runtime}" != libcudart.so.12* ]]; then
        echo "error: ${lammps_path} links to ${cuda_runtime:-no CUDA runtime}" >&2
        echo "Izar requires the CUDA 12 C++ libtorch pinned in environment.yml; recreate the environment" >&2
        conda list 2>/dev/null | awk '$1 == "cuda-version" || $1 == "libtorch" || $1 == "lammps-metatomic"'
        exit 2
    fi

    echo "LAMMPS:        ${lammps_path}; C++ CUDA runtime: ${cuda_runtime}"
}


check_jax_cuda() {
    python - <<'PY'
import jax

devices = jax.devices()
print(f"JAX: {jax.__version__}; devices: {devices}")
if not any(device.platform == "gpu" for device in devices):
    raise RuntimeError("JAX cannot initialize the allocated CUDA device")
PY
}


run_md() {
    local command=(python run.py --config "${MOF_CONFIG}" --device cuda)

    if [[ -n "${MOF_STEPS}" ]]; then
        command+=(--steps "${MOF_STEPS}")
    fi

    if [[ -n "${MOF_OUTPUT_DIR}" ]]; then
        command+=(--output-dir "${MOF_OUTPUT_DIR}")
    fi

    if [[ -n "${MOF_PREFIX}" ]]; then
        command+=(--prefix "${MOF_PREFIX}")
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

    case "${MOF_RESUME,,}" in
        1|true|yes)
            command+=(--resume)
            ;;
        0|false|no)
            ;;
        *)
            echo "error: MOF_RESUME must be 0/1, false/true, or no/yes" >&2
            exit 2
            ;;
    esac

    if [[ "${MOF_RERUN,,}" =~ ^(1|true|yes)$ && "${MOF_RESUME,,}" =~ ^(1|true|yes)$ ]]; then
        echo "error: MOF_RERUN and MOF_RESUME cannot both be enabled" >&2
        exit 2
    fi

    srun --ntasks=1 "${command[@]}"
}


run_heat_capacity() {
    local command=(python utils/heat_capacity.py --config "${MOF_CONFIG}")

    if [[ -n "${MOF_HEAT_FRAMES}" ]]; then
        command+=(--frames "${MOF_HEAT_FRAMES}")
    fi

    if [[ -n "${MOF_HEAT_FRAME_INDICES}" ]]; then
        command+=(--frame-indices "${MOF_HEAT_FRAME_INDICES}")
    fi

    if [[ -n "${MOF_HEAT_OUTPUT}" ]]; then
        command+=(--output "${MOF_HEAT_OUTPUT}")
    fi

    srun --ntasks=1 "${command[@]}"
}


case "${MOF_STAGE}" in
    md)
        check_pytorch_cuda
        check_lammps_cuda
        run_md
        ;;
    heat-capacity)
        check_jax_cuda
        run_heat_capacity
        ;;
    all)
        check_pytorch_cuda
        check_lammps_cuda
        check_jax_cuda
        run_md
        run_heat_capacity
        ;;
    *)
        echo "error: MOF_STAGE must be md, heat-capacity, or all" >&2
        exit 2
        ;;
esac
