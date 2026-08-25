#!/usr/bin/env bash

# Submit final Hessian-corrected classical C_P assembly after MD/Hessian jobs.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODEL="pet-mad"
LOADING=100
REPLICAS="1"
TEMPERATURES="100,200,300,400,500"
PARTITION="${MOF_ANALYSIS_PARTITION:-gpu}"
QOS="${MOF_ANALYSIS_QOS:-normal}"
WALL_TIME="${MOF_ANALYSIS_TIME:-00:30:00}"
CPUS_PER_TASK="${MOF_ANALYSIS_CPUS:-4}"
ZERO_THRESHOLD_CM1="1.0"
MAX_NEAR_ZERO_MODES=3
SLURM_OUTPUT_DIR="${MOF_SLURM_OUTPUT_DIR:-${PROJECT_DIR}/output/slurm}"
DRY_RUN=0


run_hybrid_worker() {
    local model_label=""
    local loading=""
    local replicas=""
    local temperatures=""
    local output=""
    local zero_threshold=""
    local max_near_zero=""

    shift
    while (($#)); do
        case "$1" in
            --model-label) model_label="$2"; shift 2 ;;
            --loading) loading="$2"; shift 2 ;;
            --replicas) replicas="$2"; shift 2 ;;
            --temperatures) temperatures="$2"; shift 2 ;;
            --output) output="$2"; shift 2 ;;
            --zero-threshold-cm1) zero_threshold="$2"; shift 2 ;;
            --max-near-zero-modes) max_near_zero="$2"; shift 2 ;;
            *) echo "error: unknown hybrid-worker argument: $1" >&2; exit 2 ;;
        esac
    done

    if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
        echo "error: the internal worker must be launched with sbatch" >&2
        exit 2
    fi
    if [[ -z "${model_label}" || -z "${loading}" || -z "${replicas}" \
        || -z "${temperatures}" || -z "${output}" \
        || -z "${zero_threshold}" || -z "${max_near_zero}" ]]; then
        echo "error: incomplete internal hybrid-worker arguments" >&2
        exit 2
    fi

    local environment_prefix="${MOF_ENV_PREFIX:-${HOME}/.conda/envs/mof-heat-capacity-izar}"
    local analysis_python="${environment_prefix}/bin/python"
    if [[ ! -x "${analysis_python}" ]]; then
        echo "error: analysis Python not found: ${analysis_python}" >&2
        echo "set MOF_ENV_PREFIX to the installed Conda environment" >&2
        exit 2
    fi

    cd "${SLURM_SUBMIT_DIR}"
    if [[ ! -f mof_heat_capacity/analysis/hybrid.py ]]; then
        echo "error: submit from the mof-heat-capacity repository root" >&2
        exit 2
    fi

    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
    export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
    export PYTHONUNBUFFERED=1

    echo "Job ID:      ${SLURM_JOB_ID}"
    echo "Node:        ${SLURMD_NODENAME}"
    echo "Environment: ${environment_prefix}"
    echo "Model:       ${model_label}"
    echo "Output:      ${output}"
    echo "GPU allocated: 1 (required by Izar normal QOS; assembly is CPU-based)"

    srun --ntasks=1 "${analysis_python}" -m mof_heat_capacity.analysis.hybrid \
        --model-label "${model_label}" \
        --loading "${loading}" \
        --replicas "${replicas}" \
        --temperatures "${temperatures}" \
        --output "${output}" \
        --zero-threshold-cm1 "${zero_threshold}" \
        --max-near-zero-modes "${max_near_zero}"
}


if [[ "${1:-}" == "--internal-hybrid-worker" ]]; then
    run_hybrid_worker "$@"
    exit 0
fi


usage() {
    cat <<'EOF'
Usage: scripts/properties/submit_hybrid_analysis.sh [options]

Options:
  --model NAME            pet-mad, pet-sol, or both (default: pet-mad).
  --loading N             Positive methane loading (default: 100).
  --replicas LIST         Classical MD replicas (default: 1).
  --temperatures LIST     Classical MD temperatures (default: 100,200,300,400,500).
  --partition NAME        Slurm partition (default: gpu).
  --qos NAME              Slurm QOS (default: normal).
  --time HH:MM:SS         Wall time per model (default: 00:30:00).
  --cpus N                CPUs per task (default: 4).
  --zero-threshold-cm1 X  Imaginary/near-zero cutoff (default: 1.0 cm^-1).
  --max-near-zero-modes N Maximum allowed near-zero modes (default: 3).
  --slurm-output-dir PATH Slurm log directory (default: output/slurm).
  --dry-run               Print submissions without calling sbatch.
  -h, --help              Show this help.
EOF
}


require_value() {
    if (($# < 2)); then
        echo "error: $1 requires a value" >&2
        exit 2
    fi
}


while (($#)); do
    case "$1" in
        --model) require_value "$@"; MODEL="$2"; shift 2 ;;
        --loading) require_value "$@"; LOADING="$2"; shift 2 ;;
        --replicas) require_value "$@"; REPLICAS="$2"; shift 2 ;;
        --temperatures) require_value "$@"; TEMPERATURES="$2"; shift 2 ;;
        --partition) require_value "$@"; PARTITION="$2"; shift 2 ;;
        --qos) require_value "$@"; QOS="$2"; shift 2 ;;
        --time) require_value "$@"; WALL_TIME="$2"; shift 2 ;;
        --cpus) require_value "$@"; CPUS_PER_TASK="$2"; shift 2 ;;
        --zero-threshold-cm1) require_value "$@"; ZERO_THRESHOLD_CM1="$2"; shift 2 ;;
        --max-near-zero-modes) require_value "$@"; MAX_NEAR_ZERO_MODES="$2"; shift 2 ;;
        --slurm-output-dir) require_value "$@"; SLURM_OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done


case "${MODEL}" in
    pet-mad) MODEL_LABELS=("pet-mad-1.5-s-40nn") ;;
    pet-sol) MODEL_LABELS=("pet-sol-s-best") ;;
    both) MODEL_LABELS=("pet-mad-1.5-s-40nn" "pet-sol-s-best") ;;
    *) echo "error: --model must be pet-mad, pet-sol, or both" >&2; exit 2 ;;
esac
if [[ ! "${LOADING}" =~ ^[1-9][0-9]*$ \
    || ! "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: loading and CPUs must be positive integers" >&2
    exit 2
fi
if [[ ! "${ZERO_THRESHOLD_CM1}" =~ ^[0-9]+([.][0-9]+)?$ \
    || ! "${MAX_NEAR_ZERO_MODES}" =~ ^[0-9]+$ ]]; then
    echo "error: spectral thresholds must be non-negative numbers" >&2
    exit 2
fi
IFS=',' read -r -a REPLICA_VALUES <<< "${REPLICAS}"
if ((${#REPLICA_VALUES[@]} == 0)); then
    echo "error: --replicas must not be empty" >&2
    exit 2
fi
declare -A SEEN_REPLICAS=()
for replica in "${REPLICA_VALUES[@]}"; do
    if [[ ! "${replica}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: --replicas must contain unique positive integers" >&2
        exit 2
    fi
    if [[ -n "${SEEN_REPLICAS[${replica}]:-}" ]]; then
        echo "error: --replicas must contain unique positive integers" >&2
        exit 2
    fi
    SEEN_REPLICAS[${replica}]=1
done
IFS=',' read -r -a TEMPERATURE_VALUES <<< "${TEMPERATURES}"
if ((${#TEMPERATURE_VALUES[@]} < 3)); then
    echo "error: --temperatures must contain at least three values" >&2
    exit 2
fi
declare -A SEEN_TEMPERATURES=()
previous_temperature=0
for temperature in "${TEMPERATURE_VALUES[@]}"; do
    if [[ ! "${temperature}" =~ ^[1-9][0-9]*$ \
        || -n "${SEEN_TEMPERATURES[${temperature}]:-}" \
        || "${temperature}" -le "${previous_temperature}" ]]; then
        echo "error: --temperatures must contain unique, increasing positive integers" >&2
        exit 2
    fi
    SEEN_TEMPERATURES[${temperature}]=1
    previous_temperature="${temperature}"
done
if [[ "${SLURM_OUTPUT_DIR}" != /* ]]; then
    SLURM_OUTPUT_DIR="${PROJECT_DIR}/${SLURM_OUTPUT_DIR}"
fi
if ((!DRY_RUN)) && ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch is unavailable; submit from a cluster login node" >&2
    exit 2
fi
if ((!DRY_RUN)); then
    mkdir -p "${SLURM_OUTPUT_DIR}"
fi


cd "${PROJECT_DIR}"
for model_label in "${MODEL_LABELS[@]}"; do
    output="output/hybrid/${model_label}/${LOADING}ch4/hybrid-heat-capacity.npz"
    command=(
        sbatch --parsable
        --job-name="mof5-hybrid-analysis-${model_label}"
        --partition="${PARTITION}" --qos="${QOS}"
        --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}"
        --gres=gpu:1 --time="${WALL_TIME}"
        --output="${SLURM_OUTPUT_DIR}/slurm-%x-%j.out"
        "${SCRIPT_DIR}/submit_hybrid_analysis.sh"
        --internal-hybrid-worker
        --model-label "${model_label}"
        --loading "${LOADING}"
        --replicas "${REPLICAS}"
        --temperatures "${TEMPERATURES}"
        --output "${output}"
        --zero-threshold-cm1 "${ZERO_THRESHOLD_CM1}"
        --max-near-zero-modes "${MAX_NEAR_ZERO_MODES}"
    )
    if ((DRY_RUN)); then
        printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
    else
        submission=$("${command[@]}")
        echo "Submitted ${model_label}: ${submission%%;*}"
    fi
done
