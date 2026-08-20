#!/usr/bin/env bash

# Submit final Hessian-corrected classical C_P assembly after MD/Hessian jobs.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODEL="pet-mad"
LOADING=100
REPLICAS="1,2,3,4,5"
PARTITION="${MOF_ANALYSIS_PARTITION:-gpu}"
QOS="${MOF_ANALYSIS_QOS:-normal}"
WALL_TIME="${MOF_ANALYSIS_TIME:-00:30:00}"
CPUS_PER_TASK="${MOF_ANALYSIS_CPUS:-4}"
SLURM_OUTPUT_DIR="${MOF_SLURM_OUTPUT_DIR:-${PROJECT_DIR}/output/slurm}"
DRY_RUN=0


usage() {
    cat <<'EOF'
Usage: scripts/submit_hybrid_analysis.sh [options]

Options:
  --model NAME            pet-mad, pet-sol, or both (default: pet-mad).
  --loading N             Positive methane loading (default: 100).
  --replicas LIST         Classical MD replicas (default: 1,2,3,4,5).
  --partition NAME        Slurm partition (default: gpu).
  --qos NAME              Slurm QOS (default: normal).
  --time HH:MM:SS         Wall time per model (default: 00:30:00).
  --cpus N                CPUs per task (default: 4).
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
        --partition) require_value "$@"; PARTITION="$2"; shift 2 ;;
        --qos) require_value "$@"; QOS="$2"; shift 2 ;;
        --time) require_value "$@"; WALL_TIME="$2"; shift 2 ;;
        --cpus) require_value "$@"; CPUS_PER_TASK="$2"; shift 2 ;;
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
REPLICAS_EXPORT="${REPLICAS//,/:}"
for model_label in "${MODEL_LABELS[@]}"; do
    output="output/hybrid/${model_label}/${LOADING}ch4/hybrid-heat-capacity.npz"
    command=(
        sbatch --parsable
        --job-name="mof5-hybrid-analysis-${model_label}"
        --partition="${PARTITION}" --qos="${QOS}"
        --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}"
        --gres=gpu:1 --time="${WALL_TIME}"
        --output="${SLURM_OUTPUT_DIR}/slurm-%x-%j.out"
        --export="ALL,MOF_STAGE=hybrid-analysis,MOF_HYBRID_MODEL_LABEL=${model_label},MOF_HYBRID_LOADING=${LOADING},MOF_HYBRID_REPLICAS=${REPLICAS_EXPORT},MOF_HYBRID_OUTPUT=${output}"
        "${SCRIPT_DIR}/izar_job.sh"
    )
    if ((DRY_RUN)); then
        printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
    else
        submission=$("${command[@]}")
        echo "Submitted ${model_label}: ${submission%%;*}"
    fi
done
