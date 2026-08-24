#!/usr/bin/env bash

# Validate and submit loaded classical-NPT runs to Slurm.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
JOB_SCRIPT="${PROJECT_DIR}/scripts/izar_job.sh"
ENV_PREFIX="${MOF_ENV_PREFIX:-${HOME}/.conda/envs/mof-heat-capacity-izar}"
VALIDATE_PYTHON="${MOF_CAMPAIGN_PYTHON:-${ENV_PREFIX}/bin/python}"
MODEL="pet-mad"
LOADING=100
TEMPERATURES=(100 200 300 400 500)
REPLICAS=1
PARTITION="${MOF_MD_PARTITION:-gpu}"
QOS="${MOF_MD_QOS:-normal}"
WALL_TIME="${MOF_MD_TIME:-24:00:00}"
CPUS_PER_TASK="${MOF_MD_CPUS:-8}"
SLURM_OUTPUT_DIR="${MOF_SLURM_OUTPUT_DIR:-${PROJECT_DIR}/output/slurm}"
STEPS=""
DEBUG=0
CALIBRATION=0
RESUME=0
DRY_RUN=0


usage() {
    cat <<'EOF'
Usage: simulation/submit_loaded_md.sh [options]

Options:
  --model NAME         pet-mad, pet-sol, or both (default: pet-mad).
  --loading N          Positive methane count (default: 100).
  --temperatures LIST  Comma-separated temperatures (default: 100,200,300,400,500).
  --replicas N         Replica count (default: 1).
  --partition NAME     Slurm partition (default: gpu).
  --qos NAME           Slurm QOS (default: normal).
  --time HH:MM:SS      Wall time per run (default: 24:00:00).
  --cpus N             CPUs per task (default: 8).
  --steps N            Override the configured final step.
  --debug              Isolated 10-step, 300 K replica-01 debug run.
  --calibration        Isolated 1000-step, 300 K replica-01 timing run.
  --resume             Continue from the latest numeric LAMMPS restart.
  --dry-run            Validate and print all commands without submission.
  -h, --help           Show this help.
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
        --temperatures) require_value "$@"; IFS=',' read -r -a TEMPERATURES <<< "$2"; shift 2 ;;
        --replicas) require_value "$@"; REPLICAS="$2"; shift 2 ;;
        --partition) require_value "$@"; PARTITION="$2"; shift 2 ;;
        --qos) require_value "$@"; QOS="$2"; shift 2 ;;
        --time) require_value "$@"; WALL_TIME="$2"; shift 2 ;;
        --cpus) require_value "$@"; CPUS_PER_TASK="$2"; shift 2 ;;
        --steps) require_value "$@"; STEPS="$2"; shift 2 ;;
        --debug) DEBUG=1; shift ;;
        --calibration) CALIBRATION=1; shift ;;
        --resume) RESUME=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done


case "${MODEL}" in
    pet-mad) MODELS=(pet-mad); MODEL_LABELS=(pet-mad-1.5-s-40nn) ;;
    pet-sol) MODELS=(pet-sol); MODEL_LABELS=(pet-sol-s-best) ;;
    both)
        MODELS=(pet-mad pet-sol)
        MODEL_LABELS=(pet-mad-1.5-s-40nn pet-sol-s-best)
        ;;
    *) echo "error: --model must be pet-mad, pet-sol, or both" >&2; exit 2 ;;
esac
if [[ ! "${LOADING}" =~ ^[1-9][0-9]*$ \
    || ! "${REPLICAS}" =~ ^[1-9][0-9]*$ \
    || ! "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: loading, replicas, and CPUs must be positive integers" >&2
    exit 2
fi
if [[ -n "${STEPS}" && ! "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --steps must be a positive integer" >&2
    exit 2
fi
if ((DEBUG && CALIBRATION)); then
    echo "error: --debug and --calibration are mutually exclusive" >&2
    exit 2
fi
if ((DEBUG || CALIBRATION)) && ((RESUME)); then
    echo "error: debug/calibration cannot be resumed" >&2
    exit 2
fi
if ((DEBUG)); then
    TEMPERATURES=(300); REPLICAS=1; QOS=debug; WALL_TIME=00:30:00; CPUS_PER_TASK=4; STEPS=10
elif ((CALIBRATION)); then
    TEMPERATURES=(300); REPLICAS=1; WALL_TIME=02:00:00; CPUS_PER_TASK=4; STEPS=1000
fi
if ((${#TEMPERATURES[@]} == 0)); then
    echo "error: --temperatures must not be empty" >&2
    exit 2
fi
if [[ "${SLURM_OUTPUT_DIR}" != /* ]]; then
    SLURM_OUTPUT_DIR="${PROJECT_DIR}/${SLURM_OUTPUT_DIR}"
fi
if [[ ! -x "${VALIDATE_PYTHON}" ]]; then
    echo "error: validation Python is unavailable: ${VALIDATE_PYTHON}" >&2
    exit 2
fi
if ((!DRY_RUN)) && ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch is unavailable; submit from a cluster login node" >&2
    exit 2
fi


CONFIGS=()
RUN_MODELS=()
RUN_TEMPERATURES=()
RUN_REPLICAS=()
for model_index in "${!MODELS[@]}"; do
    for temperature in "${TEMPERATURES[@]}"; do
        if [[ ! "${temperature}" =~ ^[1-9][0-9]*$ ]]; then
            echo "error: temperatures must be positive integers" >&2
            exit 2
        fi
        for ((replica=1; replica<=REPLICAS; replica++)); do
            printf -v replica_tag '%02d' "${replica}"
            stem="mof5-${LOADING}ch4-${MODEL_LABELS[model_index]}-npt-${temperature}K-rep${replica_tag}"
            config="configs/${stem}.toml"
            if [[ ! -f "${PROJECT_DIR}/${config}" ]]; then
                echo "error: missing ${config}; run python -m mof_heat_capacity.protocols.loaded" >&2
                exit 2
            fi
            CONFIGS+=("${config}")
            RUN_MODELS+=("${MODELS[model_index]}")
            RUN_TEMPERATURES+=("${temperature}")
            RUN_REPLICAS+=("${replica_tag}")
        done
    done
done


cd "${PROJECT_DIR}"
"${VALIDATE_PYTHON}" - "${CONFIGS[@]}" <<'PY'
from pathlib import Path
import sys

from mof_heat_capacity.config import load_run_config

models = {}
for path_text in sys.argv[1:]:
    path = Path(path_text)
    config = load_run_config(path)
    if config.md_driver != "lammps" or config.md_ensemble != "npt-flexible":
        raise SystemExit(f"error: expected classical NPT configuration: {path}")
    if not config.structure.is_file():
        raise SystemExit(f"error: structure is missing: {config.structure}")
    if not config.stress_validated:
        raise SystemExit(f"error: stress validation is not recorded: {path}")
    if not config.exported_model.is_file():
        raise SystemExit(f"error: exported model is missing: {config.exported_model}")
    if config.exported_model not in models:
        import metatomic.torch as metatomic_torch

        models[config.exported_model] = metatomic_torch.load_atomistic_model(
            str(config.exported_model)
        )
    outputs = set(models[config.exported_model].capabilities().outputs)
    if not outputs.intersection({"stress", "non_conservative_stress"}):
        raise SystemExit(f"error: model has no stress output: {config.exported_model}")
PY

if ((!DRY_RUN)); then
    mkdir -p "${SLURM_OUTPUT_DIR}"
fi
for index in "${!CONFIGS[@]}"; do
    config="${CONFIGS[index]}"
    run_model="${RUN_MODELS[index]}"
    temperature="${RUN_TEMPERATURES[index]}"
    replica_tag="${RUN_REPLICAS[index]}"
    stem=$(basename "${config}" .toml)
    stage=production
    prefix_suffix=""
    rerun=0
    if ((DEBUG)); then
        stage=debug; prefix_suffix=-debug; rerun=1
    elif ((CALIBRATION)); then
        stage=calibration; prefix_suffix=-calibration; rerun=1
    fi
    output_dir="output/classical/${stage}/${LOADING}ch4/${stem}"
    output_prefix="${stem}${prefix_suffix}"
    command=(
        sbatch --parsable
        --job-name="mof5-${run_model}-npt-${temperature}K-r${replica_tag}"
        --partition="${PARTITION}" --qos="${QOS}"
        --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}"
        --gres=gpu:1 --time="${WALL_TIME}"
        --output="${SLURM_OUTPUT_DIR}/slurm-%x-%j.out"
        --export="ALL,MOF_STAGE=md,MOF_CONFIG=${config},MOF_STEPS=${STEPS},MOF_OUTPUT_DIR=${output_dir},MOF_PREFIX=${output_prefix},MOF_RESUME=${RESUME},MOF_RERUN=${rerun}"
        "${JOB_SCRIPT}"
    )
    if ((DRY_RUN)); then
        printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
    else
        submission=$("${command[@]}")
        echo "Submitted ${temperature} K replica ${replica_tag}: ${submission%%;*}"
    fi
done
