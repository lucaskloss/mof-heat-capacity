#!/usr/bin/env bash

# Submit generated paper-protocol classical or PIMD configurations to Slurm.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
JOB_SCRIPT="${PROJECT_DIR}/scripts/izar_job.sh"
ENV_PREFIX="${MOF_ENV_PREFIX:-${HOME}/.conda/envs/mof-heat-capacity-izar}"
VALIDATE_PYTHON="${MOF_PAPER_PYTHON:-${ENV_PREFIX}/bin/python}"
METHOD="classical"
MODEL="pet-mad"
LOADING=100
TEMPERATURES=(100 200 300 400 500)
REPLICAS=""
REPLICAS_SET=0
TEMPERATURES_SET=0
PARTITION="${MOF_PAPER_PARTITION:-gpu}"
QOS="${MOF_PAPER_QOS:-normal}"
WALL_TIME="${MOF_PAPER_TIME:-24:00:00}"
CPUS_PER_TASK="${MOF_PAPER_CPUS:-8}"
SLURM_OUTPUT_DIR="${MOF_SLURM_OUTPUT_DIR:-${PROJECT_DIR}/output/slurm}"
STEPS=""
DEBUG=0
CALIBRATION=0
RESUME=0
DRY_RUN=0


usage() {
    cat <<'EOF'
Usage: scripts/submit_paper_protocol.sh [options]

Options:
  --method NAME        classical or pimd (default: classical).
  --model NAME         pet-mad or pet-sol (default: pet-mad).
  --loading N          Methane count; use 0 for pristine MOF-5 (default: 100).
  --temperatures LIST  Comma-separated temperatures (default: 100,200,300,400,500).
  --replicas N         Replica count (default: 5 classical, 30 PIMD).
  --partition NAME     Slurm partition (default: gpu).
  --qos NAME           Slurm QOS (default: normal).
  --time HH:MM:SS      Wall time per replica (default: 24:00:00; benchmark first).
  --cpus N             CPUs per task (default: 8).
  --slurm-output-dir PATH
                       Slurm log directory (default: output/slurm).
  --steps N            Override the configured MD step count.
  --debug              Isolated 10-step debug-QOS test (30 minutes, 4 CPUs,
                       one 300 K replica unless temperatures/replicas are set).
  --calibration        Isolated 1000-step timing run (2 hours, 4 CPUs, one
                       300 K replica unless temperatures/replicas are set).
  --resume             Continue from each run's latest restart.
  --dry-run            Validate and print commands without submission.
  -h, --help           Show this help.

Run scripts/prepare_paper_protocol.py first. PIMD uses one MLIP client on one
GPU by default and is extremely expensive: benchmark a shortened replica before
submitting the full 64-bead campaign.
EOF
}


while (($#)); do
    case "$1" in
        --method) METHOD="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --loading) LOADING="$2"; shift 2 ;;
        --temperatures) IFS=',' read -r -a TEMPERATURES <<< "$2"; TEMPERATURES_SET=1; shift 2 ;;
        --replicas) REPLICAS="$2"; REPLICAS_SET=1; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --time) WALL_TIME="$2"; shift 2 ;;
        --cpus) CPUS_PER_TASK="$2"; shift 2 ;;
        --slurm-output-dir) SLURM_OUTPUT_DIR="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --debug) DEBUG=1; shift ;;
        --calibration) CALIBRATION=1; shift ;;
        --resume) RESUME=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done


if [[ "${METHOD}" != "classical" && "${METHOD}" != "pimd" ]]; then
    echo "error: --method must be classical or pimd" >&2
    exit 2
fi
case "${MODEL}" in
    pet-mad) MODEL_LABEL="pet-mad-1.5-s-40nn" ;;
    pet-sol) MODEL_LABEL="pet-sol-s-best" ;;
    *) echo "error: --model must be pet-mad or pet-sol" >&2; exit 2 ;;
esac
if [[ -z "${REPLICAS}" ]]; then
    [[ "${METHOD}" == "classical" ]] && REPLICAS=5 || REPLICAS=30
fi
if [[ ! "${LOADING}" =~ ^[0-9]+$ || ! "${REPLICAS}" =~ ^[1-9][0-9]*$ || ! "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: loading must be a non-negative integer; replicas and CPUs must be positive integers" >&2
    exit 2
fi
if [[ -n "${STEPS}" && ! "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --steps must be a positive integer" >&2
    exit 2
fi
if [[ -z "${SLURM_OUTPUT_DIR}" ]]; then
    echo "error: --slurm-output-dir must not be empty" >&2
    exit 2
fi
if [[ "${SLURM_OUTPUT_DIR}" != /* ]]; then
    SLURM_OUTPUT_DIR="${PROJECT_DIR}/${SLURM_OUTPUT_DIR}"
fi
if ((DEBUG && CALIBRATION)); then
    echo "error: --debug and --calibration are mutually exclusive" >&2
    exit 2
fi
if ((DEBUG)); then
    QOS="debug"
    WALL_TIME="00:30:00"
    CPUS_PER_TASK=4
    STEPS=10
    ((TEMPERATURES_SET)) || TEMPERATURES=(300)
    ((REPLICAS_SET)) || REPLICAS=1
    if ((RESUME)); then
        echo "error: --debug cannot be combined with --resume" >&2
        exit 2
    fi
fi
if ((CALIBRATION)); then
    WALL_TIME="02:00:00"
    CPUS_PER_TASK=4
    [[ -n "${STEPS}" ]] || STEPS=1000
    ((TEMPERATURES_SET)) || TEMPERATURES=(300)
    ((REPLICAS_SET)) || REPLICAS=1
    if ((RESUME)); then
        echo "error: --calibration cannot be combined with --resume" >&2
        exit 2
    fi
fi
if ((!DRY_RUN)) && ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch is unavailable; submit from a cluster login node" >&2
    exit 2
fi
if [[ ! -x "${VALIDATE_PYTHON}" ]]; then
    echo "error: validation Python is unavailable: ${VALIDATE_PYTHON}" >&2
    echo "set MOF_ENV_PREFIX or MOF_PAPER_PYTHON to the project environment" >&2
    exit 2
fi


cd "${PROJECT_DIR}"
if ((!DRY_RUN)); then
    mkdir -p "${SLURM_OUTPUT_DIR}"
fi
if ((LOADING == 0)); then
    SYSTEM_LABEL="pristine MOF-5"
else
    SYSTEM_LABEL="MOF-5 + ${LOADING} CH4"
fi
echo "Protocol: ${SYSTEM_LABEL}, ${MODEL}/${METHOD}; replicas=${REPLICAS}; temperatures=${TEMPERATURES[*]} K"
echo "Resources per run: ${PARTITION}/${QOS}, GPU=1, CPUs=${CPUS_PER_TASK}, time=${WALL_TIME}"
echo "Slurm output: ${SLURM_OUTPUT_DIR}/slurm-%x-%j.out"
[[ -n "${STEPS}" ]] && echo "MD step override: ${STEPS}"

for temperature in "${TEMPERATURES[@]}"; do
    [[ "${temperature}" =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid temperature" >&2; exit 2; }
    for ((replica=1; replica<=REPLICAS; replica++)); do
        printf -v rep "%02d" "${replica}"
        config="configs/mof5-${LOADING}ch4-paper-${MODEL_LABEL}-${METHOD}-${temperature}K-rep${rep}.toml"
        if [[ ! -f "${config}" ]]; then
            echo "error: missing ${config}; run scripts/prepare_paper_protocol.py" >&2
            exit 2
        fi
        "${VALIDATE_PYTHON}" - "${config}" <<'PY'
from pathlib import Path
import sys

from mof_heat_capacity.config import load_run_config

config = load_run_config(Path(sys.argv[1]))
missing = []
if not config.structure.is_file():
    missing.append(str(config.structure))
if not config.exported_model.is_file() and (
    config.checkpoint is None or not config.checkpoint.is_file()
):
    missing.append(
        f"{config.exported_model} (or export source {config.checkpoint})"
    )
if missing:
    raise SystemExit("error: missing runtime file(s): " + ", ".join(missing))
if not config.stress_validated:
    raise SystemExit(
        "error: NPT configuration has model.stress_validated=false; "
        "prepare it with a stress/virial-validated model"
    )
if config.exported_model.is_file():
    import metatomic.torch as metatomic_torch

    model = metatomic_torch.load_atomistic_model(str(config.exported_model))
    outputs = set(model.capabilities().outputs)
    if not outputs.intersection({"stress", "non_conservative_stress"}):
        raise SystemExit(
            "error: exported model does not advertise a stress output: "
            + str(config.exported_model)
        )
PY
        output_dir=""
        debug_prefix=""
        rerun=0
        if ((DEBUG)); then
            stem=${config##*/}
            stem=${stem%.toml}
            output_dir="output/debug/${stem}"
            debug_prefix="${stem}-debug"
            rerun=1
        elif ((CALIBRATION)); then
            stem=${config##*/}
            stem=${stem%.toml}
            output_dir="output/calibration/${stem}"
            debug_prefix="${stem}-calibration"
            rerun=1
        fi
        command=(
            sbatch --parsable
            --job-name="mof5-${MODEL}-${METHOD}-${temperature}K-r${rep}"
            --partition="${PARTITION}" --qos="${QOS}"
            --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}"
            --gres=gpu:1 --time="${WALL_TIME}"
            --output="${SLURM_OUTPUT_DIR}/slurm-%x-%j.out"
            --export="ALL,MOF_STAGE=md,MOF_CONFIG=${config},MOF_STEPS=${STEPS},MOF_OUTPUT_DIR=${output_dir},MOF_PREFIX=${debug_prefix},MOF_RESUME=${RESUME},MOF_RERUN=${rerun}"
            "${JOB_SCRIPT}"
        )
        if ((DRY_RUN)); then
            printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
        else
            submission=$("${command[@]}")
            echo "Submitted ${temperature} K replica ${rep}: ${submission%%;*}"
        fi
    done
done
