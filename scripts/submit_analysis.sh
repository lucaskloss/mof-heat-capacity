#!/usr/bin/env bash

# Validate and submit the CPU-based classical-trajectory analysis job.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
JOB_SCRIPT="${SCRIPT_DIR}/izar_analysis_job.sh"
ENV_PREFIX="${MOF_ENV_PREFIX:-${HOME}/.conda/envs/mof-heat-capacity-izar}"
VALIDATE_PYTHON="${MOF_ANALYSIS_PYTHON:-${ENV_PREFIX}/bin/python}"
RUNS=""
MODEL="both"
LOADINGS="0"
TEMPERATURES="100,200,300,400,500"
REPLICAS="1"
DISCARD_PS="100"
ANALYSIS_DIR="output/analysis"
PARTITION="${MOF_ANALYSIS_PARTITION:-gpu}"
QOS="${MOF_ANALYSIS_QOS:-normal}"
WALL_TIME="${MOF_ANALYSIS_TIME:-01:15:00}"
CPUS_PER_TASK="${MOF_ANALYSIS_CPUS:-4}"
SLURM_OUTPUT_DIR="${MOF_SLURM_OUTPUT_DIR:-${PROJECT_DIR}/output/slurm}"
NO_PLOTS=0
DRY_RUN=0


usage() {
    cat <<'EOF'
Usage: scripts/submit_analysis.sh [options]

Options:
  --model NAME            pet-mad, pet-sol, or both (default: both).
  --loading LIST          Comma-separated methane counts (default: 0).
  --temperatures LIST     Comma-separated temperatures in K.
  --replicas LIST         Comma-separated replica numbers (default: 1).
  --runs PATTERN          Advanced: explicit run-name glob(s); overrides selectors.
  --discard-ps VALUE      Initial trajectory time to discard (default: 100 ps).
  --analysis-dir PATH     Analysis output directory (default: output/analysis).
  --partition NAME        Slurm partition (default: gpu).
  --qos NAME              Slurm QOS (default: normal).
  --time HH:MM:SS         Wall time (default: 01:15:00).
  --cpus N                CPUs for the serial analysis job (default: 4).
  --slurm-output-dir PATH Slurm log directory (default: output/slurm).
  --no-plots              Skip PNG generation.
  --dry-run               Validate inputs and print the sbatch command.
  -h, --help              Show this help.

Izar's normal QOS requires one allocated GPU, although the analysis itself is
CPU-based. By default the job analyzes the ten pristine classical replica-01
trajectories for both MLIPs and all five temperatures. For example, use
`--loading 0,100 --model both` to analyze both available methane loadings and
potentials in one job.
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
        --loading) require_value "$@"; LOADINGS="$2"; shift 2 ;;
        --temperatures) require_value "$@"; TEMPERATURES="$2"; shift 2 ;;
        --replicas) require_value "$@"; REPLICAS="$2"; shift 2 ;;
        --runs) require_value "$@"; RUNS="$2"; shift 2 ;;
        --discard-ps) require_value "$@"; DISCARD_PS="$2"; shift 2 ;;
        --analysis-dir) require_value "$@"; ANALYSIS_DIR="$2"; shift 2 ;;
        --partition) require_value "$@"; PARTITION="$2"; shift 2 ;;
        --qos) require_value "$@"; QOS="$2"; shift 2 ;;
        --time) require_value "$@"; WALL_TIME="$2"; shift 2 ;;
        --cpus) require_value "$@"; CPUS_PER_TASK="$2"; shift 2 ;;
        --slurm-output-dir) require_value "$@"; SLURM_OUTPUT_DIR="$2"; shift 2 ;;
        --no-plots) NO_PLOTS=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done


if [[ -z "${RUNS}" ]]; then
    case "${MODEL}" in
        both) MODEL_NAMES=("pet-mad-1.5-s-40nn" "pet-sol-s-best") ;;
        pet-mad) MODEL_NAMES=("pet-mad-1.5-s-40nn") ;;
        pet-sol) MODEL_NAMES=("pet-sol-s-best") ;;
        *) echo "error: --model must be pet-mad, pet-sol, or both" >&2; exit 2 ;;
    esac
    IFS=',' read -r -a LOADING_VALUES <<< "${LOADINGS}"
    IFS=',' read -r -a TEMPERATURE_VALUES <<< "${TEMPERATURES}"
    IFS=',' read -r -a REPLICA_VALUES <<< "${REPLICAS}"
    RUN_PATTERNS=()
    for loading in "${LOADING_VALUES[@]}"; do
        if [[ ! "${loading}" =~ ^[0-9]+$ ]]; then
            echo "error: --loading must contain non-negative integers" >&2
            exit 2
        fi
        for model_name in "${MODEL_NAMES[@]}"; do
            for temperature in "${TEMPERATURE_VALUES[@]}"; do
                if [[ ! "${temperature}" =~ ^[1-9][0-9]*$ ]]; then
                    echo "error: --temperatures must contain positive integers" >&2
                    exit 2
                fi
                for replica in "${REPLICA_VALUES[@]}"; do
                    if [[ ! "${replica}" =~ ^[1-9][0-9]*$ ]]; then
                        echo "error: --replicas must contain positive integers" >&2
                        exit 2
                    fi
                    printf -v replica_tag '%02d' "${replica}"
                    RUN_PATTERNS+=(
                        "mof5-${loading}ch4-paper-${model_name}-classical-${temperature}K-rep${replica_tag}"
                    )
                done
            done
        done
    done
    RUNS=$(IFS=','; echo "${RUN_PATTERNS[*]}")
fi


if [[ ! "${DISCARD_PS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "error: --discard-ps must be a non-negative number" >&2
    exit 2
fi
if [[ ! "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --cpus must be a positive integer" >&2
    exit 2
fi
if [[ -z "${RUNS}" || -z "${ANALYSIS_DIR}" || -z "${SLURM_OUTPUT_DIR}" ]]; then
    echo "error: run pattern and output directories must not be empty" >&2
    exit 2
fi
if [[ "${SLURM_OUTPUT_DIR}" != /* ]]; then
    SLURM_OUTPUT_DIR="${PROJECT_DIR}/${SLURM_OUTPUT_DIR}"
fi
if [[ ! -x "${VALIDATE_PYTHON}" ]]; then
    echo "error: validation Python is unavailable: ${VALIDATE_PYTHON}" >&2
    echo "set MOF_ENV_PREFIX or MOF_ANALYSIS_PYTHON" >&2
    exit 2
fi
if ((!DRY_RUN)) && ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch is unavailable; submit from a cluster login node" >&2
    exit 2
fi


cd "${PROJECT_DIR}"
"${VALIDATE_PYTHON}" - "${RUNS}" <<'PY'
from pathlib import Path
import fnmatch
import sys

from mof_heat_capacity.analysis.lammps import read_lammps_thermo
from mof_heat_capacity.analysis.results import discover_runs

patterns = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
if not patterns:
    raise SystemExit("error: --runs must contain at least one glob pattern")
runs, _ = discover_runs(Path("configs"), patterns)
if not runs:
    raise SystemExit("error: no completed trajectories match --runs")
selected_names = [config.name for _, config, _ in runs]
missing_patterns = [
    pattern for pattern in patterns
    if not any(fnmatch.fnmatch(name, pattern) for name in selected_names)
]
if missing_patterns:
    raise SystemExit(
        "error: no completed trajectory matches requested pattern(s): "
        + ", ".join(missing_patterns)
    )
print(f"Validated {len(runs)} completed trajectory/thermo-log selections")
for _, config, trajectory in runs:
    log = config.output_dir / f"{config.md_prefix}.lammps.log"
    if config.md_driver == "lammps" and not log.is_file():
        raise SystemExit(f"error: missing LAMMPS thermo log: {log}")
    if config.md_driver == "lammps":
        thermo = read_lammps_thermo(log)
        final_step = int(thermo["md_step"][-1])
        if final_step < config.md_steps:
            raise SystemExit(
                f"error: {config.name} ends at step {final_step}, "
                f"before configured step {config.md_steps}"
            )
        unique_frames = len(set(zip(thermo["md_step"], thermo["time_ps"])))
        duplicates = len(thermo["md_step"]) - unique_frames
        detail = (
            f"{unique_frames} unique thermo frames, step {final_step}, "
            f"{duplicates} restart duplicate(s) to remove"
        )
    else:
        detail = "trajectory present"
    print(f"  {config.name}: {trajectory} ({detail})")
PY

command=(
    sbatch --parsable
    --job-name=mof5-classical-analysis
    --partition="${PARTITION}" --qos="${QOS}"
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}"
    --gres=gpu:1
    --time="${WALL_TIME}"
    --output="${SLURM_OUTPUT_DIR}/slurm-%x-%j.out"
    "${JOB_SCRIPT}"
    --runs "${RUNS}"
    --discard-ps "${DISCARD_PS}"
    --analysis-dir "${ANALYSIS_DIR}"
)
if ((NO_PLOTS)); then
    command+=(--no-plots)
fi

echo "Resources: ${PARTITION}/${QOS}, GPU=1, CPUs=${CPUS_PER_TASK}, time=${WALL_TIME}"
echo "Slurm output: ${SLURM_OUTPUT_DIR}/slurm-%x-%j.out"
if ((DRY_RUN)); then
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
else
    mkdir -p "${SLURM_OUTPUT_DIR}"
    submission=$("${command[@]}")
    echo "Submitted analysis job: ${submission%%;*}"
fi
