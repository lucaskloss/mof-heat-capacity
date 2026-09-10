#!/usr/bin/env bash

# Build one calibrated PET LLPR shallow ensemble in a Slurm GPU allocation.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
ENV_PREFIX="${MOF_ENV_PREFIX:-${HOME}/.conda/envs/mof-heat-capacity-izar}"
PYTHON="${MOF_LLPR_PYTHON:-${ENV_PREFIX}/bin/python}"
MODEL=""
TRAINING_SET=""
VALIDATION_SET=""
ENERGY_KEY="energy"
ENERGY_UNIT="eV"
LENGTH_UNIT="angstrom"
MEMBERS=32
BATCH_SIZE=4
NUM_WORKERS=0
SEED=2025
CALIBRATION_METHOD="absolute_residuals"
REGULARIZER=""
CHECKPOINT=""
OUTPUT=""
WORK_DIR=""
VALIDATION_FRAMES=8
ENSEMBLE_MEAN_TOLERANCE_EV=0.0001
PARTITION="${MOF_LLPR_PARTITION:-gpu}"
QOS="${MOF_LLPR_QOS:-normal}"
WALL_TIME="${MOF_LLPR_TIME:-04:00:00}"
CPUS_PER_TASK="${MOF_LLPR_CPUS:-4}"
DEFAULT_OUTPUT_ROOT="/work/cosmo/dealmeid/mof-heat-capacity/output"
OUTPUT_ROOT="${MOF_OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
SLURM_OUTPUT_DIR="${MOF_SLURM_OUTPUT_DIR:-${OUTPUT_ROOT}/slurm}"
DRY_RUN=0


usage() {
    cat <<'EOF'
Usage: scripts/setup/submit_llpr_ensemble.sh [options]

Required:
  --model NAME             pet-mad or pet-sol.
  --training-set PATH      Labeled structures for the LLPR covariance.
  --validation-set PATH    Separate labeled structures for scale calibration.

Data and LLPR options:
  --energy-key NAME        Energy property in both files (default: energy).
  --energy-unit UNIT       Energy unit (default: eV).
  --length-unit UNIT       Structure length unit (default: angstrom).
  --members N              Persistent shallow-ensemble members (default: 32).
  --batch-size N           Structures per GPU batch (default: 4).
  --num-workers N          Data-loader worker processes (default: 0).
  --seed N                 Ensemble seed (default: 2025).
  --calibration-method M   absolute_residuals, squared_residuals, or crps
                           (default: absolute_residuals).
  --regularizer VALUE      Optional positive covariance regularizer.
  --checkpoint PATH        Override the selected PET checkpoint.
  --output PATH            Override the generated ensemble .pt path.
  --work-dir PATH          Override generated options/checkpoint directory.
  --validation-frames N    Frames used for post-export checks (default: 8).
  --ensemble-mean-tolerance-eV VALUE
                           Export consistency tolerance (default: 0.0001 eV).

Slurm options:
  --partition NAME         Partition (default: gpu).
  --qos NAME               QOS (default: normal).
  --time HH:MM:SS          Wall time (default: 04:00:00).
  --cpus N                 CPUs per task (default: 4).
  --slurm-output-dir PATH  Slurm log root (default: /work/cosmo/dealmeid/mof-heat-capacity/output/slurm).
  --dry-run                Validate and print options/submission without writing.
  -h, --help               Show this help.

The input files must carry reference energies from the same electronic-structure
definition as the base model. The validation file must be disjoint from the
covariance/training file. This command sets num_epochs to null, samples LLPR
last-layer weights without additional gradient training, and exports energy,
energy_uncertainty, and energy_ensemble for CEA post-processing. Basic LLPR
without num_ensemble_members exports no member-resolved energy_ensemble and is
therefore insufficient for CEA.
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
        --training-set) require_value "$@"; TRAINING_SET="$2"; shift 2 ;;
        --validation-set) require_value "$@"; VALIDATION_SET="$2"; shift 2 ;;
        --energy-key) require_value "$@"; ENERGY_KEY="$2"; shift 2 ;;
        --energy-unit) require_value "$@"; ENERGY_UNIT="$2"; shift 2 ;;
        --length-unit) require_value "$@"; LENGTH_UNIT="$2"; shift 2 ;;
        --members) require_value "$@"; MEMBERS="$2"; shift 2 ;;
        --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
        --num-workers) require_value "$@"; NUM_WORKERS="$2"; shift 2 ;;
        --seed) require_value "$@"; SEED="$2"; shift 2 ;;
        --calibration-method) require_value "$@"; CALIBRATION_METHOD="$2"; shift 2 ;;
        --regularizer) require_value "$@"; REGULARIZER="$2"; shift 2 ;;
        --checkpoint) require_value "$@"; CHECKPOINT="$2"; shift 2 ;;
        --output) require_value "$@"; OUTPUT="$2"; shift 2 ;;
        --work-dir) require_value "$@"; WORK_DIR="$2"; shift 2 ;;
        --validation-frames) require_value "$@"; VALIDATION_FRAMES="$2"; shift 2 ;;
        --ensemble-mean-tolerance-eV) require_value "$@"; ENSEMBLE_MEAN_TOLERANCE_EV="$2"; shift 2 ;;
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


if [[ -z "${MODEL}" || -z "${TRAINING_SET}" || -z "${VALIDATION_SET}" ]]; then
    echo "error: --model, --training-set, and --validation-set are required" >&2
    exit 2
fi
case "${MODEL}" in
    pet-mad|pet-sol) ;;
    *) echo "error: --model must be pet-mad or pet-sol" >&2; exit 2 ;;
esac
if [[ ! "${MEMBERS}" =~ ^[0-9]+$ || "${MEMBERS}" -lt 2 \
    || ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ \
    || ! "${NUM_WORKERS}" =~ ^[0-9]+$ \
    || ! "${VALIDATION_FRAMES}" =~ ^[1-9][0-9]*$ \
    || ! "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: members must be at least two; batch size, validation frames, and CPUs must be positive" >&2
    exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "error: LLPR Python is unavailable: ${PYTHON}" >&2
    exit 2
fi
if [[ "${SLURM_OUTPUT_DIR}" != /* ]]; then
    SLURM_OUTPUT_DIR="${PROJECT_DIR}/${SLURM_OUTPUT_DIR}"
fi

cd "${PROJECT_DIR}"
llpr_args=(
    --model "${MODEL}"
    --training-set "${TRAINING_SET}"
    --validation-set "${VALIDATION_SET}"
    --energy-key "${ENERGY_KEY}"
    --energy-unit "${ENERGY_UNIT}"
    --length-unit "${LENGTH_UNIT}"
    --members "${MEMBERS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --calibration-method "${CALIBRATION_METHOD}"
    --validation-frames "${VALIDATION_FRAMES}"
    --ensemble-mean-tolerance-eV "${ENSEMBLE_MEAN_TOLERANCE_EV}"
)
if [[ -n "${REGULARIZER}" ]]; then llpr_args+=(--regularizer "${REGULARIZER}"); fi
if [[ -n "${CHECKPOINT}" ]]; then llpr_args+=(--checkpoint "${CHECKPOINT}"); fi
if [[ -n "${OUTPUT}" ]]; then llpr_args+=(--output "${OUTPUT}"); fi
if [[ -n "${WORK_DIR}" ]]; then llpr_args+=(--work-dir "${WORK_DIR}"); fi

"${PYTHON}" -m mof_heat_capacity.llpr "${llpr_args[@]}" --dry-run

worker_command=("${PYTHON}" -m mof_heat_capacity.llpr "${llpr_args[@]}")
printf -v worker '%q ' "${worker_command[@]}"
printf -v wrapped 'cd %q && exec %s' "${PROJECT_DIR}" "${worker}"
log_dir="${SLURM_OUTPUT_DIR}/llpr/${MODEL}"
submit_command=(
    sbatch --parsable
    --job-name="mof5-${MODEL}-llpr"
    --partition="${PARTITION}" --qos="${QOS}"
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}"
    --gres=gpu:1 --time="${WALL_TIME}"
    --output="${log_dir}/%j.out" --wrap "${wrapped}"
)
if ((DRY_RUN)); then
    printf 'DRY RUN:'; printf ' %q' "${submit_command[@]}"; printf '\n'
    exit 0
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch is unavailable; submit from an Izar login node" >&2
    exit 2
fi

mkdir -p "${log_dir}"
submission=$("${submit_command[@]}")
echo "Submitted ${MODEL} LLPR ensemble: ${submission%%;*}"
