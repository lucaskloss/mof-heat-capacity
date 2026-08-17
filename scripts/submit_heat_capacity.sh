#!/usr/bin/env bash

# Submit sparse SADMOF harmonic heat capacities as one Slurm array task per run.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
JOB_SCRIPT="${SCRIPT_DIR}/izar_job.sh"
MODEL="both"
LOADINGS="0,100"
MD_TEMPERATURES="100,200,300,400,500"
REPLICAS="1"
FRAME_TIMES_PS="200,350,500"
CV_TEMPERATURES="100:500:10"
PARTITION="${MOF_HEAT_PARTITION:-gpu}"
QOS="${MOF_HEAT_QOS:-normal}"
WALL_TIME="${MOF_HEAT_TIME:-00:45:00}"
CPUS_PER_TASK="${MOF_HEAT_CPUS:-8}"
MAX_CONCURRENT="${MOF_HEAT_MAX_CONCURRENT:-4}"
SLURM_OUTPUT_DIR="${MOF_SLURM_OUTPUT_DIR:-${PROJECT_DIR}/output/slurm}"
OVERWRITE=0
DEBUG=0
DRY_RUN=0
HEAT_OUTPUT=""


usage() {
    cat <<'EOF'
Usage: scripts/submit_heat_capacity.sh [options]

Options:
  --model NAME            pet-mad, pet-sol, or both (default: both).
  --loading LIST          Comma-separated methane counts (default: 0,100).
  --md-temperatures LIST  Comma-separated MD temperatures in K.
  --replicas LIST         Comma-separated replica numbers (default: 1).
  --frame-times-ps LIST   Physical trajectory times (default: 200,350,500 ps).
  --cv-temperatures RANGE Harmonic C_V grid (default: 100:500:10 K).
  --partition NAME        Slurm partition (default: gpu).
  --qos NAME              Slurm QOS (default: normal).
  --time HH:MM:SS         Time per trajectory/task (default: 00:45:00).
  --cpus N                CPUs per trajectory/task (default: 8).
  --max-concurrent N      Maximum simultaneous array tasks (default: 4).
  --slurm-output-dir PATH Slurm log directory (default: output/slurm).
  --overwrite             Intentionally replace matching existing NPZ outputs.
  --debug                 One PET-MAD 100-CH4 300 K frame at 275 ps, debug QOS.
  --dry-run               Validate files and print the array submission only.
  -h, --help              Show this help.

The defaults submit 20 array tasks: two MLIPs x two methane loadings x five MD
temperatures x replica 1. Each task processes three well-separated production
frames serially and writes one NPZ archive inside that run's output directory.
No SADMOF, JAX, model, trajectory, or Hessian calculation runs before Slurm.
EOF
}


require_value() {
    if (($# < 2)); then
        echo "error: $1 requires a value" >&2
        exit 2
    fi
}


require_unique_values() {
    local option_name="$1"
    shift
    local -A seen=()
    local value

    for value in "$@"; do
        if [[ -z "${value}" ]]; then
            echo "error: ${option_name} contains an empty value" >&2
            exit 2
        fi
        if [[ -n "${seen["${value}"]+present}" ]]; then
            echo "error: ${option_name} contains duplicate value: ${value}" >&2
            exit 2
        fi
        seen["${value}"]=1
    done
}


while (($#)); do
    case "$1" in
        --model) require_value "$@"; MODEL="$2"; shift 2 ;;
        --loading) require_value "$@"; LOADINGS="$2"; shift 2 ;;
        --md-temperatures) require_value "$@"; MD_TEMPERATURES="$2"; shift 2 ;;
        --replicas) require_value "$@"; REPLICAS="$2"; shift 2 ;;
        --frame-times-ps) require_value "$@"; FRAME_TIMES_PS="$2"; shift 2 ;;
        --cv-temperatures) require_value "$@"; CV_TEMPERATURES="$2"; shift 2 ;;
        --partition) require_value "$@"; PARTITION="$2"; shift 2 ;;
        --qos) require_value "$@"; QOS="$2"; shift 2 ;;
        --time) require_value "$@"; WALL_TIME="$2"; shift 2 ;;
        --cpus) require_value "$@"; CPUS_PER_TASK="$2"; shift 2 ;;
        --max-concurrent) require_value "$@"; MAX_CONCURRENT="$2"; shift 2 ;;
        --slurm-output-dir) require_value "$@"; SLURM_OUTPUT_DIR="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --debug) DEBUG=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done


if ((DEBUG)); then
    MODEL="pet-mad"
    LOADINGS="100"
    MD_TEMPERATURES="300"
    REPLICAS="1"
    FRAME_TIMES_PS="275"
    QOS="debug"
    WALL_TIME="00:30:00"
    MAX_CONCURRENT=1
    HEAT_OUTPUT="output/debug/heat-capacity-pet-mad-100ch4-300K-275ps.npz"
fi
case "${MODEL}" in
    both) MODEL_NAMES=("pet-mad-1.5-s-40nn" "pet-sol-s-best") ;;
    pet-mad) MODEL_NAMES=("pet-mad-1.5-s-40nn") ;;
    pet-sol) MODEL_NAMES=("pet-sol-s-best") ;;
    *) echo "error: --model must be pet-mad, pet-sol, or both" >&2; exit 2 ;;
esac
if [[ ! "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ \
    || ! "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --cpus and --max-concurrent must be positive integers" >&2
    exit 2
fi
if [[ ! "${CV_TEMPERATURES}" =~ ^[0-9]+([.][0-9]+)?:[0-9]+([.][0-9]+)?:[0-9]+([.][0-9]+)?$ ]]; then
    echo "error: --cv-temperatures must be start:stop:step" >&2
    exit 2
fi
if [[ -z "${SLURM_OUTPUT_DIR}" ]]; then
    echo "error: --slurm-output-dir must not be empty" >&2
    exit 2
fi
if [[ "${SLURM_OUTPUT_DIR}" != /* ]]; then
    SLURM_OUTPUT_DIR="${PROJECT_DIR}/${SLURM_OUTPUT_DIR}"
fi
if ((!DRY_RUN)) && ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch is unavailable; submit from a cluster login node" >&2
    exit 2
fi


IFS=',' read -r -a LOADING_VALUES <<< "${LOADINGS}"
IFS=',' read -r -a MD_TEMPERATURE_VALUES <<< "${MD_TEMPERATURES}"
IFS=',' read -r -a REPLICA_VALUES <<< "${REPLICAS}"
IFS=',' read -r -a FRAME_TIME_VALUES <<< "${FRAME_TIMES_PS}"
require_unique_values "--loading" "${LOADING_VALUES[@]}"
require_unique_values "--md-temperatures" "${MD_TEMPERATURE_VALUES[@]}"
require_unique_values "--replicas" "${REPLICA_VALUES[@]}"
previous_time=-1
for time_ps in "${FRAME_TIME_VALUES[@]}"; do
    if [[ ! "${time_ps}" =~ ^[0-9]+$ \
        || ${time_ps} -lt 100 || ${time_ps} -gt 500 ]]; then
        echo "error: --frame-times-ps must contain integer times from 100 to 500 ps" >&2
        exit 2
    fi
    if ((time_ps <= previous_time)); then
        echo "error: --frame-times-ps must be unique and strictly increasing" >&2
        exit 2
    fi
    previous_time=${time_ps}
done

CONFIGS=()
cd "${PROJECT_DIR}"
for loading in "${LOADING_VALUES[@]}"; do
    [[ "${loading}" =~ ^[0-9]+$ ]] || {
        echo "error: --loading must contain non-negative integers" >&2
        exit 2
    }
    for model_name in "${MODEL_NAMES[@]}"; do
        for temperature in "${MD_TEMPERATURE_VALUES[@]}"; do
            [[ "${temperature}" =~ ^[1-9][0-9]*$ ]] || {
                echo "error: --md-temperatures must contain positive integers" >&2
                exit 2
            }
            for replica in "${REPLICA_VALUES[@]}"; do
                [[ "${replica}" =~ ^[1-9][0-9]*$ ]] || {
                    echo "error: --replicas must contain positive integers" >&2
                    exit 2
                }
                printf -v replica_tag '%02d' "${replica}"
                run="mof5-${loading}ch4-paper-${model_name}-classical-${temperature}K-rep${replica_tag}"
                config="configs/${run}.toml"
                trajectory="output/${run}/${run}.lammpstrj"
                thermo_log="output/${run}/${run}.lammps.log"
                for required in "${config}" "${trajectory}" "${thermo_log}"; do
                    if [[ ! -f "${required}" ]]; then
                        echo "error: missing required completed-run file: ${required}" >&2
                        exit 2
                    fi
                done
                time_tag=""
                for time_ps in "${FRAME_TIME_VALUES[@]}"; do
                    time_tag+="${time_tag:+-}${time_ps}ps"
                done
                heat_output="${HEAT_OUTPUT:-output/${run}/heat-capacity-times-${time_tag}.npz}"
                if ((!OVERWRITE)) && [[ -e "${heat_output}" ]]; then
                    echo "error: heat-capacity output already exists: ${heat_output}" >&2
                    echo "use --overwrite only if replacing it is intentional" >&2
                    exit 2
                fi
                if [[ "${model_name}" == "pet-mad-1.5-s-40nn" ]]; then
                    jax_model="models/pet-mad-1.5-s_40nn_jax"
                else
                    jax_model="models/pet_sol-s-best_nostress_jax"
                fi
                if [[ ! -f "${jax_model}/model.msgpack" \
                    || ! -f "${jax_model}/metadata.yaml" ]]; then
                    echo "error: incomplete PET-JAX model directory: ${jax_model}" >&2
                    exit 2
                fi
                CONFIGS+=("${config}")
            done
        done
    done
done
if ((${#CONFIGS[@]} == 0)); then
    echo "error: selection produced no heat-capacity tasks" >&2
    exit 2
fi


CONFIG_LIST=$(IFS=':'; echo "${CONFIGS[*]}")
LAST_TASK=$((${#CONFIGS[@]} - 1))
export MOF_STAGE="heat-capacity"
export MOF_HEAT_CONFIGS="${CONFIG_LIST}"
export MOF_HEAT_FRAME_TIMES_PS="${FRAME_TIMES_PS}"
export MOF_HEAT_TEMPERATURES="${CV_TEMPERATURES}"
export MOF_HEAT_OVERWRITE="${OVERWRITE}"
export MOF_HEAT_OUTPUT="${HEAT_OUTPUT}"

command=(
    sbatch --parsable
    --job-name=mof5-harmonic-cv
    --partition="${PARTITION}" --qos="${QOS}"
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}"
    --gres=gpu:1 --time="${WALL_TIME}"
    --array="0-${LAST_TASK}%${MAX_CONCURRENT}"
    --output="${SLURM_OUTPUT_DIR}/slurm-%x-%A_%a.out"
    --export=ALL
    "${JOB_SCRIPT}"
)

echo "Selected ${#CONFIGS[@]} trajectories; ${#FRAME_TIME_VALUES[@]} frame time(s) each: ${FRAME_TIMES_PS} ps"
for index in "${!CONFIGS[@]}"; do
    echo "  array task ${index}: ${CONFIGS[index]}"
done
echo "Harmonic C_V grid: ${CV_TEMPERATURES} K"
echo "Resources per array task: ${PARTITION}/${QOS}, GPU=1, CPUs=${CPUS_PER_TASK}, time=${WALL_TIME}"
echo "Array concurrency: ${MAX_CONCURRENT}; Slurm output: ${SLURM_OUTPUT_DIR}/slurm-%x-%A_%a.out"
if ((DRY_RUN)); then
    printf 'DRY RUN ENV:'
    printf ' %q=%q' MOF_STAGE "${MOF_STAGE}"
    printf ' %q=%q' MOF_HEAT_CONFIGS "${MOF_HEAT_CONFIGS}"
    printf ' %q=%q' MOF_HEAT_FRAME_TIMES_PS "${MOF_HEAT_FRAME_TIMES_PS}"
    printf ' %q=%q' MOF_HEAT_TEMPERATURES "${MOF_HEAT_TEMPERATURES}"
    printf ' %q=%q' MOF_HEAT_OUTPUT "${MOF_HEAT_OUTPUT}"
    printf ' %q=%q\n' MOF_HEAT_OVERWRITE "${MOF_HEAT_OVERWRITE}"
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
else
    mkdir -p "${SLURM_OUTPUT_DIR}"
    submission=$("${command[@]}")
    echo "Submitted heat-capacity array: ${submission%%;*} (tasks 0-${LAST_TASK})"
fi
