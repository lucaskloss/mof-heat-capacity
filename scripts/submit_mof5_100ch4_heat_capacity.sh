#!/usr/bin/env bash

# Submit harmonic C_V calculations for the completed MOF-5 + 100 CH4 runs.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
JOB_SCRIPT="${PROJECT_DIR}/scripts/izar_job.sh"
COMMAND_NAME="${MOF_CV_COMMAND_NAME:-submit_mof5_100ch4_heat_capacity.sh}"
RUN_LABEL="${MOF_CV_RUN_LABEL:-MOF-5 + 100 CH4}"
MODEL_LABEL="${MOF_CV_MODEL_LABEL:-PET-MAD}"
CONFIG_PATTERN="${MOF_CV_CONFIG_PATTERN:-configs/mof5_100ch4_%sK_test.toml}"
OUTPUT_PATTERN="${MOF_CV_OUTPUT_PATTERN:-output/mof5-100ch4-%sK-test}"
PREFIX_PATTERN="${MOF_CV_PREFIX_PATTERN:-mof5-100ch4-%sK-test}"
JOB_PATTERN="${MOF_CV_JOB_PATTERN:-mof5-cv-%sK}"
RESULT_PATTERN="${MOF_CV_RESULT_PATTERN:-heat-capacity-frame-%s.npz}"
DEFAULT_TEMPERATURES=(100 200 300 400 500)
TEMPERATURES=("${DEFAULT_TEMPERATURES[@]}")
FRAME_INDEX="${MOF_CV_FRAME:-4000}"
PARTITION="${MOF_CV_PARTITION:-gpu}"
QOS="${MOF_CV_QOS:-normal}"
WALL_TIME="${MOF_CV_TIME:-04:00:00}"
CPUS_PER_TASK="${MOF_CV_CPUS:-8}"
DRY_RUN=0
FORCE=0


usage() {
    cat <<EOF
Usage: scripts/${COMMAND_NAME} [options]

Submit one PET-JAX/SADMOF harmonic C_V job for the final frame of each
completed ${RUN_LABEL} test trajectory. Each output contains a C_V curve
from 100 to 500 K in 10 K increments, as configured in the matching TOML file.

Options:
  --temperatures LIST  Comma-separated MD temperatures to analyze
                       (default: 100,200,300,400,500).
  --frame INDEX        Trajectory frame to analyze (default: 4000).
  --dry-run            Validate inputs and print commands without submitting.
  --force              Allow an existing result file to be overwritten.
  --partition NAME     Slurm partition (default: gpu).
  --qos NAME           Slurm QOS (default: normal).
  --time HH:MM:SS      Wall time per Hessian job (default: 04:00:00).
  --cpus N             CPUs per task (default: 8).
  -h, --help           Show this help text.

Examples:
  # Recommended first diagnostic: one Hessian from the 300 K trajectory
  ./scripts/${COMMAND_NAME} --temperatures 300 --dry-run
  ./scripts/${COMMAND_NAME} --temperatures 300

  # Submit all five independent Hessians after sizing resources
  ./scripts/${COMMAND_NAME}

Defaults can also be set with MOF_CV_FRAME, MOF_CV_PARTITION, MOF_CV_QOS,
MOF_CV_TIME, and MOF_CV_CPUS.
EOF
}


parse_temperatures() {
    local specification=$1
    local item

    IFS=',' read -r -a TEMPERATURES <<< "${specification}"
    if ((${#TEMPERATURES[@]} == 0)); then
        echo "error: --temperatures must not be empty" >&2
        exit 2
    fi

    for item in "${TEMPERATURES[@]}"; do
        if [[ ! "${item}" =~ ^(100|200|300|400|500)$ ]]; then
            echo "error: unsupported temperature ${item}; choose from 100,200,300,400,500" >&2
            exit 2
        fi
    done
}


while (($#)); do
    case "$1" in
        --temperatures)
            [[ $# -ge 2 ]] || { echo "error: --temperatures needs a value" >&2; exit 2; }
            parse_temperatures "$2"
            shift 2
            ;;
        --frame)
            [[ $# -ge 2 ]] || { echo "error: --frame needs a value" >&2; exit 2; }
            FRAME_INDEX="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --partition)
            [[ $# -ge 2 ]] || { echo "error: --partition needs a value" >&2; exit 2; }
            PARTITION="$2"
            shift 2
            ;;
        --qos)
            [[ $# -ge 2 ]] || { echo "error: --qos needs a value" >&2; exit 2; }
            QOS="$2"
            shift 2
            ;;
        --time)
            [[ $# -ge 2 ]] || { echo "error: --time needs a value" >&2; exit 2; }
            WALL_TIME="$2"
            shift 2
            ;;
        --cpus)
            [[ $# -ge 2 ]] || { echo "error: --cpus needs a value" >&2; exit 2; }
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done


if [[ ! "${FRAME_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "error: --frame must be a non-negative integer" >&2
    exit 2
fi

if [[ ! "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --cpus must be a positive integer" >&2
    exit 2
fi

cd "${PROJECT_DIR}"

if [[ ! -f "${JOB_SCRIPT}" ]]; then
    echo "error: Izar job script not found: ${JOB_SCRIPT}" >&2
    exit 2
fi

if ((!DRY_RUN)) && ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch is unavailable; run this script on an Izar login node" >&2
    exit 2
fi

echo "${RUN_LABEL} harmonic C_V submission"
echo "Potential: ${MODEL_LABEL}"
echo "MD temperatures: ${TEMPERATURES[*]} K"
echo "Trajectory frame: ${FRAME_INDEX}"
echo "C_V grid: 100:500:10 K"
echo "Resources: partition=${PARTITION}, qos=${QOS}, cpus=${CPUS_PER_TASK}, gpu=1, time=${WALL_TIME}"
echo "Method: one PET-JAX/SADMOF Hessian per selected trajectory"

for temperature in "${TEMPERATURES[@]}"; do
    printf -v config "${CONFIG_PATTERN}" "${temperature}"
    printf -v output_dir "${OUTPUT_PATTERN}" "${temperature}"
    printf -v prefix "${PREFIX_PATTERN}" "${temperature}"
    printf -v job_name "${JOB_PATTERN}" "${temperature}"
    printf -v result_name "${RESULT_PATTERN}" "${FRAME_INDEX}"
    trajectory="${output_dir}/${prefix}.traj"
    result="${output_dir}/${result_name}"

    if [[ ! -f "${config}" ]]; then
        echo "error: configuration not found: ${config}" >&2
        exit 2
    fi

    if [[ ! -s "${trajectory}" ]]; then
        echo "error: completed trajectory not found: ${trajectory}" >&2
        exit 2
    fi

    if [[ -e "${result}" && ${FORCE} -eq 0 ]]; then
        echo "error: result already exists: ${result}" >&2
        echo "use --force only when you intentionally want to replace it" >&2
        exit 2
    fi

    echo "${temperature} K output: ${result}"

    command=(
        sbatch
        --parsable
        --job-name="${job_name}"
        --partition="${PARTITION}"
        --qos="${QOS}"
        --nodes=1
        --ntasks=1
        --cpus-per-task="${CPUS_PER_TASK}"
        --gres=gpu:1
        --time="${WALL_TIME}"
        --export="ALL,MOF_STAGE=heat-capacity,MOF_CONFIG=${config},MOF_HEAT_FRAME_INDICES=${FRAME_INDEX},MOF_HEAT_OUTPUT=${result}"
        "${JOB_SCRIPT}"
    )

    if ((DRY_RUN)); then
        printf 'DRY RUN:'
        printf ' %q' "${command[@]}"
        printf '\n'
    else
        submission=$("${command[@]}")
        job_id=${submission%%;*}
        echo "Submitted ${temperature} K frame ${FRAME_INDEX}: job ${job_id}"
    fi
done
