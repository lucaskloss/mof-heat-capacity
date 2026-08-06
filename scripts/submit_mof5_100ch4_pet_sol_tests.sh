#!/usr/bin/env bash

# Submit the five short MOF-5 + 100 CH4 PET-SOL NVT validation jobs to Izar.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
JOB_SCRIPT="${PROJECT_DIR}/scripts/izar_job.sh"
MODEL_CHECKPOINT="${PROJECT_DIR}/models/pet_sol-s-best_nostress.ckpt"
TEMPERATURES=(100 200 300 400 500)
PARTITION="${MOF_TEST_PARTITION:-gpu}"
QOS="${MOF_TEST_QOS:-normal}"
WALL_TIME="${MOF_TEST_TIME:-02:00:00}"
CPUS_PER_TASK="${MOF_TEST_CPUS:-4}"
RERUN="${MOF_TEST_RERUN:-1}"
DRY_RUN=0


usage() {
    cat <<'EOF'
Usage: scripts/submit_mof5_100ch4_pet_sol_tests.sh [options]

Submit one 2 ps PET-SOL NVT job at each of 100, 200, 300, 400, and 500 K.
The jobs reuse output/structures/mof5-100ch4-seed2025.pdb from the original
prepared sweep, but write trajectories to separate PET-SOL output directories.

Options:
  --dry-run          Print sbatch commands without submitting jobs.
  --partition NAME   Slurm partition (default: gpu).
  --qos NAME         Slurm QOS (default: normal).
  --time HH:MM:SS    Wall time per job (default: 02:00:00).
  --cpus N           CPUs per task (default: 4).
  --no-rerun         Do not overwrite an existing trajectory.
  -h, --help         Show this help text.

The defaults can also be set with MOF_TEST_PARTITION, MOF_TEST_QOS,
MOF_TEST_TIME, MOF_TEST_CPUS, and MOF_TEST_RERUN.
EOF
}


while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
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
        --no-rerun)
            RERUN=0
            shift
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


if [[ ! "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --cpus must be a positive integer" >&2
    exit 2
fi

if [[ "${RERUN,,}" != "0" && "${RERUN,,}" != "1" && \
      "${RERUN,,}" != "false" && "${RERUN,,}" != "true" && \
      "${RERUN,,}" != "no" && "${RERUN,,}" != "yes" ]]; then
    echo "error: MOF_TEST_RERUN must be 0/1, false/true, or no/yes" >&2
    exit 2
fi

cd "${PROJECT_DIR}"

if [[ ! -f "${JOB_SCRIPT}" ]]; then
    echo "error: Izar job script not found: ${JOB_SCRIPT}" >&2
    exit 2
fi

if [[ ! -f "${MODEL_CHECKPOINT}" ]]; then
    echo "error: PET-SOL checkpoint not found: ${MODEL_CHECKPOINT}" >&2
    exit 2
fi

if ((!DRY_RUN)) && ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch is unavailable; run this script on an Izar login node" >&2
    exit 2
fi

echo "Submitting MOF-5 + 100 CH4 PET-SOL NVT tests"
echo "Model: pet_sol-s-best_nostress"
echo "Temperatures: ${TEMPERATURES[*]} K"
echo "MD settings: 4000 steps x 0.5 fs = 2 ps"
echo "Resources: partition=${PARTITION}, qos=${QOS}, cpus=${CPUS_PER_TASK}, gpu=1, time=${WALL_TIME}"

for temperature in "${TEMPERATURES[@]}"; do
    config="configs/mof5_100ch4_${temperature}K_pet_sol_test.toml"
    if [[ ! -f "${config}" ]]; then
        echo "error: configuration not found: ${config}" >&2
        exit 2
    fi

    command=(
        sbatch
        --parsable
        --job-name="mof5-ch4-${temperature}K-pet-sol"
        --partition="${PARTITION}"
        --qos="${QOS}"
        --nodes=1
        --ntasks=1
        --cpus-per-task="${CPUS_PER_TASK}"
        --gres=gpu:1
        --time="${WALL_TIME}"
        --export="ALL,MOF_STAGE=md,MOF_CONFIG=${config},MOF_RERUN=${RERUN}"
        "${JOB_SCRIPT}"
    )

    if ((DRY_RUN)); then
        printf 'DRY RUN:'
        printf ' %q' "${command[@]}"
        printf '\n'
    else
        submission=$("${command[@]}")
        job_id=${submission%%;*}
        echo "Submitted ${temperature} K: job ${job_id}"
    fi
done
