#!/usr/bin/env bash

# CPU-based trajectory analysis for the MOF-5 workflow on SCITAS Izar.
# The normal QOS requires allocating one GPU, although this code does not use it.
# Normally submit this through scripts/submit_analysis.sh.

#SBATCH --job-name=mof5-analysis
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:15:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail


RUNS="mof5-0ch4-paper-*-classical-*-rep01"
DISCARD_PS="100"
ANALYSIS_DIR="output/analysis"
NO_PLOTS=0


usage() {
    cat <<'EOF'
Usage: scripts/izar_analysis_job.sh [options]

Options:
  --runs PATTERN       Run-name glob or comma-separated globs.
  --discard-ps VALUE   Initial trajectory time to discard (default: 100 ps).
  --analysis-dir PATH  Analysis output directory (default: output/analysis).
  --no-plots           Skip PNG generation.
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
        --runs) require_value "$@"; RUNS="$2"; shift 2 ;;
        --discard-ps) require_value "$@"; DISCARD_PS="$2"; shift 2 ;;
        --analysis-dir) require_value "$@"; ANALYSIS_DIR="$2"; shift 2 ;;
        --no-plots) NO_PLOTS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done


if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "error: submit this script with sbatch from the repository root" >&2
    exit 2
fi
if [[ ! "${DISCARD_PS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "error: --discard-ps must be a non-negative number" >&2
    exit 2
fi
if [[ -z "${RUNS}" || -z "${ANALYSIS_DIR}" ]]; then
    echo "error: --runs and --analysis-dir must not be empty" >&2
    exit 2
fi


MOF_ENV_PREFIX="${MOF_ENV_PREFIX:-${HOME}/.conda/envs/mof-heat-capacity-izar}"
ANALYSIS_PYTHON="${MOF_ENV_PREFIX}/bin/python"
if [[ ! -x "${ANALYSIS_PYTHON}" ]]; then
    echo "error: analysis Python not found: ${ANALYSIS_PYTHON}" >&2
    echo "set MOF_ENV_PREFIX to the installed Conda environment" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f mof_heat_capacity/analysis/results.py ]]; then
    echo "error: submit from the mof-heat-capacity repository root" >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${ANALYSIS_DIR}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

echo "Job ID:          ${SLURM_JOB_ID}"
echo "Node:            ${SLURMD_NODENAME}"
echo "Environment:     ${MOF_ENV_PREFIX}"
echo "Run selection:   ${RUNS}"
echo "Discard:         ${DISCARD_PS} ps"
echo "Analysis output: ${ANALYSIS_DIR}"
echo "GPU allocated:   1 (required by Izar normal QOS; analysis is CPU-based)"

command=(
    "${ANALYSIS_PYTHON}" -m mof_heat_capacity.analysis.results
    --runs "${RUNS}"
    --discard-ps "${DISCARD_PS}"
    --analysis-dir "${ANALYSIS_DIR}"
)
if ((NO_PLOTS)); then
    command+=(--no-plots)
fi

srun --ntasks=1 "${command[@]}"
