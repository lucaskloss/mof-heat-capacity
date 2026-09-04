#!/usr/bin/env bash

# Validate and submit one CPU-based analysis job per classical trajectory.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
ENV_PREFIX="${MOF_ENV_PREFIX:-${HOME}/.conda/envs/mof-heat-capacity-izar}"
VALIDATE_PYTHON="${MOF_ANALYSIS_PYTHON:-${ENV_PREFIX}/bin/python}"
RUNS=""
MODEL="both"
LOADINGS="100"
TEMPERATURES="200,225,250,275,300,325,350,375,400"
REPLICAS="1"
DISCARD_PS="100"
ANALYSIS_DIR="output/post-processing/trajectory-analysis"
PARTITION="${MOF_ANALYSIS_PARTITION:-gpu}"
QOS="${MOF_ANALYSIS_QOS:-normal}"
WALL_TIME="${MOF_ANALYSIS_TIME:-01:15:00}"
CPUS_PER_TASK="${MOF_ANALYSIS_CPUS:-4}"
SLURM_OUTPUT_DIR="${MOF_SLURM_OUTPUT_DIR:-${PROJECT_DIR}/output/slurm}"
NO_PLOTS=0
MODEL_UNCERTAINTY=0
UNCERTAINTY_MODEL=""
UNCERTAINTY_STRIDE=20
UNCERTAINTY_BATCH_SIZE=4
UNCERTAINTY_CENTRAL_TOLERANCE_EV=0.01
DRY_RUN=0


run_analysis_worker() {
    local runs=""
    local discard_ps="100"
    local analysis_dir="output/post-processing/trajectory-analysis"
    local no_plots=0
    local model_uncertainty=0
    local uncertainty_model=""
    local uncertainty_stride=20
    local uncertainty_batch_size=4
    local uncertainty_central_tolerance_eV=0.01
    local run_only=0
    local aggregate_only=0

    shift
    while (($#)); do
        case "$1" in
            --runs) runs="$2"; shift 2 ;;
            --discard-ps) discard_ps="$2"; shift 2 ;;
            --analysis-dir) analysis_dir="$2"; shift 2 ;;
            --no-plots) no_plots=1; shift ;;
            --model-uncertainty) model_uncertainty=1; shift ;;
            --uncertainty-model) uncertainty_model="$2"; shift 2 ;;
            --uncertainty-stride) uncertainty_stride="$2"; shift 2 ;;
            --uncertainty-batch-size) uncertainty_batch_size="$2"; shift 2 ;;
            --uncertainty-central-tolerance-eV) uncertainty_central_tolerance_eV="$2"; shift 2 ;;
            --run-only) run_only=1; shift ;;
            --aggregate-only) aggregate_only=1; shift ;;
            *) echo "error: unknown analysis-worker argument: $1" >&2; exit 2 ;;
        esac
    done

    if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
        echo "error: the internal worker must be launched with sbatch" >&2
        exit 2
    fi
    if [[ ! "${discard_ps}" =~ ^[0-9]+([.][0-9]+)?$ \
        || ! "${uncertainty_stride}" =~ ^[1-9][0-9]*$ \
        || ! "${uncertainty_batch_size}" =~ ^[1-9][0-9]*$ \
        || ! "${uncertainty_central_tolerance_eV}" =~ ^[0-9]+([.][0-9]+)?$ \
        || -z "${runs}" || -z "${analysis_dir}" ]] \
        || ((run_only && aggregate_only)); then
        echo "error: invalid internal analysis-worker arguments" >&2
        exit 2
    fi

    local analysis_python="${ENV_PREFIX}/bin/python"
    if [[ ! -x "${analysis_python}" ]]; then
        echo "error: analysis Python not found: ${analysis_python}" >&2
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
    export MPLCONFIGDIR="${analysis_dir}/.matplotlib/${SLURM_JOB_ID}"
    mkdir -p "${MPLCONFIGDIR}"

    echo "Job ID:          ${SLURM_JOB_ID}"
    echo "Node:            ${SLURMD_NODENAME}"
    echo "Environment:     ${ENV_PREFIX}"
    if ((aggregate_only)); then
        echo "Stage:           aggregate completed trajectory analyses"
    else
        echo "Stage:           analyze one trajectory"
    fi
    echo "Run selection:   ${runs}"
    echo "Discard:         ${discard_ps} ps"
    echo "Analysis output: ${analysis_dir}"
    if ((model_uncertainty)); then
        echo "Model UQ:        enabled (configured energy ensemble)"
    else
        echo "Model UQ:        disabled"
    fi

    local command=(
        "${analysis_python}" -m mof_heat_capacity.analysis.results
        --runs "${runs}"
        --discard-ps "${discard_ps}"
        --analysis-dir "${analysis_dir}"
    )
    if ((no_plots)); then
        command+=(--no-plots)
    fi
    if ((model_uncertainty)); then
        command+=(
            --model-uncertainty
            --uncertainty-stride "${uncertainty_stride}"
            --uncertainty-batch-size "${uncertainty_batch_size}"
            --uncertainty-central-tolerance-eV "${uncertainty_central_tolerance_eV}"
        )
        if [[ -n "${uncertainty_model}" ]]; then
            command+=(--uncertainty-model "${uncertainty_model}")
        fi
    fi
    if ((run_only)); then
        command+=(--run-only)
    elif ((aggregate_only)); then
        command+=(--aggregate-only)
    fi
    srun --ntasks=1 "${command[@]}"
}


if [[ "${1:-}" == "--internal-analysis-worker" ]]; then
    run_analysis_worker "$@"
    exit 0
fi


usage() {
    cat <<'EOF'
Usage: scripts/properties/submit_analysis.sh [options]

Options:
  --model NAME            pet-mad, pet-sol, or both (default: both).
  --loading LIST          Comma-separated positive methane counts (default: 100).
  --temperatures LIST     Comma-separated temperatures in K
                          (default: 200 to 400 K in 25 K steps).
  --replicas LIST         Comma-separated replica numbers (default: 1).
  --runs PATTERN          Advanced: explicit run-name glob(s); overrides selectors.
  --discard-ps VALUE      Initial trajectory time to discard (default: 100 ps).
  --analysis-dir PATH     Analysis output override (default:
                          output/post-processing/trajectory-analysis/<model>/<loading>ch4).
  --partition NAME        Slurm partition (default: gpu).
  --qos NAME              Slurm QOS (default: normal).
  --time HH:MM:SS         Wall time (default: 01:15:00).
  --cpus N                CPUs for each trajectory-analysis job (default: 4).
  --slurm-output-dir PATH Slurm log directory (default: output/slurm).
  --no-plots              Skip PNG generation.
  --model-uncertainty     Evaluate energy_ensemble on production frames and
                          propagate model uncertainty into classical C_P.
  --uncertainty-model PATH
                          Calibrated ensemble model override; use with one MLIP.
  --uncertainty-stride N  Use every Nth production frame for UQ (default: 20).
  --uncertainty-batch-size N
                          Structures per model-inference batch (default: 4).
  --uncertainty-central-tolerance-eV VALUE
                          Maximum centered ensemble-mean/MD energy residual
                          (default: 0.01 eV).
  --dry-run               Validate inputs and print the sbatch commands.
  -h, --help              Show this help.

The trajectory jobs use Izar's shortest production QOS, normal, by default and
allocate one GPU. Standard diagnostics are CPU-based; optional ensemble
inference uses that GPU. Each selected trajectory is analyzed in its own Slurm
job. A small dependent job assembles
the combined CSV, manifest, and temperature-sweep plot. By default, loaded
classical replica 1 is selected from 200 to 400 K in 25 K steps for both MLIPs.
Empty MOF-5 has no MD stage in the hybrid workflow.
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
        --model-uncertainty) MODEL_UNCERTAINTY=1; shift ;;
        --uncertainty-model) require_value "$@"; UNCERTAINTY_MODEL="$2"; shift 2 ;;
        --uncertainty-stride) require_value "$@"; UNCERTAINTY_STRIDE="$2"; shift 2 ;;
        --uncertainty-batch-size) require_value "$@"; UNCERTAINTY_BATCH_SIZE="$2"; shift 2 ;;
        --uncertainty-central-tolerance-eV) require_value "$@"; UNCERTAINTY_CENTRAL_TOLERANCE_EV="$2"; shift 2 ;;
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
        if [[ ! "${loading}" =~ ^[1-9][0-9]*$ ]]; then
            echo "error: --loading must contain positive integers" >&2
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
                        "mof5-${loading}ch4-${model_name}-npt-${temperature}K-rep${replica_tag}"
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
if [[ ! "${UNCERTAINTY_STRIDE}" =~ ^[1-9][0-9]*$ \
    || ! "${UNCERTAINTY_BATCH_SIZE}" =~ ^[1-9][0-9]*$ \
    || ! "${UNCERTAINTY_CENTRAL_TOLERANCE_EV}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "error: invalid uncertainty stride, batch size, or central tolerance" >&2
    exit 2
fi
if [[ -n "${UNCERTAINTY_MODEL}" && ! "${MODEL_UNCERTAINTY}" -eq 1 ]]; then
    echo "error: --uncertainty-model requires --model-uncertainty" >&2
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
selection_output=$(
"${VALIDATE_PYTHON}" - "${RUNS}" "${MODEL_UNCERTAINTY}" "${UNCERTAINTY_MODEL}" <<'PY'
from pathlib import Path
import fnmatch
import re
import sys

from mof_heat_capacity.analysis.lammps import read_lammps_thermo
from mof_heat_capacity.analysis.results import discover_runs
from mof_heat_capacity.config import find_classical_output_file

patterns = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
model_uncertainty = bool(int(sys.argv[2]))
uncertainty_model = Path(sys.argv[3]).expanduser().resolve() if sys.argv[3] else None
if not patterns:
    raise SystemExit("error: --runs must contain at least one glob pattern")
runs, _ = discover_runs(Path("configs"), patterns)
if not runs:
    raise SystemExit("error: no completed trajectories match --runs")
selected_names = [config.name for _, config, _ in runs]
if model_uncertainty:
    try:
        import metatomic.torch as metatomic_torch
    except ImportError as error:
        raise SystemExit(
            "error: --model-uncertainty requires metatomic.torch in the analysis environment"
        ) from error
    checked = set()
    required_outputs = {"energy", "energy_ensemble"}
    configured_models = {config.exported_model.resolve() for _, config, _ in runs}
    if uncertainty_model is not None and len(configured_models) != 1:
        raise SystemExit(
            "error: one --uncertainty-model override cannot analyze multiple MLIPs; "
            "submit each --model separately"
        )
    for _, config, _ in runs:
        model_path = uncertainty_model or config.exported_model.resolve()
        if model_path in checked:
            continue
        if not model_path.is_file():
            raise SystemExit(f"error: uncertainty model is missing: {model_path}")
        model = metatomic_torch.load_atomistic_model(str(model_path))
        outputs = set(model.capabilities().outputs)
        missing = sorted(required_outputs.difference(outputs))
        if missing:
            raise SystemExit(
                f"error: {model_path} cannot propagate model uncertainty; "
                "missing output(s): " + ", ".join(missing)
            )
        del model
        checked.add(model_path)
missing_patterns = [
    pattern for pattern in patterns
    if not any(fnmatch.fnmatch(name, pattern) for name in selected_names)
]
if missing_patterns:
    raise SystemExit(
        "error: no completed trajectory matches requested pattern(s): "
        + ", ".join(missing_patterns)
    )
print(
    f"Validated {len(runs)} completed trajectory/thermo-log selections",
    file=sys.stderr,
)
for _, config, trajectory in runs:
    log = find_classical_output_file(
        config, "md.lammps.log", f"{config.md_prefix}.lammps.log"
    )
    if config.md_driver == "lammps" and log is None:
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
    print(f"  {config.name}: {trajectory} ({detail})", file=sys.stderr)
run_pattern = re.compile(
    r"^mof5-(?P<loading>[1-9][0-9]*)ch4-(?P<model>.+)-npt-"
    r"[1-9][0-9]*K-rep[0-9]+$"
)
for name in selected_names:
    match = run_pattern.fullmatch(name)
    if match is None:
        raise SystemExit(f"error: cannot group nonstandard run name by MLIP: {name}")
    print(f"{name}\t{match.group('model')}\t{match.group('loading')}")
PY
)
mapfile -t SELECTED_RUN_RECORDS <<< "${selection_output}"

echo "Analysis campaign: ${#SELECTED_RUN_RECORDS[@]} trajectory job(s)"
echo "Resources per job: ${PARTITION}/${QOS}, GPU=1, CPUs=${CPUS_PER_TASK}, time=${WALL_TIME}"
echo "Slurm output: ${SLURM_OUTPUT_DIR}/slurm-%x-%j.out"
if ((!DRY_RUN)); then
    mkdir -p "${SLURM_OUTPUT_DIR}"
fi

GROUP_KEYS=()
declare -A GROUP_RUNS=()
declare -A GROUP_JOB_IDS=()
declare -A GROUP_DIRS=()
for record in "${SELECTED_RUN_RECORDS[@]}"; do
    IFS=$'\t' read -r run_name model_label loading <<< "${record}"
    group_key="${model_label}/${loading}ch4"
    if [[ -z "${GROUP_DIRS[${group_key}]:-}" ]]; then
        GROUP_KEYS+=("${group_key}")
        GROUP_DIRS[${group_key}]="${ANALYSIS_DIR}/${model_label}/${loading}ch4"
        GROUP_RUNS[${group_key}]="${run_name}"
    else
        GROUP_RUNS[${group_key}]="${GROUP_RUNS[${group_key}]},${run_name}"
    fi
    run_analysis_dir="${GROUP_DIRS[${group_key}]}"
    if [[ "${run_name}" =~ -npt-([0-9]+)K-rep([0-9]+)$ ]]; then
        slurm_run_dir="${SLURM_OUTPUT_DIR}/trajectory-analysis/${model_label}/${loading}ch4/${BASH_REMATCH[1]}K/rep${BASH_REMATCH[2]}"
    else
        echo "error: cannot determine Slurm log directory for ${run_name}" >&2
        exit 2
    fi
    command=(
        sbatch --parsable
        --job-name="mof5-analysis-${run_name#mof5-}"
        --partition="${PARTITION}" --qos="${QOS}"
        --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}"
        --gres=gpu:1
        --time="${WALL_TIME}"
        --output="${slurm_run_dir}/%j.out"
        "${SCRIPT_DIR}/submit_analysis.sh"
        --internal-analysis-worker
        --runs "${run_name}"
        --discard-ps "${DISCARD_PS}"
        --analysis-dir "${run_analysis_dir}"
        --run-only
    )
    if ((NO_PLOTS)); then
        command+=(--no-plots)
    fi
    if ((MODEL_UNCERTAINTY)); then
        command+=(
            --model-uncertainty
            --uncertainty-stride "${UNCERTAINTY_STRIDE}"
            --uncertainty-batch-size "${UNCERTAINTY_BATCH_SIZE}"
            --uncertainty-central-tolerance-eV "${UNCERTAINTY_CENTRAL_TOLERANCE_EV}"
        )
        if [[ -n "${UNCERTAINTY_MODEL}" ]]; then
            command+=(--uncertainty-model "${UNCERTAINTY_MODEL}")
        fi
    fi
    if ((DRY_RUN)); then
        printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
    else
        mkdir -p "${slurm_run_dir}"
        submission=$("${command[@]}")
        job_id=${submission%%;*}
        if [[ -z "${GROUP_JOB_IDS[${group_key}]:-}" ]]; then
            GROUP_JOB_IDS[${group_key}]="${job_id}"
        else
            GROUP_JOB_IDS[${group_key}]="${GROUP_JOB_IDS[${group_key}]}:${job_id}"
        fi
        echo "Submitted ${run_name}: ${job_id}"
    fi
done

for group_key in "${GROUP_KEYS[@]}"; do
    model_label=${group_key%%/*}
    loading_label=${group_key#*/}
    if ((DRY_RUN)); then
        dependency="afterok:<${model_label}-${loading_label}-analysis-jobs>"
    else
        dependency="afterok:${GROUP_JOB_IDS[${group_key}]}"
    fi
    aggregate_command=(
        sbatch --parsable
        --job-name="mof5-analysis-${model_label}-${loading_label}-summary"
        --dependency="${dependency}"
        --partition="${PARTITION}" --qos="${QOS}"
        --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}"
        --gres=gpu:1
        --time="${WALL_TIME}"
        --output="${SLURM_OUTPUT_DIR}/trajectory-analysis/${model_label}/${loading_label}/summary/%j.out"
        "${SCRIPT_DIR}/submit_analysis.sh"
        --internal-analysis-worker
        --runs "${GROUP_RUNS[${group_key}]}"
        --discard-ps "${DISCARD_PS}"
        --analysis-dir "${GROUP_DIRS[${group_key}]}"
        --aggregate-only
    )
    if ((NO_PLOTS)); then
        aggregate_command+=(--no-plots)
    fi
    if ((MODEL_UNCERTAINTY)); then
        aggregate_command+=(
            --model-uncertainty
            --uncertainty-stride "${UNCERTAINTY_STRIDE}"
            --uncertainty-batch-size "${UNCERTAINTY_BATCH_SIZE}"
            --uncertainty-central-tolerance-eV "${UNCERTAINTY_CENTRAL_TOLERANCE_EV}"
        )
        if [[ -n "${UNCERTAINTY_MODEL}" ]]; then
            aggregate_command+=(--uncertainty-model "${UNCERTAINTY_MODEL}")
        fi
    fi
    if ((DRY_RUN)); then
        printf 'DRY RUN:'; printf ' %q' "${aggregate_command[@]}"; printf '\n'
    else
        mkdir -p "${SLURM_OUTPUT_DIR}/trajectory-analysis/${model_label}/${loading_label}/summary"
        submission=$("${aggregate_command[@]}")
        echo "Submitted dependent ${model_label}/${loading_label} summary: ${submission%%;*}"
    fi
done
