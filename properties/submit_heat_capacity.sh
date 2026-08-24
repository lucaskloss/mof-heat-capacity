#!/usr/bin/env bash

# Relax empty/loaded structures and submit AD Hessians for the hybrid workflow.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
JOB_SCRIPT="${PROJECT_DIR}/scripts/izar_job.sh"
MODEL="pet-mad"
LOADING=100
SOURCE_TEMPERATURE=300
REPLICAS="1"
EMPTY_STRUCTURE="input/mof5.pdb"
INCLUDE_EMPTY=1
CV_TEMPERATURES="100:500:10"
FMAX="0.01"
RELAX_STEPS=2000
OPTIMIZER="fire"
CONTINUE_LOADED=0
HESSIAN_ONLY=0
HESSIAN_DTYPE=""
HESSIAN_HOPS=""
HESSIAN_CHUNK_SIZE=""
HESSIAN_TAG=""
PARTITION="${MOF_HEAT_PARTITION:-gpu}"
QOS="${MOF_HEAT_QOS:-normal}"
WALL_TIME="${MOF_HEAT_TIME:-08:00:00}"
CPUS_PER_TASK="${MOF_HEAT_CPUS:-8}"
SLURM_OUTPUT_DIR="${MOF_SLURM_OUTPUT_DIR:-${PROJECT_DIR}/output/slurm}"
OVERWRITE=0
DRY_RUN=0


usage() {
    cat <<'EOF'
Usage: properties/submit_heat_capacity.sh [options]

Options:
  --model NAME            pet-mad, pet-sol, or both (default: pet-mad).
  --loading N             Positive methane loading (default: 100).
  --source-temperature N  Loaded-MD temperature used to choose minima (default: 300 K).
  --replicas LIST         Independent loaded replicas to quench (default: 1).
  --empty-structure PATH  Equilibrated empty MOF-5 structure (default: input/mof5.pdb).
  --skip-empty            Do not submit the one empty-reference Hessian per model.
  --cv-temperatures RANGE Harmonic C_V grid (default: 100:500:10 K).
  --fmax VALUE            Fixed-cell relaxation threshold in eV/A (default: 0.01).
  --relax-steps N         Maximum optimizer steps (default: 2000).
  --optimizer NAME        fire or lbfgs-linesearch (default: fire).
  --continue-loaded       Continue from existing loaded optimized structures.
  --hessian-only          Reuse existing minima and replace only Hessian archives.
  --dtype NAME            Hessian precision: float32 or float64 (default: config).
  --hops N                Hessian sparsity graph hops (default: config).
  --chunk-size N          Hessian AD chunk size (default: config).
  --hessian-tag NAME      Write a non-canonical archive for a convergence test.
  --partition NAME        Slurm partition (default: gpu).
  --qos NAME              Slurm QOS (default: normal).
  --time HH:MM:SS         Time per relaxation plus Hessian (default: 08:00:00).
  --cpus N                CPUs per task (default: 8).
  --slurm-output-dir PATH Slurm log directory (default: output/slurm).
  --overwrite             Replace existing relaxation and Hessian outputs.
  --dry-run               Validate inputs and print all submissions.
  -h, --help              Show this help.

Each loaded task reads the final structure from one completed classical MD
replica, relaxes it at fixed cell with the same MLIP, and computes one AD
Hessian from the resulting minimum. Empty MOF-5 is relaxed directly from the
supplied equilibrated structure; no empty-MOF MD trajectory is used.
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
        --source-temperature) require_value "$@"; SOURCE_TEMPERATURE="$2"; shift 2 ;;
        --replicas) require_value "$@"; REPLICAS="$2"; shift 2 ;;
        --empty-structure) require_value "$@"; EMPTY_STRUCTURE="$2"; shift 2 ;;
        --skip-empty) INCLUDE_EMPTY=0; shift ;;
        --cv-temperatures) require_value "$@"; CV_TEMPERATURES="$2"; shift 2 ;;
        --fmax) require_value "$@"; FMAX="$2"; shift 2 ;;
        --relax-steps) require_value "$@"; RELAX_STEPS="$2"; shift 2 ;;
        --optimizer) require_value "$@"; OPTIMIZER="$2"; shift 2 ;;
        --continue-loaded) CONTINUE_LOADED=1; shift ;;
        --hessian-only) HESSIAN_ONLY=1; shift ;;
        --dtype) require_value "$@"; HESSIAN_DTYPE="$2"; shift 2 ;;
        --hops) require_value "$@"; HESSIAN_HOPS="$2"; shift 2 ;;
        --chunk-size) require_value "$@"; HESSIAN_CHUNK_SIZE="$2"; shift 2 ;;
        --hessian-tag) require_value "$@"; HESSIAN_TAG="$2"; shift 2 ;;
        --partition) require_value "$@"; PARTITION="$2"; shift 2 ;;
        --qos) require_value "$@"; QOS="$2"; shift 2 ;;
        --time) require_value "$@"; WALL_TIME="$2"; shift 2 ;;
        --cpus) require_value "$@"; CPUS_PER_TASK="$2"; shift 2 ;;
        --slurm-output-dir) require_value "$@"; SLURM_OUTPUT_DIR="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done


case "${MODEL}" in
    pet-mad) MODEL_NAMES=("pet-mad-1.5-s-40nn") ;;
    pet-sol) MODEL_NAMES=("pet-sol-s-best") ;;
    both) MODEL_NAMES=("pet-mad-1.5-s-40nn" "pet-sol-s-best") ;;
    *) echo "error: --model must be pet-mad, pet-sol, or both" >&2; exit 2 ;;
esac
if [[ ! "${LOADING}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --loading must be a positive methane count" >&2
    exit 2
fi
if [[ ! "${SOURCE_TEMPERATURE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --source-temperature must be a positive integer" >&2
    exit 2
fi
if [[ ! "${RELAX_STEPS}" =~ ^[1-9][0-9]*$ \
    || ! "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --relax-steps and --cpus must be positive integers" >&2
    exit 2
fi
if [[ ! "${FMAX}" =~ ^[0-9]+([.][0-9]+)?$ \
    || "${FMAX}" =~ ^0+([.]0+)?$ ]]; then
    echo "error: --fmax must be a positive number" >&2
    exit 2
fi
if [[ ! "${CV_TEMPERATURES}" =~ ^[0-9]+([.][0-9]+)?:[0-9]+([.][0-9]+)?:[0-9]+([.][0-9]+)?$ ]]; then
    echo "error: --cv-temperatures must be start:stop:step" >&2
    exit 2
fi
if [[ "${OPTIMIZER}" != "fire" && "${OPTIMIZER}" != "lbfgs-linesearch" ]]; then
    echo "error: --optimizer must be fire or lbfgs-linesearch" >&2
    exit 2
fi
if [[ -n "${HESSIAN_DTYPE}" \
    && "${HESSIAN_DTYPE}" != "float32" \
    && "${HESSIAN_DTYPE}" != "float64" ]]; then
    echo "error: --dtype must be float32 or float64" >&2
    exit 2
fi
if [[ -n "${HESSIAN_HOPS}" && ! "${HESSIAN_HOPS}" =~ ^[0-9]+$ ]]; then
    echo "error: --hops must be a non-negative integer" >&2
    exit 2
fi
if [[ -n "${HESSIAN_CHUNK_SIZE}" \
    && ! "${HESSIAN_CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --chunk-size must be a positive integer" >&2
    exit 2
fi
if [[ -n "${HESSIAN_TAG}" && ! "${HESSIAN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "error: --hessian-tag may contain only letters, numbers, dot, underscore, and dash" >&2
    exit 2
fi
if ((CONTINUE_LOADED && !OVERWRITE)); then
    echo "error: --continue-loaded requires --overwrite for the canonical outputs" >&2
    exit 2
fi
if ((CONTINUE_LOADED && HESSIAN_ONLY)); then
    echo "error: --continue-loaded and --hessian-only are mutually exclusive" >&2
    exit 2
fi
if ((CONTINUE_LOADED)) && ((INCLUDE_EMPTY)); then
    echo "error: combine --continue-loaded with --skip-empty; the empty result is independent" >&2
    exit 2
fi
if [[ "${SLURM_OUTPUT_DIR}" != /* ]]; then
    SLURM_OUTPUT_DIR="${PROJECT_DIR}/${SLURM_OUTPUT_DIR}"
fi
if [[ "${EMPTY_STRUCTURE}" != /* ]]; then
    EMPTY_STRUCTURE="${PROJECT_DIR}/${EMPTY_STRUCTURE}"
fi
if ((INCLUDE_EMPTY)) && [[ ! -f "${EMPTY_STRUCTURE}" ]]; then
    echo "error: empty MOF-5 structure not found: ${EMPTY_STRUCTURE}" >&2
    exit 2
fi
if ((!DRY_RUN)) && ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch is unavailable; submit from a cluster login node" >&2
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
        echo "error: --replicas must contain positive integers" >&2
        exit 2
    fi
    if [[ -n "${SEEN_REPLICAS[${replica}]:-}" ]]; then
        echo "error: duplicate replica: ${replica}" >&2
        exit 2
    fi
    SEEN_REPLICAS[${replica}]=1
done


CONFIGS=()
INPUTS=()
RELAXED=()
HESSIANS=()
LABELS=()
ALLOW_SUBSETS=()
cd "${PROJECT_DIR}"
for model_name in "${MODEL_NAMES[@]}"; do
    carrier_config=""
    for replica in "${REPLICA_VALUES[@]}"; do
        printf -v replica_tag '%02d' "${replica}"
        run="mof5-${LOADING}ch4-${model_name}-npt-${SOURCE_TEMPERATURE}K-rep${replica_tag}"
        config="configs/${run}.toml"
        trajectory="output/classical/production/${LOADING}ch4/${run}/${run}.final.data"
        if [[ ! -f "${config}" || ! -f "${trajectory}" ]]; then
            echo "error: completed loaded run is required: ${config} and ${trajectory}" >&2
            exit 2
        fi
        carrier_config="${carrier_config:-${config}}"
        base="output/hybrid/${model_name}/${LOADING}ch4"
        relaxed="${base}/minima/rep${replica_tag}.optimized.extxyz"
        if ((CONTINUE_LOADED)); then
            if [[ ! -f "${relaxed}" ]]; then
                echo "error: loaded minimum to continue is missing: ${relaxed}" >&2
                exit 2
            fi
            relaxation_input="${relaxed}"
        else
            relaxation_input="${trajectory}"
        fi
        CONFIGS+=("${config}")
        INPUTS+=("${relaxation_input}")
        RELAXED+=("${relaxed}")
        hessian_suffix="${HESSIAN_TAG:+.${HESSIAN_TAG}}"
        HESSIANS+=("${base}/hessians/rep${replica_tag}${hessian_suffix}.npz")
        LABELS+=("${LOADING}ch4-r${replica_tag}")
        ALLOW_SUBSETS+=(0)
    done
    if ((INCLUDE_EMPTY)); then
        base="output/hybrid/${model_name}/0ch4"
        CONFIGS+=("${carrier_config}")
        INPUTS+=("${EMPTY_STRUCTURE}")
        RELAXED+=("${base}/minima/empty.optimized.extxyz")
        hessian_suffix="${HESSIAN_TAG:+.${HESSIAN_TAG}}"
        HESSIANS+=("${base}/hessians/empty${hessian_suffix}.npz")
        LABELS+=("empty")
        ALLOW_SUBSETS+=(1)
    fi
done


for index in "${!CONFIGS[@]}"; do
    if ((!OVERWRITE && !HESSIAN_ONLY)) && [[ -e "${HESSIANS[index]}" ]]; then
        echo "error: Hessian output exists; use --overwrite to replace it: ${HESSIANS[index]}" >&2
        exit 2
    fi
    if ((HESSIAN_ONLY)) && [[ ! -e "${RELAXED[index]}" ]]; then
        echo "error: --hessian-only requires an existing minimum: ${RELAXED[index]}" >&2
        exit 2
    fi
    if ((!OVERWRITE || HESSIAN_ONLY)) && [[ -e "${RELAXED[index]}" ]]; then
        echo "Will reuse converged relaxed structure: ${RELAXED[index]}"
    fi
done


echo "Hybrid Hessian campaign: ${#CONFIGS[@]} fixed-cell relaxation/Hessian task(s)"
echo "Loaded source: ${LOADING} CH4 at ${SOURCE_TEMPERATURE} K; replicas ${REPLICAS}"
echo "Harmonic grid: ${CV_TEMPERATURES} K; fmax=${FMAX} eV/A; optimizer=${OPTIMIZER}"
echo "Hessian overrides: dtype=${HESSIAN_DTYPE:-config}; hops=${HESSIAN_HOPS:-config}; chunk=${HESSIAN_CHUNK_SIZE:-config}"
echo "Hessian output: ${HESSIAN_TAG:-canonical}"
if ((CONTINUE_LOADED)); then
    echo "Continuation: existing loaded optimized structure(s)"
fi
if ((HESSIAN_ONLY)); then
    echo "Relaxation: reuse existing minimum; overwrite Hessian only"
fi
if ((!DRY_RUN)); then
    mkdir -p "${SLURM_OUTPUT_DIR}"
fi

for index in "${!CONFIGS[@]}"; do
    config="${CONFIGS[index]}"
    input="${INPUTS[index]}"
    relaxed="${RELAXED[index]}"
    hessian="${HESSIANS[index]}"
    label="${LABELS[index]}"
    allow_subset="${ALLOW_SUBSETS[index]}"
    relax_overwrite="${OVERWRITE}"
    heat_overwrite="${OVERWRITE}"
    if ((HESSIAN_ONLY)); then
        relax_overwrite=0
        heat_overwrite=1
    fi
    command=(
        sbatch --parsable
        --job-name="mof5-hybrid-${label}"
        --partition="${PARTITION}" --qos="${QOS}"
        --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}"
        --gres=gpu:1 --time="${WALL_TIME}"
        --output="${SLURM_OUTPUT_DIR}/slurm-%x-%j.out"
        --export="ALL,MOF_STAGE=relax-and-heat-capacity,MOF_CONFIG=${config},MOF_RELAX_INPUT=${input},MOF_RELAX_INDEX=-1,MOF_RELAX_OUTPUT=${relaxed},MOF_RELAX_FMAX=${FMAX},MOF_RELAX_STEPS=${RELAX_STEPS},MOF_RELAX_OPTIMIZER=${OPTIMIZER},MOF_RELAX_ALLOW_ELEMENT_SUBSET=${allow_subset},MOF_RELAX_OVERWRITE=${relax_overwrite},MOF_HEAT_TEMPERATURES=${CV_TEMPERATURES},MOF_HEAT_OUTPUT=${hessian},MOF_HEAT_DTYPE=${HESSIAN_DTYPE},MOF_HEAT_HOPS=${HESSIAN_HOPS},MOF_HEAT_CHUNK_SIZE=${HESSIAN_CHUNK_SIZE},MOF_HEAT_OVERWRITE=${heat_overwrite}"
        "${JOB_SCRIPT}"
    )
    if ((DRY_RUN)); then
        printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
    else
        submission=$("${command[@]}")
        echo "Submitted ${label}: ${submission%%;*}"
    fi
done
