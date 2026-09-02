#!/usr/bin/env bash

# Prepare, validate, calibrate, and submit loaded classical-NPT runs to Slurm.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
GPU_RUNTIME="${PROJECT_DIR}/scripts/slurm/izar_gpu_runtime.sh"
ENV_PREFIX="${MOF_ENV_PREFIX:-${HOME}/.conda/envs/mof-heat-capacity-izar}"
VALIDATE_PYTHON="${MOF_CAMPAIGN_PYTHON:-${ENV_PREFIX}/bin/python}"
MODEL="both"
LOADING=100
TEMPERATURES=(200 225 250 275 300 325 350 375 400)
REPLICAS=1
PARTITION="${MOF_MD_PARTITION:-gpu}"
QOS="${MOF_MD_QOS:-normal}"
WALL_TIME="${MOF_MD_TIME:-2-23:50:00}"
WALL_TIME_SET=0
if [[ -n "${MOF_MD_TIME:-}" ]]; then
    WALL_TIME_SET=1
fi
CPUS_PER_TASK="${MOF_MD_CPUS:-8}"
SLURM_OUTPUT_DIR="${MOF_SLURM_OUTPUT_DIR:-${PROJECT_DIR}/output/slurm}"
STEPS=""
DEBUG=0
CALIBRATION=0
PRODUCTION_AFTER_CALIBRATION=""
RESUME=0
AUTO_RESUME=1
DRY_RUN=0
CALIBRATION_STEPS=1000
CALIBRATION_MARGIN_PERCENT=125
CALIBRATION_STARTUP_SECONDS=600
NORMAL_QOS_WALL_TIME="2-23:50:00"
AUTO_RESUME_BUFFER_SECONDS=600


usage() {
    cat <<'EOF'
Usage: scripts/md/submit_loaded_md.sh [options]

Options:
  --model NAME         pet-mad, pet-sol, or both (default: both).
  --loading N          Positive methane count (default: 100).
  --temperatures LIST  Comma-separated temperatures
                       (default: 200 to 400 K in 25 K steps).
  --replicas N         Replica count (default: 1).
  --partition NAME     Slurm partition (default: gpu).
  --qos NAME           Production Slurm QOS (default: normal).
  --time [D-]HH:MM:SS Production wall-time override (default for normal QOS:
                       2-23:50:00, ten minutes below its three-day limit).
  --cpus N             CPUs per task (default: 8).
  --steps N            Override the configured final step.
  --debug              Manually submit an isolated 10-step debug run.
  --calibration        Manually submit an isolated 1000-step timing run.
  --resume             Submit only unfinished production runs, continuing from
                       their latest numeric LAMMPS restart.
  --no-auto-resume     Do not automatically submit another production job when
                       an MD segment reaches its wall-time buffer.
  --dry-run            Prepare missing inputs, validate, and print the full pipeline.
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
        --time) require_value "$@"; WALL_TIME="$2"; WALL_TIME_SET=1; shift 2 ;;
        --cpus) require_value "$@"; CPUS_PER_TASK="$2"; shift 2 ;;
        --steps) require_value "$@"; STEPS="$2"; shift 2 ;;
        --debug) DEBUG=1; shift ;;
        --calibration) CALIBRATION=1; shift ;;
        --production-after-calibration)
            require_value "$@"; PRODUCTION_AFTER_CALIBRATION="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --no-auto-resume) AUTO_RESUME=0; shift ;;
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
if ((DEBUG && CALIBRATION)) || { [[ -n "${PRODUCTION_AFTER_CALIBRATION}" ]] && ((DEBUG || CALIBRATION)); }; then
    echo "error: --debug, --calibration, and --production-after-calibration are mutually exclusive" >&2
    exit 2
fi
if ((DEBUG || CALIBRATION)) && ((RESUME)); then
    echo "error: debug/calibration cannot be resumed" >&2
    exit 2
fi
if ((DEBUG)); then
    TEMPERATURES=(300); REPLICAS=1; QOS=debug; WALL_TIME=00:30:00; CPUS_PER_TASK=4; STEPS=10; AUTO_RESUME=0
elif ((CALIBRATION)); then
    TEMPERATURES=(300); REPLICAS=1; QOS=debug; WALL_TIME=01:00:00; CPUS_PER_TASK=4; STEPS=${CALIBRATION_STEPS}; AUTO_RESUME=0
fi
if ((RESUME && !WALL_TIME_SET)); then
    WALL_TIME="${NORMAL_QOS_WALL_TIME}"
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


prepare_campaign() {
    local model
    local temperature_list

    local IFS=,
    temperature_list="${TEMPERATURES[*]}"
    for model in "${MODELS[@]}"; do
        echo "Preparing ${model} campaign inputs"
        "${VALIDATE_PYTHON}" -m mof_heat_capacity.protocols.loaded \
            --model "${model}" \
            --loading "${LOADING}" \
            --temperatures "${temperature_list}" \
            --replicas "${REPLICAS}" \
            --skip-existing
    done
}


CONFIGS=()
RUN_MODELS=()
RUN_MODEL_LABELS=()
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
            config="configs/${MODEL_LABELS[model_index]}/${LOADING}ch4/${temperature}K-rep${replica_tag}.toml"
            if [[ ! -f "${config}" && -f "configs/${stem}.toml" ]]; then
                config="configs/${stem}.toml"
            fi
            CONFIGS+=("${config}")
            RUN_MODELS+=("${MODELS[model_index]}")
            RUN_MODEL_LABELS+=("${MODEL_LABELS[model_index]}")
            RUN_TEMPERATURES+=("${temperature}")
            RUN_REPLICAS+=("${replica_tag}")
        done
    done
done


cd "${PROJECT_DIR}"
echo "Loaded MD campaign: model=${MODEL}; loading=${LOADING}; replicas=${REPLICAS}"
printf 'Temperatures:'
printf ' %s' "${TEMPERATURES[@]}"
printf ' K\n'
prepare_campaign
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


estimate_production_wall_time() {
    local calibration_seconds
    local production_steps
    local estimate_seconds
    local estimated_wall_time
    local first_config=${CONFIGS[0]}

    if [[ ! -f "${PRODUCTION_AFTER_CALIBRATION}" ]]; then
        echo "error: calibration timing file is missing: ${PRODUCTION_AFTER_CALIBRATION}" >&2
        exit 2
    fi
    calibration_seconds=$(<"${PRODUCTION_AFTER_CALIBRATION}")
    if [[ ! "${calibration_seconds}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: invalid calibration duration in ${PRODUCTION_AFTER_CALIBRATION}" >&2
        exit 2
    fi
    production_steps=$("${VALIDATE_PYTHON}" - "${first_config}" "${STEPS}" <<'PY'
from pathlib import Path
import sys

from mof_heat_capacity.config import load_run_config

config = load_run_config(Path(sys.argv[1]))
print(sys.argv[2] or config.md_steps)
PY
    )
    if [[ ! "${production_steps}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: invalid production step count: ${production_steps}" >&2
        exit 2
    fi
    estimate_seconds=$((
        (calibration_seconds * production_steps * CALIBRATION_MARGIN_PERCENT
            + CALIBRATION_STEPS * 100 - 1) / (CALIBRATION_STEPS * 100)
        + CALIBRATION_STARTUP_SECONDS
    ))
    printf -v estimated_wall_time '%02d:%02d:%02d' \
        "$((estimate_seconds / 3600))" \
        "$(((estimate_seconds / 60) % 60))" \
        "$((estimate_seconds % 60))"
    if [[ "${QOS}" == "normal" ]]; then
        WALL_TIME="${NORMAL_QOS_WALL_TIME}"
        echo "Calibrated production estimate: ${estimated_wall_time}."
        if ((AUTO_RESUME)); then
            echo "Using the default normal-QOS allocation ${WALL_TIME}; unfinished MD will automatically continue in a new job."
        else
            echo "Using the default normal-QOS allocation ${WALL_TIME}; automatic continuation is disabled."
        fi
    else
        WALL_TIME="${estimated_wall_time}"
    fi
    echo "Calibration: ${calibration_seconds}s for ${CALIBRATION_STEPS} steps; production: ${production_steps} steps; requested wall time: ${WALL_TIME}"
}


submit_automatic_pipeline() {
    local model_index
    local model
    local model_label
    local config
    local stem
    local calibration_dir
    local calibration_log_dir
    local planner_log_dir
    local timing_file
    local calibration_submission
    local planner_submission
    local continuation
    local -a calibration_command
    local -a continuation_command

    for model_index in "${!MODELS[@]}"; do
        model=${MODELS[model_index]}
        model_label=${MODEL_LABELS[model_index]}
        stem="mof5-${LOADING}ch4-${MODEL_LABELS[model_index]}-npt-${TEMPERATURES[0]}K-rep01"
        config="configs/${MODEL_LABELS[model_index]}/${LOADING}ch4/${TEMPERATURES[0]}K-rep01.toml"
        if [[ ! -f "${config}" && -f "configs/${stem}.toml" ]]; then
            config="configs/${stem}.toml"
        fi
        calibration_dir="output/md/calibration/${model_label}/${LOADING}ch4/${TEMPERATURES[0]}K/rep01"
        timing_file="${calibration_dir}/elapsed-seconds.txt"
        calibration_log_dir="${SLURM_OUTPUT_DIR}/simulation/${model_label}/${LOADING}ch4/calibration"
        planner_log_dir="${SLURM_OUTPUT_DIR}/simulation/${model_label}/${LOADING}ch4/planner"
        calibration_command=(
            sbatch --parsable --job-name="mof5-${LOADING}ch4-${model}-calibration"
            --partition="${PARTITION}" --qos=debug
            --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=01:00:00
            --output="${calibration_log_dir}/%j.out"
            --export="ALL,MOF_STAGE=md,MOF_CONFIG=${config},MOF_STEPS=${CALIBRATION_STEPS},MOF_OUTPUT_DIR=${calibration_dir},MOF_PREFIX=md-calibration,MOF_RERUN=1,MOF_TIMING_FILE=${timing_file}"
            "${GPU_RUNTIME}"
        )
        continuation_command=(
            "${PROJECT_DIR}/scripts/md/submit_loaded_md.sh"
            --model "${model}" --loading "${LOADING}"
            --temperatures "$(IFS=,; echo "${TEMPERATURES[*]}")" --replicas "${REPLICAS}"
            --partition "${PARTITION}" --qos "${QOS}" --cpus "${CPUS_PER_TASK}"
            --production-after-calibration "${timing_file}"
        )
        if [[ -n "${STEPS}" ]]; then
            continuation_command+=(--steps "${STEPS}")
        fi
        if ((WALL_TIME_SET)); then
            continuation_command+=(--time "${WALL_TIME}")
        fi
        if ((!AUTO_RESUME)); then
            continuation_command+=(--no-auto-resume)
        fi
        printf -v continuation '%q ' "${continuation_command[@]}"
        if ((DRY_RUN)); then
            printf 'DRY RUN:'; printf ' %q' "${calibration_command[@]}"; printf '\n'
            printf 'DRY RUN: sbatch --dependency=afterok:<calibration-job> --partition=%q --qos=debug --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --time=00:20:00 --wrap %q\n' "${PARTITION}" "cd ${PROJECT_DIR} && exec ${continuation}"
            continue
        fi
        mkdir -p "${calibration_log_dir}" "${planner_log_dir}"
        calibration_submission=$("${calibration_command[@]}")
        planner_submission=$(sbatch --parsable --job-name="mof5-${LOADING}ch4-${model}-production-plan" \
            --dependency="afterok:${calibration_submission%%;*}" \
            --partition="${PARTITION}" --qos=debug --nodes=1 --ntasks=1 \
            --cpus-per-task=1 --gres=gpu:1 --time=00:20:00 \
            --output="${planner_log_dir}/%j.out" \
            --wrap="cd ${PROJECT_DIR} && exec ${continuation}")
        echo "Submitted ${model} pipeline: debug-QOS calibration ${calibration_submission%%;*} -> debug-QOS production planner ${planner_submission%%;*}"
        echo "The planner submits the independent production jobs under QOS ${QOS}."
    done
}

if ((!DRY_RUN)); then
    mkdir -p "${SLURM_OUTPUT_DIR}"
fi
if [[ -z "${PRODUCTION_AFTER_CALIBRATION}" ]] && ((!DEBUG && !CALIBRATION && !RESUME)); then
    submit_automatic_pipeline
    exit 0
fi
if [[ -n "${PRODUCTION_AFTER_CALIBRATION}" ]]; then
    if ((WALL_TIME_SET)); then
        echo "Using explicit production wall-time override: ${WALL_TIME}"
    else
        estimate_production_wall_time
    fi
fi


wall_time_seconds() {
    local specification=$1
    local days=0
    local clock=${specification}
    local hours
    local minutes
    local seconds
    local extra

    if [[ "${clock}" == *-* ]]; then
        days=${clock%%-*}
        clock=${clock#*-}
    fi
    IFS=: read -r hours minutes seconds extra <<< "${clock}"
    if [[ -n "${extra:-}" || ! "${days}" =~ ^[0-9]+$ \
        || ! "${hours:-}" =~ ^[0-9]+$ || ! "${minutes:-}" =~ ^[0-9]+$ \
        || ! "${seconds:-}" =~ ^[0-9]+$ || "${minutes}" -ge 60 \
        || "${seconds}" -ge 60 ]]; then
        echo "error: invalid wall time: ${specification}" >&2
        return 2
    fi
    printf '%s\n' "$((days * 86400 + hours * 3600 + minutes * 60 + seconds))"
}


segment_seconds=""
if ((AUTO_RESUME && !DEBUG && !CALIBRATION)); then
    allocation_seconds=$(wall_time_seconds "${WALL_TIME}")
    if ((allocation_seconds <= AUTO_RESUME_BUFFER_SECONDS)); then
        echo "error: automatic resume requires a wall time longer than ${AUTO_RESUME_BUFFER_SECONDS} seconds" >&2
        exit 2
    fi
    segment_seconds=$((allocation_seconds - AUTO_RESUME_BUFFER_SECONDS))
fi
for index in "${!CONFIGS[@]}"; do
    config="${CONFIGS[index]}"
    run_model="${RUN_MODELS[index]}"
    model_label="${RUN_MODEL_LABELS[index]}"
    temperature="${RUN_TEMPERATURES[index]}"
    replica_tag="${RUN_REPLICAS[index]}"
    stem="mof5-${LOADING}ch4-${model_label}-npt-${temperature}K-rep${replica_tag}"
    stage=production
    prefix_suffix=""
    rerun=0
    if ((DEBUG)); then
        stage=debug; prefix_suffix=-debug; rerun=1
    elif ((CALIBRATION)); then
        stage=calibration; prefix_suffix=-calibration; rerun=1
    fi
    output_dir="output/md/${stage}/${model_label}/${LOADING}ch4/${temperature}K/rep${replica_tag}"
    historical_output_dir="output/classical/${stage}/${model_label}/${LOADING}ch4/${stem}"
    legacy_output_dir="output/classical/${stage}/${LOADING}ch4/${stem}"
    output_prefix="md${prefix_suffix}"
    if [[ "${stage}" == "production" && -d "${historical_output_dir}" ]]; then
        output_dir="${historical_output_dir}"
        output_prefix="${stem}${prefix_suffix}"
        echo "Using existing historical run directory: ${output_dir}"
    elif [[ "${stage}" == "production" && -d "${legacy_output_dir}" ]]; then
        output_dir="${legacy_output_dir}"
        output_prefix="${stem}${prefix_suffix}"
        echo "Using existing legacy run directory: ${output_dir}"
    fi
    final_restart="${output_dir}/${output_prefix}.restart.final"
    if ((RESUME)) && [[ -f "${final_restart}" ]]; then
        echo "Skipping completed production run: ${stem}"
        continue
    fi
    slurm_run_dir="${SLURM_OUTPUT_DIR}/simulation/${model_label}/${LOADING}ch4/${temperature}K/rep${replica_tag}"
    command=(
        sbatch --parsable
        --job-name="mof5-${LOADING}ch4-${run_model}-npt-${temperature}K-r${replica_tag}"
        --partition="${PARTITION}" --qos="${QOS}"
        --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}"
        --gres=gpu:1 --time="${WALL_TIME}"
        --output="${slurm_run_dir}/%j.out"
        --export="ALL,MOF_STAGE=md,MOF_CONFIG=${config},MOF_STEPS=${STEPS},MOF_OUTPUT_DIR=${output_dir},MOF_PREFIX=${output_prefix},MOF_RESUME=${RESUME},MOF_RERUN=${rerun},MOF_AUTO_RESUME=${AUTO_RESUME},MOF_MD_SEGMENT_SECONDS=${segment_seconds},MOF_SUBMIT_JOB_NAME=mof5-${LOADING}ch4-${run_model}-npt-${temperature}K-r${replica_tag},MOF_SUBMIT_PARTITION=${PARTITION},MOF_SUBMIT_QOS=${QOS},MOF_SUBMIT_TIME=${WALL_TIME},MOF_SUBMIT_CPUS=${CPUS_PER_TASK},MOF_SUBMIT_OUTPUT=${slurm_run_dir}/%j.out,MOF_RUNTIME_PATH=${GPU_RUNTIME}"
        "${GPU_RUNTIME}"
    )
    if ((DRY_RUN)); then
        printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
    else
        mkdir -p "${slurm_run_dir}"
        submission=$("${command[@]}")
        echo "Submitted ${temperature} K replica ${replica_tag}: ${submission%%;*}"
    fi
done
