#!/usr/bin/env bash

# Remove equilibration frames from LAMMPS trajectories without a temporary copy.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
DEFAULT_OUTPUT_ROOT="${SCRATCH:+${SCRATCH}/mof-heat-capacity/output}"
OUTPUT_ROOT="${MOF_OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT:-${PROJECT_DIR}/output}}"
TRAJECTORY_ROOT=""
FIRST_STEP=200000
APPLY=0

usage() {
    cat <<'EOF'
Usage: scripts/md/truncate_trajectories.sh [options]

Options:
  --output-root DIR  Root containing md/production.
  --trajectory-root DIR  Process only this production subdirectory.
  --first-step N     First timestep to retain (default: 200000 = 100 ps).
  --apply            Perform the irreversible trajectory trimming.
  -h, --help         Show this help.

Without --apply this is a dry run. Stop jobs writing trajectories first.
Runs that ended before --first-step are left untouched.
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
        --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
        --trajectory-root) require_value "$@"; TRAJECTORY_ROOT="$2"; shift 2 ;;
        --first-step) require_value "$@"; FIRST_STEP="$2"; shift 2 ;;
        --apply) APPLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${PROJECT_DIR}/${OUTPUT_ROOT}"
fi
if [[ ! "${FIRST_STEP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --first-step must be a positive integer" >&2
    exit 2
fi

first_kept_offset() {
    local trajectory=$1
    local timestep_offset
    local frame_offset
    local header_length=15  # "ITEM: TIMESTEP" plus its newline

    timestep_offset=$(LC_ALL=C grep -abo "^${FIRST_STEP}$" "${trajectory}" \
        | head -n 1 | cut -d: -f1 || true)
    if [[ ! "${timestep_offset}" =~ ^[0-9]+$ ]] \
        || ((timestep_offset < header_length)); then
        return
    fi
    frame_offset=$((timestep_offset - header_length))
    if [[ $(dd if="${trajectory}" bs=1 skip="${frame_offset}" count=14 status=none) \
        != "ITEM: TIMESTEP" ]]; then
        return
    fi
    printf '%s\n' "${frame_offset}"
}

if [[ -z "${TRAJECTORY_ROOT}" ]]; then
    trajectory_root="${OUTPUT_ROOT}/md/production"
elif [[ "${TRAJECTORY_ROOT}" == /* ]]; then
    trajectory_root="${TRAJECTORY_ROOT}"
else
    trajectory_root="${PROJECT_DIR}/${TRAJECTORY_ROOT}"
fi
if [[ ! -d "${trajectory_root}" ]]; then
    echo "error: production trajectory directory does not exist: ${trajectory_root}" >&2
    exit 2
fi

echo "Output root: ${OUTPUT_ROOT}"
echo "Discarding frames before timestep ${FIRST_STEP}"
if ((APPLY == 0)); then
    echo "Dry run: pass --apply after reviewing this output."
fi

changed=0
skipped=0
while IFS= read -r -d '' trajectory; do
    offset=$(first_kept_offset "${trajectory}")
    if [[ -z "${offset}" || "${offset}" == "0" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    changed=$((changed + 1))
    if ((APPLY)); then
        size=$(stat --format='%s' "${trajectory}")
        retained_size=$((size - offset))
        dd if="${trajectory}" of="${trajectory}" bs=1048576 \
            iflag=skip_bytes,count_bytes oflag=seek_bytes \
            skip="${offset}" seek=0 count="${retained_size}" \
            conv=notrunc,fsync status=none
        truncate --size="${retained_size}" "${trajectory}"
        echo "Trimmed ${trajectory}: removed ${offset} bytes"
    else
        echo "Would trim ${trajectory}: remove ${offset} bytes"
    fi
done < <(find "${trajectory_root}" -type f -name 'trajectory.lammpstrj' -print0)

if ((APPLY)); then
    echo "Trajectories trimmed: ${changed}; no frame at/after the limit: ${skipped}"
else
    echo "Trajectories to trim: ${changed}; no frame at/after the limit: ${skipped}"
fi
