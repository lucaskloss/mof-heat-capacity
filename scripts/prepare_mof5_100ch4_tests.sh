#!/usr/bin/env bash

# Prepare one MOF-5 + 100 CH4 structure and five short NVT test configs.
# The generated runs use 100, 200, 300, 400, and 500 K; 4,000 steps; and a
# 0.5 fs timestep, corresponding to 2 ps per temperature.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BASE_CONFIG="${PROJECT_DIR}/configs/mof5_pet_mad.toml"
STRUCTURE_DIR="${PROJECT_DIR}/output/structures"
STRUCTURE_PATH="${STRUCTURE_DIR}/mof5-100ch4-seed2025.pdb"
DATA_PATH="${STRUCTURE_DIR}/mof5-100ch4-seed2025.data"
TEMPERATURES=(100 200 300 400 500)
DRY_RUN=0
FORCE=0


usage() {
    cat <<'EOF'
Usage: scripts/prepare_mof5_100ch4_tests.sh [--dry-run] [--force]

Creates:
  output/structures/mof5-100ch4-seed2025.{pdb,data}
  configs/mof5_100ch4_{100,200,300,400,500}K_test.toml

Options:
  --dry-run  Print the planned outputs without creating them.
  --force    Replace the generated structure and any differing test configs.
  -h, --help Show this help text.
EOF
}


while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        --force)
            FORCE=1
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
    shift
done


cd "${PROJECT_DIR}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "error: base configuration not found: ${BASE_CONFIG}" >&2
    exit 2
fi

if ((DRY_RUN)); then
    echo "Would prepare a 924-atom MOF-5 + 100 CH4 structure:"
    echo "  ${STRUCTURE_PATH}"
    echo "  ${DATA_PATH}"
    echo "Would generate 2 ps, 0.5 fs NVT configurations:"
    for temperature in "${TEMPERATURES[@]}"; do
        echo "  configs/mof5_100ch4_${temperature}K_test.toml (${temperature} K)"
    done
    exit 0
fi

if [[ -e "${STRUCTURE_PATH}" || -e "${DATA_PATH}" ]]; then
    if ((FORCE)); then
        echo "Replacing the existing generated methane structure."
    elif [[ -f "${STRUCTURE_PATH}" && -f "${DATA_PATH}" ]]; then
        echo "Reusing existing methane structure: ${STRUCTURE_PATH}"
    else
        echo "error: only one generated structure file exists; use --force to replace it" >&2
        exit 2
    fi
fi

if ((FORCE)) || [[ ! -f "${STRUCTURE_PATH}" || ! -f "${DATA_PATH}" ]]; then
    python utils/insert_methane.py \
        --nmol 100 \
        --seed 2025 \
        --try 20000 \
        --min-distance 1.5 \
        --output "${STRUCTURE_PATH}" \
        --data-output "${DATA_PATH}"
fi

python - "${STRUCTURE_PATH}" <<'PY'
from pathlib import Path
import sys

from ase.io import read

path = Path(sys.argv[1])
atoms = read(path)
if len(atoms) != 924:
    raise RuntimeError(f"expected 924 atoms for MOF-5 + 100 CH4, found {len(atoms)}")
print(f"Validated loaded structure: {len(atoms)} atoms; {atoms.get_chemical_formula()}")
PY

tmp_config=""
trap '[[ -z "${tmp_config}" ]] || rm -f -- "${tmp_config}"' EXIT

for temperature in "${TEMPERATURES[@]}"; do
    target="${PROJECT_DIR}/configs/mof5_100ch4_${temperature}K_test.toml"
    tmp_config=$(mktemp)

    sed \
        -e "s/^name = \"mof5-pet-mad\"$/name = \"mof5-100ch4-${temperature}K-test\"/" \
        -e "s#^output_dir = \"../output/mof5-pet-mad\"\$#output_dir = \"../output/mof5-100ch4-${temperature}K-test\"#" \
        -e 's#^path = "../input/mof5.cif"$#path = "../output/structures/mof5-100ch4-seed2025.pdb"#' \
        -e "s/^temperature_K = 300.0$/temperature_K = ${temperature}.0/" \
        -e 's/^steps = 10$/steps = 4000/' \
        -e 's/^timestep_fs = 0.25$/timestep_fs = 0.5/' \
        -e "s/^prefix = \"mof5-md\"$/prefix = \"mof5-100ch4-${temperature}K-test\"/" \
        "${BASE_CONFIG}" > "${tmp_config}"

    if ! grep -Fq "temperature_K = ${temperature}.0" "${tmp_config}" || \
       ! grep -Fq 'steps = 4000' "${tmp_config}" || \
       ! grep -Fq 'timestep_fs = 0.5' "${tmp_config}" || \
       ! grep -Fq 'path = "../output/structures/mof5-100ch4-seed2025.pdb"' "${tmp_config}"; then
        echo "error: failed to generate the expected settings for ${temperature} K" >&2
        exit 2
    fi

    if [[ -f "${target}" ]] && cmp -s "${tmp_config}" "${target}"; then
        echo "Configuration is already current: ${target}"
        rm -f -- "${tmp_config}"
    elif [[ -e "${target}" ]] && ((!FORCE)); then
        echo "error: configuration exists and differs: ${target}" >&2
        echo "rerun with --force to replace generated test configurations" >&2
        exit 2
    else
        mv -- "${tmp_config}" "${target}"
        echo "Wrote configuration: ${target}"
    fi
    tmp_config=""
done

echo "Preparation complete. Preview submissions with:"
echo "  scripts/submit_mof5_100ch4_tests.sh --dry-run"

