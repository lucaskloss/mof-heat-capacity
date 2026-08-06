#!/usr/bin/env bash

# Submit harmonic C_V calculations for the PET-SOL MOF-5 + 100 CH4 runs.

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export MOF_CV_COMMAND_NAME="submit_mof5_100ch4_pet_sol_heat_capacity.sh"
export MOF_CV_RUN_LABEL="MOF-5 + 100 CH4 PET-SOL"
export MOF_CV_MODEL_LABEL="pet_sol-s-best_nostress"
export MOF_CV_CONFIG_PATTERN="configs/mof5_100ch4_%sK_pet_sol_test.toml"
export MOF_CV_OUTPUT_PATTERN="output/mof5-100ch4-%sK-pet-sol-s-best-nostress-test"
export MOF_CV_PREFIX_PATTERN="mof5-100ch4-%sK-pet-sol-s-best-nostress-test"
export MOF_CV_JOB_PATTERN="mof5-cv-%sK-pet-sol"
export MOF_CV_RESULT_PATTERN="heat-capacity-pet-sol-s-best-nostress-frame-%s.npz"

exec "${SCRIPT_DIR}/submit_mof5_100ch4_heat_capacity.sh" "$@"
