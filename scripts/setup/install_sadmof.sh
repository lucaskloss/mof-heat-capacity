#!/usr/bin/env bash

set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
SADMOF_SOURCE=${SADMOF_SOURCE:-"${PROJECT_DIR}/../repos/sadmof-work"}
SADMOF_DIR="${PROJECT_DIR}/external/sadmof"
DEPENDENCY_DIR="${SADMOF_DIR}/deps"


if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "error: activate the project conda environment before running this script" >&2
    exit 1
fi

if [[ ! -f "${SADMOF_SOURCE}/pyproject.toml" ]]; then
    echo "error: SADMOF source not found at ${SADMOF_SOURCE}" >&2
    echo "set SADMOF_SOURCE=/path/to/sadmof-work and run this script again" >&2
    exit 1
fi

if [[ -e "$SADMOF_DIR" ]]; then
    echo "Removing existing SADMOF checkout: $SADMOF_DIR"
    rm -rf -- "$SADMOF_DIR"
fi

mkdir -p "${PROJECT_DIR}/external"
git clone --no-recurse-submodules "$SADMOF_SOURCE" "$SADMOF_DIR"
mkdir -p "$DEPENDENCY_DIR"
git clone https://github.com/adrhill/asdex.git "$DEPENDENCY_DIR/asdex"
git clone https://github.com/lab-cosmo/pet-jax.git "$DEPENDENCY_DIR/pet-jax"
git clone https://github.com/sirmarcel/marathon.git "$DEPENDENCY_DIR/marathon"
git -C "$DEPENDENCY_DIR/asdex" checkout 3209417
git -C "$DEPENDENCY_DIR/pet-jax" checkout ed0add4
git -C "$DEPENDENCY_DIR/marathon" checkout c316527

python -m pip install --upgrade pip
python -m pip install flax e3nn-jax jaxtyping scipy numba
python -m pip install -e "$DEPENDENCY_DIR/asdex"
python -m pip install -e "$DEPENDENCY_DIR/marathon"
python -m pip install -e "$DEPENDENCY_DIR/pet-jax"
python -m pip install --no-deps -e "$SADMOF_DIR"
# Keep the CUDA plugin ABI matched with the JAX version pinned in
# environment.yml. Installing flax above can otherwise select an incompatible
# plugin release without changing jaxlib.
python -m pip install --force-reinstall --no-deps \
    "jax==0.11.0" "jaxlib==0.11.0" \
    "jax-cuda12-plugin==0.11.0" "jax-cuda12-pjrt==0.11.0"

python -c 'from petjax.select import _selection; import asdex, sadmof; print("Verified SADMOF imports:", sadmof.__file__)'
echo "SADMOF and its pinned public dependencies are installed."
