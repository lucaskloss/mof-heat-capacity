# MOF-5 heat capacity with MLIP Hessians

This repository implements a focused, economical workflow for methane-loaded
MOF-5:

\[
C_P^{\mathrm{approx}}(T)
= \frac{d\langle E+P_{\mathrm{ext}}V\rangle_{\mathrm{cl}}}{dT}
+ C_{\mathrm{qn}}^{\mathrm{har}}(T)-C_{\mathrm{cl}}^{\mathrm{har}}.
\]

Classical NPT molecular dynamics of the **loaded** material captures methane
and host--guest anharmonicity. Automatic-differentiation Hessians of optimized
loaded minima provide the harmonic quantum correction. The equilibrated empty
MOF-5 structure goes directly to fixed-cell relaxation and one reference
Hessian; no empty-MOF MD is required.

See [KAPIL_2019.md](docs/KAPIL_2019.md) for the scientific rationale,
assumptions, and limits, and [IZAR.md](docs/IZAR.md) for the cluster procedure.

## Layout

```text
mof_heat_capacity/   reusable simulation, structure, and analysis code
simulation/          molecular-dynamics Slurm submission command and guide
properties/          trajectory and heat-capacity calculation commands
configs/             versioned templates plus ignored generated run TOMLs
input/               source MOF-5 and methane structures
scripts/             shared environment and Slurm infrastructure
docs/                scientific and Izar documentation
models/              local MLIP artifacts (ignored)
external/            local SADMOF checkout (ignored)
output/              all generated structures, trajectories, logs, and results
```

## Environment

```bash
conda env create --file environment.yml
conda activate mof-heat-capacity
./scripts/install_sadmof.sh
```

The environment is pinned for Izar's V100 GPUs. Keep the Python PyTorch,
LAMMPS/libtorch, and JAX CUDA versions aligned with `environment.yml`.

## Independent workflows

The two stages have separate entry-point folders. `simulation/` produces files
under `output/`; `properties/` only reads completed simulation products and
writes diagnostics or heat-capacity results. Neither folder invokes commands
from the other. Shared package code and TOML configurations remain at the root
to avoid duplicating model and scientific settings.

### 1. Molecular dynamics

Run every command from the repository root. First preview the loaded campaign;
then generate its structures and TOMLs:

```bash
python -m mof_heat_capacity.protocols.loaded --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1 --dry-run
python -m mof_heat_capacity.protocols.loaded --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1
```

Submit a short integration check, then the classical NPT jobs:

```bash
./simulation/submit_loaded_md.sh --model pet-mad --loading 100 --debug --dry-run
./simulation/submit_loaded_md.sh --model pet-mad --loading 100 --debug
./simulation/submit_loaded_md.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1 --dry-run
./simulation/submit_loaded_md.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1
```

### 2. Property calculations

Inspect completed trajectories before interpreting thermodynamics:

```bash
./properties/submit_analysis.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1 --dry-run
./properties/submit_analysis.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1
```

Quench representative loaded final structures, calculate their AD Hessians,
and calculate the empty reference Hessian from `input/mof5.pdb`:

```bash
./properties/submit_heat_capacity.sh --model pet-mad --loading 100 \
  --source-temperature 300 --replicas 1 --dry-run
./properties/submit_heat_capacity.sh --model pet-mad --loading 100 \
  --source-temperature 300 --replicas 1
```

Finally assemble the hybrid heat capacity from all classical replicas and the
loaded Hessian corrections:

```bash
./properties/submit_hybrid_analysis.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1 --dry-run
./properties/submit_hybrid_analysis.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1
```

Results are stored under `output/hybrid/<model>/<loading>ch4/` as NPZ, CSV,
and JSON files. Never mix independent campaigns in one output directory.

## Local checks

```bash
python -m compileall -q mof_heat_capacity
bash -n scripts/*.sh simulation/*.sh properties/*.sh
python -m mof_heat_capacity.structures.methane --dry-run
```
