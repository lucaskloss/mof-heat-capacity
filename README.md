# Loaded-MOF heat capacity with MLIP Hessians

This repository implements a workflow for MOFs loaded with CH₄, CO₂, or H₂O.
The original methane/MOF-5 campaign remains the default:

$$C_P^{\mathrm{approx}}(T) = \frac{d\langle E+P_{\mathrm{ext}}V\rangle_{\mathrm{cl}}}{dT} + C_{\mathrm{qn}}^{\mathrm{har}}(T)-C_{\mathrm{cl}}^{\mathrm{har}}. $$

Classical NPT molecular dynamics of the **loaded** material captures guest
and host--guest anharmonicity. Automatic-differentiation Hessians of optimized
loaded minima provide the harmonic quantum correction. The equilibrated empty
MOF structure goes directly to fixed-cell relaxation and one reference
Hessian; no empty-MOF MD is required.

See [MOF-5.md](docs/MOF-5.md) for the scientific rationale,
assumptions, and limits, [SADMOF.md](docs/SADMOF.md) for the
Hessian and harmonic heat-capacity implementation,
[PET.md](docs/PET.md) for the PET paper summary and model architecture,
[LLPR.md](docs/LLPR.md) for the supplied CEA ensemble checkpoints and their
use, and [IZAR.md](docs/IZAR.md) for the cluster procedure.

## Layout

```text
mof_heat_capacity/   reusable simulation, structure, and analysis code
simulation/          molecular-dynamics implementation namespace
properties/          property-calculation implementation namespace
configs/             versioned templates plus ignored generated run TOMLs
input/               source MOF structures and single-molecule guest templates
scripts/             Bash entry points and workflow guides, grouped by role
docs/                scientific and Izar documentation
models/              local MLIP artifacts (ignored)
external/            local SADMOF checkout (ignored)
output/              all generated structures, trajectories, logs, and results
```

## Environment

```bash
conda env create --file environment.yml
conda activate mof
./scripts/setup/install_sadmof.sh
```

The environment is pinned for Izar's V100 GPUs. Keep the Python PyTorch,
LAMMPS/libtorch, and JAX CUDA versions aligned with `environment.yml`.

## Workflows

Run user commands from the repository root. The MD command prepares its own
structures and configurations, then submits its automated preflight,
calibration, and production stages. Property commands consume completed MD
outputs in order: trajectory analysis, Hessians, then hybrid assembly.

Select the guest species, molecule count, and host from the command line:

```bash
./scripts/md/submit_loaded_md.sh --model pet-mad --mof mof5 --guest co2 --loading 100
./scripts/md/submit_loaded_md.sh --model pet-mad --mof mof5 --guest h2o --loading 50
./scripts/md/submit_loaded_md.sh --model pet-mad --mof uio66 --host input/uio66.cif --guest ch4 --loading 100
```

For a new MOF, supply its periodic structure with `--host`, or place it at
`input/<mof>.pdb`, `.cif`, `.gro`, or `.extxyz`. The MOF label is a lowercase
folder identifier, such as `mof5` or `uio-66`. Guest names are case-insensitive.
The supplied `mgmof74.cif` and `mof303.cif` are selected automatically with
`--mof mgmof74` and `--mof mof303`. CIF-based campaigns generate full-precision
ExtXYZ inputs, plus PDB and LAMMPS data exports, without manual conversion.
Insertion automatically retries a blocked packing by rearranging guests,
keeping the specified molecule count and minimum periodic separation.
`--loading` always counts molecules. Pass the same `--mof`, `--guest`, and
`--loading` to subsequent property commands. See the
[MD guide](scripts/md/README.md) for input overrides and folder layout.

MD and trajectory analysis default to 175–425 K in 25 K steps. The 175/425 K
runs support centered finite differences at 200/400 K; hybrid heat-capacity
outputs default to 200–400 K.

New combinations are grouped by MOF, model, molecule count/species, temperature,
and replica, for example
`output/md/production/mof5/pet-mad-1.5-s-40nn/100co2/300K/rep01/`.
The default MOF-5/CH₄ combination keeps its existing paths so completed
trajectories and restarts remain usable.

The harmonic workflow relaxes only the highest-temperature loaded structure
per replica, computes its central spectrum once, and distributes the 64 LLPR
members across eight GPU jobs before merging. Hybrid assembly reuses this
spectrum at every temperature. Submit one model/loading campaign at a time;
the [property guide](scripts/properties/README.md) includes the fresh
PET-MAD/50 CH₄ command and the merge dependency for final assembly.

The exact Bash commands and options are documented once in
[scripts/md/README.md](scripts/md/README.md) and
[scripts/properties/README.md](scripts/properties/README.md). Results are
written below `output/`; do not mix independent campaigns in one output tree.
