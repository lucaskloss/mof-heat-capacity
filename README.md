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
simulation/          molecular-dynamics implementation namespace
properties/          property-calculation implementation namespace
configs/             versioned templates plus ignored generated run TOMLs
input/               source MOF-5 and methane structures
scripts/             Bash entry points and workflow guides, grouped by role
docs/                scientific and Izar documentation
models/              local MLIP artifacts (ignored)
external/            local SADMOF checkout (ignored)
output/              all generated structures, trajectories, logs, and results
```

## Environment

```bash
conda env create --file environment.yml
conda activate mof-heat-capacity
./scripts/setup/install_sadmof.sh
```

The environment is pinned for Izar's V100 GPUs. Keep the Python PyTorch,
LAMMPS/libtorch, and JAX CUDA versions aligned with `environment.yml`.

## Workflows

Run user commands from the repository root. The MD command prepares its own
structures and configurations, then submits its automated preflight,
calibration, and production stages. Property commands consume completed MD
outputs in order: trajectory analysis, Hessians, then hybrid assembly.

The exact Bash commands and options are documented once in
[scripts/md/README.md](scripts/md/README.md) and
[scripts/properties/README.md](scripts/properties/README.md). Results are
written below `output/`; do not mix independent campaigns in one output tree.
