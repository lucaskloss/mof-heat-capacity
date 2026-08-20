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
configs/             versioned templates plus ignored generated run TOMLs
input/               source MOF-5 and methane structures
scripts/             preparation, Slurm submission, and setup commands
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

## Workflow

Run every command from the repository root. First preview the loaded campaign;
then generate its structures and TOMLs:

```bash
python scripts/prepare_loaded_campaign.py --model pet-mad --loading 100 \
  --replicas 5 --dry-run
python scripts/prepare_loaded_campaign.py --model pet-mad --loading 100 \
  --replicas 5
```

Submit a short integration check, then the classical NPT jobs:

```bash
./scripts/submit_loaded_md.sh --model pet-mad --loading 100 --debug --dry-run
./scripts/submit_loaded_md.sh --model pet-mad --loading 100 --debug
./scripts/submit_loaded_md.sh --model pet-mad --loading 100 --replicas 5 --dry-run
./scripts/submit_loaded_md.sh --model pet-mad --loading 100 --replicas 5
```

Inspect the completed trajectories before interpreting thermodynamics:

```bash
./scripts/submit_analysis.sh --model pet-mad --loading 100 \
  --replicas 1,2,3,4,5 --dry-run
./scripts/submit_analysis.sh --model pet-mad --loading 100 \
  --replicas 1,2,3,4,5
```

Quench representative loaded final structures, calculate their AD Hessians,
and calculate the empty reference Hessian from `input/mof5.pdb`:

```bash
./scripts/submit_heat_capacity.sh --model pet-mad --loading 100 \
  --source-temperature 300 --replicas 1,2,3 --dry-run
./scripts/submit_heat_capacity.sh --model pet-mad --loading 100 \
  --source-temperature 300 --replicas 1,2,3
```

Finally assemble the hybrid heat capacity from all classical replicas and the
loaded Hessian corrections:

```bash
./scripts/submit_hybrid_analysis.sh --model pet-mad --loading 100 \
  --replicas 1,2,3,4,5 --dry-run
./scripts/submit_hybrid_analysis.sh --model pet-mad --loading 100 \
  --replicas 1,2,3,4,5
```

Results are stored under `output/hybrid/<model>/<loading>ch4/` as NPZ, CSV,
and JSON files. Never mix independent campaigns in one output directory.

## Local checks

```bash
python -m compileall -q run.py mof_heat_capacity scripts
bash -n scripts/*.sh
python -m mof_heat_capacity.structures.methane --dry-run
```
