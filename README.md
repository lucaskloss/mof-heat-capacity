# MOF-5 heat-capacity workflow

This directory contains the runnable workflow for MOF-5 molecular dynamics and
harmonic heat-capacity analysis. Scientific background, equations, assumptions,
and convergence requirements are in [report.tex](docs/report.tex). The
simulation-focused summary of Kapil et al. (2019), including the published
parameters, current-project compatibility, and an Izar execution path, is in
[KAPIL_2019.md](docs/KAPIL_2019.md).

## Repository organization

The code is divided by responsibility:

```text
mof_heat_capacity/
├── simulation/   MD drivers and LAMMPS/i-PI execution backends
├── analysis/     harmonic, statistical, and trajectory analysis
├── structures/   structure-building operations
├── protocols/    high-level paper-protocol preparation
├── config.py     TOML loading and validation
├── io.py         structure and engine-input writers
└── models.py     MLIP export helpers
scripts/          user-facing setup, preparation, Slurm, and analysis commands
configs/          reusable TOML run specifications
input/            source structures and legacy engine inputs
docs/             scientific documentation and cluster runbooks
```

The root `run.py` and `scripts/*.py` commands are small user-facing entry
points. All reusable Python code is imported from `mof_heat_capacity`.

For CUDA jobs submitted with Slurm on EPFL's SCITAS Izar cluster, follow
[IZAR.md](docs/IZAR.md). The environment pins JAX, Python PyTorch, and LAMMPS's
separate C++ libtorch runtime to CUDA 12 builds that can run with Izar's V100
GPUs and 535-series driver. A reusable submission template is provided in
[izar_job.sh](scripts/izar_job.sh).

## 1. Create the environment

From this directory:

```bash
conda env create --file environment.yml
conda activate mof-heat-capacity
```

Install SADMOF and the compatible dependencies used by the harmonic-analysis
command:

```bash
./scripts/install_sadmof.sh
```

By default, the installer clones the SADMOF source at `../repos/sadmof-work`
into `external/sadmof`, replaces any previous checkout there, and installs its
pinned public dependencies into the active Conda environment. To use SADMOF
from a different location:

```bash
SADMOF_SOURCE=/path/to/sadmof-work ./scripts/install_sadmof.sh
```

## 2. Run the bundled MOF-5 example

Run the short MD smoke test:

```bash
python run.py --config configs/mof5_pet_mad.toml
```

Use a CPU or longer test explicitly when needed:

```bash
python run.py --config configs/mof5_pet_mad.toml --device cpu --steps 100 --rerun
```

The trajectory and log are written below the configured output directory.

## 3. Insert methane with ASE

The insertion script reads the periodic MOF and methane template with ASE,
places randomly oriented methane molecules, checks minimum atom distances, and
writes a combined structure.

```bash
python -m mof_heat_capacity.structures.methane --nmol 1 --seed 2025
```

The default outputs are `output/mof5-pet-mad/mof5-md.pdb` and
`output/mof5-pet-mad/mof5-md.data`, matching the bundled MD output directory
and prefix. Useful options are:

```bash
python -m mof_heat_capacity.structures.methane \
  --nmol 4 --try 5000 --min-distance 1.5
python -m mof_heat_capacity.structures.methane --nmol 1 --dry-run
```

Use the generated structure in a copied TOML configuration:

```bash
cp configs/metatomic_md_template.toml configs/mof5-ch4.toml
```

Set the copied configuration’s structure path to:

```toml
[structure]
path = "../output/mof5-pet-mad/mof5-md.pdb"
```

Set model paths, output directory, temperature, timestep, and trajectory prefix
in the same file before running `run.py`.

The paper-protocol workflow requires a stress/virial-validated MLIP because it
runs constant-pressure dynamics. The two updated PET checkpoints advertise an
explicit stress output and are recorded as stress-validated in generated paper
configurations. Their filenames retain the historical `nostress` suffix, which
is not used as a capability test. Prepare an independent structure and
configuration for each temperature, replica, and MLIP. For the first classical
300 K debug tests, activate the environment described below and run:

```bash
./scripts/prepare_paper_protocol.py --model pet-mad \
  --method classical --temperatures 300 --replicas 1 --configs-only
./scripts/prepare_paper_protocol.py --model pet-sol \
  --method classical --temperatures 300 --replicas 1 --configs-only

./scripts/submit_paper_protocol.sh --model pet-mad --debug --dry-run
./scripts/submit_paper_protocol.sh --model pet-sol --debug --dry-run
./scripts/submit_paper_protocol.sh --model pet-mad --debug
./scripts/submit_paper_protocol.sh --model pet-sol --debug
```

`--debug` fixes the request at the `debug` QOS, 30 minutes, four CPUs, one GPU,
and ten MD steps. Unless explicitly overridden, it selects only the 300 K,
replica-01 configuration and writes under `output/debug/`, separate from the
production output directory. The dry run validates the generated structure,
model paths, NPT stress-validation flag, and TOML schema before printing the
exact `sbatch` command. Confirm the live Izar partition/QOS configuration before
submitting. Use `--help` for loading, temperature, replica, and resource
overrides.

After debug succeeds, use `--calibration` for timing. It defaults to 1,000
steps and writes under `output/calibration/`, so a shortened trajectory cannot
be reused accidentally as production output.

The complete Izar runbook—including environment checks, preparation options,
debug and calibration commands, staged 100 ps equilibration, restart-to-500 ps
production, replica expansion, measured resource estimates, storage sizing,
monitoring, and completion checks—is in
[Run the MOF-5 paper-protocol campaign](docs/IZAR.md#run-the-mof-5-paper-protocol-campaign).
It also documents the pristine-MOF workflow: pass `--loading 0` to both the
preparation and submission scripts to run without methane while keeping those
outputs isolated from loaded-system results.
Paper-protocol Slurm stdout/stderr is stored under `output/slurm/` by default;
use `--slurm-output-dir PATH` to place it on scratch or project storage.

## 4. Calculate harmonic heat capacity

After generating a trajectory, run the analysis using the matching TOML file:

```bash
python -m mof_heat_capacity.analysis.harmonic \
  --config configs/mof5_pet_mad.toml
```

For a small diagnostic run:

```bash
python -m mof_heat_capacity.analysis.harmonic \
  --config configs/mof5_pet_mad.toml \
  --frame-indices 0 \
  --hops 3
```

`--frames 1` also selects one evenly spaced frame, while `--frame-indices 10`
selects trajectory frame 10 exactly. The analysis output is written to the
configured output directory. Use `--help` on either script to see all available
overrides.

For the completed MOF-5 + 100 CH4 validation trajectories, first preview and
submit the 300 K final-frame Hessian as a resource-sizing test:

```bash
./scripts/submit_mof5_100ch4_heat_capacity.sh --temperatures 300 --dry-run
./scripts/submit_mof5_100ch4_heat_capacity.sh --temperatures 300
```

If that succeeds and the Slurm accounting data supports the chosen resources,
submit the remaining independent jobs (the existing 300 K result is protected
from accidental replacement):

```bash
./scripts/submit_mof5_100ch4_heat_capacity.sh --temperatures 100,200,400,500
```

Each job analyzes explicit trajectory frame 4000 and writes a unique
`heat-capacity-frame-4000.npz` below its MD output directory. The complete
analysis command in the next section collects these files, reports each value
at its matching MD temperature, and preserves the full temperature curves.

For the PET-SOL trajectories, use the separate submission script so the
PET-SOL configs, trajectories, and output directories are selected:

```bash
./scripts/submit_mof5_100ch4_pet_sol_heat_capacity.sh --temperatures 300 --dry-run
./scripts/submit_mof5_100ch4_pet_sol_heat_capacity.sh --temperatures 300
./scripts/submit_mof5_100ch4_pet_sol_heat_capacity.sh --temperatures 100,200,400,500
```

Each PET-SOL result is written as
`heat-capacity-pet-sol-s-best-nostress-frame-4000.npz` inside its matching
PET-SOL output directory, so neither its directory nor filename overlaps the
PET-MAD results.

The generated per-run `harmonic_heat_capacity.npz` files retain the complete
100--500 K curves for comparisons at a common analysis temperature. These are
harmonic constant-volume results from individual thermally displaced
structures. For methane-loaded MOF-5 they are diagnostics, not a reproduction
of the paper's anharmonic, quantum-mechanical constant-pressure heat capacity.

## 5. Inspect results

Run the complete analysis for every completed trajectory described by a TOML
file:

```bash
python scripts/analyze_all_results.py
```

To analyze only one run, or to choose an explicit equilibration interval:

```bash
python scripts/analyze_all_results.py \
  --runs 'mof5-100ch4-300K-test' \
  --discard-ps 1.0
```

Results are written under `output/analysis/`. Each run gets framewise
thermodynamic and structural CSV files, autocorrelation/RDF/MSD and harmonic
heat-capacity NPZ files, a JSON convergence summary, and diagnostic plots.
The top-level `runs.csv`, `temperature_sweep.png`, and
`paper_requirements.json` compare runs and identify calculations that the
current data cannot support. By default, the first half of each trajectory is
treated as equilibration; choose this cutoff from observed stationarity for
production calculations. Use `--runs 'mof5-100ch4-300K-*'` or another glob to
limit the analysis, and `--no-plots` for machine-only output.

Density convergence is included in the same command. Inspect
`timeseries.csv`, `summary.json`, and `timeseries.png` in each run's analysis
directory for density, volume, running means, drift, autocorrelation-aware
uncertainty, and fixed-volume detection. The current ASE runs are fixed-cell
NVT calculations, so their density is constant by construction; meaningful
density convergence requires variable-cell NPT trajectories with a
stress-trained and validated potential.

The generated PNG files provide the standard visual diagnostics without a
notebook. The CSV, JSON, and NPZ products can also be loaded directly by
plotting or manuscript scripts. Inspect the structure, forces, temperature,
energy, autocorrelation times, effective sample counts, methane distributions,
and trajectory stability before interpreting heat-capacity results.

Use a separate output directory and prefix for each structure, model, and
loading so that trajectories cannot be confused with one another.
