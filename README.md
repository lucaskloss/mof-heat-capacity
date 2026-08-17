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

Run SADMOF only inside a GPU allocation. The submission wrapper defaults to
both MLIPs, pristine and 100-CH4 MOF-5, all five MD temperatures, and replica
1. It selects three physical times—200, 350, and 500 ps—from each trajectory:

```bash
./scripts/submit_heat_capacity.sh --dry-run
./scripts/submit_heat_capacity.sh
```

This is one Slurm array with 20 tasks and at most four running simultaneously.
Each task uses one GPU and processes the three frames from one trajectory
serially. With a measured cost near ten minutes per frame, the default
45-minute limit includes compilation and shutdown margin. The requested times
are past the 100 ps equilibration interval and separated by 150 ps, much longer
than the measured autocorrelation times. Three frames are the minimum needed
for a frame-to-frame standard deviation and standard error.

Before the full array, run the isolated one-frame debug test on the larger
100-CH4 system at 275 ps:

```bash
./scripts/submit_heat_capacity.sh --debug --dry-run
./scripts/submit_heat_capacity.sh --debug
```

The debug output is isolated at
`output/debug/heat-capacity-pet-mad-100ch4-300K-275ps.npz`, so it is not
collected with the three production frames. To select a subset, use for example:

```bash
./scripts/submit_heat_capacity.sh --model pet-sol --loading 100 \
  --md-temperatures 300 --frame-times-ps 200,350,500 --dry-run
```

Physical times are mapped through each LAMMPS thermo log, so the duplicated
100 ps restart boundary cannot shift frame selection. Output names record the
times, for example `heat-capacity-times-200ps-350ps-500ps.npz`, inside each
run directory. Duplicate selectors and existing outputs are rejected before
submission; pass `--overwrite` only when replacement is intentional.

After every array task succeeds, rerun `submit_analysis.sh` for each loading.
It collects the NPZ archives, reports the value at the matching MD temperature,
and retains the complete 100--500 K harmonic curves. These are harmonic
constant-volume results from individual thermally displaced structures. For
methane-loaded MOF-5 they are diagnostics, not a reproduction of the paper's
anharmonic, quantum-mechanical constant-pressure heat capacity.

## 5. Inspect results

On Izar, submit trajectory analysis as a CPU-based Slurm job that allocates
the one GPU required by the `normal` QOS. Preview the validated selection and
scheduler command first, then submit it:

```bash
./scripts/submit_analysis.sh --dry-run
./scripts/submit_analysis.sh
```

The defaults select the pristine MOF-5 classical replica-01 trajectories from
both MLIPs at all five temperatures, discard the first 100 ps, request four
CPUs for 1 hour 15 minutes, allocate the one GPU required by Izar's `normal` QOS, and
write scheduler output below `output/slurm/`. The analysis remains CPU-based.
Select both potentials and multiple methane loadings explicitly with:

```bash
./scripts/submit_analysis.sh --model both --loading 0,100 --time 01:30:00 \
  --analysis-dir output/analysis-0ch4-100ch4 --dry-run
./scripts/submit_analysis.sh --model both --loading 0,100 --time 01:30:00 \
  --analysis-dir output/analysis-0ch4-100ch4
```

Use `--model pet-mad` or `--model pet-sol` for one potential, and combine
`--loading`, `--temperatures`, and `--replicas` comma-separated lists as
needed. Every requested combination must have a completed trajectory and
thermodynamic log or submission stops before `sbatch`.
To run the same analysis outside Slurm, use:

```bash
python -m mof_heat_capacity.analysis.results \
  --runs 'mof5-0ch4-paper-*-classical-*-rep01' \
  --discard-ps 100
```

To analyze only one run, or to choose an explicit equilibration interval:

```bash
./scripts/submit_analysis.sh \
  --runs 'mof5-0ch4-paper-pet-mad-1.5-s-40nn-classical-300K-rep01' \
  --discard-ps 250 --analysis-dir output/analysis-cutoff250 --dry-run
```

Results are written under `output/analysis/`. Each run gets framewise
thermodynamic and structural CSV files, autocorrelation/RDF/MSD and harmonic
heat-capacity NPZ files, a JSON convergence summary, and diagnostic plots.
For LAMMPS runs, global thermodynamic quantities are read from the matching
`.lammps.log` and aligned with positions, velocities, forces, and cells from
the dump trajectory. Resumed thermo blocks are retained in order, while a
repeated restart-boundary step/frame is removed before calculating statistics.
The top-level `runs.csv`, `temperature_sweep.png`, and
`paper_requirements.json` compare runs and identify calculations that the
current data cannot support. The cluster wrapper discards the first 100 ps by
default; choose this cutoff from observed stationarity for production
calculations. Use `--runs` for advanced custom globs, and `--no-plots` for
machine-only output.

Density convergence is included in the same command. Inspect
`timeseries.csv`, `summary.json`, `timeseries.png`, and `cell.png` in each
run's analysis directory for density, volume, cell lengths/angles/tilts,
pressure components, running means, drift, autocorrelation-aware uncertainty,
and fixed-volume detection. Flexible-cell NPT results include equilibrium
density and cell-response diagnostics when the selected model records validated
stress support; fixed-cell NVT density is constant by construction.

The generated PNG files provide the standard visual diagnostics without a
notebook. The CSV, JSON, and NPZ products can also be loaded directly by
plotting or manuscript scripts. Inspect the structure, forces, temperature,
energy, autocorrelation times, effective sample counts, methane distributions,
and trajectory stability before interpreting heat-capacity results.

Use a separate output directory and prefix for each structure, model, and
loading so that trajectories cannot be confused with one another.
