# MOF-5 heat-capacity workflow

This directory contains the runnable workflow for MOF-5 molecular dynamics and
harmonic heat-capacity analysis. Scientific background, equations, assumptions,
and convergence requirements are in [report.tex](docs/report.tex). The
simulation-focused summary of Kapil et al. (2019), including the published
parameters, current-project compatibility, and an Izar execution path, is in
[KAPIL_2019.md](docs/KAPIL_2019.md).

For CUDA jobs submitted with Slurm on EPFL's SCITAS Izar cluster, follow
[IZAR.md](docs/IZAR.md). The environment pins both JAX and PyTorch to CUDA 12 builds
that can target Izar's V100 GPUs; allowing pip to select the current default
PyTorch wheel installs CUDA 13 and fails on Izar's 535-series driver. A reusable
submission template is provided in [izar_job.sh](scripts/izar_job.sh).

## 1. Create the environment

From this directory:

```bash
conda env create --file environment.yml
conda activate mof-heat-capacity
```

Install SADMOF and the compatible dependencies used by `utils/heat_capacity.py`:

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
python utils/insert_methane.py --nmol 1 --seed 2025
```

The default outputs are `output/mof5-pet-mad/mof5-md.pdb` and
`output/mof5-pet-mad/mof5-md.data`, matching the bundled MD output directory
and prefix. Useful options are:

```bash
python utils/insert_methane.py --nmol 4 --try 5000 --min-distance 1.5
python utils/insert_methane.py --nmol 1 --dry-run
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

For the short MOF-5 + 100 CH4 Izar validation sweep at 100, 200, 300, 400,
and 500 K, prepare the shared loaded structure and five 2 ps configurations,
preview the Slurm commands, and submit them with:

```bash
./scripts/prepare_mof5_100ch4_tests.sh
./scripts/submit_mof5_100ch4_tests.sh --dry-run
./scripts/submit_mof5_100ch4_tests.sh
```

The submission script defaults to the `gpu` partition, `normal` QOS, one GPU,
four CPUs, and two hours per job. Use `--help` to override those values after
checking Izar's live QOS and partition configuration.

To repeat the same five MD simulations with the
`pet_sol-s-best_nostress.ckpt` potential, reuse the prepared methane-loaded
structure and run:

```bash
./scripts/submit_mof5_100ch4_pet_sol_tests.sh --dry-run
./scripts/submit_mof5_100ch4_pet_sol_tests.sh
```

These runs use PET-SOL-specific configuration names, trajectory prefixes, and
`output/mof5-100ch4-<temperature>K-pet-sol-s-best-nostress-test/`
directories. The portable `.pt` model is exported automatically from the
checkpoint on its first run. For harmonic analysis, the matching PET-JAX model
is likewise converted automatically from the same checkpoint on first use.

## 4. Calculate harmonic heat capacity

After generating a trajectory, run the analysis using the matching TOML file:

```bash
python utils/heat_capacity.py --config configs/mof5_pet_mad.toml
```

For a small diagnostic run:

```bash
python utils/heat_capacity.py \
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
`heat-capacity-frame-4000.npz` below its MD output directory. After all jobs
finish, print the heat capacity from each MD frame at its matching temperature:

```bash
python scripts/summarize_mof5_100ch4_heat_capacity.py
```

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

Use `--analysis-temperature 300` to compare all five local Hessians at the
same analysis temperature. These are harmonic constant-volume results from
individual thermally displaced structures. For methane-loaded MOF-5 they are
diagnostics, not a reproduction of the paper's anharmonic, quantum-mechanical
constant-pressure heat capacity.

## 5. Inspect results

Open the trajectory notebook from this directory:

```bash
jupyter lab
```

Then open `utils/visualize.ipynb`. Inspect the structure, forces, temperature,
energy, and trajectory stability before interpreting heat-capacity results.

Use a separate output directory and prefix for each structure, model, and
loading so that trajectories cannot be confused with one another.
