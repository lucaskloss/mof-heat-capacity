# MOF-5 heat-capacity workflow

This directory contains the runnable workflow for MOF-5 molecular dynamics and
harmonic heat-capacity analysis. Scientific background, equations, assumptions,
and convergence requirements are in [report.tex](report.tex).

## 1. Create the environment

From this directory:

```bash
conda env create --file environment.yml
conda activate mof-heat-capacity
```

Install SADMOF and the compatible dependencies used by `heat_capacity.py`:

```bash
./install_sadmof.sh
```

By default, the installer clones the SADMOF source at `../repos/sadmof-work`
into `external/sadmof`, replaces any previous checkout there, and installs its
pinned public dependencies into the active Conda environment. To use SADMOF
from a different location:

```bash
SADMOF_SOURCE=/path/to/sadmof-work ./install_sadmof.sh
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
python insert_methane.py --nmol 1 --seed 2025
```

The default outputs are `output/mof5-pet-mad/mof5-md.pdb` and
`output/mof5-pet-mad/mof5-md.data`, matching the bundled MD output directory
and prefix. Useful options are:

```bash
python insert_methane.py --nmol 4 --try 5000 --min-distance 1.5
python insert_methane.py --nmol 1 --dry-run
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

## 4. Calculate harmonic heat capacity

After generating a trajectory, run the analysis using the matching TOML file:

```bash
python heat_capacity.py --config configs/mof5_pet_mad.toml
```

For a small diagnostic run:

```bash
python heat_capacity.py \
  --config configs/mof5_pet_mad.toml \
  --frames 1 \
  --hops 3
```

The analysis output is written to the configured output directory. Use
`--help` on either script to see all available overrides.

## 5. Inspect results

Open the trajectory notebook from this directory:

```bash
jupyter lab
```

Then open `visualize.ipynb`. Inspect the structure, forces, temperature,
energy, and trajectory stability before interpreting heat-capacity results.

Use a separate output directory and prefix for each structure, model, and
loading so that trajectories cannot be confused with one another.
