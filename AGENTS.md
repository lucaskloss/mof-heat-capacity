# MOF-5 heat-capacity project

This repository runs molecular dynamics for periodic MOF-5 with a PET-MAD
machine-learned interatomic potential and computes harmonic heat capacities
from PET-JAX/SADMOF Hessians. The bundled configuration is a short integration
smoke test, not a converged production calculation.

## Repository layout

- `mof_heat_capacity/` is the importable implementation package. Its
  `simulation/`, `analysis/`, `structures/`, and `protocols/` subpackages
  separate the major workflow responsibilities; `config.py`, `io.py`, and
  `models.py` contain shared infrastructure.
- `run.py` is the backward-compatible ASE/metatomic molecular-dynamics entry
  point.
- `configs/` holds reusable TOML run specifications. Paths in a TOML file are
  resolved relative to that file.
- `input/` holds source structures and legacy LAMMPS/i-PI input files.
- `scripts/install_sadmof.sh` installs the SADMOF/PET-JAX stack; it expects the
  sibling checkout at `../repos/sadmof-work` unless `SADMOF_SOURCE` is set.
- `scripts/izar_job.sh` is the Slurm entry point for Izar.
- `scripts/analyze_all_results.py` generates the thermodynamic, statistical,
  structural, density, and heat-capacity convergence reports.
- `scripts/README.md` maps all user-facing commands to their responsibilities.
- `docs/` contains the scientific report and the Izar operations guide.
- `models/`, `output/`, and `external/` are generated, supplied, or local
  dependency directories and are intentionally ignored by Git.

## Local workflow

Run commands from the repository root. Create the Conda environment with
`conda env create --file environment.yml`, activate it, then install the
SADMOF dependencies with `./scripts/install_sadmof.sh` before using harmonic
heat-capacity analysis.

```bash
python run.py --config configs/mof5_pet_mad.toml
python -m mof_heat_capacity.structures.methane --nmol 1 --seed 2025
python -m mof_heat_capacity.analysis.harmonic --config configs/mof5_pet_mad.toml
```

The default configuration uses `input/mof5.cif`, CUDA model evaluation, and a
ten-step NVT trajectory. It writes results under `output/mof5-pet-mad/`. Copy
`configs/metatomic_md_template.toml` for a new material, adsorbate, model, or
output directory; do not overwrite or mix production results.

The active MD path is ASE/metatomic. The i-PI/LAMMPS helper and input files are
legacy utilities; do not make them part of the default workflow without a
separate validation.

## Model and scientific constraints

- Keep the model checkpoint, exported metatomic model, and PET-JAX conversion
  paths consistent when changing models.
- Harmonic analysis currently supports only `ad_backend = "pet-jax"`; it
  computes selected trajectory-frame Hessians serially on one JAX device.
- Treat `hops`, `chunk_size`, precision, frame selection, relaxation, and
  frequency handling as scientific convergence parameters. Do not silently
  change them to address resource limits.
- Inspect trajectories, forces, temperatures, and structural stability before
  interpreting a heat capacity. Use separate prefixes and output directories
  for independent runs.

## Izar / GPU guidance

Read `docs/IZAR.md` before running on EPFL SCITAS Izar. The Slurm template
uses one node, one task, and one GPU because neither entry point is MPI or
multi-GPU. Validate CUDA only in an allocation, never on a login node.

The environment deliberately pins CUDA-12-compatible JAX and
`torch==2.5.1+cu121` for Izar's V100 GPUs and 535-series driver. Do not replace
that Torch pin with the current default PyPI wheel or load an unrelated CUDA
toolkit module. Measure representative jobs before changing CPU, memory, or
wall-time requests. Use separate dependent MD and heat-capacity jobs for
production unless a short end-to-end test specifically needs `MOF_STAGE=all`.

## Change and validation guidance

- Preserve the implementation/entry-point separation above. When moving a
  file, update Python imports, compatibility entry points, TOML paths, shell
  commands, Markdown links, and this guide together.
- Keep generated weights, converted models, trajectories, analysis products,
  logs, Slurm output, and caches out of source control.
- Use `python -m compileall -q run.py mof_heat_capacity scripts` after Python changes and
  `bash -n scripts/izar_job.sh scripts/install_sadmof.sh` after shell changes.
- A lightweight dependency-independent smoke check is
  `python -m mof_heat_capacity.structures.methane --dry-run`. Full MD requires
  the configured
  model files and runtime dependencies; full harmonic analysis also requires
  SADMOF/PET-JAX.
- Keep functions focused, use `Path` for filesystem paths, and preserve clear
  validation and error messages. Use blank lines to separate imports,
  constants, functions, validation, file preparation, and runtime execution.
