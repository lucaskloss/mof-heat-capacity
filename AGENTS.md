# MOF-5 heat-capacity project

This repository computes an approximate heat capacity for methane-loaded
periodic MOF-5. Classical NPT captures loaded-system anharmonicity, while
PET-JAX/SADMOF Hessians provide the harmonic quantum correction.

## Repository layout

- `mof_heat_capacity/` is the importable implementation package. Its
  `simulation/`, `analysis/`, `structures/`, and `protocols/` subpackages
  separate the major workflow responsibilities; `config.py`, `io.py`, and
  `models.py` contain shared infrastructure.
- `run.py` is the single-run ASE/metatomic molecular-dynamics entry point.
- `configs/` holds reusable TOML run specifications. Paths in a TOML file are
  resolved relative to that file.
- `input/` holds source MOF-5 and methane structures.
- `scripts/install_sadmof.sh` installs the SADMOF/PET-JAX stack; it expects the
  sibling checkout at `../repos/sadmof-work` unless `SADMOF_SOURCE` is set.
- `scripts/izar_job.sh` is the Slurm entry point for Izar.
- `scripts/submit_analysis.sh` and `scripts/izar_analysis_job.sh` validate,
  submit, and execute CPU-based trajectory analysis on Izar. The normal QOS
  requires allocating one GPU even though this analysis does not use it.
- `scripts/prepare_loaded_campaign.py` and `scripts/submit_loaded_md.sh`
  generate and submit loaded classical-NPT campaigns.
- `scripts/submit_heat_capacity.sh` quenches representative loaded structures
  and computes loaded and empty-reference Hessians.
- `scripts/submit_hybrid_analysis.sh` assembles the final hybrid curve.
- `scripts/README.md` maps all user-facing commands to their responsibilities.
- `docs/` contains the scientific rationale, source paper, and Izar guide.
- `models/`, `output/`, and `external/` are generated, supplied, or local
  dependency directories and are intentionally ignored by Git.

## Local workflow

Run commands from the repository root. Create the Conda environment with
`conda env create --file environment.yml`, activate it, then install the
SADMOF dependencies with `./scripts/install_sadmof.sh` before using harmonic
heat-capacity analysis.

```bash
python scripts/prepare_loaded_campaign.py --model pet-mad --loading 100 --dry-run
./scripts/submit_loaded_md.sh --model pet-mad --loading 100 --debug --dry-run
./scripts/submit_heat_capacity.sh --model pet-mad --loading 100 --dry-run
./scripts/submit_hybrid_analysis.sh --model pet-mad --loading 100 --dry-run
```

The active production MD path is LAMMPS/metatomic classical NPT. Generated
campaign TOMLs are ignored; the versioned hybrid template is their source.

## Model and scientific constraints

- Keep the model checkpoint, exported metatomic model, and PET-JAX conversion
  paths consistent when changing models.
- Harmonic analysis supports only `ad_backend = "pet-jax"`; production
  Hessians are computed on fixed-cell optimized minima, not raw thermal frames.
- Do not run an empty-MOF MD campaign. Relax the equilibrated empty structure
  directly and use its Hessian as a separate reference.
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
wall-time requests. Keep MD, Hessian, and hybrid assembly as separate jobs.

## Change and validation guidance

- Preserve the implementation/entry-point separation above. When moving a
  file, update Python imports, compatibility entry points, TOML paths, shell
  commands, Markdown links, and this guide together.
- Keep generated weights, converted models, trajectories, analysis products,
  logs, Slurm output, and caches out of source control.
- Use `python -m compileall -q run.py mof_heat_capacity scripts` after Python
  changes and `bash -n scripts/*.sh` after shell changes.
- A lightweight dependency-independent smoke check is
  `python -m mof_heat_capacity.structures.methane --dry-run`. Full MD requires
  the configured
  model files and runtime dependencies; full harmonic analysis also requires
  SADMOF/PET-JAX.
- Keep functions focused, use `Path` for filesystem paths, and preserve clear
  validation and error messages. Use blank lines to separate imports,
  constants, functions, validation, file preparation, and runtime execution.
