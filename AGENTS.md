# MOF-5 heat-capacity project

This repository computes an approximate heat capacity for methane-loaded
periodic MOF-5. Classical NPT captures loaded-system anharmonicity, while
PET-JAX/SADMOF Hessians provide the harmonic quantum correction.

## Repository layout

- `mof_heat_capacity/` is the importable implementation package. Its
  `simulation/`, `analysis/`, `structures/`, and `protocols/` subpackages
  separate the major workflow responsibilities; `config.py`, `io.py`, and
  `models.py` contain shared infrastructure.
- `simulation/` contains the MD implementation namespace. Campaign preparation
  and single-run execution use the package modules
  `mof_heat_capacity.protocols.loaded` and `mof_heat_capacity.simulation.md`
  directly; do not add forwarding Python scripts for them.
- `properties/` contains the property-calculation implementation namespace; it
  consumes completed outputs without starting MD.
- `configs/` holds reusable TOML run specifications. Paths in a TOML file are
  resolved relative to that file.
- `input/` holds source MOF-5 and methane structures.
- `scripts/setup/install_sadmof.sh` installs the SADMOF/PET-JAX stack; it expects the
  sibling checkout at `../repos/sadmof-work` unless `SADMOF_SOURCE` is set.
- `scripts/slurm/izar_gpu_runtime.sh` is the shared GPU runtime for MD, relaxation, and
  Hessian stages on Izar.
- `scripts/properties/submit_analysis.sh` validates, submits, and executes CPU-based
  trajectory analysis on Izar. Its Slurm worker mode is internal to the same
  file. The normal QOS requires allocating one GPU even though this analysis
  does not use it.
- `scripts/md/submit_loaded_md.sh` invokes
  `mof_heat_capacity.protocols.loaded` to generate and submit loaded
  classical-NPT campaigns.
- `scripts/properties/submit_heat_capacity.sh` quenches representative loaded structures
  and computes loaded and empty-reference Hessians.
- `scripts/properties/submit_hybrid_analysis.sh` submits and assembles the final hybrid
  curve without a separate worker script.
- `scripts/` contains every Bash entry point, grouped into `setup/`, `slurm/`,
  `md/`, and `properties/`; each workflow also has its own README.
- `docs/` contains the scientific rationale, source paper, and Izar guide.
- `models/`, `output/`, and `external/` are generated, supplied, or local
  dependency directories and are intentionally ignored by Git.

## Local workflow

Run commands from the repository root. Create the Conda environment with
`conda env create --file environment.yml`, activate it, then install the
SADMOF dependencies with `./scripts/setup/install_sadmof.sh` before using harmonic
heat-capacity analysis.

```bash
./scripts/md/submit_loaded_md.sh --model pet-mad --loading 100 --dry-run
./scripts/properties/submit_heat_capacity.sh --model pet-mad --loading 100 --dry-run
./scripts/properties/submit_hybrid_analysis.sh --model pet-mad --loading 100 --dry-run
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
- Use `python -m compileall -q mof_heat_capacity` after Python changes and
  `find scripts -name '*.sh' -exec bash -n {} +`
  after shell changes.
- A lightweight dependency-independent smoke check is
  `python -m mof_heat_capacity.structures.methane --dry-run`. Full MD requires
  the configured
  model files and runtime dependencies; full harmonic analysis also requires
  SADMOF/PET-JAX.
- Keep functions focused, use `Path` for filesystem paths, and preserve clear
  validation and error messages. Use blank lines to separate imports,
  constants, functions, validation, file preparation, and runtime execution.
- Use `$$...$$` for equations in Markdown instead of `\[...\]` to avoid conflicts with the Markdown parser.
  Put each complete display equation, including both `$$` delimiters, on one source line: GitHub does
  not recognize the newline-spanning form used here and can parse a continuation beginning with `+`
  or `-` as a list item. Also use `$...$` for inline expressions, especially when writing units like
  `$J g^{-1} K^{-1}$`.
