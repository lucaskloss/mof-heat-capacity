# Molecular-dynamics simulations

This folder is the user-facing Bash entry point and guide for the loaded-MOF
classical-NPT campaign. By default it writes structures, trajectories, restarts,
and logs under `/work/cosmo/dealmeid/mof-heat-capacity/output`. It does not run
property analysis.

Run commands from the repository root:

```bash
./scripts/md/submit_loaded_md.sh --model both --loading 50 --replicas 1
```

## Select the MOF and guest

`--mof` selects the host label (default `mof5`), `--guest` selects `ch4`,
`co2`, or `h2o` (default `ch4`), and `--loading N` sets a positive molecule
count. Each invocation prepares one host/species/count campaign. To run several
species or counts, invoke it separately for each selection.

```bash
./scripts/md/submit_loaded_md.sh --model pet-mad --mof mof5 --guest co2 --loading 100
./scripts/md/submit_loaded_md.sh --model pet-sol --mof mof5 --guest h2o --loading 50
# Supply a periodic structure for a new host:
./scripts/md/submit_loaded_md.sh --model both --mof uio66 \
  --host input/uio66.cif --guest co2 --loading 100
```

MOF labels must start with a lowercase letter and contain only lowercase
letters, digits, underscores, and hyphens. Without `--host`, the command looks
for `input/<mof>.pdb`, `.cif`, `.gro`, then `.extxyz`, in that order. The
repository currently includes `mof5.pdb`, `mgmof74.cif`, and `mof303.cif`.
Select these with `--mof mof5`, `--mof mgmof74`, or `--mof mof303`; no
`--host` or manual conversion is needed. Other hosts must be supplied by the user. The host
must have a nonzero cell and be periodic in all three directions.

For the supplied CIF hosts, using PET-MAD and 50 guest molecules:

```bash
./scripts/md/submit_loaded_md.sh --model pet-mad --mof mgmof74 --guest ch4 --loading 50
./scripts/md/submit_loaded_md.sh --model pet-mad --mof mgmof74 --guest co2 --loading 50
./scripts/md/submit_loaded_md.sh --model pet-mad --mof mgmof74 --guest h2o --loading 50
./scripts/md/submit_loaded_md.sh --model pet-mad --mof mof303 --guest ch4 --loading 50
./scripts/md/submit_loaded_md.sh --model pet-mad --mof mof303 --guest co2 --loading 50
./scripts/md/submit_loaded_md.sh --model pet-mad --mof mof303 --guest h2o --loading 50
```

Preparation reads the CIF, expands its symmetry, inserts the selected number
of molecules, and automatically writes `structure.extxyz`, `structure.pdb`,
and `structure.data` in the campaign's structures directory. MD configurations
use the full-precision ExtXYZ structure for CIF inputs, preserving cell vectors
and atom coordinates through the conversion to LAMMPS. The PDB is an additional
export with the rounding inherent to that format. MOF-5 continues to use its
existing PDB input and generated PDB path. Element/type mappings include Mg
for `mgmof74`, and Al and N for `mof303`.

Mg-MOF-74 runs first relax the entire loaded structure at fixed cell with
LAMMPS FIRE and the selected MD model. This applies to CH4, CO2, and H2O in
fresh calibration and production runs; restart continuations skip relaxation.
Generated Mg-MOF-74 configurations record these adjustable settings:

```toml
[initial_relaxation]
enabled = true
force_tolerance_eV_A = 0.05
max_iterations = 2000
max_evaluations = 20000
max_displacement_A = 0.05
```

The tolerance bounds the largest absolute force component. Relaxation stops
the job before NPT if that tolerance is not reached. Its log, relaxed LAMMPS
data, and final coordinates/forces are saved alongside MD outputs as
`<prefix>.lammps.relaxation.log`, `<prefix>.lammps.relaxed.data`, and
`<prefix>.lammps.relaxed.lammpstrj`. The MD clock and timestep are reset after
relaxation, retaining the 500 ps simulation and 100 ps equilibration window.
The calibration wall-time measurement includes relaxation, making its
production-time estimate conservative. Other MOFs retain their previous
initialization unless relaxation is explicitly enabled in their TOML.

The default guest templates are `input/ch4.gro`, `input/co2.gro`, and
`input/h2o.gro`. Use `--molecule PATH` to substitute your own single-molecule
geometry; its elemental composition must match the chosen species. The
`--methane` spelling remains a compatibility alias. These are insertion
geometries for the existing topology-free MLIP workflow; no classical water or
CO₂ force field is added. The selected model must support the host/guest
elements. Numerical settings such as timestep and insertion distance retain
their existing defaults and should be checked for each new physical system.

Insertion defaults to `--insertion-method auto`: it first tries sequential
random placement, then rearranges previously placed guests if that packing
gets blocked. Repacking removes and reinserts a random quarter of the guests,
with a complete restart every fourth round, up to `--packing-restarts 20`.
The framework, requested molecule count, rigid molecule geometries, and
1.5 Å minimum periodic separation remain fixed. Distances account for the
triclinic cell and are independently validated with ASE before writing files.
Failure reports how many molecules could be placed and never writes a partial
loaded structure. A failed search does not prove the requested loading cannot
fit; it may require more packing rounds or a different initial arrangement.

Use the same command to retry MOF-303 with 50 methane molecules:

```bash
./scripts/md/submit_loaded_md.sh --model pet-mad --mof mof303 --guest ch4 --loading 50
```

Select `--insertion-method random` for a single sequential placement pass or
`--insertion-method repacking` to explicitly select the rearrangement method.
Both the Python preparation command and the MD submission command accept
`--insertion-method` and `--packing-restarts`; the standalone insertion command
accepts them too. New `inputs.json` files record the requested/resolved method,
attempt budget, and number of repacking rounds. Existing structures are reused
with `auto`, preserving previous metadata. An explicit method conflicting
with an existing structure is rejected; Python `--force` regenerates inputs
only when intentionally requested. Packing success establishes geometric
separation, not equilibration or stable MD with the chosen model.

LAMMPS neighbor-list capacity grows with the atom count to handle short
periodic cells and long model cutoffs. This changes the allowed storage limit;
it does not remove neighbors or alter the model cutoff.

For input and configuration inspection without writing files or submitting
jobs:

```bash
python -m mof_heat_capacity.protocols.loaded --model pet-mad \
  --mof mof5 --guest h2o --loading 50 --dry-run
```

The Bash `--dry-run` prepares missing structures/configurations and prints
submissions, so it requires the configured model and runtime but never calls
`sbatch`. Calibration planners and automatic continuations retain the selected
host, guest, and paths. Repeat the same selection with `--resume` for recovery.

Generated TOMLs record MOF, guest, molecule count, host atom count, source paths,
and source-file hashes. Each inserted structure has an `inputs.json` alongside
it with its insertion settings. Changed inputs are rejected when reusing a
campaign; use a distinct MOF label for a different cell or structure variant.
The Python preparation command's `--force` explicitly replaces generated inputs.

## Output storage on Izar

All generated workflow data defaults to the shared work location. To override
that location for a separate campaign, set one shared output root in the login
shell:

```bash
export MOF_OUTPUT_ROOT="/work/cosmo/dealmeid/mof-heat-capacity/output"
export MOF_SLURM_OUTPUT_DIR="${MOF_OUTPUT_ROOT}/slurm"
```

`submit_loaded_md.sh`, automatic continuations, and heat-capacity submission
then use this location for MD data and Slurm logs. The setting is inherited by
the Slurm workers. Keep it set for all later analysis commands so they find the
work-resident trajectories.

The default MD grid is 175 to 425 K in 25 K steps (11 temperatures).
The 175 and 425 K runs supply the neighboring enthalpy averages needed for
centered finite differences at 200 and 400 K. Hybrid heat-capacity results
default to the 200–400 K reporting range. Use `--temperatures` to run an
explicit convergence or scope-extension grid.

Production runs use 1,000,000 steps at 0.5 fs (500 ps), including the 200,000
step (100 ps) equilibration period. To remove equilibration frames from
existing trajectories without making a full temporary copy, first inspect then
apply:

```bash
./scripts/md/truncate_trajectories.sh
./scripts/md/truncate_trajectories.sh --apply
```

With both MLIPs, the eleven-temperature default grid, and one replica, the two
production planners submit 22 independent MD jobs: one job for every
`(model, temperature)` pair. The short debug and calibration jobs and the two
production planners are additional pipeline jobs and are not part of that
production count.

For every combination except the existing MOF-5/CH₄ default, output paths are:

```text
output/md/production/<mof>/<model>/<N><guest>/<temperature>K/repNN/
output/md/calibration/<mof>/<model>/<N><guest>/<temperature>K/repNN/
output/md/debug/<mof>/<model>/<N><guest>/<temperature>K/repNN/
output/md/structures/<mof>/<N><guest>/<temperature>K/repNN/
configs/<mof>/<model>/<N><guest>/<temperature>K-repNN.toml
output/slurm/simulation/<mof>/<model>/<N><guest>/...
```

For example, CO₂-loaded MOF-5 uses
`output/md/production/mof5/pet-mad-1.5-s-40nn/100co2/300K/rep01/`.
Structures are shared across MLIPs for the same host/guest/count/temperature/
replica and use identical initial coordinates. Guest-specific structures and
all model results remain separate.

MOF-5/CH₄ preserves its established model/loading hierarchy:

```text
/work/cosmo/dealmeid/mof-heat-capacity/output/md/production/pet-mad-1.5-s-40nn/50ch4/<temperature>K/repNN/
/work/cosmo/dealmeid/mof-heat-capacity/output/md/production/pet-sol-s-best/50ch4/<temperature>K/repNN/
```

Calibration and manual debug stages use the same model/loading hierarchy below
`/work/cosmo/dealmeid/mof-heat-capacity/output/md/calibration/` and
`/work/cosmo/dealmeid/mof-heat-capacity/output/md/debug/`. Run
directories use concise role names such as `trajectory.lammpstrj`,
`md.lammps.log`, and `md.final.data`; the directory supplies model, loading,
temperature, and replica context.

The reusable implementation lives in `mof_heat_capacity/simulation/`.
Generated configurations follow the same hierarchy:

```text
configs/<model>/<loading>ch4/<temperature>K-repNN.toml
```

Reusable templates remain directly below `configs/`. Each generated TOML
retains the unique full run name and the settings needed to interpret results.

## Commands

| Command | Responsibility |
| --- | --- |
| `scripts/md/submit_loaded_md.sh` | Prepare, validate, and submit loaded classical-NPT jobs. |

The submission command automatically invokes
`mof_heat_capacity.protocols.loaded` to create any missing independent loaded
structures and classical-NPT TOMLs, preserving inputs that already exist. It
then submits one 1,000-step calibration per MLIP under the high-priority debug
QOS and one dependent debug-QOS production planner. Calibration includes the
CUDA, LAMMPS, model, and stress checks, so a separate automatic 10-step job
would be redundant. The planner estimates and reports the total runtime, then
submits production under the requested production QOS only after calibration
succeeds. Under the default `normal` QOS, every production segment requests
`2-23:50:00`, ten minutes below Izar's three-day limit. The worker reserves the
last ten minutes for an orderly stop and submits exactly one new normal-QOS job
when the configured final MD step has not yet been reached. The successor reads
the latest numeric LAMMPS restart, appends the log and trajectory, and uses
LAMMPS `run ... upto` to retain the original absolute million-step target. This
repeats automatically until LAMMPS writes the final restart. Runtime or model
failures do not automatically resubmit.

If calibration fails, its planner remains pending with
`DependencyNeverSatisfied`: production requires a successful calibration.
Inspect the calibration's Slurm and LAMMPS logs, correct the failure, cancel
the obsolete planner with `scancel <planner-job-id>`, and submit the campaign
again. Do not use `--resume` to bypass a failed stability check. Renaming a MOF
input or campaign on disk does not update commands already submitted to Slurm;
submit subsequent Mg-MOF-74 campaigns with `--mof mgmof74`.

Pass `--time TIME` to override the allocation length or `--no-auto-resume` to
disable automatic continuation. The manual recovery command remains available:

```bash
./scripts/md/submit_loaded_md.sh --model both --loading 50 --replicas 1 --resume
```

Manual resume skips runs containing the final restart and submits only
unfinished trajectories. Because periodic restarts are written every 100,000
steps, a continuation resumes from the latest checkpoint rather than the exact
instant at which the previous segment was stopped.

The automatic calibration should fit Izar's one-hour debug-QOS limit; the
10-step `--debug` mode remains available for an isolated manual preflight. A
pending job has no Slurm output file until it starts; use `squeue --me` or
`sacct -j <job-id>` to inspect the complete chain.

Empty MOF-5 is intentionally absent from MD preparation: it is a direct
reference input for the property workflow and does not need an MD trajectory.
