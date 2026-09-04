# Molecular-dynamics simulations

This folder is the user-facing Bash entry point and guide for the loaded MOF-5
classical-NPT campaign. By default it writes structures, generated TOML files,
trajectories, restarts, and logs under the repository-level `output/`
directory. It does not run property analysis.

Run commands from the repository root:

```bash
./scripts/md/submit_loaded_md.sh --model both --loading 50 --replicas 1
```

## Scratch storage on Izar

Large trajectories and restart files should be kept on SCITAS scratch rather
than in home. When `$SCRATCH` is defined, the submission script defaults to
`${SCRATCH}/mof-heat-capacity/output`. To use a different scratch location,
set one shared output root in the login shell:

```bash
export MOF_OUTPUT_ROOT="${SCRATCH}/mof-heat-capacity/output"
export MOF_SLURM_OUTPUT_DIR="${MOF_OUTPUT_ROOT}/slurm"
```

`submit_loaded_md.sh`, automatic continuations, and heat-capacity submission
then use this location for MD data and Slurm logs. The setting is inherited by
the Slurm workers. Keep it set for all later analysis commands so they find the
scratch-resident trajectories. Existing repository `output/` data can be
copied with `rsync -a --info=progress2 output/ "${MOF_OUTPUT_ROOT}/"`; verify
the copy before removing the home-folder source.

The default production grid is 200 to 400 K in 25 K steps. This spacing
provides neighboring enthalpy averages for numerical differentiation while
keeping the hybrid approximation in the temperature range where Kapil et al.
found it reliable for their 100-methane system. Use `--temperatures` to run an
explicit convergence or scope-extension grid.

Production runs use 1,000,000 steps at 0.5 fs (500 ps), including the 200,000
step (100 ps) equilibration period. To remove equilibration frames from
existing trajectories without making a full temporary copy, first inspect then
apply:

```bash
./scripts/md/truncate_trajectories.sh
./scripts/md/truncate_trajectories.sh --apply
```

With both MLIPs, the nine-temperature default grid, and one replica, the two
production planners submit 18 independent MD jobs: one job for every
`(model, temperature)` pair. The short debug and calibration jobs and the two
production planners are additional pipeline jobs and are not part of that
production count.

Simulation outputs are separated first by MLIP and then by loading:

```text
output/md/production/pet-mad-1.5-s-40nn/50ch4/<temperature>K/repNN/
output/md/production/pet-sol-s-best/50ch4/<temperature>K/repNN/
```

Calibration and manual debug stages use the same model/loading hierarchy below
`output/md/calibration/` and `output/md/debug/`. Historical output locations
remain readable so that already-running campaigns can finish safely. Run
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
