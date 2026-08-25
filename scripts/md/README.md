# Molecular-dynamics simulations

This folder is the user-facing Bash entry point and guide for the loaded MOF-5
classical-NPT campaign. It writes structures, generated TOML files,
trajectories, restarts, and logs under the repository-level `output/`
directory. It does not run property analysis.

Run commands from the repository root:

```bash
./scripts/md/submit_loaded_md.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1
```

The reusable implementation lives in `mof_heat_capacity/simulation/`. Run
configurations are kept in the shared `configs/` directory because they also
record the model and harmonic settings needed to interpret existing results.

## Commands

| Command | Responsibility |
| --- | --- |
| `scripts/md/submit_loaded_md.sh` | Prepare, validate, and submit loaded classical-NPT jobs. |

The submission command automatically invokes
`mof_heat_capacity.protocols.loaded` to create any missing independent loaded
structures and classical-NPT TOMLs, preserving inputs that already exist. It
then submits a 10-step debug job, a dependent 1,000-step calibration job, and
a dependent production planner. The planner estimates wall time from the
calibration rate, adds a 25% margin plus 10 minutes for startup, prints the
result, and submits production only after calibration succeeds.
Pass `--time HH:MM:SS` only to override that computed production wall time.

Empty MOF-5 is intentionally absent from MD preparation: it is a direct
reference input for the property workflow and does not need an MD trajectory.
