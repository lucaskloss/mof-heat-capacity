# Molecular-dynamics simulations

This folder is the complete user-facing entry point for preparing and running
the loaded MOF-5 classical-NPT campaign. It writes structures, generated TOML
files, trajectories, restarts, and logs under the repository-level `output/`
directory. It does not run property analysis.

Run commands from the repository root:

```bash
python -m mof_heat_capacity.protocols.loaded --model pet-mad --loading 100 --dry-run
./simulation/submit_loaded_md.sh --model pet-mad --loading 100 --debug --dry-run
```

For a single run, use:

```bash
python -m mof_heat_capacity.simulation.md --config configs/RUN.toml
```

The reusable implementation lives in `mof_heat_capacity/simulation/`. Run
configurations are kept in the shared `configs/` directory because they also
record the model and harmonic settings needed to interpret existing results.

## Commands

| Command | Responsibility |
| --- | --- |
| `submit_loaded_md.sh` | Validate and submit loaded classical-NPT jobs. |

Campaign preparation and single-run MD are direct package entry points rather
than forwarding scripts. `mof_heat_capacity.protocols.loaded` generates
independent loaded structures and classical-NPT TOMLs;
`mof_heat_capacity.simulation.md` executes one configured run.

Remove `--dry-run` only after inspecting the full command matrix. Empty MOF-5
is intentionally absent from MD preparation: it is a direct reference input
for the property workflow and does not need an MD trajectory.
