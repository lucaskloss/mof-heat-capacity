# Shared project utilities

The two scientific workflows have their own command folders:

- `simulation/` prepares and runs molecular dynamics.
- `properties/` inspects results and calculates heat capacities.

This directory contains only infrastructure shared by those workflows:

- `install_sadmof.sh` installs the PET-JAX/SADMOF dependencies required by
  harmonic property calculations.
- `izar_job.sh` contains the shared GPU runtime checks used by MD, relaxation,
  and Hessian submission. CPU trajectory analysis and final hybrid assembly
  are self-contained in their respective submission scripts.

See each workflow's README for its public commands.
