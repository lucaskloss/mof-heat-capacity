# Bash entry points

All Bash scripts live here and are grouped by responsibility:

- `setup/install_sadmof.sh` installs the PET-JAX/SADMOF dependencies required
  by harmonic property calculations.
- `slurm/izar_gpu_runtime.sh` is the shared single-GPU runtime for MD, relaxation, and
  Hessian stages on Izar.
- `md/submit_loaded_md.sh` validates and submits loaded classical-NPT campaigns.
- `properties/submit_analysis.sh` submits trajectory diagnostics.
- `properties/submit_heat_capacity.sh` submits relaxation and Hessian jobs.
- `properties/submit_hybrid_analysis.sh` submits final hybrid-curve assembly.

Run these commands from the repository root. See `md/README.md` and
`properties/README.md` for workflow-specific usage; reusable Python
implementation remains in `mof_heat_capacity/`.
