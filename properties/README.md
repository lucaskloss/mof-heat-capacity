# Property calculations

This folder is the complete user-facing entry point for inspecting completed
trajectories and calculating heat capacities. It reads existing configurations
and simulation results; it never starts or resumes molecular dynamics.

Run commands from the repository root:

```bash
./properties/submit_analysis.sh --model pet-mad --loading 100 --dry-run
./properties/submit_heat_capacity.sh --model pet-mad --loading 100 --dry-run
./properties/submit_hybrid_analysis.sh --model pet-mad --loading 100 --dry-run
```

The first command produces trajectory diagnostics. The second relaxes selected
structures and computes harmonic Hessians. The third combines classical
enthalpy derivatives with the harmonic quantum correction. Their reusable
implementations live in `mof_heat_capacity/analysis/`.

This workflow is independent of the simulation commands: results may be
copied into the documented `output/` layout or selected explicitly where the
underlying command supports a path. Only completed inputs are required.

## Commands

| Command | Responsibility |
| --- | --- |
| `submit_analysis.sh` | Validate, submit, and execute trajectory diagnostics. |
| `submit_heat_capacity.sh` | Quench loaded configurations and compute loaded and empty-reference AD Hessians. |
| `submit_hybrid_analysis.sh` | Submit and execute classical-enthalpy differentiation plus the loaded-Hessian quantum correction. |

The analysis and hybrid commands contain their own private Slurm worker modes,
so each operation can be understood from one file. Hessian submission continues
to use the shared GPU launcher because relaxation and Hessian calculation need
the same CUDA, Conda, and model validation as MD; keeping those checks together
avoids two divergent copies of scientific runtime setup.

Hessian archives store signed frequencies. Final hybrid assembly requires the
relaxation provenance, rejects imaginary modes below the configured threshold,
and allows at most the three translational near-zero modes. If a loaded quench
needs refinement, continue from its existing minimum without rerunning the
empty reference:

```bash
./properties/submit_heat_capacity.sh --model pet-mad --loading 100 \
  --source-temperature 300 --replicas 1 --skip-empty \
  --continue-loaded --overwrite --optimizer fire \
  --fmax 0.005 --relax-steps 4000 --cv-temperatures 100:500:100 --dry-run
```

Use `--hessian-only` to reuse an accepted minimum while replacing a legacy or
diagnostic Hessian. Precision, graph hops, force tolerance, and optimizer choice
are convergence parameters. Use `--dtype`, `--hops`, and `--chunk-size` for
explicit Hessian tests rather than changing campaign settings silently. A tag
such as `--hessian-tag fp64-h4` preserves a non-canonical comparison; hybrid
assembly reads only canonical `repNN.npz` archives.
