# Property calculations

This folder contains the user-facing property-calculation Bash commands and
guide. They read existing configurations and simulation results and never
start or resume molecular dynamics.

For the SADMOF/PET-JAX Hessian algorithm, heat-capacity equations, archive
contents, and debugging checks, see
[`docs/SADMOF_HESSIANS.md`](../../docs/SADMOF_HESSIANS.md).

Run commands from the repository root:

```bash
./scripts/properties/submit_analysis.sh --model pet-mad --loading 100 \
  --replicas 1
./scripts/properties/submit_heat_capacity.sh --model pet-mad --loading 100 \
  --source-temperature 300 --replicas 1
./scripts/properties/submit_hybrid_analysis.sh --model pet-mad --loading 100 \
  --replicas 1
```

Trajectory-analysis output is grouped automatically by loading. For example,
`--loading 100` writes below `output/analysis/100ch4/`; an explicit
`--analysis-dir` remains available when a custom location is needed.

The first command produces trajectory diagnostics. The second relaxes selected
structures and computes harmonic Hessians. The third combines classical
enthalpy derivatives with the harmonic quantum correction. Their reusable
implementations live in `mof_heat_capacity/analysis/`.

Hybrid NPZ and CSV outputs contain the classical term, harmonic correction,
and final approximate $C_P$ in both gravimetric $J g^{-1} K^{-1}$ and
volumetric $J cm^{-3} K^{-1}$ units. The volumetric values use the loaded NPT
mean volume and cell mass at each temperature; the accompanying JSON notes the
uncertainty assumptions.

Classical analysis and hybrid assembly default to the same 200–400 K grid in
25 K steps as MD submission. Interior enthalpy derivatives are centered; the
200 and 400 K values use second-order one-sided estimates. The independently
configurable harmonic diagnostic grid remains broader because normal-mode
$C_V(T)$ is evaluated analytically and does not require neighboring MD runs.

This workflow is independent of the simulation commands: results may be
copied into the documented `output/` layout or selected explicitly where the
underlying command supports a path. Only completed inputs are required.

## Commands

| Command | Responsibility |
| --- | --- |
| `scripts/properties/submit_analysis.sh` | Validate, submit, and execute trajectory diagnostics. |
| `scripts/properties/submit_heat_capacity.sh` | Quench loaded configurations and compute loaded and empty-reference AD Hessians. |
| `scripts/properties/submit_hybrid_analysis.sh` | Submit and execute classical-enthalpy differentiation plus the loaded-Hessian quantum correction. |

The analysis and hybrid commands contain their own private Slurm worker modes,
so each operation can be understood from one file. Hessian submission continues
to use the shared GPU launcher because relaxation and Hessian calculation need
the same CUDA, Conda, and model validation as MD; keeping those checks together
avoids two divergent copies of scientific runtime setup.

Before Hessian jobs are submitted, `submit_heat_capacity.sh` automatically
submits a lightweight GPU/JAX preflight for each model. The relaxation/Hessian
jobs run only when that preflight succeeds. Their wall time continues to use
the configured default (or an explicit `--time` override).

Hessian archives store signed frequencies. Final hybrid assembly requires the
relaxation provenance, rejects imaginary modes below the configured threshold,
and allows at most the three translational near-zero modes. If a loaded quench
needs refinement, continue from its existing minimum without rerunning the
empty reference:

Use `--hessian-only` to reuse an accepted minimum while replacing a legacy or
diagnostic Hessian. Precision, graph hops, force tolerance, and optimizer choice
are convergence parameters. Use `--dtype`, `--hops`, and `--chunk-size` for
explicit Hessian tests rather than changing campaign settings silently. A tag
such as `--hessian-tag fp64-h4` preserves a non-canonical comparison; hybrid
assembly reads only canonical `repNN.npz` archives.
