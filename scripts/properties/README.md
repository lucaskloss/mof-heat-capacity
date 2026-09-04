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
./scripts/properties/submit_analysis.sh --model pet-mad --loading 100 \
  --replicas 1 --model-uncertainty \
  --uncertainty-model models/pet-mad-1.5-s-llpr-ensemble.pt
./scripts/properties/submit_heat_capacity.sh --model both --loading 100 \
  --source-temperatures 200,225,250,275,300,325,350,375,400 --replicas 1
./scripts/properties/submit_hybrid_analysis.sh --model both --loading 100 \
  --replicas 1
```

Trajectory analysis submits one Slurm job per selected trajectory. A dependent
summary job runs after every trajectory job succeeds and assembles the combined
CSV, manifest, workflow requirements, and temperature-sweep plot. These jobs use
the production-appropriate `normal` QOS by default; the shorter `debug` QOS is
reserved for testing. Output is grouped automatically by MLIP and loading; for
example, `--model pet-mad --loading 100` writes below
`output/post-processing/trajectory-analysis/pet-mad-1.5-s-40nn/100ch4/`. An explicit `--analysis-dir`
changes the base directory while retaining the model/loading subdirectories.
Each run is below `<temperature>K/repNN/`, with concise products such as
`summary.json`, `timeseries.csv`, and `structure.png`.

`--model-uncertainty` additionally requires each configured exported model to
provide a calibrated system-level `energy_ensemble` output. The optional
analytical `energy_uncertainty` output is archived when present. Use
`--uncertainty-model` to select a calibrated LLPR/ensemble export
without changing the central model recorded in the MD configuration. One
override can analyze only one MLIP per submission; invoke `--model pet-mad` and
`--model pet-sol` separately when their ensemble exports differ. The analysis
evaluates every 20th production frame by default; change
this convergence parameter with `--uncertainty-stride`, and control GPU memory
with `--uncertainty-batch-size`. The analysis also removes a constant energy
offset and requires the ensemble mean to reproduce the trajectory-driving
potential within `--uncertainty-central-tolerance-eV` (default: 0.01 eV) on
every selected frame. Each run writes `model_uncertainty.npz` with
the framewise central, analytical, and member energies plus exact-reweighting
overlap diagnostics. The dependent summary job writes
`model_uncertainty_heat_capacity.{npz,csv,png}` after averaging replicas and
differentiating each persistent member's enthalpy curve. The CSV reports both
direct and CEA results; use the CEA committee standard deviation as the primary
large-system MLIP error bar and inspect `minimum_direct_effective_samples` and
`maximum_dimensionless_delta_variance` before interpreting it.

The same exported ensemble file, verified by SHA-256, must be used at every
temperature and in every replica so member identities remain correlated across
the derivative. This model band applies to the classical NPT contribution. It
does not replace the existing sampling uncertainty or quantify uncertainty in
the separate PET-JAX harmonic correction; those terms must remain separately
labeled or be combined only under an explicitly justified independence
assumption.

After a model-uncertainty analysis has completed, include its classical CEA
band in final assembly with:

```bash
./scripts/properties/submit_hybrid_analysis.sh --model pet-mad --loading 100 \
  --replicas 1 --model-uncertainty
```

The hybrid NPZ and CSV then retain the original sampling `standard_error`, the
classical `model_standard_deviation`, and a
`combined_standard_uncertainty`. The plotted classical and hybrid bands use
the combined value. The combination is a quadrature summary under an explicit
independence assumption; the component arrays should be used when correlations
cannot be neglected.

The first command produces trajectory diagnostics. The second relaxes selected
structures and computes harmonic Hessians. The third combines classical
enthalpy derivatives with the harmonic quantum correction. Their reusable
implementations live in `mof_heat_capacity/analysis/`.

Hybrid NPZ and CSV outputs contain the classical term, harmonic correction,
and final approximate $C_P$ in both gravimetric $J g^{-1} K^{-1}$ and
volumetric $J cm^{-3} K^{-1}$ units. A matching PNG plots all three curves and
their uncertainties against temperature in both unit systems. The volumetric
values use the loaded NPT mean volume and cell mass at each temperature; the
accompanying JSON notes the uncertainty assumptions.

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

With `--model both`, final hybrid assembly submits one independent job for
PET-MAD and one for PET-SOL. There is no dependency between those jobs, and
their outputs remain separated below their respective model directories. Use
`--afterok JOB1,JOB2,...` to hold both jobs until an upstream Hessian campaign
has completed successfully.

Before Hessian jobs are submitted, `submit_heat_capacity.sh` automatically
submits a lightweight GPU/JAX preflight for each model. The relaxation/Hessian
jobs run only when that preflight succeeds. Every selected model, loaded
temperature, and replica combination is submitted as its own independent job;
each model's empty reference is also independent. They run in parallel as
resources become available. Their wall time continues to use the configured
default (or an explicit `--time` override).

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
assembly reads only canonical `TEMPERATUREK/repNN/hessian.npz` archives matching each
classical-MD temperature.
