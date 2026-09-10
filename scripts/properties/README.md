# Property calculations

This folder contains the user-facing property-calculation Bash commands and
guide. They read existing configurations and simulation results and never
start or resume molecular dynamics.

For the SADMOF/PET-JAX Hessian algorithm, heat-capacity equations, archive
contents, and debugging checks, see
[`docs/SADMOF_HESSIANS.md`](../../docs/SADMOF_HESSIANS.md).

Run commands from the repository root:

```bash
./scripts/setup/submit_llpr_ensemble.sh --model pet-mad \
  --training-set /path/to/covariance.extxyz \
  --validation-set /path/to/calibration.extxyz --dry-run
./scripts/properties/submit_analysis.sh --model pet-mad --loading 100 \
  --replicas 1
./scripts/properties/submit_analysis.sh --model pet-mad --loading 100 \
  --replicas 1 --model-uncertainty
./scripts/properties/submit_heat_capacity.sh --model both --loading 100 \
  --source-temperatures 200,225,250,275,300,325,350,375,400 --replicas 1
./scripts/properties/submit_hybrid_analysis.sh --model both --loading 100 \
  --replicas 1
```

The LLPR preparation command requires reference-labeled structures that are
not distributed with this repository. It wraps the selected PET checkpoint,
constructs the last-layer feature covariance from `--training-set`, calibrates
the uncertainty scale on the separate `--validation-set`, samples one
persistent 32-member shallow ensemble, and validates that its arithmetic mean
reproduces the central energy. It writes the exported model below `models/`
and a matching provenance JSON containing model/data hashes and software
versions. See [`docs/LLPR_ENSEMBLE.md`](../../docs/LLPR_ENSEMBLE.md) before
preparing production uncertainty results.

Trajectory analysis submits one Slurm job per selected trajectory. A dependent
summary job runs after every trajectory job succeeds and assembles the combined
CSV, manifest, workflow requirements, and temperature-sweep plot. These jobs use
the production-appropriate `normal` QOS by default; the shorter `debug` QOS is
reserved for testing. Output is grouped automatically by MLIP and loading; for
example, `--model pet-mad --loading 100` writes below
`/work/cosmo/dealmeid/mof-heat-capacity/output/post-processing/trajectory-analysis/pet-mad-1.5-s-40nn/100ch4/`. An explicit `--analysis-dir`
changes the base directory while retaining the model/loading subdirectories.
Each run is below `<temperature>K/repNN/`, with concise products such as
`summary.json`, `timeseries.csv`, and `structure.png`.
LAMMPS thermo logs retain equilibration records, whereas the coordinate dump
intentionally begins at the configured production start. Analysis aligns the
two by their LAMMPS timestep, so their frame counts are not expected to match.

`--model-uncertainty` uses the matching calibrated LLPR checkpoint by default
and combines its persistent readouts with system-level last-layer features
from the configured central export. Analytical `energy_uncertainty` is also
reconstructed and archived. Use `--uncertainty-model` to select another LLPR
checkpoint or a compatible exported ensemble
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

The CEA implementation follows the Atomistic Cookbook's
[PET-MAD uncertainty example](https://atomistic-cookbook.org/examples/pet-mad-uq/pet-mad-uq.html#cumulant-expansion-approximation-cea),
extended from its NVT RDF example to member-specific NPT enthalpy. The reported
`direct_effective_samples` is a Kish count of normalized reweighting weights;
it diagnoses overlap but is not adjusted for trajectory autocorrelation. The
condition `maximum_dimensionless_delta_variance < 1` only avoids the code's
severe warning. CEA formally requires this quantity to be much smaller than
one, especially before differentiating the enthalpy curves into heat capacity.

The same LLPR ensemble, verified by SHA-256, must be used at every
temperature and in every replica so member identities remain correlated across
the derivative. It does not replace the existing sampling uncertainty. The
Hessian workflow propagates the same persistent readouts through PET-JAX at
the central minimum and reports the harmonic model contribution separately.
Member-specific geometry relaxation remains outside the uncertainty estimate.

After a model-uncertainty analysis has completed, include its classical CEA
band in final assembly with:

```bash
./scripts/properties/submit_hybrid_analysis.sh --model pet-mad --loading 100 \
  --replicas 1 --model-uncertainty
```

The hybrid NPZ and CSV then retain the original sampling `standard_error`, the
classical and harmonic `model_standard_deviation` components, and a
`combined_standard_uncertainty`. Classical and harmonic deviations are added
member by member before the hybrid model spread is evaluated; only the final
combination of sampling and LLPR model uncertainty uses quadrature.

The first command produces trajectory diagnostics. The second relaxes selected
structures and computes harmonic Hessians. The third combines classical
enthalpy derivatives with the harmonic quantum correction. Their reusable
implementations live in `mof_heat_capacity/analysis/`.

Hessian submission defaults to fixed-cell `lbfgs-linesearch`, at most 20,000
optimizer steps, and a maximum force of $0.001\ \mathrm{eV\ \AA^{-1}}$.
These are deliberate minimum-validation settings: a saved Hessian with modes
below the imaginary-frequency threshold is not accepted for hybrid assembly.
Each archive includes the central Hessian spectrum and all 64 LLPR-member
spectra by default, so its wall time is substantially longer than a central-only
calculation. Benchmark a representative allocation before selecting `--time`;
use `--no-model-uncertainty` only for an explicitly central-only diagnostic.
LLPR Hessians use `float64` because the sampled last-layer weights contain
cancellation-sensitive covariance directions; central-only runs retain the
configured precision unless `--dtype` is supplied.
Use `--optimizer`, `--relax-steps`, and `--fmax` only as explicit convergence
tests, and record the overrides with the resulting archives.
For a mixed recovery campaign, `--reuse-relaxed` retains every existing
`optimized.extxyz` minimum and recomputes its Hessian, while submitting the
longer relaxation only for cases without a saved converged minimum. This is
different from `--hessian-only`, which requires an existing minimum for every
selected case.

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

This workflow is independent of the simulation commands: results are read from
the shared work output tree or selected explicitly where the underlying command
supports a path. Only completed inputs are required.

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
