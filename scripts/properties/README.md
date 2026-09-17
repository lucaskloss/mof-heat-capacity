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
./scripts/properties/submit_heat_capacity.sh --model pet-mad --loading 50 \
  --source-temperature 400 --replicas 1 --skip-empty --llpr-jobs 8 \
  --cv-temperatures 200:400:25
# Replace MERGE_JOB_ID with the final merge ID printed by the command above.
./scripts/properties/submit_hybrid_analysis.sh --model pet-mad --loading 50 \
  --replicas 1 --hessian-source-temperature 400 --afterok MERGE_JOB_ID
```

Submit one model/loading campaign at a time. The command above is the fresh
PET-MAD/50 CH₄ run after removal of its old relaxation outputs: it starts from
`output/md/production/pet-mad-1.5-s-40nn/50ch4/400K/rep01/md.final.data`.
It submits a preflight, one 400 K relaxation/central-Hessian job, eight LLPR
workers with eight members each, and a final merge. Dependencies ensure that
the eight workers start after the central calculation and that the merge
starts after all workers succeed. They do not serialize separate campaigns;
wait for this campaign to finish before submitting the next model/loading.
The central relaxation/Hessian job and each LLPR worker default to a one-day
wall time (`1-00:00:00`); override it with `--time` or `MOF_HEAT_TIME`.
The preflight retains its 15-minute limit and the merge its 30-minute limit.
`--skip-empty` avoids an additional empty-reference campaign; the loaded hybrid
correction uses the loaded-system spectrum.

For a fresh relaxation, omit `--hessian-only`, `--reuse-relaxed`, and continuation
options. With saved outputs present, an ordinary submission may reuse the
minimum; intentionally replacing it requires `--overwrite` and clearing
incompatible Hessian restart files first. To recover an interrupted campaign
with unchanged settings, repeat the command with `--continue-unfinished`, after
checking that no matching jobs are still active. Both loading-100 campaigns
already have complete 400 K central/64-member spectra and need no Hessian rerun.

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
The analysis also writes `enthalpy_convergence.{csv,png}` at 25 ps cumulative
production-time intervals. These files compare the autocorrelation-corrected
sampling standard error with the LLPR committee standard deviation and show
the prefix mean's deviation from the complete-run mean. Change the interval
with `--enthalpy-convergence-step-ps`; this is accumulated trajectory time in
ps, not the MD integration timestep in fs.
LAMMPS thermo logs retain equilibration records, whereas the coordinate dump
intentionally begins at the configured production start. Analysis aligns the
two by their LAMMPS timestep, so their frame counts are not expected to match.

Trajectory analysis uses the matching calibrated LLPR checkpoint by default.
`--model-uncertainty` is retained as an explicit no-op for compatibility;
`--no-model-uncertainty` requests a central-model-only diagnostic. LLPR mode
combines its persistent readouts with system-level last-layer features
from the configured central export. Analytical `energy_uncertainty` is also
reconstructed and archived. Use `--uncertainty-model` to select another LLPR
checkpoint or a compatible exported ensemble
without changing the central model recorded in the MD configuration. One
override can analyze only one MLIP per submission; invoke `--model pet-mad` and
`--model pet-sol` separately when their ensemble exports differ. The analysis
evaluates every 20th production frame by default; change
this convergence parameter with `--uncertainty-stride`, and control GPU memory
with `--uncertainty-batch-size`. The analysis records the centered residual
between the ensemble mean and the trajectory-driving potential but does not
reject a trajectory based on that diagnostic. Each run writes
`model_uncertainty.npz` with
the framewise central, analytical, and member energies plus exact-reweighting
overlap diagnostics. The dependent summary job writes
`model_uncertainty_heat_capacity.{npz,csv,png}` after averaging replicas and
differentiating each persistent member's enthalpy curve. The CSV reports both
direct and CEA results; use the CEA committee standard deviation as the primary
large-system MLIP error bar and inspect `minimum_direct_effective_samples` and
`maximum_dimensionless_delta_variance` before interpreting it.
The same aggregate also writes `model_uncertainty_enthalpy.{csv,png}`; its CEA
and direct enthalpy columns and plot include the corresponding member standard
deviations in eV at every temperature.
While LLPR inference is running, `model_uncertainty.progress.npz` is updated
atomically after each inference batch. If a Slurm job reaches its wall-time
limit, resubmitting the same trajectory analysis resumes from its completed
LLPR frames; the progress file is removed after a successful final archive is
written.

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
Prefix convergence from one trajectory is a retrospective truncation test,
not independent validation: use it to nominate a shorter duration, then check
that duration with independent replicas before changing the production target.

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

The first command produces trajectory diagnostics. The second relaxes only the
highest selected source temperature for each model, loading, and replica and
computes its harmonic Hessian and eigenfrequencies once. The same spectrum is
evaluated over the full `--cv-temperatures` grid. The third combines classical
enthalpy derivatives with the harmonic quantum correction. Their reusable
implementations live in `mof_heat_capacity/analysis/`.

Hybrid assembly uses the highest temperature in its MD grid as the shared
Hessian source by default, including the corresponding LLPR member spectra.
If the Hessian submission used a higher source temperature than the assembly
grid, pass `--hessian-source-temperature N` to `submit_hybrid_analysis.sh`.
Lower-temperature minima and Hessian archives are not needed or read. Each
harmonic sum uses its evaluation temperature with the shared eigenfrequencies;
both quantum and classical harmonic terms use the same retained modes.

Hessian submission defaults to fixed-cell `lbfgs-linesearch`, at most 10,000
optimizer steps, and a maximum force of $0.002\ \mathrm{eV\ \AA^{-1}}$.
These are deliberate minimum-validation settings: a saved Hessian with modes
below the imaginary-frequency threshold is not accepted for hybrid assembly.
Each archive includes the central Hessian spectrum and all 64 LLPR-member
spectra by default. Submission now uses one relaxation/central-Hessian job,
then a Slurm array of eight GPU jobs, each computing eight LLPR member Hessians
and their eigenfrequencies on that shared minimum. A final merge job waits for
all eight tasks and writes the canonical archive expected by hybrid assembly.
Use the printed **final merge job ID** with hybrid `--afterok`, rather than the
central job or array ID. The merge checks shared geometry, numerical settings,
LLPR checkpoint identity, and exact coverage of all 64 member indices; it pools
matrix moments and restores member order for correlated hybrid uncertainties.

Intermediate files are `hessian.central.npz` and `hessian.llpr-0.npz` through
`hessian.llpr-7.npz` beside the final `hessian.npz` (tags are preserved).
Each batch has its own restart checkpoint. `--continue-unfinished` reuses
completed central/batch archives and resumes unfinished batches. Changing the
geometry or calculation settings requires a fresh output tag or deliberately
removing incompatible intermediate/restart files. Existing serial jobs retain
their original execution plan. `--llpr-jobs 1` selects the serial workflow;
`--no-model-uncertainty` submits just the relaxation/central-Hessian job.
All jobs use one node, one task, and one GPU; the merge is CPU-based but retains
the GPU allocation required by Izar's normal QOS.

The one-day limit applies to each job, rather than the entire dependent chain.
Inspect measured runtimes when selecting a different `--time`;
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
configurable harmonic diagnostic grid defaults to `200:400:25`; normal-mode
$C_V(T)$ is evaluated analytically and can use another grid without additional
Hessian calculations.

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
jobs run only when that preflight succeeds. Every selected model and replica
gets one loaded job at the highest selected source temperature;
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
assembly reads the canonical `SOURCE_TEMPERATUREK/repNN/hessian.npz` archive
at the shared highest source temperature for each replica.
