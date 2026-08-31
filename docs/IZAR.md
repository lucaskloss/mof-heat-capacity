# Running the hybrid MOF-5 workflow on Izar

This runbook covers only the current workflow: classical NPT for methane-loaded
MOF-5, fixed-cell relaxation and AD Hessians, and assembly of the harmonic
quantum correction. Empty MOF-5 does not receive an MD job.

Run commands from the repository root on an Izar login node. CUDA validation
belongs inside a Slurm allocation, not on the login node.

## 1. Validate and submit loaded classical MD

Use the MD submission command documented in
[`scripts/md/README.md`](../scripts/md/README.md).

The command automatically submits one 1,000-step debug-QOS calibration per
MLIP. Each calibration performs the CUDA, LAMMPS, model, and stress preflight
before measuring the MD rate. After it succeeds, a dependent debug-QOS planner
estimates and reports the production runtime, then submits the production jobs
under the requested production QOS. The default `normal` jobs request
`2-23:50:00`, ten minutes below the three-day limit. MD is stopped ten minutes
before the allocation ends; if its absolute final step is still incomplete,
the worker automatically submits one successor under the same QOS. Successors
resume from the latest numeric LAMMPS restart and repeat this process until the
final restart is written. A scientific or runtime failure is not automatically
retried. Inspect the calibration logs, LAMMPS thermodynamics, trajectory, cell,
forces, and restart before interpreting the final results.

Production output is organized as:

```text
output/classical/production/<loading>ch4/<run-name>/
```

`--resume` remains available for manual recovery. It continues production from
the latest numeric LAMMPS restart toward the TOML's absolute step target.

Useful scheduler overrides are `--partition`, `--qos`, `--time`, and `--cpus`.
Each program uses one task and one GPU; requesting multiple GPUs does not make
it parallel.

## 2. Inspect trajectories

Trajectory diagnostics are required before heat capacities are interpreted;
use the property commands documented in
[`scripts/properties/README.md`](../scripts/properties/README.md).

Check equilibration, temperature and enthalpy stationarity, autocorrelation,
effective sample counts, density/cell behavior, forces, framework stability,
and methane mobility. The analysis itself is CPU-based, although Izar's normal
QOS may require allocating one GPU.

## 3. Relax minima and compute Hessians

Choose representative loaded replicas at one source temperature. The Hessian
command uses each completed run's final structure, quenches it at fixed cell,
verifies force convergence, and computes one PET-JAX Hessian. It also processes
the equilibrated empty structure directly.

The command first submits a short GPU/JAX preflight. Each Hessian job depends
on that check succeeding; the configured default wall time remains in use
unless `--time` is supplied.

Do not use raw thermal frames as normal-mode structures. Inspect relaxation
logs, maximum residual forces, imaginary modes, precision, graph hops, and mode
cutoff sensitivity. Existing relaxation/Hessian products are protected unless
`--overwrite` is supplied.

Hessian archives retain the sign of unstable modes. Hybrid assembly requires
signed spectra and relaxation metadata, rejects any mode below the negative
frequency threshold, and rejects more than three near-zero translational modes.
An old archive produced with the zero-clipped convention must be recomputed.

If a loaded quench reaches the force threshold but its signed spectrum is
unstable, refine that same minimum without repeating the empty calculation.
The overwrite is intentional for the canonical loaded minimum and Hessian; it
does not touch the empty reference.
Use `--hessian-only` when the minimum itself is accepted and only a legacy or
diagnostic Hessian must be replaced. Non-finite relaxation results are rejected
before they can overwrite a valid minimum; line-search optimizers require
particular care with float32 model energies.
For a precision or graph-depth comparison, use an explicit tag such as
`--hessian-tag fp64-h4 --dtype float64 --hops 4`; tagged archives are retained
for inspection but are never consumed automatically by hybrid assembly.

Outputs are stored below:

```text
output/hybrid/<model>/<loading>ch4/minima/
output/hybrid/<model>/<loading>ch4/hessians/
output/hybrid/<model>/0ch4/
```

The empty result is a reference and Hessian-pipeline check. The correction in
the loaded heat capacity uses the complete loaded-system spectra.

## 4. Assemble the hybrid heat capacity

After every requested temperature/replica and loaded Hessian is complete, the
hybrid command differentiates replica-averaged classical NPT enthalpy and adds
the loaded harmonic quantum-minus-classical correction. It writes NPZ, CSV,
and JSON provenance below `output/hybrid/<model>/<loading>ch4/`.

## Troubleshooting

- Missing generated TOMLs: rerun `scripts/md/submit_loaded_md.sh` with the
  same model, loading, temperatures, and replica count.
- Missing stress output: stop; NPT is not valid with that exported model.
- CUDA initialization failure: recreate the pinned environment and confirm the
  Slurm job has one V100 allocation.
- Relaxation has not converged or has imaginary modes: do not silently discard
  them; improve the minimum or convergence settings and rerun.
- Noisy enthalpy derivative: extend sampling, increase independent replicas,
  or revisit the temperature spacing as a scientific convergence parameter.
- Existing outputs: choose a new campaign directory unless replacement is
  intentional; only then use the explicit rerun/overwrite option.
