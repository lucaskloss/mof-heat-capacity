# Running the hybrid MOF-5 workflow on Izar

This runbook covers only the current workflow: classical NPT for methane-loaded
MOF-5, fixed-cell relaxation and AD Hessians, and assembly of the harmonic
quantum correction. Empty MOF-5 does not receive an MD job.

Run commands from the repository root on an Izar login node. CUDA validation
belongs inside a Slurm allocation, not on the login node.

## One-time setup

Create the pinned environment and install SADMOF/PET-JAX:

```bash
conda env create --file environment.yml
conda activate mof-heat-capacity
./scripts/install_sadmof.sh
```

The environment intentionally combines a CUDA-12 Python PyTorch wheel with the
separately pinned LAMMPS/libtorch build and CUDA-12 JAX packages. Do not load an
unrelated CUDA module or replace those pins with current defaults.

The Slurm scripts default to:

```text
$HOME/.conda/envs/mof-heat-capacity-izar
```

Override this, when necessary, with `MOF_ENV_PREFIX`. If Conda is not available
after `module purge`, set `MOF_CONDA_SH` to the base installation's
`etc/profile.d/conda.sh`.

Before production, verify that the selected model artifacts exist under
`models/` and that the model's stress/virial output has been validated. The
preparation command records this fact for bundled presets; custom paths require
an explicit `--stress-validated` after validation.

## 1. Prepare loaded runs

Preview the initial 100--500 K campaign with 100 K spacing and one replica:

```bash
python -m mof_heat_capacity.protocols.loaded --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1 --dry-run
```

Then create the independent methane arrangements and generated TOMLs:

```bash
python -m mof_heat_capacity.protocols.loaded --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1
```

Generated configurations remain local and are ignored by Git. Use a distinct
model/loading combination or output tree for every independent campaign.

## 2. Validate and submit loaded classical MD

First run the isolated ten-step path:

```bash
./simulation/submit_loaded_md.sh --model pet-mad --loading 100 --debug --dry-run
./simulation/submit_loaded_md.sh --model pet-mad --loading 100 --debug
```

Inspect its Slurm log, LAMMPS thermodynamics, trajectory, cell, forces, and
restart. Next use `--calibration` to measure representative throughput and set
a defensible wall time:

```bash
./simulation/submit_loaded_md.sh --model pet-mad --loading 100 --calibration --dry-run
./simulation/submit_loaded_md.sh --model pet-mad --loading 100 --calibration
```

Submit production only after both checks pass:

```bash
./simulation/submit_loaded_md.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1 --dry-run
./simulation/submit_loaded_md.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1
```

Production output is organized as:

```text
output/classical/production/<loading>ch4/<run-name>/
```

`--resume` continues from the latest numeric LAMMPS restart toward the TOML's
absolute step target. Do not combine it with `--debug` or `--calibration`.

Useful scheduler overrides are `--partition`, `--qos`, `--time`, and `--cpus`.
Each program uses one task and one GPU; requesting multiple GPUs does not make
it parallel.

## 3. Inspect trajectories

Trajectory diagnostics are required before heat capacities are interpreted:

```bash
./properties/submit_analysis.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1 --dry-run
./properties/submit_analysis.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1
```

Check equilibration, temperature and enthalpy stationarity, autocorrelation,
effective sample counts, density/cell behavior, forces, framework stability,
and methane mobility. The analysis itself is CPU-based, although Izar's normal
QOS may require allocating one GPU.

## 4. Relax minima and compute Hessians

Choose representative loaded replicas at one source temperature. The command
uses each completed run's final structure, quenches it at fixed cell, verifies
force convergence, and computes one PET-JAX Hessian. It also processes the
equilibrated empty structure directly:

```bash
./properties/submit_heat_capacity.sh --model pet-mad --loading 100 \
  --source-temperature 300 --replicas 1 --dry-run
./properties/submit_heat_capacity.sh --model pet-mad --loading 100 \
  --source-temperature 300 --replicas 1
```

Do not use raw thermal frames as normal-mode structures. Inspect relaxation
logs, maximum residual forces, imaginary modes, precision, graph hops, and mode
cutoff sensitivity. Existing relaxation/Hessian products are protected unless
`--overwrite` is supplied.

Hessian archives retain the sign of unstable modes. Hybrid assembly requires
signed spectra and relaxation metadata, rejects any mode below the negative
frequency threshold, and rejects more than three near-zero translational modes.
An old archive produced with the zero-clipped convention must be recomputed.

If a loaded quench reaches the force threshold but its signed spectrum is
unstable, refine that same minimum without repeating the empty calculation:

```bash
./properties/submit_heat_capacity.sh --model pet-mad --loading 100 \
  --source-temperature 300 --replicas 1 --skip-empty \
  --continue-loaded --overwrite --optimizer fire \
  --fmax 0.005 --relax-steps 4000 --cv-temperatures 100:500:100 --dry-run
```

Remove `--dry-run` after checking that the relaxation input and output are the
same existing loaded minimum. The overwrite is intentional for the canonical
loaded minimum and Hessian; it does not touch the empty reference.
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

## 5. Assemble the hybrid heat capacity

After every requested temperature/replica and loaded Hessian is complete:

```bash
./properties/submit_hybrid_analysis.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1 --dry-run
./properties/submit_hybrid_analysis.sh --model pet-mad --loading 100 \
  --temperatures 100,200,300,400,500 --replicas 1
```

The analysis differentiates replica-averaged classical NPT enthalpy and adds
the loaded harmonic quantum-minus-classical correction. It writes NPZ, CSV,
and JSON provenance below `output/hybrid/<model>/<loading>ch4/`.

## Troubleshooting

- Missing generated TOMLs: rerun
  `python -m mof_heat_capacity.protocols.loaded` with the same model, loading,
  temperatures, and replica count.
- Missing stress output: stop; NPT is not valid with that exported model.
- CUDA initialization failure: recreate the pinned environment and confirm the
  Slurm job has one V100 allocation.
- Relaxation has not converged or has imaginary modes: do not silently discard
  them; improve the minimum or convergence settings and rerun.
- Noisy enthalpy derivative: extend sampling, increase independent replicas,
  or revisit the temperature spacing as a scientific convergence parameter.
- Existing outputs: choose a new campaign directory unless replacement is
  intentional; only then use the explicit rerun/overwrite option.
