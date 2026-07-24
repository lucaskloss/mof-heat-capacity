# MOF-5 PET-MAD molecular-dynamics smoke test

This directory is the MD starting point for a future automatic-differentiation
heat-capacity workflow.  It deliberately performs **only** a short classical
NVT trajectory.  No finite-difference estimator, Hessian, eigenvalue, or heat
capacity is calculated here yet.

```text
periodic MOF-5 structure
        |
        v
ASE: Langevin NVT integration
        |
        v
metatomic-ase: PET-MAD v1.5 exported model on CUDA
```

## Inputs

`models/pet-mad-1.5-s_40nn_nostress.ckpt` is the supplied PET-MAD v1.5
checkpoint.  On its first use, `run.py` exports it to a local `.pt` file
through `upet.save_upet`; `metatomic-ase` loads that exported file directly.

A MOF-5 crystal structure is intentionally not bundled.  Provide a vetted,
fully periodic CIF, PDB, or other ASE-readable structure with C, H, O, and Zn
only.  It is best to start with the 106-atom conventional/primitive cell whose
cell and activation state are documented alongside the source.  The script
does not claim that any arbitrary C/H/O/Zn structure is MOF-5; it only checks
the periodic cell and element set.

`data/mof5-test.cif` is included as a tiny synthetic four-atom input for
checking file parsing and input generation.  It is not a physical MOF-5
structure and must not be used for scientific MD or heat-capacity results.

## Run the smoke test

Use the repository environment with `metatomic-ase` installed.

```bash
conda activate mof-heat-capacity
python run.py --rerun
```

The default structure is `data/mof5.cif`.  Use `--structure` only when you
want to run a different structure.

The defaults are intentionally lightweight: ten steps, a 0.25 fs timestep,
and CUDA model evaluation. The run evaluates energy and forces directly with
`metatomic-ase`, then uses ASE's Langevin integrator. It writes
`mof5-ase-smoke.traj` and `mof5-ase-smoke.log` under `output/`.

CUDA model evaluation is the default on this machine.  Use `--device cpu` if
you need a CPU-only run.

Use `--device cpu` for a CPU test. Properties, logs, and trajectories are
written below `output/`.

For a longer exploratory trajectory, change `--steps` only after inspecting a
successful smoke test.  Converge timestep, equilibration, trajectory length,
thermostat, model applicability, and structure stability before interpreting
the energy or any eventual Hessian-based heat capacity.

## Analyze a trajectory

After a run, start JupyterLab from this directory and open `visualize.ipynb`:

```bash
jupyter lab
```

The notebook automatically selects the newest ASE trajectory in `output/` and
its matching log. It plots potential/total energy, temperature, and the maximum
atomic force over time. It also opens every stored trajectory frame in the
Chemiscope 3D widget and saves a portable `output/<trajectory>.json.gz` dataset
that can be opened at <https://chemiscope.org>.
