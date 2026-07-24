# MOF-5 PET-MAD molecular-dynamics smoke test

This directory is the MD starting point for a future automatic-differentiation
heat-capacity workflow.  It deliberately performs **only** a short classical
NVT trajectory.  No finite-difference estimator, Hessian, eigenvalue, or heat
capacity is calculated here yet.

```text
periodic MOF-5 structure
        |
        v
i-PI: one-bead NVT integration
        | Unix-domain socket
        v
LAMMPS: fix ipi force client
        |
        v
metatomic: PET-MAD v1.5 exported model
```

## Inputs

`models/pet-mad-1.5-s_40nn_nostress.ckpt` is the supplied PET-MAD v1.5
checkpoint.  On its first use, `mof-md.py` exports it to a local `.pt` file
through `upet.save_upet`; LAMMPS loads that exported file through
`pair_style metatomic`.

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

Use the repository's metatomic-enabled environment.  Plain LAMMPS does not
provide `pair_style metatomic`.

```bash
conda activate mof-heat-capacity
python work/mof-heat-capacity/mof-md.py --rerun
```

The default structure is `data/mof5.cif`.  Use `--structure` only when you
want to run a different structure.

The defaults are intentionally lightweight: one bead, one force client, ten
steps, a 0.25 fs timestep, and CUDA model evaluation.  They only test the
structure reader, checkpoint export, i-PI socket, LAMMPS/metatomic type map,
and short integration.  They cannot establish stability or yield a meaningful
thermodynamic quantity.

CUDA model evaluation is the default on this machine.  Use `--device cpu` if
you need a CPU-only run.

The fixed LAMMPS mapping is:

```text
LAMMPS type 1: C  -> 6
LAMMPS type 2: H  -> 1
LAMMPS type 3: O  -> 8
LAMMPS type 4: Zn -> 30
```

The four input files are kept below `data/` (or `--input-dir`): `mof5.pdb` and
`input.xml` for i-PI, plus `mof5.data` and `in.lmp` for LAMMPS.  Properties,
restart data, logs, and trajectories are written below `output/` (or
`--output-dir`).  If the structure or run settings change, regenerate the four
inputs explicitly with `--prepare-inputs`.

For a longer exploratory trajectory, change `--steps` only after inspecting a
successful smoke test.  Converge timestep, equilibration, trajectory length,
thermostat, model applicability, and structure stability before interpreting
the energy or any eventual Hessian-based heat capacity.
