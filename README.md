# MOF-5 PET-MAD MD and harmonic heat capacity

This directory contains two related but scientifically distinct workflows:

- `run.py` performs a short classical NVT trajectory with ASE and the
  metatomic PET-MAD export.
- `heat_capacity.py` post-processes selected structures with SADMOF to compute
  automatic-differentiation Hessians, harmonic frequencies, and constant-volume
  heat capacities.

The MD trajectory is useful for stability checks and selecting candidate
geometries. It is not itself the heat-capacity estimator used here.

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

`data/mof5.cif` is the bundled default structure. Verify its source, cell,
activation state, and atom count before scientific use. The loader only checks
that the structure is periodic and contains C, H, O, and Zn; passing these
checks does not establish that an arbitrary file is a valid MOF-5 structure.

## Install SADMOF for Hessian heat capacities

SADMOF is research software rather than a released, stable application. Its
`pyproject.toml` declares `asdex`, `pet-jax`, and `marathon-train`; all three
are developed as editable source checkouts. The top-level SADMOF README
currently mentions only the first two, but the `marathon` checkout is also
required.

Use HTTPS and install everything in the same conda environment as this project:

```bash
conda env create -f environment.yml
conda activate mof-heat-capacity

python -m pip install --upgrade pip
python -m pip install jax jaxlib flax e3nn-jax jaxtyping scipy numba

mkdir -p external
git clone https://github.com/sirmarcel/sadmof-work.git external/sadmof
git clone https://github.com/adrhill/asdex.git external/sadmof/deps/asdex
git clone https://github.com/lab-cosmo/pet-jax.git external/sadmof/deps/pet-jax
git clone https://github.com/sirmarcel/marathon.git external/sadmof/deps/marathon
git -C external/sadmof checkout e351103
git -C external/sadmof/deps/asdex checkout 3209417
git -C external/sadmof/deps/pet-jax checkout f33ddf6
git -C external/sadmof/deps/marathon checkout c316527

python -m pip install -e external/sadmof/deps/asdex
python -m pip install -e external/sadmof/deps/marathon
python -m pip install -e external/sadmof/deps/pet-jax
python -m pip install --no-deps -e external/sadmof
```

The PET adapter uses internal `pet-jax` functions, so independently cloning the
latest commit of each repository can produce an API mismatch. Record and pin a
known-working set:

```bash
git -C external/sadmof rev-parse HEAD
git -C external/sadmof/deps/asdex rev-parse HEAD
git -C external/sadmof/deps/pet-jax rev-parse HEAD
git -C external/sadmof/deps/marathon rev-parse HEAD
```

The known-working checkout is `sadmof@e351103`, `asdex@3209417`,
`pet-jax@f33ddf6`, and `marathon@c316527`. If an import such as
`petjax.select._selection` fails, the SADMOF and `pet-jax` revisions do not
match. Use the revisions recorded by SADMOF instead of silently patching the
scientific adapter.

JAX GPU support is independent of PyTorch/metatomic CUDA support. Check it with:

```bash
python -c "import jax; print(jax.devices())"
```

A CPU-only `jaxlib` can run small tests, but a production MOF Hessian normally
needs CUDA-enabled JAX and substantial accelerator memory.

## How the SADMOF library works

SADMOF calculates position Hessians of atomistic machine-learning potentials
and converts them into harmonic constant-volume heat capacities:

```text
ASE Atoms
   |
   v
model-specific atoms_to_inputs
   |
   +-- positions (N, 3)  <---- differentiated
   +-- cell (3, 3)       <---- passed to the model
   `-- graph             <---- fixed topology, periodic shifts, and masks
            |
            v
energy_fn(params, positions, cell, graph) -> (scalar energy, auxiliary data)
            |
            +-- dense HVP sweep
            `-- graph pattern -> coloring -> sparse HVP reconstruction
                                |
                                v
                    Cartesian Hessian in eV/A^2
                                |
                                v
              mass weighting -> frequencies in cm^-1
                                |
                                v
                quantum-harmonic Cv(T) in J/(g K)
```

Neighbor topology is fixed during one Hessian evaluation. Positions remain JAX
variables, while neighbor indices, masks, atomic numbers, and periodic cell
shifts live in `graph`. All model adapters expose the effective contract
`energy_fn(params, positions, cell, graph) -> (energy, auxiliary)`.

### Source package map

The important code is under `external/sadmof/src/sadmof/`:

| Module | Role and principal API |
|---|---|
| `models/` | Model adapters and ASE calculators. `get_calculator()` selects MACE, MACE+D3, or PET. |
| `models/pet/` | Loads pet-jax checkpoints, builds raw and selected PET graphs, and exposes `get_energy_fn()`. |
| `models/mace/` | JAX MACE model, padded ASE inputs, and a Verlet-cached ASE calculator. |
| `models/d3/` | Differentiable JAX D3(BJ) with energy and coordination neighbor lists. |
| `dense/` | `get_dense_hessian_fn()`: one HVP per Cartesian coordinate. |
| `sparse/` | Graph reachability, asdex coloring, and sparse reconstruction. |
| `observables/` | Pure-NumPy Hessian-to-frequency and frequency-to-Cv functions. |
| `relax.py` | Joint cell-and-position relaxation with ASE `FrechetCellFilter`. |
| `scripts/relax.py` | Installed `sadmof-relax` command. |
| `provenance.py` | Git revision, dirty state, and package-version records. |
| `utils.py` | Shared JAX-pytree precision conversion. |

The top-level package exports only its version. Import functionality from the
relevant subpackage:

```python
from sadmof.dense import get_dense_hessian_fn
from sadmof.sparse import get_hessian_fn, sparsity_pattern
from sadmof.observables import cv_from_hessian
from sadmof.models.pet import atoms_to_inputs, get_energy_fn, load_pet
```

### Model adapters

SADMOF supports three energy components:

- **PET:** `load_pet()` reads a pet-jax directory containing `model.msgpack`
  and `metadata.yaml`. `atoms_to_inputs()` constructs both the raw neighbor
  ball and PET's adaptively selected adjacency. `get_energy_fn()` keeps
  adaptive selection inside JAX. Sparse PET requires `no_shadow=True`; shadow
  derivatives add couplings beyond the selected graph and require a dense
  Hessian.
- **MACE:** checkpoints are converted `model.yaml`/`model.msgpack` pairs. The
  adapter builds a padded fixed-radius graph. Position-independent per-element
  reference energies do not contribute to its Hessian.
- **D3(BJ):** a differentiable correction with separate dispersion-energy and
  coordination-number half-neighbor lists. It is treated as a separate dense
  Hessian. For MACE+D3, compute both Hessians and add them before
  diagonalization. Do not add D3 to PET; that would double count dispersion.

The ASE calculators cache neighbor graphs with a Verlet skin and rebuild them
when atoms or the cell invalidate the reference. That layer is used for
relaxation and inference. Hessian code calls the JAX energy directly.

### Dense Hessian path

`get_dense_hessian_fn()` uses forward-over-reverse Hessian-vector products. For
`N` padded atoms it evaluates `3N` HVPs. One-hot tangent vectors are created
inside JAX, so a full identity matrix is never materialized.

`chunk_size` controls parallel HVPs. Larger chunks can be faster but consume
more memory. `remat=True` applies `jax.checkpoint`, trading recomputation for a
lower memory peak. SADMOF's benchmarks found roughly a 21% compute penalty, but
rematerialization was necessary for many systems above about 100 atoms on 24 GB
devices. Dense mode is the reference path and is preferable when the graph
pattern is already close to full.

### Sparse Hessian path

The sparse path has three stages:

1. `sparsity_pattern(graph, hops)` builds atom reachability
   `(I + A)^hops` and expands every atom pair to a `3 x 3`
   Cartesian block.
2. `asdex.hessian_coloring_from_sparsity(..., mode="fwd_over_rev")` groups
   independent Hessian columns into colors.
3. `get_hessian_fn()` evaluates one HVP per color and reconstructs a JAX sparse
   `BCOO` Hessian.

The exact architectural support established in `play/hops/` is:

| Model | Message-passing layers `L` | Exact Hessian hops |
|---|---:|---:|
| MACE with node readout | `L` | `2L` |
| PET with edge readout | `L` | `2L + 1` |
| MACE-MP-0 medium | 2 | 4 |
| PET-MAD-XS | 2 | 5 |
| PET-MAD-S | 3 | 7 |

Fewer hops deliberately truncate long-range mixed derivatives. Trained
couplings often decay by about an order of magnitude per hop, but the error has
to be measured in the final observable. A truncated sparse Hessian is an
approximation, not merely a faster representation of the exact matrix.

Sparse reconstruction helps only while the pattern remains meaningfully
sparse. Production examples calculate real-atom pattern fill first and usually
skip sparse conditions above 0.7 fill; near-dense sparse work pays coloring
overhead without eliminating much model work.

### Hessian to frequencies and heat capacity

`observables/` has no JAX or file-format dependency. Its one-call interface is:

```python
frequencies_cm1, cv = cv_from_hessian(
    hessian_eV_per_A2,
    atoms.get_masses(),
    temperatures_K,
    enforce_asr=True,
)
```

It symmetrizes the Hessian, optionally imposes the translational acoustic sum
rule, mass-weights it as `H_ij / sqrt(m_i m_j)`, diagonalizes the dynamical
matrix, and converts eigenvalues to frequencies in `cm^-1`. Each retained mode
contributes `k_B [x / (2 sinh(x/2))]^2`, with `x = h nu / (k_B T)`. Dividing by
total atomic mass gives gravimetric `C_V` in J/(g K).

Frequencies below `1e-3 cm^-1` are excluded by default. Negative eigenvalues
become zero-frequency modes and are excluded. This avoids treating imaginary
modes as real vibrations, but they must still be inspected and reported because
they commonly indicate an unconverged or unstable geometry.

`freq_scale` applies a documented empirical frequency correction. SADMOF
mentions 1.181 for one Gonnheimer MACE-MP-0 comparison; do not apply this
implicitly to PET or another protocol.

### Relaxation before a Hessian

A conventional harmonic calculation should use a stationary, model-matched
geometry, not an arbitrary thermally displaced MD frame. `sadmof.relax.relax()`
optimizes positions and cell together through `FrechetCellFilter` and returns
convergence, time, steps, and final filter-gradient information.

| Optimizer | Intended use |
|---|---|
| `bfgs` | Robust dense inverse-Hessian method for small systems; `O(N^3)` work becomes prohibitive for large MOFs. |
| `lbfgs` | Fast limited-memory method, but can overshoot from a poor initial structure. |
| `lbfgs-ls` | Limited-memory method with a line search; robust default for large or unrelaxed cells. |

The SADMOF default is `fmax=0.005` eV/A. Cell relaxation needs stress and is
normally run in float64:

```bash
sadmof-relax data/mof5.cif \
  --calculator pet \
  --checkpoint models/pet-mad-1.5-s_40nn_jax \
  --optimizer lbfgs-ls \
  --dtype float64 \
  --fmax 0.005 \
  --trajectory \
  --output-dir output/relaxed-pet
```

This writes `relaxed.xyz`, optionally `optimization.traj`, and
`relax_summary.json` with provenance.

## What the example folders demonstrate

SADMOF uses `work/` for official workflows and `play/` for exploratory
benchmarks. They are experiment drivers rather than stable library interfaces.

- `work/relax-goennheimer/` fans out model/optimizer/precision combinations
  over 233 structures, resumes around completed outputs, and aggregates
  summaries. Its literature reproduction uses MACE-MP-0+D3, BFGS, float64,
  `FrechetCellFilter`, and `fmax=0.005` eV/A.
- `work/relax-big-mofs/` compares optimizers on MOF-177 and MIL-101. It shows
  why dense BFGS scaling dominates for thousands of atoms and why `lbfgs-ls`
  is the practical large-system default.
- `work/cv-ref-goennheimer/` is the simplest full reference chain: relaxed
  primitive -> dense MACE Hessian + dense D3 Hessian -> real-block sum ->
  `cv_from_hessian()` -> one NPZ per structure. Its 13-240 atom primitives are
  effectively dense at these interaction ranges.
- `play/hops/` uses controlled chain graphs to establish exact model-dependent
  hop counts and distinguish exact support from useful truncation.
- `play/initial-hessian-benchmark/` compares cold and warm dense/sparse costs,
  validates against `jax.hessian`, records pattern/color/model statistics, and
  shows why each structure shape should run in its own process.
- `play/hessians-goennheimer/` and `play/hessians-big-mofs/` are the most
  production-like drivers. They select model-matched relaxed structures, tile
  converged supercells, cache colorings and raw Hessians, skip saturated sparse
  patterns, isolate conditions in separate processes, resume from artifacts,
  and write configuration, timings, environment, and provenance.

The production-style output layout is:

```text
output/<structure>/<condition>/
    supercell.xyz
    coloring.npz          # sparse only
    hessian_raw.npy
    observables.npz
    results.yaml
```

`results.yaml` is the completion marker. The raw Hessian is cached before
observables so a stopped run can restart without recomputing it. D3 is computed
once per structure and combined with cached MACE Hessians on the host.

### Periodic supercells

The examples do not assume a primitive cell is sufficient. For MACE-MP-0
medium they choose the smallest diagonal supercell whose perpendicular widths
are at least `2 L r_c = 24 A`, using `L = 2` and `r_c = 6 A`. This prevents an atom
from coupling to its own periodic image within the Hessian support.

PET requires its own cell/hop convergence study. Sharing a MACE-defined
supercell across models is useful for controlled comparisons, but it is not a
universal PET convergence rule.

## How this project uses SADMOF

`heat_capacity.py` implements the PET sparse path:

1. Read selected ASE trajectory frames.
2. Convert the metatrain PET-MAD checkpoint to a pet-jax directory if needed.
3. Load PET parameters and metadata.
4. Build the raw PET graph and selected adaptive adjacency.
5. Build and color the selected-graph reachability pattern.
6. Evaluate and densify the real-atom Hessian block.
7. Apply the acoustic sum rule and compute frequencies and `C_V(T)`.
8. Save frame indices, temperatures, spectra, `C_V`, and configuration to NPZ.

A memory-conscious diagnostic command is:

```bash
conda activate mof-heat-capacity

python heat_capacity.py \
  --trajectory output/long-run/mof5-ase-100.traj \
  --frames 1 \
  --temperatures 100:500:10 \
  --dtype float32 \
  --chunk-size 1 \
  --hops 3 \
  --remat \
  --output output/heat-capacity-test.npz
```

Read the result with:

```bash
python - <<'PY'
import numpy as np

data = np.load("output/heat-capacity-test.npz")
print(data["frame_indices"])
print(data["temperatures_K"])
print(data["cv_J_per_gK"])
PY
```

`--frames` accepts a count, `all`, or comma-separated indices; `--stride`
selects every nth frame. Multiple displaced frames characterize curvature
variation. Averaging their harmonic curves is not automatically a rigorous
anharmonic heat capacity.

The `hops=3` command is a deliberately truncated diagnostic. PET-MAD-S has
three message-passing layers and needs seven hops for the exact no-shadow
Hessian. On this 424-atom cell, the seven-hop pattern attempted a roughly
150 GB CPU allocation. Before production, relax the structure, converge the
cell and hop count, inspect pattern fill and color count, use CUDA-enabled JAX,
and compare truncated `C_V` with a feasible dense or higher-hop reference.

## Scientific and reproducibility checklist

Record and converge all of the following before reporting a heat capacity:

- structure source, activation state, atom ordering, and periodic cell;
- optimizer, force/stress threshold, precision, and geometry convergence;
- PET-MAD version and pet-jax conversion revision;
- primitive versus supercell choice and cell-width convergence criterion;
- dense/sparse mode, hops, fill, colors, chunk size, rematerialization, dtype,
  and shadow/no-shadow setting;
- acoustic-sum-rule choice, excluded/imaginary modes, threshold, and any
  frequency scaling;
- temperature grid, units, device, wall time, and peak memory;
- SADMOF, asdex, pet-jax, marathon, JAX, and JAXLIB versions and Git revisions.

The result is a harmonic constant-volume `C_V`. It does not include thermal
expansion (`C_P - C_V`), explicit anharmonicity, defects, adsorbates, model
error, or uncertainty from an out-of-domain structure.

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
