# SADMOF Hessians and harmonic heat capacity

This guide explains how this project uses SADMOF, PET-JAX, JAX, and asdex to
compute harmonic vibrational properties. It is intended as a map of the code
and a debugging guide, not as an alternative command-line workflow. Submit
the supported jobs through
[`scripts/properties/submit_heat_capacity.sh`](../scripts/properties/submit_heat_capacity.sh),
as documented in
[`scripts/properties/README.md`](../scripts/properties/README.md).

## What each library does

| Component | Responsibility |
| --- | --- |
| This repository | Selects structures, relaxes them, calls SADMOF, preserves signed frequencies, validates minima, stores provenance, and assembles the hybrid heat capacity. |
| SADMOF | Adapts PET models to JAX, describes the Hessian sparsity pattern, reconstructs a sparse Hessian, and evaluates harmonic heat capacity. |
| PET-JAX | Loads and evaluates the JAX version of the PET potential. |
| JAX | Differentiates the PET energy and compiles the Hessian calculation for the GPU. |
| asdex | Colors the Hessian sparsity pattern and reconstructs the sparse matrix from compressed Hessian-vector products. |
| ASE | Stores atomic structures, performs fixed-cell relaxation, supplies masses and units, and reads/writes structure files. |

SADMOF does **not** decide which thermal structure is scientifically valid,
relax the structure, prove that it is a minimum, or assemble the final hybrid
$C_P$. Those responsibilities remain in this repository.

## End-to-end data flow

```text
final loaded MD structure                     equilibrated empty MOF
              |                                        |
              +------ fixed-cell MLIP relaxation ------+
                                     |
                            optimized ASE structure
                                     |
                     PET checkpoint -> PET-JAX model
                                     |
                 PET neighbour graph and sparse pattern
                                     |
                    JAX + asdex sparse AD Hessian
                                     |
                mass weighting and eigendecomposition
                                     |
                     signed frequencies and harmonic Cv
                                     |
            validation + quantum-minus-classical correction
                                     |
          classical NPT d<Etot + Pext V>/dT + correction
                                     |
                         approximate hybrid Cp
```

The production Hessian is computed at a fixed-cell local minimum, not at a raw
finite-temperature MD frame. A thermal frame generally has nonzero forces and
can produce imaginary modes that describe the local slope rather than a
meaningful normal-mode spectrum.

## Installation and model conversion

[`scripts/setup/install_sadmof.sh`](../scripts/setup/install_sadmof.sh) installs
an editable SADMOF checkout and pinned copies of its direct dependencies. In
particular, this project pins asdex, PET-JAX, JAX, `jaxlib`, and the CUDA 12 JAX
plugins so that they remain compatible with Izar's V100 GPUs.

The MD model and Hessian model are two representations of the same potential:

- the exported metatomic/PyTorch model is used for MD and relaxation;
- the PET-JAX checkpoint directory is used for automatic differentiation.

A PET-JAX directory contains `model.msgpack` and `metadata.yaml`. If it does
not exist, `ensure_jax_checkpoint()` in
[`harmonic.py`](../mof_heat_capacity/analysis/harmonic.py) converts the original
PET checkpoint. A filesystem lock prevents simultaneous Slurm jobs from
performing the same conversion. Checkpoint, exported-model, and JAX-conversion
paths must always refer to the same trained potential.

## How the sparse Hessian is constructed

For Cartesian coordinates $\mathbf R$, the force-constant matrix is

$$H_{i\alpha,j\beta} = \frac{\partial^2 E}{\partial R_{i\alpha}\,\partial R_{j\beta}}. $$

The implementation in
[`compute_frame_hessian()`](../mof_heat_capacity/analysis/harmonic.py) follows
this sequence:

1. `sadmof.models.pet.atoms_to_inputs()` converts an ASE structure into PET
   positions, cell, species, masks, and neighbour graphs. PET has both a raw
   cutoff graph and a tighter selected adaptive-neighbour graph.
2. `sadmof.sparse.sparsity_pattern()` expands graph reachability to the chosen
   number of `hops` and then expands each atom pair to a Cartesian $3\times3$
   block.
3. `asdex.hessian_coloring_from_sparsity()` groups independent Hessian columns
   into colors. One compressed differentiation can recover multiple columns.
4. `sadmof.models.pet.get_energy_fn(..., no_shadow=True)` constructs the scalar
   PET total-energy function used by automatic differentiation.
5. `sadmof.sparse.get_hessian_fn()` builds the colored Hessian-vector-product
   calculation. JAX compiles it with `jax.jit` and runs it on the GPU.
6. The returned JAX sparse matrix is densified, and padding atoms introduced
   by PET-JAX are removed. The result has shape $(3N,3N)$ for the real atoms.

This is sparse **reconstruction**, not a different physical definition of the
Hessian. It is exact only when the supplied sparsity pattern includes every
nonzero coupling relevant to the model.

At library level, the central calls look like this simplified sketch:

```python
import asdex
import jax

from sadmof.models.pet import atoms_to_inputs, get_energy_fn, load_pet
from sadmof.sparse import get_hessian_fn, sparsity_pattern

model, params, metadata = load_pet(jax_checkpoint, dtype="float64")
positions, cell, graph = atoms_to_inputs(atoms, model, metadata)
pattern = sparsity_pattern(
    {
        "centers": graph["sel_centers"],
        "others": graph["sel_others"],
        "atomic_numbers": graph["atomic_numbers"],
    },
    hops=3,
)
coloring = asdex.hessian_coloring_from_sparsity(
    pattern, mode="fwd_over_rev"
)
energy = get_energy_fn(model, metadata, no_shadow=True)
hessian_fn = jax.jit(
    get_hessian_fn(energy, coloring, chunk_size=1, remat=True)
)
sparse_hessian = jax.block_until_ready(
    hessian_fn(params, positions, cell, graph)
)
```

The project wrapper adds precision selection, conversion locking, padding
removal, acoustic-sum-rule enforcement, signed frequencies, output metadata,
and validation. For production, use the Bash workflow rather than copying this
sketch into a separate script.

### Parameters that control the calculation

`hops`
: Controls how far coupling is propagated through the neighbour graph. Too
  few hops omit Hessian blocks and change frequencies; more hops increase the
  color count, runtime, and memory. SADMOF's source identifies seven hops as
  the structurally exact pattern for PET-MAD-S with three message-passing
  layers. This project's configured value of three is therefore a truncation
  and must be treated as a convergence parameter, not as a universal exact
  setting. Other PET architectures require their own validation.

`chunk_size`
: Number of colors evaluated together. It changes peak GPU memory and
  throughput, but should not change a converged numerical result. A larger
  value can be faster and can also cause an out-of-memory failure.

`dtype`
: `float32` uses less memory; `float64` promotes PET parameters and enables JAX
  64-bit mode. Compare frequencies and the final correction before deciding
  that single precision is adequate.

`remat`
: Enables JAX checkpointing (`jax.checkpoint`). It saves intermediate memory
  by recomputing parts of the energy evaluation, trading time for memory.

`no_shadow=True`
: Stops derivatives through PET's adaptive cutoff. This confines coupling to
  the selected graph and makes the sparse path possible. Shadow derivatives
  extend beyond that graph and require a dense treatment; this repository
  rejects `shadow=true` for the sparse workflow.

Changing any of these settings to make a job fit is a scientific/numerical
change. Preserve the old output under a separate Hessian tag and compare it.

## From the Hessian to signed frequencies

The project symmetrizes the dense real-atom Hessian and applies a
force-constant acoustic sum rule. The latter forces a uniform translation to
have zero restoring force. It then constructs the mass-weighted dynamical
matrix

$$D_{i\alpha,j\beta} = \frac{H_{i\alpha,j\beta}}{\sqrt{m_i m_j}}, $$

diagonalizes $D$, and converts its eigenvalues $\omega^2$ to frequencies
in cm$^{-1}$.

This repository deliberately differs from SADMOF's compatibility helper for
unstable modes. SADMOF's default observable maps negative eigenvalues to zero;
[`signed_frequencies_from_hessian()`](../mof_heat_capacity/analysis/harmonic.py)
stores them as negative frequencies. Keeping the sign is essential: otherwise
an unconverged structure or saddle point could look like a valid minimum.

The normal production expectation is:

- no frequency below $-1$ cm$^{-1}$;
- no more than three modes within $\pm1$ cm$^{-1}$, corresponding to
  translations;
- all remaining modes positive.

The final hybrid analysis enforces these conditions. Do not remove imaginary
modes merely to obtain a smooth heat-capacity curve; return to the relaxation,
precision, sparsity, or model consistency instead.

## Harmonic heat capacity

For each retained positive mode, SADMOF evaluates the quantum harmonic
oscillator contribution

$$C_{V,k}^{\mathrm{qn}}(T) = k_B\frac{x_k^2 e^{-x_k}}{(1-e^{-x_k})^2}, \qquad x_k=\frac{h c\tilde\nu_k}{k_B T}. $$

Summing the modes and dividing by the total cell mass gives gravimetric
$C_V$ in $J g^{-1} K^{-1}$. High-frequency modes freeze out at low
temperature; each active mode approaches $k_B$ in the classical high-
temperature limit.

The `cv_J_per_gK` stored in a Hessian archive is a useful direct harmonic
diagnostic. It is not inserted unchanged into the final result. Hybrid
assembly reloads the signed frequencies, applies its validated common mode
mask, and recomputes both terms

$$\Delta C^{\mathrm{har}}(T) = C_{V,\mathrm{qn}}^{\mathrm{har}}(T) - C_{V,\mathrm{cl}}^{\mathrm{har}}. $$

There are intentionally two thresholds to recognize when comparing files.
SADMOF's direct diagnostic curve drops modes below its default
$10^{-3}$ cm$^{-1}$. Final hybrid assembly uses the stricter project
threshold of 1 cm$^{-1}$, rejects negative modes below that threshold, and
uses the same retained positive modes for both the quantum and classical
terms. The final correction is therefore the authoritative thermodynamic
quantity.

It then combines this correction with the loaded classical NPT enthalpy
derivative:

$$C_P^{\mathrm{approx}}(T) = \frac{d\langle E_{\mathrm{tot}}+P_{\mathrm{ext}}V\rangle}{dT} + \Delta C^{\mathrm{har}}(T). $$

The loaded Hessians determine the correction used in this expression. The
empty-MOF Hessian is a separate reference and pipeline check; it is not
subtracted from the loaded spectrum in the current hybrid formula.

## Files and provenance

For each selected loaded temperature and replica, the property workflow
normally creates:

- `minima/TEMPERATUREK/repNN/optimized.extxyz`: the optimized Hessian input;
- `minima/TEMPERATUREK/repNN/optimized.optimizer.traj`: optimization history;
- `minima/TEMPERATUREK/repNN/optimized.relax.log`: the optimizer log;
- `minima/TEMPERATUREK/repNN/optimized.relax.json`: fixed-cell flag, force target, final
  force, energies, optimizer, and step count;
- `hessians/TEMPERATUREK/repNN/hessian.npz`: signed frequencies and harmonic $C_V$,
  temperature grid, model path, frame selection, and Hessian settings.

The empty reference uses the same concise role filenames below `0ch4/`, without
a redundant temperature or replica directory. The final hybrid NPZ, CSV, and JSON record the classical term, the
harmonic correction, uncertainties, and the Hessian provenance consumed. At
each classical-MD temperature, hybrid assembly uses the Hessian obtained by
quenching the final structure from that same temperature.

Important Hessian-archive fields are:

| Field | Meaning |
| --- | --- |
| `frequencies_cm1` | Signed spectrum, normally shape `(1, 3N)` for one optimized structure. |
| `cv_J_per_gK` | Direct SADMOF quantum-harmonic diagnostic curve. |
| `temperatures_K` | Temperature grid corresponding to the diagnostic curve. |
| `trajectory` | Optimized structure used as the Hessian input. |
| `checkpoint` | PET-JAX checkpoint directory actually loaded. |
| `metadata` | Precision, hops, chunk size, rematerialization, frequency convention, ASR, and diagnostic threshold. |

## A practical debugging order

When a Hessian job or result looks wrong, check the pipeline in this order:

1. **GPU preflight:** confirm the automatic preflight log sees a JAX GPU and a
   CUDA-compatible PyTorch build. A CPU JAX device is an environment failure,
   not a reason to increase wall time.
2. **Model identity:** confirm the PyTorch checkpoint, exported model, and
   PET-JAX conversion belong to the same model. A job that runs with mismatched
   weights is still scientifically invalid.
3. **Relaxation provenance:** inspect `.relax.json` and the optimizer log.
   Confirm `fixed_cell=true`, finite energy/forces, and final maximum force no
   larger than the requested `fmax`.
4. **Structure identity:** check atom count, species, cell, total mass, and the
   selected replica. Hybrid assembly rejects a mass mismatch but cannot infer
   every possible structure-selection mistake.
5. **Spectrum:** count imaginary and near-zero modes and inspect the minimum
   frequency. Large negative modes usually indicate a saddle, poor relaxation,
   inconsistent model, or an under-converged Hessian.
6. **Numerical convergence:** compare precision, hops, chunk size, and remat
   using separately tagged outputs. `chunk_size` and `remat` should affect
   resources, while converged frequencies should be insensitive to them.
7. **Thermodynamic convergence:** compare the quantum correction across
   independent loaded minima. One numerically clean Hessian does not quantify
   minimum-to-minimum variability.

### Common failures

| Symptom | Likely cause or next check |
| --- | --- |
| JAX reports only CPU devices | CUDA plugin/JAX mismatch, no GPU allocation, or environment activation failure. |
| PET-JAX conversion is repeatedly regenerated | Incomplete `model.msgpack`/`metadata.yaml`, wrong path, or interrupted conversion. |
| GPU out of memory during compilation/evaluation | Reduce `chunk_size` first; consider `remat`; do not silently reduce `hops` or precision. |
| Job spends a long time before producing output | JAX compilation and sparsity coloring occur before evaluation; compare the Slurm log and a representative job before changing wall time. |
| Relaxation succeeds but modes are imaginary | Tighten or continue the same minimum, verify model identity, and test Hessian convergence. |
| More than three near-zero modes | Floppy/unstable structure, insufficient sparsity range, poor numerical precision, or an inappropriate minimum. |
| Direct harmonic curve looks plausible but hybrid assembly rejects it | The archive may lack signed frequencies, relaxation provenance, matching mass, or an acceptable mode spectrum. |
| Results change with `chunk_size` | Numerical instability, precision problems, or a library/version inconsistency; chunking should not change the mathematical Hessian. |

## Source map

The most useful code to read is:

- [`mof_heat_capacity/analysis/harmonic.py`](../mof_heat_capacity/analysis/harmonic.py):
  project wrapper, model conversion, sparse call path, signed frequencies, and
  Hessian archive;
- [`mof_heat_capacity/structures/relax.py`](../mof_heat_capacity/structures/relax.py):
  fixed-cell minimization and provenance;
- [`mof_heat_capacity/analysis/hybrid.py`](../mof_heat_capacity/analysis/hybrid.py):
  spectral validation and quantum-minus-classical correction;
- `external/sadmof/src/sadmof/models/pet/`: PET loading, inputs, and energy;
- `external/sadmof/src/sadmof/sparse/`: graph sparsity and colored Hessian;
- `external/sadmof/src/sadmof/observables/phonons.py`: SADMOF frequency and
  harmonic-$C_V$ helpers;
- `external/sadmof/deps/asdex/`: coloring and sparse reconstruction;
- `external/sadmof/deps/pet-jax/`: PET model implementation and checkpoint
  conversion.

The `external/` tree is generated by the installer and is intentionally not
versioned with this project. The three `mof_heat_capacity/` modules above are
the stable project-level starting points when debugging or reviewing results.
