# PET-JAX Hessian implementation

Use [README.md](README.md) for submission commands and job dependencies.
The scientific explanations and equations are in
[PET.md](../../docs/PET.md) and [SADMOF.md](../../docs/SADMOF.md).

## Model identity and conversion

MD and relaxation use the exported metatomic/PyTorch potential; Hessians use
its matching PET-JAX conversion. Keep the original checkpoint, export, and
conversion consistent. [models.py](../../mof_heat_capacity/models.py) exports
with `upet.save_upet`. `ensure_jax_checkpoint()` in
[harmonic.py](../../mof_heat_capacity/analysis/harmonic.py) creates
`model.msgpack` and `metadata.yaml`, with a filesystem lock preventing concurrent
conversion. Conversion transfers weights without retraining.

[install_sadmof.sh](../setup/install_sadmof.sh) installs SADMOF and pinned
ASDEX, PET-JAX, JAX, and CUDA dependencies; its PET-JAX revision is `ed0add4`.
The generated `external/` and model directories are excluded from Git.

The local PET-MAD and PET-SOL conversions recorded these architecture values
in `models/pet-mad-1.5-s_40nn_jax/metadata.yaml` and
`models/pet_sol-s-best_nostress_jax/metadata.yaml`. Check the selected model's
metadata when changing checkpoints; identical architecture does not imply
identical weights or training data.

| Metadata | Recorded value |
| --- | --- |
| `cutoff`, `cutoff_width` | 8.0 Å, 0.5 Å |
| `num_neighbors_adaptive` | 40 |
| `num_gnn_layers`, `num_attention_layers` | 3, 1 |
| `d_pet`, `num_heads`, `d_feedforward` | 256, 8, 512 |
| `d_node`, `d_head` | 1024, 256 |
| `num_species` | 102 embedding slots |

The inspected `metatrain` 2026.3.1 implementation differs from the original
PET paper: it embeds Cartesian components plus scalar distance, contracts
central features for attention, masks padded neighbors, and adds log-cutoff
weights to attention logits. Reverse-edge indices route outgoing tokens;
previous and new messages are averaged, and features across blocks feed
readouts. `d_head` is a readout width; each attention head has width 256/8=32.
These observations concern that installed version; the checkpoint and pinned
conversion determine compatibility.

## Calculation path

The reusable implementation stays in the importable package:

| Module | Responsibility |
| --- | --- |
| [harmonic.py](../../mof_heat_capacity/analysis/harmonic.py) | Conversion, compiled Hessians, signed frequencies, archives |
| [relax.py](../../mof_heat_capacity/structures/relax.py) | Fixed-cell minimization and provenance |
| [hybrid.py](../../mof_heat_capacity/analysis/hybrid.py) | Mode validation and quantum-minus-classical assembly |
| [llpr_merge.py](../../mof_heat_capacity/analysis/llpr_merge.py) | Validate and merge distributed member results |

`prepare_frame_hessian()` in `harmonic.py` performs the library calls:

1. `atoms_to_inputs()` creates padded positions, cell, masks, and neighbor graphs.
2. `sparsity_pattern()` uses `sel_centers`, `sel_others`, and `atomic_numbers`
   with the requested hop count.
3. `asdex.hessian_coloring_from_sparsity(..., mode="fwd_over_rev")` builds
   the symmetry-aware coloring.
4. `get_energy_fn(..., no_shadow=True)` stops adaptive-cutoff derivatives.
5. `get_hessian_fn(..., chunk_size=..., remat=...)` builds colored HVPs;
   `jax.jit` compiles their evaluation.

`evaluate_frame_hessian()` waits for completion, densifies the sparse result,
and removes padding to obtain the real-atom `(3N, 3N)` matrix. Postprocessing
symmetrizes it, enforces the force-constant acoustic sum rule, mass-weights,
and diagonalizes it. Signed frequencies retain negative eigenvalues as
negative frequencies rather than hiding unstable modes.

The stopped adaptive-cutoff derivatives define the differentiated potential.
Check agreement with MD forces; matching weights alone is insufficient.
`shadow=true` is rejected by the sparse workflow because those derivatives
introduce couplings outside the selected graph. The paper's atom-level coloring
optimization should be checked in the installed ASDEX version before assuming
it is used here.

## Numerical settings and mode validation

The [hybrid template](../../configs/mof5_100ch4_hybrid_npt.toml) specifies
`float32`, `hops=3`, `chunk_size=1`, `remat=true`, and `shadow=false`.

| Setting | Effect and check |
| --- | --- |
| `hops` | Pattern reach; three hops truncates three-layer PET-S's exact seven-hop pattern. Compare increasing hops and the final correction. |
| `dtype` | `float64` promotes parameters and enables JAX 64-bit mode; compare spectra and correction with `float32`. |
| `chunk_size` | Colors per batch; controls memory and throughput. It should not change the intended Hessian. |
| `remat` | JAX checkpointing recomputes intermediates to save memory. It should not change the intended Hessian. |

Preserve convergence trials under distinct Hessian tags. Reduce chunk size or
use rematerialization for memory limits before changing scientific precision
or pattern reach.

Production inputs are fixed-cell optimized minima. Hybrid validation rejects
frequencies below −1 cm⁻¹ and allows at most three modes within ±1 cm⁻¹.
The final correction uses the same retained positive modes in its quantum and
classical terms. The archive's direct SADMOF `cv_J_per_gK` diagnostic uses a
10⁻³ cm⁻¹ threshold; hybrid assembly recomputes with the project threshold
of 1 cm⁻¹. Use the assembled correction for the final result.

## Outputs and provenance

For each replica, the highest selected loaded temperature supplies one minimum
and spectrum reused across the evaluation grid. `SOURCEK` labels that source:

| Path | Contents |
| --- | --- |
| `minima/SOURCEK/repNN/optimized.extxyz` | Hessian geometry |
| `optimized.optimizer.traj`, `optimized.relax.log` beside it | Relaxation history |
| `optimized.relax.json` beside it | Fixed-cell flag, force target, final forces, optimizer, steps |
| `hessians/SOURCEK/repNN/hessian.npz` | Canonical central/member spectrum and provenance |

The empty reference uses the same role filenames under `0ch4/`. LLPR execution
writes `hessian.central.npz` and eight `hessian.llpr-N.npz` batches, each with
its own restart checkpoint. Merge validates shared geometry, settings,
checkpoint identity, and member coverage 0–63, then restores member order and
pools matrix moments. Use the final merge job as the hybrid dependency.
See [README.md](README.md) for restart and serial-execution options.

| Archive field | Meaning |
| --- | --- |
| `frequencies_cm1` | Signed central spectrum, normally `(1, 3N)` |
| `cv_J_per_gK`, `temperatures_K` | Direct harmonic diagnostic and its grid |
| `trajectory`, `trajectory_sha256` | Geometry path and verified identity |
| `checkpoint` | Loaded PET-JAX model |
| `central_hessians_eV_per_A2` | Matrices reused by LLPR workers |
| `llpr_member_indices` | Persistent ordered member identities |
| `llpr_frequencies_cm1` | Merged member spectra, `(1, 64, 3N)` |
| `metadata` | Precision, hops, chunking, rematerialization, ASR, mode convention |

Hybrid NPZ, CSV, and JSON retain the classical term, correction, uncertainties,
and consumed Hessian provenance. JSON distinguishes evaluation temperature
from `hessian_source_temperature_K`.

## Debugging

Check GPU preflight, checkpoint/export/conversion identity, relaxation
provenance, geometry and mass, then signed modes and numerical convergence.
A clean spectrum still requires comparison across independent loaded minima.

| Symptom | Next check |
| --- | --- |
| JAX sees only CPU | GPU allocation, CUDA plugin compatibility, environment activation |
| Repeated checkpoint conversion | Complete `model.msgpack`/`metadata.yaml`, paths, interrupted conversion |
| Out of memory | Chunk size and rematerialization |
| Slow first result | Compilation and coloring in the Slurm log |
| Imaginary or excess near-zero modes | Minimum, force tolerance, model identity, precision, pattern reach |
| Diagnostic curve passes but hybrid fails | Signed modes, provenance, mass, stricter hybrid threshold |
| Chunk size changes the spectrum | Precision, numerical stability, library consistency |

Dependency sources after installation are under `external/sadmof/`:
`src/sadmof/models/pet/` for model inputs and energy, `src/sadmof/sparse/` for
patterns and colored HVPs, `src/sadmof/observables/phonons.py` for observables,
and `deps/asdex/` and `deps/pet-jax/` for reconstruction and model evaluation.
