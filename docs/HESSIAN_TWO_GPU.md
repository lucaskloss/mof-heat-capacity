# Experimental two-GPU Hessian

The existing harmonic implementation and submission scripts remain unchanged.
New implementation: `mof_heat_capacity/analysis/harmonic_two_gpu.py`.
New allocation script: `scripts/slurm/izar_hessian_two_gpu.sh`.

The installed workflow calls SADMOF/asdex sparse coloring and decompression in
one compiled single-device function. The new implementation instead evaluates
exact automatic-differentiation Hessian-vector products for coordinate unit
vectors. Position and HVP arrays are partitioned over two devices; parameters,
cell, and graph are replicated. XLA partitions intermediate computation from
these constraints. Some intermediates may remain replicated: this is not a
64-GiB pooled allocator and fitting the full PET calculation is not guaranteed.
Odd PET position counts are handled with an extra device-placement row.
The energy function slices this row off before calling PET, preserving the
original graph, masks, and sentinel indices. The extra row has zero derivatives.
This slice may cause additional replication, so actual GPU memory must still
be measured. Columns are streamed into a host-backed `.npy` file. This uses more derivative
passes than sparse coloring and may be slower. It retains float64 and the
existing no-shadow energy convention, without truncating the Hessian by hops.

Submit a small diagnostic first, from the repository root:

```bash
sbatch scripts/slurm/izar_hessian_two_gpu.sh \
  --config configs/zif8_2x2x2/pet-mad-1.5-s-40nn/54co2/400K-rep01.toml \
  --structure output/post-processing/harmonic-correction/zif8_2x2x2/pet-mad-1.5-s-40nn/54co2/minima/400K/rep01/optimized.extxyz \
  --output output/experimental-two-gpu/zif8-54co2-probe.npy --columns 3
```

Omit `--columns` for a full matrix and choose a new output name. Existing
outputs are refused. Partial matrices are diagnostics, not spectra. The JSON
sidecar records completed columns, completion state, model-member index, and
per-device allocator statistics. An interrupted file is not resumable yet.
These archives are deliberately separate from production hybrid inputs;
frequency/heat-capacity assembly is not performed by this prototype.

For an individual LLPR Hessian, add `--llpr-checkpoint PATH --member INDEX`.
Both GPUs cooperate on that member. The default computes the central model.
The loaded ZIF-8 saved geometry did not meet its requested relaxation force
tolerance; calculations from it remain exploratory regardless of backend.

Validation before scientific use: compare a small-system full matrix against
`harmonic.compute_frame_hessian` with matching precision and energy settings;
check symmetry and directional finite differences; measure GPU memory on the
ZIF-8 probe. Two virtual CPU devices can test the sharding and column assembly:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  python -m unittest discover -s tests -p test_harmonic_two_gpu.py -v
```

## Memory-saving selected-graph version

The default `--selection-mode fixed` now performs adaptive neighbor selection
on CPU once at the reference geometry. It holds the selected pairs and
adaptive cutoffs constant during differentiation, matching the existing
`no_shadow=True` convention locally. It avoids differentiating through the
large raw-pair/probe-selection calculation. This graph is valid only at the
reference geometry; do not reuse it for MD or relaxation. An actual-model
regression test compares energy, gradients and HVPs to the original function.
The selected rectangular graph is padded with a masked atom if necessary,
and edge displacements and atomic outputs carry explicit two-device sharding
constraints. Model intermediates may still be replicated.

`--selection-mode dynamic` retains the earlier experimental algorithm.
The explicit alternative `--derivative finite-difference --fd-step 0.0001`
computes separate positive/negative displaced gradients, avoiding second-order
AD tapes. This is an approximation: compare steps 0.0001 and 0.00005 A before
using it for spectra. With fixed selection these are derivatives of the fixed
no-shadow local energy, not a relaxation potential. Output provenance records
both choices. Column timings include compilation in the first column.

Actual-model CPU regression (optional installed dependency and checkpoint):

```bash
MOF_TEST_PET_CHECKPOINT=models/pet-mad-1.5-s_40nn_jax JAX_PLATFORMS=cpu \
  python -m unittest discover -s tests -p test_pet_fixed_selection.py -v
```
