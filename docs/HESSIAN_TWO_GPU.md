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

To bypass CUDA command-buffer graph capture when the runtime fails with
`Failed to add memset node to a CUDA graph`, prefix the `sbatch` command with
`MOF_DISABLE_CUDA_GRAPHS=1`. The experimental launcher then appends
`--xla_gpu_enable_command_buffer=` to `XLA_FLAGS`. This changes execution
scheduling, not the derivative formula; it does not guarantee that GPU memory
will be sufficient. Logs and output metadata record the flag.

## ZIF-8 central model and 64 PET-MAD LLPR members

`scripts/slurm/izar_hessian_ensemble_two_gpu.sh` accepts four positional
arguments: configuration, saved structure, LLPR checkpoint, and a new output
directory. The default array is `0-64%8`; override with `sbatch --array=0-64%N ...`: task 0 is central; tasks
1–64 compute zero-based members 0–63. Each task reserves two GPUs for four
hours and computes a full fixed-selection float64 Hessian using 0.00005 A
finite differences, with CUDA graph capture disabled. The concurrency limit
allows at most eight tasks (16 GPUs) at once, matching normal QOS's current per-user GPU limit. Other running GPU jobs may reduce capacity. This selection is specifically
for the 64-member PET-MAD checkpoint; it does not select PET-SOL.

Each task then calls `analysis.two_gpu_spectrum` to record full-matrix
asymmetry, translation sums, and signed ASR-projected frequencies in
`review.json` and `spectrum.npz`. Imaginary modes are retained. Stable spectra
also report positive-mode harmonic C_v at 200–400 K; unstable spectra have
C_v set to null. These are harmonic quantities, not a final hybrid plot.
Small-column convergence does not establish full-spectrum convergence.
Loaded reference relaxation provenance is included in each review.

Outputs are `central/hessian.npy` and `member-00/hessian.npy` through
`member-63/hessian.npy`, with corresponding JSON metadata and spectra. Each
full matrix occupies about 404 MB for this ZIF-8 system (~26 GB for all 65).
The run takes about 2.2 hours per matrix based on central-model probe timing;
LLPR runtimes and queue waits can vary. Output files are never overwritten,
and interrupted matrices currently require a new output directory to retry.

## Exploratory hybrid assembly from a completed ensemble

`python -m mof_heat_capacity.analysis.hybrid_two_gpu --ensemble DIR
--analysis-archive PATH --output NEW_PREFIX.npz` validates all 65 completed
spectra, member order, common numerical settings, reference-geometry hash,
and matching trajectory/Hessian LLPR checkpoint identities. It creates a
separate compatibility archive under the ensemble's `hybrid-inputs` directory
and calls the existing hybrid analysis without changing production Hessians.
The loaded correction needs only loaded spectra; an empty spectrum is not
required for this plot. Existing output prefixes are refused.

The result covers 200–400 K with 175/425 K MD padding. All 64 harmonic member
corrections contribute to the uncertainty; classical and harmonic LLPR
deviations are combined member by member by the existing hybrid routine.
Unstable and near-zero modes are explicitly discarded for this exploratory
product. Outputs include PNG/PDF heat-capacity plots in J/g/K, CSV/NPZ data,
spectral-stability diagnostics and an ensemble-validation JSON file. Signed
input spectra and the original relaxation provenance are preserved. This
assembly does not establish full-spectrum finite-difference convergence or
correct the known trajectory-reweighting overlap limitations.
