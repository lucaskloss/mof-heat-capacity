# Running the MOF-5 workflow on SCITAS Izar

This guide adapts the desktop CUDA instructions in [README.md](../README.md) to
EPFL's SCITAS Izar cluster. It covers a single-GPU run of `run.py` followed by
`python -m mof_heat_capacity.analysis.harmonic`. Both programs are
single-process Python programs; requesting
more GPUs will not make the current implementation faster.

The commands and limits below were checked against the SCITAS documentation on
30 July 2026. Cluster configuration can change, so inspect the login message and
run `sacctmgr show qos` before the first submission.

The SCITAS cluster overview lists July 2026 as Izar's planned end-of-life date,
although the current Izar page still documents it as the academic GPU cluster.
Confirm that Izar is available and read the login message or SCITAS
announcements before planning long campaigns. If SCITAS migrates academic GPU
work to different hardware, re-check the partition, QOS, GPU architecture,
CUDA/JAX choice, and all resource measurements in this guide.

## Izar resources relevant to this workflow

The ordinary `gpu` partition contains nodes with two NVIDIA Tesla V100 GPUs
(32 GB per GPU). The `gpu-xl` subset contains two larger-memory nodes with four
V100 GPUs each. A one-GPU calculation should normally use `--partition=gpu`;
`gpu-xl` does not provide a newer GPU or more memory per GPU for this workflow.

The useful Izar QOS choices are:

- `debug`: high-priority tests only, at most one hour and two GPUs;
- the default GPU QOS: ordinary jobs of up to three days;
- `long`: jobs of up to seven days, at lower priority;
- `build`: software installation or compilation with no GPU.

The Izar overview currently names the default QOS `gpu`, while the general
SCITAS QOS page names it `normal`. To avoid relying on that documentation
discrepancy, the production templates below use the cluster's default. Check
the live configuration with `sacctmgr show qos`; after confirming its name, it
is reasonable to add either `#SBATCH --qos=gpu` or
`#SBATCH --qos=normal`. Use `#SBATCH --qos=long` only when the measured runtime
requires it. Always use `--qos=debug` for the short validation job, never for
production.

## Put code and data in the appropriate filesystems

SCITAS recommends `/scratch/$USER` for simulation I/O. Scratch is fast but is
not backed up: files older than 30 days are deleted, and earlier cleanup is
possible when occupancy is high. Keep source, environments, input structures,
and irreplaceable model files in `/home` or an available lab `/work` space, and
copy completed results out of scratch promptly. `/home` has a 100 GB default
quota; `/work` is shared lab storage and is not backed up by default.

From a desktop on the EPFL network or VPN, transfer the project with `rsync`:

```bash
rsync -azP /path/to/project/ \
  <gaspar-username>@izar.hpc.epfl.ch:/home/<gaspar-username>/project/
```

On Izar, stage the runnable tree to scratch:

```bash
mkdir -p "$SCRATCH/project"
rsync -a "$HOME/project/mof-heat-capacity/" \
  "$SCRATCH/project/mof-heat-capacity/"
cd "$SCRATCH/project/mof-heat-capacity"
```

Make sure the paths named under `[model]` in the selected TOML configuration
exist after staging. The default configuration expects the checkpoint below
`../models/`.

## Create an Izar-compatible environment

Izar's V100 is a Volta GPU with compute capability 7.0. CUDA 13 dropped Volta
library and offline-compilation support, and its drivers require release 580 or
newer. Izar job 3099805 instead reported driver 535.154.5. The repository's
`environment.yml` therefore keeps every GPU runtime in the CUDA 12 family:
JAX 0.11.0 and its matching plugin/PJRT packages, PyTorch 2.5.1's CUDA 12.1
wheel, and LAMMPS's independently linked C++ libtorch 2.12.1 CUDA 12.9 build.
The LAMMPS and libtorch versions are also pinned to matching ABIs. Do not
remove these pins: Conda can otherwise select a CUDA 13 C++ libtorch even when
Python reports the correct CUDA 12.1 PyTorch wheel.

Create the environment under `/home` or `/work`, not under scratch. Package
installation can be run in Izar's GPU-free `build` QOS rather than consuming a
GPU. The following requests eight CPU cores; omit `--account` if the correct
course or project account is already your default:

```bash
salloc --partition=gpu --qos=build --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=04:00:00
srun --pty bash -l

# CUDA 12 minor-version compatibility allows this CUDA 12.9 user-space
# runtime to run with Izar's 535-series driver. The override is needed because
# this build allocation has no GPU and Conda cannot detect the driver.
CONDA_OVERRIDE_CUDA=12.9 conda env create \
  --prefix "$HOME/.conda/envs/mof-heat-capacity-izar" \
  --file "$HOME/project/mof-heat-capacity/environment.yml"
conda activate "$HOME/.conda/envs/mof-heat-capacity-izar"

cd "$HOME/project/mof-heat-capacity"
./scripts/install_sadmof.sh
exit
exit
```

If `conda` is not already available, first use `module spider` to inspect the
live software stack or install Miniforge in your user space. Record the result
of `conda info --base`; batch scripts need its exact
`etc/profile.d/conda.sh` path. The SCITAS module stack changes annually, so do
not copy a versioned module name from another cluster without checking it on
Izar.

`scripts/install_sadmof.sh` expects the SADMOF source at
`$HOME/project/repos/sadmof-work` for the layout above. If it is elsewhere,
run it as:

```bash
SADMOF_SOURCE=/persistent/path/to/sadmof-work ./scripts/install_sadmof.sh
```

Do not load a separate CUDA toolkit module when using `jax[cuda12]`: these JAX
wheels bring their CUDA user-space libraries, and an incompatible
`LD_LIBRARY_PATH` can override them. The NVIDIA driver remains supplied by the
cluster.

Do not repair an existing environment only by changing JAX or pip Torch.
LAMMPS loads Conda's C++ libtorch, not the pip wheel imported by Python. To
correct an existing prefix, explicitly replace the complete C++ CUDA stack:

```bash
CONDA_OVERRIDE_CUDA=12.9 conda install --yes \
  --prefix "$HOME/.conda/envs/mof-heat-capacity-izar" \
  --override-channels -c metatensor -c conda-forge \
  'lammps-metatomic=2026.07.04.mta1=*cpu*mpi_openmpi*' \
  'libtorch=2.12.1=cuda129_mkl_*' \
  'cuda-version=12.9' 'mkl>=2026,<2027'
```

Then confirm both Torch stacks:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
conda list | awk '$1 == "cuda-version" || $1 == "libtorch" || $1 == "lammps-metatomic"'
ldd "$(command -v lmp)" | grep libcudart
```

Python must report `2.5.1+cu121` and CUDA `12.1`; Conda must report libtorch
2.12.1 with a `cuda129` build and `cuda-version` 12.9; and `ldd` must resolve
`libcudart.so.12`, never `.so.13`. If the explicit replacement does not produce those
exact results, create a fresh prefix using the preceding recipe rather than
submitting another GPU job.

The environment pins Vesin to 0.6.1 because `metatomic-ase` 0.1.2 requires
that API, and pins `nvalchemi-toolkit-ops` to 0.3.1 for compatibility with
Torch 2.5.1. The runner uses that GPU neighbor backend by default, avoiding
Vesin's NVRTC JIT path on Izar's V100. Set `MOF_USE_NVALCHEMIOPS=0` only for
diagnostics on a different CUDA stack.

## Validate CUDA in a debug allocation

Never test GPU availability on the login node. Request one GPU interactively:

```bash
Sinteract -p gpu -q debug -g gpu:1 -c 4 -t 00:30:00
```

Run `Sinteract -h` to inspect the installed wrapper's options if this command
is rejected.

On the allocated node:

```bash
module purge
source <conda-base>/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/mof-heat-capacity-izar"

nvidia-smi
python -c 'import jax; print(jax.devices())'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

cd "$SCRATCH/project/mof-heat-capacity"
python run.py --config configs/mof5_pet_mad.toml \
  --device cuda --steps 10 --output-dir output/debug --prefix debug --rerun
```

Replace `<conda-base>` with the literal path reported by `conda info --base`;
angle-bracket placeholders are not valid shell paths. The JAX output must list
a CUDA device, the PyTorch check must print `True` and a V100 device name, and
the ten-step trajectory must finish before proceeding.

CUDA 12 supports minor-version compatibility with Linux drivers 525 through
the 570 series, subject to NVIDIA's documented feature restrictions, so the
reported 535.154.5 driver belongs to the correct major compatibility range.
Do not switch to CUDA 13: it requires driver 580 or newer and cannot target the
V100. If JAX or the pinned CUDA 12 PyTorch build still cannot initialize CUDA,
preserve the full job output and contact SCITAS support.

## Choose the Slurm resources

There is no universal resource request based only on the number of atoms or MD
steps. Choose resources from the program's parallelism and measurements from a
representative short run. Slurm command-line options override matching
`#SBATCH` lines, so most tests do not require editing `scripts/izar_job.sh`.

First inspect the live cluster rather than relying only on this file:

```bash
sinfo -N -p gpu -o "%N %c CPUs %m MB %G %t"
scontrol show partition gpu
sacctmgr show qos
```

### What each directive means here

| Directive | Slurm meaning | Choice for this workflow |
| --- | --- | --- |
| `--partition=gpu` | Set of nodes on which the job may run. | Use `gpu` for normal work. Do not use `test` for production. |
| `--qos` | Priority and resource/time limits. | Use `debug` only for tests up to one hour. Use the live default for jobs up to three days and `long` for justified jobs up to seven days. |
| `--nodes=1` | Number of physical compute nodes. | Keep `1`. Neither Python entry point distributes work across nodes. |
| `--ntasks=1` | Number of processes or MPI ranks. | Keep `1`. The programs are not MPI applications. |
| `--cpus-per-task=N` | CPU cores available to that one Python process and its native-library threads. | Start with 4 for MD and 8 for Hessian analysis, then benchmark. |
| `--gres=gpu:1` | Number of GPUs allocated per node. | Keep `1`. The programs select one CUDA device and do not distribute over multiple GPUs. |
| `--time=HH:MM:SS` | Hard wall-clock limit, not an estimate of CPU time. | Measure a short run, scale it, add a modest margin, and stay within the selected QOS. |
| `--mem` | Host RAM allocation. This is separate from GPU memory. | Normally omit it on SCITAS; the default is already the configured maximum per requested CPU. Measure `MaxRSS` first. |
| `--account` | Course or project charged/authorized for the job. | Add it only when the correct account is not already the default. |
| `--output` | Slurm standard-output/error file. | Keep `%j` so every job has a unique log. |

`--ntasks` and `--cpus-per-task` are not interchangeable. Tasks are separate
program instances; CPUs per task are cores available to one instance. The
template also calls `srun --ntasks=1`, so increasing only the allocation's
`--ntasks` would reserve unused tasks. Removing that override would instead
launch multiple copies that could overwrite the same trajectory or NPZ file.

Similarly, a second node or GPU does not combine its memory or speed with the
first. The current ASE MD loop and PET-JAX Hessian path have no distributed
implementation. On Izar each V100 has 32 GB of device memory, including in the
`gpu-xl` partition. Requesting `gpu:2` therefore does not fix a one-GPU
out-of-memory failure in this code.

### Starting requests by simulation type

These are calibration starting points, not production guarantees:

| Work | Nodes | Tasks | CPUs/task | GPUs | Initial wall time and QOS |
| --- | ---: | ---: | ---: | ---: | --- |
| Environment creation | 1 | 1 | 8 | 0 | Up to `04:00:00`, `build` |
| CUDA and ten-step MD smoke test | 1 | 1 | 4 | 1 | `00:30:00`, `debug` |
| Short or calibration MD | 1 | 1 | 4 | 1 | `00:30:00` to `02:00:00` |
| Production MD | 1 | 1 | 4 initially | 1 | Scale from a representative trajectory segment |
| One-frame Hessian diagnostic | 1 | 1 | 8 | 1 | Measured near 10 minutes; request `00:30:00`, `debug` |
| Three-frame heat capacity | 1 | 1 | 8 | 1 | Request `00:45:00` per trajectory from the measured one-frame cost |
| Both stages in one job | 1 | 1 | 8 initially | 1 | Sum both measured times plus startup margin |

The MD stage is a serial ASE integration loop with one GPU calculator. Extra
CPU cores can help only CPU-side work such as structure handling, neighbor
lists, logging, and threaded libraries; they do not create additional MD
replicas. The heat-capacity stage uses one JAX device and processes selected
frames one after another. Its dense Hessian conversion and vibrational
analysis can need more host RAM and CPU work than MD, which is why eight CPUs
is a reasonable first measurement rather than a guaranteed optimum.

Separate dependent MD and heat-capacity jobs are preferable to `MOF_STAGE=all`
for production. Each stage can then request its own CPUs and wall time, and a
short MD does not hold an oversized Hessian allocation.

### Benchmark CPU cores

Run the same small, representative calculation with 2, 4, and 8 CPUs, one at a
time. Use a unique output configuration for each run, or deliberately overwrite
only a disposable smoke-test trajectory. For example:

```bash
sbatch --qos=debug --cpus-per-task=2 --time=00:30:00 \
  --export=ALL,MOF_STAGE=md,MOF_STEPS=100,MOF_RERUN=1 \
  scripts/izar_job.sh

sbatch --qos=debug --cpus-per-task=4 --time=00:30:00 \
  --export=ALL,MOF_STAGE=md,MOF_STEPS=100,MOF_RERUN=1 \
  scripts/izar_job.sh
```

Do not submit these simultaneously against the same output. Compare elapsed
time only after the model has already been exported, because first-time model
conversion is startup work rather than per-step MD cost. Keep the smallest CPU
count whose runtime is close to the fastest result. If 8 CPUs are no faster
than 4, requesting 8 only consumes fair-share and can increase queue time.

The template sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` equal to
`SLURM_CPUS_PER_TASK`, following SCITAS guidance for threaded applications.
Because this is one task, do not introduce MPI rank binding or
`--ntasks-per-node` settings.

### Estimate wall time

For MD, time a representative number of steps after one-time export and warmup.
An initial estimate is:

```text
target time = measured time * target steps / measured steps
requested time = target time * 1.2 to 1.5
```

For example, if 1,000 representative steps take 18 minutes, 20,000 steps are
estimated at 360 minutes. A 30% margin gives about 7 hours 48 minutes, so
`--time=08:00:00` is sensible. Avoid requesting three days for an eight-hour
job: SCITAS notes that a realistic short limit can improve backfilling and
reduce queue time.

Heat-capacity timing is less linear. Each frame can incur JAX tracing or
compilation work, and system size, sparsity/coloring, `hops`, `chunk_size`,
precision, and frame geometry can all change cost. Measure one frame, then two
or three frames, before extrapolating. `chunk_size=1` and `remat=true` in the
bundled configuration are already conservative memory choices.

A job is killed when it reaches `--time`. The classical LAMMPS production
driver writes periodic numeric restart files and supports `--resume`; the ASE
smoke-test driver does not provide equivalent continuation. Keep each segment
within a safe wall-time limit and verify that a numeric restart exists before
resuming. Jobs longer than seven days require a different workflow or prior
SCITAS coordination; merely requesting a larger value will not bypass the QOS
limit.

### Size host and GPU memory

SCITAS sets default memory per CPU equal to its maximum allowed memory per CPU,
so `--mem` is normally unnecessary. More `cpus-per-task` also grants more
default host RAM, even when the Python program does not make effective use of
the extra cores. After a job, inspect the batch and `srun` steps:

```bash
sacct -j <job-id> --units=G \
  --format=JobID,JobName,AllocCPUS,Elapsed,TotalCPU,ReqMem,MaxRSS,State,ExitCode
```

If the job ends with `OUT_OF_MEMORY`, inspect `MaxRSS` and the live partition
limits before changing the request. SCITAS can reject `--mem` when it exceeds
`MaxMemPerCPU * allocated CPUs`; request enough CPUs to make the required host
RAM legal. This does not imply that those added cores accelerate the program.

Izar's normal nodes have either 196 or 384 GB of host RAM; the two `gpu-xl`
nodes have 755 GB. Use `gpu-xl` only when measured host-memory requirements
justify its much smaller pool of nodes. It still supplies 32 GB per V100, so it
does not solve device-memory exhaustion.

GPU out-of-memory messages are different from Slurm `OUT_OF_MEMORY`: they refer
to the V100's device RAM and do not appear as host `MaxRSS`. During a running
job, find its node and inspect the GPU from the login node:

```bash
job_node=$(squeue -h -j <job-id> -o "%N")
ssh "$job_node" nvidia-smi
```

SCITAS permits node access only while one of your jobs is running there. If a
single heat-capacity frame exceeds 32 GB, splitting the frame list across jobs
will not reduce that frame's peak. Reducing system size, precision, `hops`, or
changing differentiation settings can alter the scientific calculation and
requires validation rather than being treated as a scheduler adjustment.

### Scale independent simulations with separate jobs

Independent structures, loadings, temperatures, or replicas should use
separate one-GPU jobs with unique configuration files, trajectory prefixes,
and output directories. This is throughput scaling: it does not require more
nodes, tasks, or GPUs inside one job. Avoid submitting thousands of individual
jobs; SCITAS recommends job arrays or contacting support for very large
campaigns.

Selected trajectories are independent. The heat-capacity submission wrapper
uses one array element per trajectory and evaluates that trajectory's selected
frames serially, avoiding repeated JAX compilation for every frame:

```bash
./scripts/submit_heat_capacity.sh --dry-run
./scripts/submit_heat_capacity.sh
```

The defaults create 20 tasks for two MLIPs, loadings 0 and 100, five MD
temperatures, and replica 1, with at most four tasks active. Each task evaluates
200, 350, and 500 ps and writes one archive in its run directory. Use
`--max-concurrent` to lower concurrency if fair-share or job limits require it.

### Read the accounting results

Use `Elapsed`, `TotalCPU`, and `AllocCPUS` to decide whether added CPU cores
were useful. A low `TotalCPU` relative to `Elapsed * AllocCPUS` suggests that
many allocated cores were idle, although a GPU-bound job can still be working
efficiently overall. Use `MaxRSS` for host-memory sizing and inspect the Slurm
state and exit code before trusting output.

For pending jobs, `squeue` and `scontrol show job <job-id>` report reasons such
as `Priority`, `Resources`, or a QOS limit. SCITAS scheduling is not FIFO:
requested job size and wall time, QOS, age, and fair-share all affect priority.
Canceling and resubmitting a correctly configured pending job usually loses
age without making it start sooner.

## Run the MOF-5 paper-protocol campaign

Run every command in this section from the repository root on an Izar login
node unless an allocation is stated explicitly:

```bash
cd "$HOME/project/mof-heat-capacity"
```

If the runnable tree is staged on scratch, use its repository root instead.
Keep the Conda environment and irreplaceable models outside scratch, and copy
completed results to persistent storage promptly.

### Verify the runtime before submitting

The submission script uses
`$HOME/.conda/envs/mof-heat-capacity-izar` by default. Confirm that Python and
LAMMPS use the two intended, independent Torch runtimes:

```bash
"$HOME/.conda/envs/mof-heat-capacity-izar/bin/python" -c \
  'import torch; print(torch.__version__, torch.version.cuda)'
conda list --prefix "$HOME/.conda/envs/mof-heat-capacity-izar" | \
  awk '$1 == "cuda-version" || $1 == "libtorch" || $1 == "lammps-metatomic"'
ldd "$HOME/.conda/envs/mof-heat-capacity-izar/bin/lmp" | grep libcudart
```

The expected results are Python PyTorch `2.5.1+cu121`/CUDA 12.1, Conda
libtorch 2.12.1 with a `cuda129` build, `cuda-version` 12.9, and
`libcudart.so.12`. Do not submit if LAMMPS resolves `libcudart.so.13`.

### Prepare structures and configurations

The classical defaults are five temperatures (100, 200, 300, 400, and 500 K)
and five independent replicas at each temperature. Generate the loaded
structures once with the first model, then reuse them when generating the
second model's configurations:

```bash
./scripts/prepare_paper_protocol.py --model pet-mad --method classical
./scripts/prepare_paper_protocol.py --model pet-sol --method classical \
  --configs-only
```

Preparation refuses to overwrite existing files. Do not use `--force` unless
regenerating those exact structures and configurations is intentional. The
important preparation options are:

| Option | Meaning |
| --- | --- |
| `--model pet-mad\|pet-sol` | Select the checkpoint, exported model, JAX conversion, and name prefix. |
| `--method classical\|pimd` | Select flexible-cell classical NPT or 64-bead Suzuki–Chin PIMD. |
| `--temperatures 100,200,...` | Generate only the comma-separated temperatures listed. |
| `--replicas N` | Generate replicas numbered 1 through `N`; classical defaults to 5. |
| `--loading N` | Set the methane count; the present campaign uses 100. |
| `--configs-only` | Generate TOML files while reusing structures already prepared by the other model. |
| `--dry-run` | Print planned paths without writing anything. |

### Run the debug and calibration jobs

First validate one 300 K replica for each model. Always preview submissions:

```bash
./scripts/submit_paper_protocol.sh --model pet-mad --debug --dry-run
./scripts/submit_paper_protocol.sh --model pet-sol --debug --dry-run

./scripts/submit_paper_protocol.sh --model pet-mad --debug
./scripts/submit_paper_protocol.sh --model pet-sol --debug
```

`--debug` enforces ten steps, four CPUs, one V100, 30 minutes, the `debug`
QOS, 300 K, and replica 1 unless temperature or replica options are explicitly
given. It writes disposable results below `output/debug/`.

After both debug jobs finish, measure a representative 1,000-step run:

```bash
./scripts/submit_paper_protocol.sh --model pet-mad --calibration
./scripts/submit_paper_protocol.sh --model pet-sol --calibration
```

Calibration results are isolated below `output/calibration/`. Obtain the
measured rates with:

```bash
rg 'Loop time|Performance:|Total wall time' output/calibration/*/*.lammps.log
```

The August 2026 measurements for the 924-atom system on one V100 and four CPUs
were 3.31 steps/s for PET-MAD and 3.67 steps/s for PET-SOL. Recalibrate after
changing the model, system, GPU type, software environment, or important MD
settings.

### Equilibrate one replica at every temperature

The classical configuration uses a 0.5 fs timestep. Thus 200,000 steps are
100 ps, including the configured equilibration interval. Submit five jobs per
model—one replica at each temperature—with a 24-hour limit:

```bash
./scripts/submit_paper_protocol.sh --model pet-mad --replicas 1 \
  --steps 200000 --cpus 4 --time 24:00:00 --qos normal --dry-run
./scripts/submit_paper_protocol.sh --model pet-sol --replicas 1 \
  --steps 200000 --cpus 4 --time 24:00:00 --qos normal --dry-run

./scripts/submit_paper_protocol.sh --model pet-mad --replicas 1 \
  --steps 200000 --cpus 4 --time 24:00:00 --qos normal
./scripts/submit_paper_protocol.sh --model pet-sol --replicas 1 \
  --steps 200000 --cpus 4 --time 24:00:00 --qos normal
```

Confirm `normal` is the live ordinary QOS name with `sacctmgr show qos`. Inspect
temperature, energy, volume, density, cell lengths/tilts, and structural
stability before continuing. Instantaneous pressure in this small flexible
cell can fluctuate by thousands of bar and can be negative; assess its mean
with 5--10 ps blocks. In the completed 100 ps pilot runs, final-50-ps mean
pressures were -12 to +17 bar with block uncertainties of roughly 3--17 bar,
consistent with the 1 bar target.

### Continue replica 1 to 500 ps

Once the 100 ps pilot is stable, continue from its latest numeric restart to
the configured final target of 1,000,000 steps (500 ps):

```bash
./scripts/submit_paper_protocol.sh --model pet-mad --replicas 1 \
  --steps 1000000 --resume --cpus 4 --time 96:00:00 --qos long --dry-run
./scripts/submit_paper_protocol.sh --model pet-sol --replicas 1 \
  --steps 1000000 --resume --cpus 4 --time 96:00:00 --qos long --dry-run

./scripts/submit_paper_protocol.sh --model pet-mad --replicas 1 \
  --steps 1000000 --resume --cpus 4 --time 96:00:00 --qos long
./scripts/submit_paper_protocol.sh --model pet-sol --replicas 1 \
  --steps 1000000 --resume --cpus 4 --time 96:00:00 --qos long
```

For LAMMPS, `--steps` is the absolute final timestep because the generated
input uses `run ... upto`. It is not a number of additional steps: resuming a
200,000-step run with `--steps 1000000` advances it by 800,000 steps. The
`--resume` option appends to the existing trajectory and thermodynamic log.
Never combine `--resume` with `--debug` or `--calibration`.

At the measured rates, complete 500 ps runs take approximately 84 hours for
PET-MAD and 76 hours for PET-SOL, before margin. This is why the continuation
uses the `long` QOS and a 96-hour limit. Verify the live `long` limit before
submission.

### Expand to replicas 2 through 5

After replica 1 is stable across all temperatures, start the 100 ps stage for
all five replicas:

```bash
./scripts/submit_paper_protocol.sh --model pet-mad --replicas 5 \
  --steps 200000 --cpus 4 --time 24:00:00 --qos normal --dry-run
./scripts/submit_paper_protocol.sh --model pet-sol --replicas 5 \
  --steps 200000 --cpus 4 --time 24:00:00 --qos normal --dry-run
```

The script interprets `--replicas 5` as replicas 1 through 5. If replica 1
already has a trajectory, its submitted job detects and reuses it without
overwriting it; replicas 2--5 begin normally. Remove `--dry-run` after checking
all commands. Once these runs are stable, continue all replicas:

```bash
./scripts/submit_paper_protocol.sh --model pet-mad --replicas 5 \
  --steps 1000000 --resume --cpus 4 --time 96:00:00 --qos long --dry-run
./scripts/submit_paper_protocol.sh --model pet-sol --replicas 5 \
  --steps 1000000 --resume --cpus 4 --time 96:00:00 --qos long --dry-run
```

Remove `--dry-run` only after every selected replica has a numeric restart.
The full two-model classical campaign contains 50 one-GPU trajectories and is
approximately 4,000 GPU-hours at the measured rate. Check fair-share and job
count policies before submitting all of them simultaneously. The existing
100 ps trajectories occupy approximately 295--336 MB each; at the same output
stride, 50 complete 500 ps trajectories will occupy roughly 75--85 GB before
restarts, logs, analysis, and models. This is too close to a default 100 GB
home quota, so place production output on scratch or project storage and copy
the results that must be retained to persistent storage.

### Run pristine MOF-5 without methane

Use `--loading 0` consistently during both preparation and submission. This
mode copies only the 424-atom periodic MOF-5 host and creates names containing
`0ch4`, keeping pristine and methane-loaded results separate. Prepare the
structures with the first MLIP and reuse them for the second MLIP's configs:

```bash
./scripts/prepare_paper_protocol.py --model pet-mad --method classical \
  --loading 0 --replicas 1
./scripts/prepare_paper_protocol.py --model pet-sol --method classical \
  --loading 0 --replicas 1 --configs-only
```

Changing the atom count can change both stability and performance, so run a
pristine-system debug and calibration even if the loaded system already
passed:

```bash
./scripts/submit_paper_protocol.sh --model pet-mad --loading 0 \
  --debug --dry-run
./scripts/submit_paper_protocol.sh --model pet-sol --loading 0 \
  --debug --dry-run
./scripts/submit_paper_protocol.sh --model pet-mad --loading 0 --debug
./scripts/submit_paper_protocol.sh --model pet-sol --loading 0 --debug

./scripts/submit_paper_protocol.sh --model pet-mad --loading 0 --calibration
./scripts/submit_paper_protocol.sh --model pet-sol --loading 0 --calibration
```

Use the pristine calibration rate to choose the wall time. The campaign that
matches the loaded-system workflow uses one trajectory at each of five
temperatures for each MLIP: ten trajectories total. Preview both five-job,
500 ps submissions first:

```bash
./scripts/submit_paper_protocol.sh --model pet-mad --loading 0 \
  --steps 1000000 --replicas 1 --cpus 4 \
  --time 96:00:00 --qos long --dry-run
./scripts/submit_paper_protocol.sh --model pet-sol --loading 0 \
  --steps 1000000 --replicas 1 --cpus 4 \
  --time 96:00:00 --qos long --dry-run
```

If the measured pristine runtime plus margin fits within the ordinary QOS,
reduce `--time` and use its live QOS name instead. Otherwise, after checking
the previews and the cluster's job-count policy, submit by removing
`--dry-run`:

```bash
./scripts/submit_paper_protocol.sh --model pet-mad --loading 0 \
  --steps 1000000 --replicas 1 --cpus 4 \
  --time 96:00:00 --qos long
./scripts/submit_paper_protocol.sh --model pet-sol --loading 0 \
  --steps 1000000 --replicas 1 --cpus 4 \
  --time 96:00:00 --qos long
```

These are fresh runs, so do not add `--resume`. The template's 0.5 fs
timestep makes 1,000,000 steps equal to 500 ps, with the first 200,000 steps
(100 ps) marked as equilibration. For a safer staged campaign, first submit
`--steps 200000` with the ordinary QOS, inspect equilibration, and then submit
the same selections with `--steps 1000000 --resume` and the measured long-job
resources.

Additional replicas are optional and should be submitted only if independent
sampling and uncertainty estimates are required. For only one 300 K, 500 ps
trajectory per MLIP instead of the five-temperature sweep, add
`--temperatures 300` to the two production commands.

### Analyze completed classical trajectories

Run the thermodynamic and structural extraction as a CPU-based Slurm job rather
than on the login node. The defaults select the ten pristine replica-01 runs,
discard the first 100 ps, and request four CPUs for 1 hour 15 minutes. Izar's `normal`
QOS enforces a minimum GRES allocation, so the job requests one GPU even though
the analysis code does not use it:

```bash
./scripts/submit_analysis.sh --dry-run
./scripts/submit_analysis.sh
```

The default is both potentials, pristine MOF-5 (`--loading 0`), replica 1,
and 100--500 K. Analyze another loading or combine loadings with:

```bash
./scripts/submit_analysis.sh --model both --loading 100 --time 01:15:00 \
  --analysis-dir output/analysis-100ch4 --dry-run
./scripts/submit_analysis.sh --model both --loading 100 --time 01:15:00 \
  --analysis-dir output/analysis-100ch4

./scripts/submit_analysis.sh --model both --loading 0,100 --time 01:30:00 \
  --analysis-dir output/analysis-0ch4-100ch4 --dry-run
./scripts/submit_analysis.sh --model both --loading 0,100 --time 01:30:00 \
  --analysis-dir output/analysis-0ch4-100ch4
```

For trajectory analysis, `--model` accepts `pet-mad`, `pet-sol`, or `both`;
`--loading`, `--temperatures`, and `--replicas` accept comma-separated lists.
The wrapper validates every requested combination and refuses to submit a
partial selection. Repeated restart-boundary frames are removed automatically.

Scheduler output is written below `output/slurm/`, while analysis products are
written below `output/analysis/`. The submission fails before `sbatch` if no
completed trajectories match or a selected LAMMPS log is absent. Compare a
more conservative equilibration cutoff without overwriting the first analysis
with:

```bash
./scripts/submit_analysis.sh --discard-ps 250 \
  --analysis-dir output/analysis-cutoff250 --dry-run
./scripts/submit_analysis.sh --discard-ps 250 \
  --analysis-dir output/analysis-cutoff250
```

This stage does not calculate Hessians. Inspect its stationarity, uncertainty,
autocorrelation, force, cell, and framework results before selecting frames
for harmonic heat-capacity calculations.

### Submission options and output locations

| Option | Operational meaning |
| --- | --- |
| `--model pet-mad\|pet-sol` | Required model selection; output names remain model-specific. |
| `--method classical\|pimd` | Defaults to classical. Do not extrapolate classical timing to 64-bead PIMD. |
| `--loading N` | Select configurations for this methane count; `0` means pristine MOF-5 and the default is 100. |
| `--temperatures LIST` | Defaults to `100,200,300,400,500`. |
| `--replicas N` | Submit replicas 1 through `N`; defaults to 5 for classical and 30 for PIMD. |
| `--steps N` | Override the TOML final step target. With classical `--resume`, this remains an absolute target. |
| `--partition NAME` | Defaults to `gpu`; keep one node and one GPU per job. |
| `--qos NAME` | Use `debug` only for smoke tests, the live ordinary QOS for short runs, and `long` only when measured time requires it. |
| `--time HH:MM:SS` | Hard wall-clock limit for each independent job. |
| `--cpus N` | CPUs for one LAMMPS/Python process; four is the measured classical setting. |
| `--slurm-output-dir PATH` | Directory for Slurm stdout/stderr; defaults to `output/slurm`. Set it to scratch or project storage to move its disk usage off home. |
| `--debug` | Force an isolated ten-step, 30-minute debug-QOS run. |
| `--calibration` | Force an isolated 1,000-step timing run unless `--steps` overrides it. |
| `--resume` | Read the latest numeric restart and append output. It fails when no numeric restart exists. |
| `--dry-run` | Validate files/models and print exact `sbatch` commands without submitting. |

Normal production files are written under the `output_dir` in each generated
TOML file. Each directory contains the trajectory, `.lammps.log`, periodic
`.restart.<step>` files, `.restart.final`, `.final.data`, generated LAMMPS
input, and initial data. Debug and calibration files are kept in their own
subdirectories and are never used as production restarts.

Paper-protocol submissions write scheduler stdout/stderr as
`output/slurm/slurm-<job-name>-<job-id>.out` by default instead of placing
these files in the repository root. Override the location for an individual
submission with, for example:

```bash
./scripts/submit_paper_protocol.sh --model pet-mad --loading 0 \
  --slurm-output-dir "$SCRATCH/mof-heat-capacity/slurm" --dry-run
```

The equivalent persistent default override is
`MOF_SLURM_OUTPUT_DIR=/path/to/logs`. Moving logs into another directory on the
same filesystem only reduces root-directory clutter; it does not reduce quota
usage. Do not move Slurm files belonging to currently running jobs. Once no
jobs are writing the old root-level files, they can be organized with:

```bash
mkdir -p output/slurm
mv slurm-*.out output/slurm/
```

Monitor and account for jobs with:

```bash
squeue -u "$USER"
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,Timelimit,AllocCPUS,MaxRSS
tail -f output/slurm/slurm-<job-name>-<job-id>.out
```

A usable completion has Slurm state `COMPLETED`, exit code `0:0`, a LAMMPS
`Total wall time` line, the expected final restart/data files, and a trajectory
ending at the requested timestep. Do not infer success only from the presence
of an output directory.

## Submit the molecular-dynamics job

The repository includes [izar_job.sh](../scripts/izar_job.sh), a reusable version of
the templates below. Submit it from the directory containing `run.py`. It runs
the MD stage by default:

```bash
cd "$SCRATCH/project/mof-heat-capacity"
sbatch --export=ALL,MOF_STAGE=md scripts/izar_job.sh
```

The template defaults to the environment created above and the bundled TOML
configuration. Override these without editing the script when necessary:

```bash
sbatch --export=ALL,MOF_STAGE=md,MOF_CONFIG=configs/production.toml \
  scripts/izar_job.sh

sbatch --export=ALL,MOF_STAGE=md,MOF_STEPS=100,MOF_RERUN=1 \
  scripts/izar_job.sh
```

If Conda is not found automatically after `module purge`, pass the literal
initialization path obtained from `conda info --base`:

```bash
sbatch --export=ALL,MOF_STAGE=md,MOF_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
  scripts/izar_job.sh
```

Edit the `#SBATCH --time` and `--cpus-per-task` defaults after measuring a
representative short job. Add the live default QOS reported by
`sacctmgr show qos`, or `#SBATCH --qos=long` for a justified run longer than
three days. Do not use `debug` for production.

Create `md.slurm` in the staged `mof-heat-capacity` directory:

```bash
#!/bin/bash -l
#SBATCH --job-name=mof5-md
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module purge
source <conda-base>/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/mof-heat-capacity-izar"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
srun python run.py --config configs/mof5_pet_mad.toml --device cuda
```

Submit it from the directory containing `run.py`:

```bash
cd "$SCRATCH/project/mof-heat-capacity"
sbatch md.slurm
```

If you must select a Slurm account, use
`sbatch --account=<account> md.slurm`. Do not add a second task or GPU: the
current ASE driver has no MPI or multi-GPU distribution.

The bundled configuration is only a ten-step integration smoke test. Before a
production run, copy `configs/mof5_pet_mad.toml`, give it a unique output
directory and trajectory prefix, and set scientifically justified MD length
and analysis parameters. Choose the wall time from a measured short run.

## Submit heat-capacity analysis after MD

Do not load trajectories, initialize JAX, or run SADMOF on a login node. The
wrapper performs only lightweight file checks and submits all scientific work
to GPU compute nodes. First preview and submit the isolated one-frame test:

```bash
./scripts/submit_heat_capacity.sh --debug --dry-run
./scripts/submit_heat_capacity.sh --debug
```

The debug task uses PET-MAD MOF-5 + 100 CH4 at 300 K and the equilibrated
275 ps frame, testing the larger system. At the measured ten-minute frame cost,
the 30-minute debug allocation has adequate margin. Check its log and accounting
record; its NPZ is isolated below `output/debug/`. Then preview and submit the
full array:

```bash
./scripts/submit_heat_capacity.sh --dry-run
./scripts/submit_heat_capacity.sh
```

The production defaults select 200, 350, and 500 ps. These times are after the
100 ps equilibration cutoff and separated by much more than the measured
autocorrelation times. Physical times are mapped using each LAMMPS thermo log,
so the repeated 100 ps continuation frame does not shift selection. Each task
writes `heat-capacity-times-200ps-350ps-500ps.npz` in its unique MD output
directory and refuses to replace an existing archive unless `--overwrite` is
explicitly supplied. This collision check happens before `sbatch`, as does
rejection of duplicate loading, temperature, or replica selectors.

To restrict the calculation, combine selectors:

```bash
./scripts/submit_heat_capacity.sh --model pet-sol --loading 100 \
  --md-temperatures 300 --frame-times-ps 200,350,500 --dry-run
```

After all tasks finish, rerun the trajectory-analysis jobs. They collect every
compatible NPZ in each run directory and produce the frame-convergence tables,
curves, and uncertainty summaries.

## Monitor, diagnose, and retrieve results

Useful scheduler commands are:

```bash
squeue -u "$USER"
scontrol show job <job-id>
scancel <job-id>
sacct -j <job-id> \
  --format=JobID,JobName,AllocCPUS,Elapsed,ReqMem,MaxRSS,State,ExitCode
```

`PENDING` is normal; inspect its reason with `squeue` or `scontrol` rather than
canceling and resubmitting. A job that exceeds its wall time is killed. SCITAS
also enforces a maximum memory allocation per CPU, so if a larger memory request
is rejected, either request more CPU cores or reduce memory; do not inflate
resources without measuring `MaxRSS`.

Inspect every paper-protocol Slurm log under
`output/slurm/slurm-<name>-<job-id>.out`, together with the MD log and the
trajectory. If you selected `--slurm-output-dir`, inspect that directory
instead.
Verify physical stability before interpreting heat capacity. Then copy results
from scratch to persistent storage, for example:

```bash
mkdir -p "$HOME/project-results/mof5"
rsync -a "$SCRATCH/project/mof-heat-capacity/output/mof5-pet-mad/" \
  "$HOME/project-results/mof5/"
```

From the desktop, retrieve them with:

```bash
rsync -azP \
  <gaspar-username>@izar.hpc.epfl.ch:/home/<gaspar-username>/project-results/mof5/ \
  /local/path/mof5/
```

Record the configuration file, PET-MAD model/checkpoint version, device, GPU
model, random seed, trajectory length, timestep, temperature, selected frames,
Hessian settings, Slurm job ID, and statistical or convergence assessment with
the scientific result.

## Authoritative references

- [SCITAS: Izar hardware, connection, partitions, and QOS](https://scitas-doc.epfl.ch/supercomputers/izar/)
- [SCITAS: connecting to clusters](https://scitas-doc.epfl.ch/user-guide/using-clusters/connecting-to-the-clusters/)
- [SCITAS: running and monitoring Slurm jobs](https://scitas-doc.epfl.ch/user-guide/using-clusters/running-jobs/)
- [SCITAS: Slurm partitions and QOS](https://scitas-doc.epfl.ch/user-guide/using-clusters/slurm-qos-partitions/)
- [SCITAS: memory allocation](https://scitas-doc.epfl.ch/user-guide/using-clusters/memory-allocation/)
- [SCITAS: CPU affinity and threaded/MPI jobs](https://scitas-doc.epfl.ch/advanced-guide/cpu-affinity/)
- [SCITAS: job priorities and fair-share](https://scitas-doc.epfl.ch/user-guide/using-clusters/slurm-job-priorities/)
- [SCITAS FAQ: accounts, job arrays, and GPU requests](https://scitas-doc.epfl.ch/faq/)
- [SCITAS: cluster lifecycle overview](https://scitas-doc.epfl.ch/supercomputers/overview/)
- [SCITAS: software modules](https://scitas-doc.epfl.ch/user-guide/using-clusters/software-stack/)
- [SCITAS: Python virtual environments](https://scitas-doc.epfl.ch/user-guide/software/python/python-venv/)
- [SCITAS: choosing filesystems](https://scitas-doc.epfl.ch/user-guide/data-management/how-to-use-filesystems/)
- [SCITAS: scratch policy](https://scitas-doc.epfl.ch/storage/scratch_fs/)
- [SCITAS: transferring data](https://scitas-doc.epfl.ch/user-guide/data-management/transferring-data/)
- [JAX: CUDA installation and GPU support](https://docs.jax.dev/en/latest/installation.html)
- [NVIDIA: CUDA toolkit, driver, and architecture matrix][nvidia-matrix]

[nvidia-matrix]: https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html
