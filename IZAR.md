# Running the MOF-5 workflow on SCITAS Izar

This guide adapts the desktop CUDA instructions in [README.md](README.md) to
EPFL's SCITAS Izar cluster. It covers a single-GPU run of `run.py` followed by
`heat_capacity.py`. Both programs are single-process Python programs; requesting
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

Izar is the academic GPU cluster for courses, student projects, and Master's
work. Connect through the EPFL network or VPN:

```bash
ssh <gaspar-username>@izar.hpc.epfl.ch
```

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

Do not use the desktop environment file unchanged on Izar. Izar's V100 is a
Volta GPU with compute capability 7.0. CUDA 13 dropped Volta library and
offline-compilation support, and JAX's CUDA 13 wheels require compute capability
7.5 or newer. JAX's CUDA 12 wheels still support Volta.

Make an Izar-only copy of the environment specification in persistent storage
and change only the JAX extra:

```bash
cd "$HOME/project/mof-heat-capacity"
cp environment.yml environment-izar.yml
sed -i 's/jax\[cuda13\]/jax[cuda12]/' environment-izar.yml
diff -u environment.yml environment-izar.yml
```

The diff should contain exactly this replacement:

```diff
-      - "jax[cuda13]"
+      - "jax[cuda12]"
```

Create the environment under `/home` or `/work`, not under scratch. Package
installation can be run in Izar's GPU-free `build` QOS rather than consuming a
GPU. The following requests eight CPU cores; omit `--account` if the correct
course or project account is already your default:

```bash
salloc --partition=gpu --qos=build --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=04:00:00
srun --pty bash -l

conda env create \
  --prefix "$HOME/.conda/envs/mof-heat-capacity-izar" \
  --file "$HOME/project/mof-heat-capacity/environment-izar.yml"
conda activate "$HOME/.conda/envs/mof-heat-capacity-izar"

cd "$HOME/project/mof-heat-capacity"
./install_sadmof.sh
exit
exit
```

If `conda` is not already available, first use `module spider` to inspect the
live software stack or install Miniforge in your user space. Record the result
of `conda info --base`; batch scripts need its exact
`etc/profile.d/conda.sh` path. The SCITAS module stack changes annually, so do
not copy a versioned module name from another cluster without checking it on
Izar.

`install_sadmof.sh` expects the SADMOF source at
`$HOME/project/repos/sadmof-work` for the layout above. If it is elsewhere,
run it as:

```bash
SADMOF_SOURCE=/persistent/path/to/sadmof-work ./install_sadmof.sh
```

Do not load a separate CUDA toolkit module when using `jax[cuda12]`: these JAX
wheels bring their CUDA user-space libraries, and an incompatible
`LD_LIBRARY_PATH` can override them. The NVIDIA driver remains supplied by the
cluster.

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
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'

cd "$SCRATCH/project/mof-heat-capacity"
python run.py --config configs/mof5_pet_mad.toml \
  --device cuda --steps 10 --output-dir output/debug --prefix debug --rerun
```

Replace `<conda-base>` with the literal path reported by `conda info --base`;
angle-bracket placeholders are not valid shell paths. The JAX output must list
a CUDA device, the PyTorch check must print `True` and a V100 device name, and
the ten-step trajectory must finish before proceeding.

If JAX reports a driver error, compare the driver shown by `nvidia-smi` with
JAX's current CUDA 12 minimum (driver 525 on Linux at the time of writing). Do
not switch back to CUDA 13: it cannot target the V100. If JAX or PyTorch cannot
initialize CUDA despite the checks above, preserve the full job output and
contact SCITAS support.

## Choose the Slurm resources

There is no universal resource request based only on the number of atoms or MD
steps. Choose resources from the program's parallelism and measurements from a
representative short run. Slurm command-line options override matching
`#SBATCH` lines, so most tests do not require editing `izar_job.slurm`.

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
| One-frame Hessian diagnostic | 1 | 1 | 8 | 1 | Start at `04:00:00`; use `debug` only if proven below one hour |
| Several heat-capacity frames | 1 | 1 | 8 initially | 1 | Measure first and later frames; split independent frames if needed |
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
  izar_job.slurm

sbatch --qos=debug --cpus-per-task=4 --time=00:30:00 \
  --export=ALL,MOF_STAGE=md,MOF_STEPS=100,MOF_RERUN=1 \
  izar_job.slurm
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

A job is killed when it reaches `--time`. The current `run.py` does not resume
an interrupted trajectory: it either reuses an existing trajectory without
continuing it or overwrites it with `--rerun`. Keep MD within a safe wall-time
limit, or add and test restart support before relying on multi-day segmented
runs. Jobs longer than seven days require a different workflow or prior SCITAS
coordination; merely requesting a larger value will not bypass the QOS limit.

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

Selected heat-capacity frames are independent scientifically, although the
current `heat_capacity.py` processes them serially. The template supports a
Slurm array that maps each array index to one frame and one unique NPZ file:

```bash
sbatch --array=0,10,20%2 \
  --export=ALL,MOF_STAGE=heat-capacity,MOF_HEAT_FRAMES=array \
  izar_job.slurm
```

This runs frames 0, 10, and 20, with at most two jobs active at once. Results
are written as `output/heat-capacity-frame-<index>.npz`. Confirm those indices
exist, wait for MD to finish, and combine or analyze the files explicitly;
automatic NPZ merging is not implemented. Each array element repeats JAX
startup/compilation, so an array is useful for wall-time or throughput limits,
not automatically more resource-efficient than several frames in one job.

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

## Submit the molecular-dynamics job

The repository includes [izar_job.slurm](izar_job.slurm), a reusable version of
the templates below. Submit it from the directory containing `run.py`. It runs
the MD stage by default:

```bash
cd "$SCRATCH/project/mof-heat-capacity"
sbatch --export=ALL,MOF_STAGE=md izar_job.slurm
```

The template defaults to the environment created above and the bundled TOML
configuration. Override these without editing the script when necessary:

```bash
sbatch --export=ALL,MOF_STAGE=md,MOF_CONFIG=configs/production.toml \
  izar_job.slurm

sbatch --export=ALL,MOF_STAGE=md,MOF_STEPS=100,MOF_RERUN=1 \
  izar_job.slurm
```

If Conda is not found automatically after `module purge`, pass the literal
initialization path obtained from `conda info --base`:

```bash
sbatch --export=ALL,MOF_STAGE=md,MOF_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
  izar_job.slurm
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

The sparse Hessian can require substantially more GPU memory, host memory, and
time than the short MD test. Start with one frame and the existing
`chunk_size = 1`, then use the observed resource usage to size the production
job. Create `heat-capacity.slurm`:

```bash
#!/bin/bash -l
#SBATCH --job-name=mof5-cv
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module purge
source <conda-base>/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/mof-heat-capacity-izar"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
srun python heat_capacity.py \
  --config configs/mof5_pet_mad.toml \
  --frames 1 \
  --hops 3
```

Submit it only after a successful MD job, or create a Slurm dependency:

```bash
md_job=$(sbatch --parsable --export=ALL,MOF_STAGE=md izar_job.slurm)
md_job=${md_job%%;*}
sbatch --dependency="afterok:${md_job}" \
  --export=ALL,MOF_STAGE=heat-capacity izar_job.slurm
```

`afterok` prevents analysis from starting if MD fails. For production, remove
the diagnostic overrides only after checking convergence and GPU memory use.
This analysis writes one output file, so simultaneous jobs must use distinct
`--output` paths or distinct TOML output directories.

To run both stages sequentially inside one allocation, use:

```bash
sbatch --export=ALL,MOF_STAGE=all izar_job.slurm
```

Separate dependent jobs are normally preferable because MD and Hessian
analysis can need different wall times and CPU allocations. The combined mode
is convenient for a short end-to-end validation.

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

Inspect every `slurm-<name>-<job-id>.out`, the MD log, and the trajectory.
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
