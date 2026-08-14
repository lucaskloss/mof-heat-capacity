# Kapil et al. (2019): MOF-5 heat-capacity simulations

This guide extracts the simulation choices and practical conclusions from
Kapil et al., *Modeling the Structural and Thermal Properties of Loaded
Metal-Organic Frameworks: An Interplay of Quantum and Anharmonic
Fluctuations*, J. Chem. Theory Comput. **15**, 3237–3249 (2019), DOI
[10.1021/acs.jctc.8b01297](https://doi.org/10.1021/acs.jctc.8b01297). It is
based on the [supplied article PDF](<Kapil et al. - 2019 - Modeling the Structural and Thermal Properties of Loaded Metal-Organic Frameworks. An Interplay of Q.pdf>)
and the public [Supporting Information](https://acs.figshare.com/articles/journal_contribution/Modeling_the_Structural_and_Thermal_Properties_of_Loaded_Metal_Organic_Frameworks_An_Interplay_of_Quantum_and_Anharmonic_Fluctuations/8061587).

The first half summarizes what the authors calculated. The second half maps
their protocol onto this repository and gives an Izar workflow that can be run
with the code as it exists today.

## Executive summary

- Empty MOF-5 is strongly affected by nuclear quantum effects, but its lattice
  vibrations are sufficiently harmonic that a quantum harmonic calculation of
  constant-volume heat capacity, \(C_V\), closely reproduces the paper's much
  more expensive path-integral result for constant-pressure heat capacity,
  \(C_P\). The calculated and experimental room-temperature values are of order
  0.7 J g\(^{-1}\) K\(^{-1}\).
- Classical MD gives an almost temperature-independent Dulong–Petit heat
  capacity and is not quantitatively adequate for pristine MOF-5 near room
  temperature.
- Methane changes the conclusion. Translational and low-frequency motion of
  methane inside the pores is strongly anharmonic, even though quantum effects
  in the intermolecular guest motion are modest. A harmonic Hessian cannot
  describe that motion.
- Loaded MOF-5 therefore requires both quantum and anharmonic physics for a
  quantitative \(C_P\). The paper's Suzuki–Chin path-integral MD (PIMD) curves
  have a loading-dependent minimum near 200 K. The minimum is traced mainly to
  the host–guest interaction: methane is localized and stores energy through
  attractive interactions at low temperature, but becomes mobile and less
  bound as temperature rises.
- This repository does **not** reproduce the paper's PIMD protocol. It runs
  fixed-cell classical NVT dynamics with PET-MAD and calculates a quantum
  harmonic \(C_V\) from PET-JAX Hessians. That is a promising route for pristine
  MOF-5 after relaxation and convergence tests. For methane-loaded MOF-5 it is
  a diagnostic approximation, not a reproduction of the published \(C_P\).

## Systems studied

The paper uses the cubic conventional cell containing eight MOF-5 inorganic
nodes. The pristine structure in this repository has the corresponding 424
atoms, formula C192H96O104Zn32, and a cubic cell parameter of 25.86584 Å.

| Methane loading | Total atoms with the repository host | Role in the paper |
| ---: | ---: | --- |
| 0 CH4 | 424 | Pristine reference |
| 50 CH4 | 674 | Low-loading regime |
| 100 CH4 | 924 | Main loaded-system comparison and decomposition |
| 150 CH4 | 1,174 | High-loading regime |

The authors note that an experimental loading near 100 bar is approximately
120 methane molecules per conventional cell. The simulated loadings are fixed
compositions; the MD and PIMD runs are not grand-canonical adsorption
simulations.

## Potential-energy model and starting structures

The paper does not use PET-MAD. It uses newly derived analytical force fields
whose covalent and noncovalent pieces can be evaluated separately.

| Item | Published choice |
| --- | --- |
| Force-field workflow | QuickFF, fitted to isolated inorganic-node and organic-linker cluster data |
| Cluster electronic structure | Gaussian 16, B3LYP |
| C, O, H basis | 6-311G(d,p) |
| Zn basis/effective core potential | LanL2DZ |
| MOF atomic charges | MBIS charges from a PBE/GPAW density, processed with Horton |
| Methane atomic charges | MBIS charges from the B3LYP all-electron density |
| MOF covalent terms | Bonds, bends, out-of-plane terms, torsions, and stretch–stretch/stretch–angle cross terms; anharmonic bond and bend forms were included |
| Methane covalent terms | Harmonic bonds and bends |
| Electrostatics | Coulomb interactions between Gaussian charge distributions |
| van der Waals model | MM3 Buckingham form with empirical mixing rules |
| van der Waals cutoff | 15 Å, with tail corrections to the potential and derivatives |
| Host–guest interaction | Noncovalent terms only |
| Initial methane placement | Random insertion with RASPA while rejecting unrealistic distances, followed by canonical Monte Carlo equilibration |
| Periodic 0 K reference | Force-field-optimized lattice parameter reported as about 26.4 Å |

The Supporting Information gives the analytical force-field forms and some
fixed anharmonic coefficients, but the paper's full parameter files are not
part of this repository. Swapping QuickFF for PET-MAD changes the potential
energy surface, so matching the integration parameters alone would not
constitute a numerical reproduction.

## Published simulation parameters

### Classical molecular dynamics

| Parameter | Published choice |
| --- | --- |
| Temperature range | 100–500 K; the complete list of simulated temperature points is not tabulated |
| Mechanical pressure | 1 bar |
| Ensemble | \(N\mathcal{P}(\boldsymbol{\sigma}_a=0)T\); fully flexible cell with no deviatoric external stress |
| Force evaluation | Covalent interactions in Yaff; expensive long-range interactions in LAMMPS |
| Integrator | Verlet |
| Timestep | 0.5 fs |
| Thermostat | One Nosé–Hoover chain containing three thermostats/beads |
| Thermostat relaxation time | 100 fs |
| Barostat | Martyna–Tobias–Klein |
| Barostat relaxation time | 1,000 fs |
| Loaded systems | Five independent 500 ps runs, with different random seeds and methane positions |
| Empty MOF-5 | One 500 ps trajectory |
| Equilibration | First 100 ps |
| Numeric seeds | Not reported |
| Trajectory/output stride | Not reported |

The classical trajectories capture qualitative volume and thermal-expansion
trends, including the loading-driven change from negative toward positive
thermal expansion. They do not recover the quantum heat capacity.

### Suzuki–Chin path-integral molecular dynamics

| Parameter | Published choice |
| --- | --- |
| Temperature and pressure | 100–500 K and 1 bar |
| Ensemble | \(N\mathcal{P}(\mathbf{h}_0)T\); volume fluctuates while cubic cell shape is retained |
| Driver and force engines | i-PI for dynamics; Yaff and LAMMPS for forces |
| Factorization | Fourth-order Suzuki–Chin |
| Thermostat | PILE-L on ring-polymer normal modes; white-noise Langevin thermostat on the cell |
| Thermostat relaxation time | 100 fs, as in the classical simulations |
| Barostat | Path-integral Bussi–Zykova–Parrinello, adapted to Suzuki–Chin |
| Barostat relaxation time | 1,000 fs, as in the classical simulations |
| Integrator | BAOAB-type multiple-time-step scheme |
| Short-range force | 64 replicas/beads, integrated every 0.25 fs |
| Long-range force | Ring-polymer contraction to 8 replicas, integrated every 1 fs |
| Loaded systems | Thirty independent 50 ps runs, with different random seeds and methane positions |
| Empty MOF-5 | Five independent 125 ps trajectories |
| Equilibration | First 25 ps |
| Heat-capacity temperature difference | Enthalpy finite difference with \(\Delta T=25\) K |
| Numeric seeds and output stride | Not reported |

The fourth-order factorization converges more rapidly in bead count than a
standard second-order path integral. Finite-difference forces and virials,
ring-polymer contraction, and multiple time stepping make the calculation
affordable without evaluating explicit Hessians during PIMD.

Although the authors derive lower-variance operator/double-virial estimators,
their production temperature curves use the centered enthalpy derivative

\[
C_P(T) \approx
\frac{\mathcal{H}(T+\Delta T)-\mathcal{H}(T-\Delta T)}{2\Delta T},
\qquad \Delta T=25\ \mathrm{K}.
\]

Consequently, every reported point depends on statistically converged
enthalpies at neighboring temperatures. It is not an energy-fluctuation
estimate from one short trajectory.

### Quantum harmonic calculation

The paper also computes the Cartesian Hessian with Yaff, performs normal-mode
analysis with TAMkin, and evaluates

\[
C_V(T)=k_B\sum_{\omega}
\left(\frac{\hbar\omega}{k_BT}\right)^2
\frac{e^{\hbar\omega/(k_BT)}}
{\left(e^{\hbar\omega/(k_BT)}-1\right)^2}.
\]

| Parameter | Published choice |
| --- | --- |
| Geometry | Energy-optimized structure |
| Empty host | QuickFF and, as a comparison, generic UFF4MOF Hessians |
| Loaded test | MOF-5 plus 100 CH4 |
| Loaded starting points | Five different Monte Carlo snapshots optimized into different local minima |
| Sensitivity to loaded minimum | Small for the resulting harmonic heat-capacity curves |
| Optimization algorithm/tolerances | Not reported in the article or Supporting Information |
| Zero/imaginary-mode threshold | Not reported |
| Brillouin-zone or supercell convergence | Not reported |

The harmonic result is \(C_V\), while the MD/PIMD result is \(C_P\). For the
empty material the paper finds them nearly equal despite MOF-5's negative
thermal expansion. This observation supports harmonic screening of pristine
MOF-5, but it does not remove the need to relax the chosen potential's
structure and converge finite-size and vibrational settings.

## Results most relevant to new simulations

1. **Pristine MOF-5:** classical MD approaches the Dulong–Petit limit and
   substantially overestimates the measured heat capacity. Suzuki–Chin PIMD
   agrees reasonably with experiments up to approximately 400 K. The quantum
   harmonic curve is close to PIMD, and UFF4MOF gives a similar order of
   magnitude to QuickFF.
2. **Methane-loaded MOF-5:** harmonic analysis misses mobile, low-frequency
   guest motion. Combining quantized framework vibrations with anharmonic
   guest motion produces a non-monotonic heat capacity with a minimum around
   200 K that becomes more pronounced with loading.
3. **Interaction decomposition:** framework contributions are predominantly
   quantum harmonic and retain almost the same shape with loading. Guest–guest
   contributions change only modestly from 100 to 500 K. Host–guest terms are
   responsible for most of the minimum; for methane these are dominated by van
   der Waals interactions.
4. **Loading dependence:** at 300 K, the mass-normalized and volumetric heat
   capacities increase approximately linearly with methane loading in the
   studied range.
5. **Low temperature:** the large contribution below 100 K includes the tail
   of a known confined-methane transition near 60 K. Extrapolating a smooth
   harmonic model into that region would miss this physics.

The authors propose an approximate loaded-system expression,

\[
C \approx C^{\mathrm{har}}_{\mathrm{qn}}
          - C^{\mathrm{har}}_{\mathrm{cl}}
          + C^{\mathrm{anh}}_{\mathrm{cl}},
\]

which combines quantum harmonic high-frequency modes with classical
anharmonic sampling. It qualitatively recovers the minimum and becomes
quantitatively useful above about 200 K in their 100-CH4 test. The current
project does not implement this combination.

## Paper protocol versus this repository

| Feature | Kapil et al. | Current project | Consequence |
| --- | --- | --- | --- |
| Potential | QuickFF analytical force field | MLIP | Results test a different potential-energy surface |
| Host cell | Variable cell at 1 bar | Fixed-cell ASE smoke test; flexible-cell 1-bar NPT backend for production templates, gated on stress/virial validation | Production volume and thermal-expansion sampling is available only with a validated stress-trained model |
| Classical ensemble | Flexible-cell NPT | ASE Langevin NVT by default; LAMMPS MTTK/Nosé–Hoover flexible NPT backend | The production backend is configured for the paper ensemble, but the bundled smoke test is not a paper trajectory |
| MD timestep | 0.5 fs | Configurable: 0.25 fs default, 0.5 fs in the classical paper template, and 0.25 fs in the PIMD template | Timestep convergence remains necessary for each backend/model |
| Thermostat | Three-element Nosé–Hoover chain | Configurable: LAMMPS NHC with three thermostats and 100 fs relaxation; PIMD PILE-L with 100 fs relaxation | Paper-matched controls are represented in the production templates |
| Initial velocities/seeds | Independent reported runs, numeric seeds omitted | Positive numeric seeds are configurable and exposed to the MD/PIMD drivers | Independent generated replicas can be reproduced from their recorded seeds and structures |
| Default duration | 50–500 ps production trajectories | 10 steps = 2.5 fs in the bundled smoke test; paper templates specify 500 ps classical and 50 ps PIMD runs with equilibration | Bundled output is still only an integration smoke test |
| Quantum nuclei | 64-bead Suzuki–Chin PIMD | 64-bead Suzuki–Chin i-PI/LAMMPS backend, with one MLIP force evaluation per bead | The backend omits the paper's 64-to-8 ring-polymer contraction and separate long-range force component |
| Heat capacity | Enthalpy derivative for \(C_P\), plus harmonic comparison | PET-JAX/SADMOF Hessian for harmonic \(C_V\), with configurable post-equilibration frame selection | Suitable first approximation for pristine MOF-5 only; it does not yet compute the paper's enthalpy-derivative (C_P) |
| Hessian geometry | Optimized reference structures | Selected MD trajectory frames | Frame 0 is not automatically an optimized minimum |
| Hessian implementation | Yaff/TAMkin dense normal modes | Sparse colored PET-JAX Hessian, ASR enforced | `hops`, precision, and sparsity require convergence |
| Parallelism | Many independent runs and path-integral replicas | One Slurm task/GPU per run; PIMD force clients are configurable and default to one | Scale replicas across jobs; extra GPUs do not accelerate a single run by default |
| Restarts/output | Production workflow not fully specified | Configurable output stride plus LAMMPS/i-PI checkpoints and `--resume` support | Benchmark restart behavior and output volume before a large campaign |

The current workspace's `output/mof5-pet-mad/mof5-md.traj` contains the initial
frame plus ten MD steps. Its `output/heat-capacity-frame-0.npz` contains a
one-frame harmonic diagnostic. Neither has relaxation, sampling, precision,
sparsity, finite-size, or statistical convergence sufficient for a production
claim. These generated outputs are ignored by Git and may not be present in a
fresh checkout.

## Recommended scientific path

### 1. Validate the pristine harmonic calculation first

This is the closest match between the paper's conclusions and the current
implementation.

1. Relax atomic coordinates with the same PET checkpoint that will be used
   for the Hessian. The repository does not yet expose a relaxation command, so
   a frame-zero Hessian from the supplied CIF remains diagnostic until a
   relaxed structure is provided. The updated PET-MAD and PET-SOL exports both
   advertise stress outputs; generated NPT configurations record the external
   validation status explicitly instead of inferring it from filenames.
2. Check residual forces and imaginary modes before interpreting \(C_V\).
3. Converge PET-JAX precision, graph `hops`, sparse reconstruction, acoustic
   sum-rule handling, cell/supercell size, and frequency treatment.
4. Compare the resulting 300 K value and full 100–500 K curve with the paper
   and experiments, while reporting that PET-MAD replaces QuickFF.

### 2. Treat methane-loaded harmonic calculations as diagnostics

`python -m mof_heat_capacity.structures.methane` can create loaded structures,
and the MD calculator
accepts C/H/O/Zn systems. Nevertheless, random non-overlapping insertion is
not equivalent to the paper's RASPA insertion plus Monte Carlo equilibration.
For each loading, use several independent initial configurations and unique
output directories. A loaded Hessian can reveal local curvature around one
minimum, but it cannot reproduce diffusive guest motion or the published
heat-capacity minimum.

### 3. Scope exact loaded-system reproduction as a separate implementation

Reproduction of the paper's loaded \(C_P\) would require all of the following:

- the published QuickFF parameterization or a separately validated replacement;
- variable-cell constant-pressure dynamics and reliable virials/stress;
- Suzuki–Chin PIMD with 64 beads, ring-polymer contraction, and MTS;
- PILE-L and BZP controls, independent seeded trajectories, and equilibration;
- enthalpy simulations at \(T-25\), \(T\), and \(T+25\) K with uncertainty
  propagation;
- restart/checkpoint, configurable output stride, and campaign-level result
  aggregation.

These are not configuration changes to the existing scripts; they are a new
simulation workflow requiring validation against the paper.

## Running the current project on SCITAS Izar

The repository already contains the detailed [Izar operations guide](IZAR.md)
and reusable [Slurm script](../scripts/izar_job.sh). The commands below are the
short path through that guide.

### Mandatory live-cluster preflight

SCITAS lists Izar's planned end-of-life as July 2026, although the current
[Izar page](https://scitas-doc.epfl.ch/supercomputers/izar/) still documents it
as the academic V100 cluster. The Izar page and the general
[QOS/partition page](https://scitas-doc.epfl.ch/user-guide/using-clusters/slurm-qos-partitions/)
also disagree on whether the default QOS is called `gpu` or `normal`. Do not
guess. Connect through the EPFL network or VPN and inspect the live scheduler:

```bash
ssh <gaspar-username>@izar.hpc.epfl.ch
sinfo -N -p gpu -o "%N %c CPUs %m MB %G %t"
scontrol show partition gpu
sacctmgr show qos
```

Confirm that Izar accepts jobs, that `gpu` remains the correct partition, and
which QOS name is live. In the examples below, replace `normal` if the live
output reports a different ordinary-production QOS.

### Transfer and stage the project

From a workstation on the EPFL network or VPN:

```bash
rsync -azP /path/to/project/ \
  <gaspar-username>@izar.hpc.epfl.ch:/home/<gaspar-username>/project/
```

Keep the environment, source checkout, and irreplaceable models under `/home`
or an appropriate `/work` allocation. Stage active simulation I/O to scratch:

```bash
mkdir -p "$SCRATCH/project"
rsync -a "$HOME/project/mof-heat-capacity/" \
  "$SCRATCH/project/mof-heat-capacity/"
cd "$SCRATCH/project/mof-heat-capacity"
```

Confirm that all three paths under `[model]` in the chosen TOML configuration
exist in the staged tree. The model files are intentionally not tracked by Git.

### Create the persistent environment

Use Izar's GPU-free build QOS. If the live configuration gives `build` an
eight-hour limit, the four-hour request below remains valid.

```bash
salloc --partition=gpu --qos=build --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=04:00:00
srun --pty bash -l

conda env create \
  --prefix "$HOME/.conda/envs/mof-heat-capacity-izar" \
  --file "$HOME/project/mof-heat-capacity/environment.yml"
conda activate "$HOME/.conda/envs/mof-heat-capacity-izar"

cd "$HOME/project/mof-heat-capacity"
./scripts/install_sadmof.sh
exit
exit
```

The installer expects SADMOF at `$HOME/project/repos/sadmof-work`. If it is
elsewhere, set its persistent path explicitly:

```bash
SADMOF_SOURCE=/persistent/path/to/sadmof-work ./scripts/install_sadmof.sh
```

Do not load a separate CUDA toolkit over this environment. It deliberately
pins CUDA-12-compatible JAX and `torch==2.5.1+cu121` for Izar's V100 GPUs and
535-series driver.

### Validate one GPU in a debug allocation

Never initialize CUDA on the login node. Request a short interactive job:

```bash
Sinteract -p gpu -q debug -g gpu:1 -c 4 -t 00:30:00
```

On the allocated node, replace `<conda-base>` with the literal value previously
reported by `conda info --base`:

```bash
module purge
source <conda-base>/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/mof-heat-capacity-izar"

nvidia-smi
python -c 'import jax; print(jax.devices())'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

cd "$SCRATCH/project/mof-heat-capacity"
python run.py --config configs/mof5_pet_mad.toml \
  --device cuda --steps 10 --output-dir output/debug \
  --prefix debug --rerun
```

The JAX check must list a GPU, and Torch must report version `2.5.1+cu121`, CUDA
12.1, `True`, and a V100 device. Inspect `output/debug/debug.log` and the
trajectory before continuing.

### Prepare a calibration configuration

Never point a production job at the bundled output directory. Copy the TOML
file and change at least the run name, output directory, trajectory prefix,
temperature, and step count:

```bash
cp configs/mof5_pet_mad.toml configs/mof5_pristine_300K_rep01.toml
```

For example, use unique values such as:

```toml
[run]
name = "mof5-pristine-300K-rep01"
output_dir = "../output/mof5-pristine-300K-rep01"

[md]
temperature_K = 300.0
steps = 4000
timestep_fs = 0.25
prefix = "mof5-pristine-300K-rep01"
```

Four thousand steps are only a 1 ps timing and stability calibration. At the
current 0.25 fs timestep, 500 ps would be 2,000,000 steps. The current runner
writes every step and cannot resume an interrupted trajectory, so it should
not be scaled blindly to that length. Add and test restart and output-stride
support before committing to paper-length MD.

For methane, first create a uniquely named structure rather than accepting the
default output path:

```bash
python -m mof_heat_capacity.structures.methane \
  --nmol 50 --seed 2025 --try 5000 \
  --output output/structures/mof5-50ch4-seed2025.pdb \
  --data-output output/structures/mof5-50ch4-seed2025.data
```

Copy `configs/metatomic_md_template.toml`, set `[structure].path` to that PDB,
set the three PET model paths consistently, and give every loading, temperature,
and replica its own run name, prefix, and output directory.

### Submit separate MD and harmonic jobs

The current programs are single-process and single-GPU. Start with one node,
one task, one GPU, four CPUs for MD, and eight CPUs for a Hessian. The Slurm
command line overrides matching defaults in `scripts/izar_job.sh`.

```bash
cd "$SCRATCH/project/mof-heat-capacity"

md_job=$(sbatch --parsable \
  --partition=gpu --qos=normal --cpus-per-task=4 --time=02:00:00 \
  --export=ALL,MOF_STAGE=md,MOF_CONFIG=configs/mof5_pristine_300K_rep01.toml \
  scripts/izar_job.sh)
md_job=${md_job%%;*}

sbatch --dependency="afterok:${md_job}" \
  --partition=gpu --qos=normal --cpus-per-task=8 --time=04:00:00 \
  --export=ALL,MOF_STAGE=heat-capacity,MOF_CONFIG=configs/mof5_pristine_300K_rep01.toml,MOF_HEAT_FRAME_INDICES=0 \
  scripts/izar_job.sh
```

`afterok` prevents harmonic analysis after a failed MD run. Frame 0 is a cheap
workflow diagnostic; it becomes a meaningful harmonic reference only when the
configured input structure has first been relaxed and validated. Benchmark a
single Hessian before selecting additional frames.

Selected frames are independent. If a validated trajectory has frames 0, 10,
and 20, an array can give each frame a unique NPZ file:

```bash
sbatch --array=0,10,20%2 \
  --partition=gpu --qos=normal --cpus-per-task=8 --time=04:00:00 \
  --export=ALL,MOF_STAGE=heat-capacity,MOF_CONFIG=configs/mof5_pristine_300K_rep01.toml,MOF_HEAT_FRAMES=array \
  scripts/izar_job.sh
```

Arrays improve throughput but repeat JAX startup/compilation and do not reduce
the memory needed for one frame. The repository does not merge the resulting
NPZ files automatically.

### Monitor, size resources, and preserve results

```bash
squeue -u "$USER"
scontrol show job <job-id>
sacct -j <job-id> --units=G \
  --format=JobID,JobName,AllocCPUS,Elapsed,TotalCPU,ReqMem,MaxRSS,State,ExitCode
```

Use elapsed time to scale the next wall-time request and `MaxRSS` for host
memory. GPU out-of-memory errors concern the V100's 32 GB device memory and are
not reported as host `MaxRSS`. More tasks, GPUs, or nodes do not accelerate the
current serial entry points.

Copy completed results out of scratch promptly:

```bash
mkdir -p "$HOME/project-results/mof5"
rsync -a "$SCRATCH/project/mof-heat-capacity/output/" \
  "$HOME/project-results/mof5/"
```

## Required convergence and provenance record

Before reporting a heat capacity, record and test:

- structure source, composition, cell, loading, and methane-placement method;
- PET-MAD checkpoint/export/JAX conversion identity and model validity;
- relaxation method, final maximum force, energy change, and cell treatment;
- timestep, thermostat, temperature, equilibration, production length,
  independent seeds, output stride, and correlation/statistical uncertainty;
- Hessian frame, precision, graph hops, chunk size, rematerialization, acoustic
  sum rule, imaginary modes, frequency threshold, and finite-size convergence;
- temperature grid, distinction between \(C_V\) and \(C_P\), and normalization;
- Slurm job ID, node/GPU, package environment, wall time, host memory, and GPU
  memory behavior.

No cluster job was submitted while preparing this guide. The commands and
resource requests above were checked against the current repository and public
SCITAS documentation, but live Izar availability and limits must be verified at
submission time.
