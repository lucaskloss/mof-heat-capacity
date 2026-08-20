# Efficient heat capacity of methane-loaded MOF-5

This document turns the conclusions of Kapil et al., *Modeling the Structural
and Thermal Properties of Loaded Metal-Organic Frameworks: An Interplay of
Quantum and Anharmonic Fluctuations*, J. Chem. Theory Comput. **15**,
3237–3249 (2019), into a focused workflow for this project. The sources are the
[supplied article PDF](<Kapil et al. - 2019 - Modeling the Structural and Thermal Properties of Loaded Metal-Organic Frameworks. An Interplay of Q.pdf>),
the [open preprint](https://arxiv.org/abs/1901.03770), and the
[Supporting Information](https://acs.figshare.com/articles/journal_contribution/Modeling_the_Structural_and_Thermal_Properties_of_Loaded_Metal_Organic_Frameworks_An_Interplay_of_Quantum_and_Anharmonic_Fluctuations/8061587).

The objective is **not** to reproduce every simulation in the paper with an
MLIP. It is to calculate the heat capacity of methane-loaded MOF-5 efficiently,
using automatic-differentiation Hessians for the quantum correction and
ordinary classical MD for the anharmonic guest motion. Full-system PIMD is a
reference method, not the default production method.

## Decision in one paragraph

Use the already-equilibrated empty MOF-5 structure directly for a relaxed,
fixed-cell AD Hessian and a quantum-harmonic reference curve; do not run empty
MOF-5 MD merely to select Hessian frames. For every methane loading of interest,
run classical MD because methane translation and host–guest interactions are
anharmonic. Separately optimize a small number of representative loaded
configurations and calculate their AD Hessians. Combine the classical,
anharmonic loaded-system heat capacity with the harmonic quantum correction
from those loaded Hessians. This is the inexpensive approximation proposed and
tested by Kapil et al. It was quantitatively accurate for their 100-CH4 system
above about 200 K, but it is not an exact replacement for PIMD at lower
temperature or for a different potential without validation.

## What the paper establishes

The relevant physical separation is:

- the MOF-5 framework is strongly quantized but predominantly harmonic;
- high-frequency intramolecular vibrations of both host and methane require a
  quantum treatment;
- low-frequency methane translation in the pores and the host–guest interaction
  are only mildly affected by nuclear quantum effects, but are strongly
  anharmonic;
- the temperature-dependent host–guest term causes the non-monotonic loaded
  heat capacity, including the minimum near 200 K in the paper;
- a Hessian at one minimum cannot represent methane hopping, diffusion, or the
  loss of binding with increasing temperature.

Consequently, neither classical MD alone nor harmonic analysis alone is
sufficient for loaded MOF-5. Their errors concern different parts of the
motion, so the paper combines the two inexpensive calculations.

The paper studied the conventional 424-atom MOF-5 cell with 50, 100, and 150
methane molecules. Its detailed PIMD acceleration scheme, force-field
decomposition, bead count, and replica campaign are useful context but are not
requirements for the MLIP workflow defined here.

## The approximation to implement

For a loaded system $L=\mathrm{MOF\text{-}5}+x\mathrm{CH_4}$, Kapil et al.
propose

\[
C^{\mathrm{approx}}(T;L)
= C^{\mathrm{anh}}_{\mathrm{cl}}(T;L)
+ \left[
    C^{\mathrm{har}}_{\mathrm{qn}}(T;L)
    - C^{\mathrm{har}}_{\mathrm{cl}}(L)
  \right].
\]

All three terms refer to the **same loaded composition and the same MLIP**:

- $C^{\mathrm{anh}}_{\mathrm{cl}}$ comes from classical MD of the loaded
  system and retains guest diffusion, host–guest binding, and all other
  classical anharmonic effects.
- $C^{\mathrm{har}}_{\mathrm{qn}}$ comes from the normal-mode frequencies of
  an optimized loaded structure. For each retained mode,

  \[
  C_{\mathrm{qn},i}^{\mathrm{har}}(T)
  = k_B\left(\frac{\hbar\omega_i}{k_BT}\right)^2
    \frac{\exp(\hbar\omega_i/k_BT)}
         {[\exp(\hbar\omega_i/k_BT)-1]^2}.
  \]

- $C^{\mathrm{har}}_{\mathrm{cl}}$ is the classical harmonic limit for the
  identical set of retained modes: $k_B$ per unconstrained vibrational mode.
  This is the Dulong–Petit term used in the paper.

It is clearest to calculate and store the bracket as a quantum correction,

\[
\Delta C^{\mathrm{har}}_{\mathrm{qn-cl}}(T;L)
= C^{\mathrm{har}}_{\mathrm{qn}}(T;L)
- C^{\mathrm{har}}_{\mathrm{cl}}(L),
\]

and then add it to the classical MD result. This correction is normally
negative because classical mechanics over-populates high-frequency modes.

This is not a decomposition by atom after diagonalizing the Hessian. Loaded
normal modes can mix framework and methane coordinates. Apply the correction
to the complete loaded-system spectrum unless a separately validated
mode-projection scheme is introduced.

### Why the empty Hessian cannot replace the loaded Hessian

The empty-framework Hessian is valuable as a reference and a validation of the
harmonic implementation, but the paper's approximation evaluates the harmonic
terms for $\mathrm{MOF\text{-}5}+x\mathrm{CH_4}$. An empty Hessian omits
quantization of methane's internal vibrations and changes the total mode count.
It therefore cannot, by itself, supply the quantum correction for the loaded
system.

An empty-host-only correction could be investigated later as an additional
cost reduction, but it would be a new approximation and must first be compared
with the full loaded-Hessian correction.

## Choose \(C_V\) or \(C_P\) before running simulations

The current Hessian calculation produces harmonic $C_V$. Kapil et al. report
classical/PIMD $C_P$, and found $C_P\approx C_V$ for empty MOF-5. That
observation does not automatically prove equality for every methane loading or
for this MLIP.

Two internally coherent routes are available:

1. **Fixed-cell \(C_V\), recommended until MLIP stresses are validated.** Run
   loaded NVT MD at the chosen cell and calculate
   $C^{\mathrm{anh}}_{V,\mathrm{cl}}$. Combine it with the fixed-cell
   harmonic correction. Report the result as approximate $C_V$.
2. **One-bar \(C_P\).** Run converged loaded NPT MD with a validated MLIP stress
   and compute $d\langle H\rangle/dT$. Add the harmonic quantum correction as
   the paper did and explicitly label the small $C_P/C_V$ mixing assumption.
   If needed, quantify $C_P-C_V$ from thermal expansion and compressibility.

Do not label an NVT energy-fluctuation result as $C_P$, and do not silently
mix per-framework, per-loaded-system, molar, volumetric, and mass-normalized
terms.

## Focused production workflow

### 1. Lock and validate the MLIP

Use one model identity for insertion relaxation, MD, geometry optimization, and
all Hessians. Record the checkpoint, metatomic export, PET-JAX conversion,
precision, and software versions. Verify stable energies and forces for the
MOF, methane, and short-range host–guest contacts. Validate stress separately
before choosing NPT.

Changing the model between MD and Hessian calculations invalidates the
additive correction, because the terms would refer to different potential
energy surfaces.

### 2. Empty MOF-5: Hessian only

Use the equilibrated empty MOF-5 structure already available to the project.
No empty-framework MD campaign is required.

1. Confirm the composition, periodic cell, and provenance of the structure.
2. With the selected MLIP, minimize the atomic coordinates at the intended
   fixed cell unless the supplied structure already meets the chosen force
   tolerance with that exact MLIP.
3. Record the maximum residual force and optimization convergence.
4. Compute one AD Hessian and the quantum-harmonic $C_V(T)$ curve.
5. Inspect the frequencies. A periodic stable minimum should have only the
   expected translational acoustic modes near zero; extra imaginary modes
   indicate a non-minimum, a numerical problem, or a genuine instability.

This is the pristine reference and an end-to-end validation of the AD Hessian
machinery. It is not a precursor MD stage for the loaded runs.

### 3. Prepare independent loaded configurations

For each loading, prepare several defensible methane arrangements. Random
non-overlapping insertion is only an initial condition, not adsorption
equilibration. Use adsorption/Monte Carlo equilibration when available, or
demonstrate that classical MD loses memory of the initial placements.

The paper optimized five independently prepared 100-CH4 configurations into
different local minima and found little sensitivity in their harmonic curves.
Use this as a starting convergence design, not as proof that one MLIP minimum is
enough. Start with two or three independent minima, compare their quantum
corrections, and expand only if the spread is material.

### 4. Run classical MD only for loaded MOF-5

Classical MD supplies the part that Hessians cannot: anharmonic guest motion.
Run it at each target loading and over a temperature grid dense enough to
resolve the heat-capacity curve. Include independent methane placements and
velocity seeds. Determine equilibration and production lengths from energy,
temperature, methane-site occupancy/diffusion, and autocorrelation diagnostics,
not from the short repository smoke test.

For fixed-cell NVT, obtain

\[
C^{\mathrm{anh}}_{V,\mathrm{cl}}
= \frac{\langle E^2\rangle-\langle E\rangle^2}{k_BT^2},
\]

with the correct treatment of constrained or removed degrees of freedom. As a
cross-check, fit the replica-averaged $\langle E\rangle(T)$ and differentiate
it.

For NPT, use $H=E+P_{\mathrm{ext}}V$ and estimate

\[
C^{\mathrm{anh}}_{P,\mathrm{cl}}(T)
= \frac{d\langle H\rangle}{dT}.
\]

A centered difference requires simulations at $T-\Delta T$ and
$T+\Delta T$. Kapil et al. used $\Delta T=25$ K for their PIMD enthalpy
derivative, but this spacing is a convergence parameter for the MLIP campaign,
not a value to copy blindly. A smooth local fit to several temperatures can be
more statistically stable. Propagate uncertainty by resampling independent
replicas or correlation-aware blocks before refitting or differentiating.

### 5. Optimize loaded minima and calculate AD Hessians

Do not calculate Hessians indiscriminately along the finite-temperature MD
trajectory. A raw thermal frame is generally not a stationary point, and its
instantaneous Hessian is not the normal-mode object used in the paper's
formula.

Instead:

1. select independent, representative loaded configurations;
2. quench each one to a local minimum with the same MLIP and cell convention;
3. verify force convergence and inspect imaginary and near-zero modes;
4. compute the loaded AD Hessian for each accepted minimum;
5. use exactly the same accepted-mode mask for
   $C^{\mathrm{har}}_{\mathrm{qn}}$ and
   $C^{\mathrm{har}}_{\mathrm{cl}}$;
6. average the resulting **quantum corrections**, and report their spread.

Converge PET-JAX precision, graph `hops`, sparse reconstruction, acoustic
sum-rule handling, and finite-size effects on the correction. Do not repair a
poor minimum by silently deleting imaginary frequencies or raising a frequency
cutoff until the result looks smooth.

The repository's harmonic entry point is:

```bash
python -m mof_heat_capacity.analysis.harmonic \
  --config <loaded-config.toml> \
  --trajectory <trajectory-containing-optimized-minima> \
  --frame-indices <indices> \
  --temperatures 100:500:10 \
  --output <unique-result.npz>
```

It writes quantum-harmonic $C_V$ and frequencies. The matching classical mode
term and the final hybrid curve are assembled after the MD and Hessian jobs by:

```bash
./scripts/submit_hybrid_analysis.sh --model pet-mad --loading 100 \
  --replicas 1,2,3,4,5
```

That analysis differentiates the replica-averaged classical NPT enthalpies,
recomputes the quantum and classical harmonic terms with one common mode mask,
and propagates MD and minimum-to-minimum uncertainty into the final curve.

### 6. Assemble and normalize the result

For every loading and temperature:

1. combine extensive heat capacities first;
2. add the mean loaded-Hessian quantum correction to the classical loaded-MD
   result;
3. only then divide by the total mass of that loaded simulation cell for
   J g\(^{-1}\) K\(^{-1}\), or by its volume for a volumetric result;
4. retain the empty harmonic curve as a separately normalized reference;
5. report statistical uncertainty from MD and sensitivity to the optimized
   minimum and numerical Hessian settings.

The mode bookkeeping in $C^{\mathrm{har}}_{\mathrm{cl}}$ must match the MD
degrees of freedom. Account consistently for removed center-of-mass motion,
constraints, frozen atoms, and excluded zero modes. Otherwise the classical
term will not cancel correctly and the hybrid result will have a constant
offset.

## Minimum useful campaign

A cost-conscious first campaign should contain:

- one optimized empty structure and one converged empty AD Hessian;
- one scientifically relevant loading, preferably 100 CH4 to compare with the
  paper's best documented test;
- a classical loaded-MD temperature grid with enough independent replicas to
  estimate uncertainty;
- two or three independently prepared and optimized loaded minima;
- one converged AD Hessian per loaded minimum;
- the hybrid curve and uncertainty, compared with classical-only,
  loaded-harmonic-only, and empty-harmonic curves.

Only after this is stable should the workflow expand to 50 and 150 CH4 or to a
denser temperature grid. There is no scientific need to reproduce the paper's
empty classical-MD or empty PIMD campaigns for this objective.

## Validation and limits

The hybrid expression is empirical. In the paper it qualitatively recovered
the heat-capacity minimum for 100 CH4 and became quantitatively accurate above
about 200 K. Below that range, localized methane and the tail of a confined-
methane transition near 60 K make a classical treatment less reliable. The
result may also depend on loading, guest species, cell size, and MLIP.

Validation should therefore include, in order of increasing cost:

1. reproduce a sensible empty-MOF harmonic curve and check against experiment
   and the paper;
2. compare quantum corrections across independently optimized loaded minima;
3. check whether loaded classical MD reproduces the expected change in methane
   mobility and host–guest energy with temperature;
4. compare the final hybrid curve with experiment or published results while
   clearly identifying the different potential;
5. if quantitative accuracy below 200 K is required, run a few targeted PIMD
   calculations as validation rather than the paper's full PIMD campaign.

Agreement with the paper is not expected numerically merely because its
temperatures or loadings are reused: PET-MAD and QuickFF define different
potential-energy surfaces.

## Required provenance record

Record the following with every reported curve:

- definition of the reported quantity ($C_V$ or $C_P$) and normalization;
- host structure, cell, loading, methane-placement method, and seeds;
- complete MLIP/PET-JAX identity and stress-validation status;
- ensemble, temperature grid, timestep, thermostat/barostat, equilibration,
  production length, output stride, and independent MD replicas;
- energy or enthalpy estimator, derivative/fit method, correlation treatment,
  and statistical uncertainty;
- source and optimization of every Hessian structure, final maximum force, and
  any imaginary modes;
- Hessian precision, `hops`, chunk size, rematerialization, acoustic sum rule,
  zero-mode policy, and finite-size checks;
- the separate values of $C^{\mathrm{anh}}_{\mathrm{cl}}$,
  $C^{\mathrm{har}}_{\mathrm{qn}}$,
  $C^{\mathrm{har}}_{\mathrm{cl}}$, and their final sum.

The central deliverable is therefore a **Hessian-corrected classical-MD heat
capacity for loaded MOF-5**, not a reproduction of the paper's entire PIMD
workflow.
