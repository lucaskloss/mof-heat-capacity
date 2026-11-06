# Efficient heat capacity of methane-loaded MOF-5

This document turns the conclusions of Kapil et al., *Modeling the Structural
and Thermal Properties of Loaded Metal-Organic Frameworks: An Interplay of
Quantum and Anharmonic Fluctuations*, J. Chem. Theory Comput. **15**,
3237–3249 (2019), into a focused workflow for this project. The sources are the
[supplied article PDF](../pdf/MOF-5.pdf),
the [open preprint](https://arxiv.org/abs/1901.03770), and the
[supplied Supporting Information](../pdf/MOF-5_APPENDIX.pdf), also available from
the [ACS data record](https://acs.figshare.com/articles/journal_contribution/Modeling_the_Structural_and_Thermal_Properties_of_Loaded_Metal_Organic_Frameworks_An_Interplay_of_Quantum_and_Anharmonic_Fluctuations/8061587).

The objective is **not** to reproduce every simulation in the paper with an
MLIP. It is to calculate the heat capacity of methane-loaded MOF-5 efficiently,
using automatic-differentiation Hessians for the quantum correction and
ordinary classical MD for the anharmonic guest motion. Full-system PIMD is a
reference method from the paper and is not part of this production workflow.

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

$$C^{\mathrm{approx}}(T;L) = C^{\mathrm{anh}}_{\mathrm{cl}}(T;L) + \left[ C^{\mathrm{har}}_{\mathrm{qn}}(T;L) - C^{\mathrm{har}}_{\mathrm{cl}}(L) \right]. $$

All three terms refer to the **same loaded composition and the same MLIP**:

- $C^{\mathrm{anh}}_{\mathrm{cl}}$ comes from classical MD of the loaded
  system and retains guest diffusion, host–guest binding, and all other
  classical anharmonic effects.
- $C^{\mathrm{har}}_{\mathrm{qn}}$ comes from the normal-mode frequencies of
  an optimized loaded structure. For each retained mode,

  $$ C_{\mathrm{qn},i}^{\mathrm{har}}(T) = k_B\left(\frac{\hbar\omega_i}{k_BT}\right)^2 \frac{\exp(\hbar\omega_i/k_BT)} {[\exp(\hbar\omega_i/k_BT)-1]^2}. $$

- $C^{\mathrm{har}}_{\mathrm{cl}}$ is the classical harmonic limit for the
  identical set of retained modes: $k_B$ per unconstrained vibrational mode.
  This is the Dulong–Petit term used in the paper.

It is clearest to calculate and store the bracket as a quantum correction,

$$\Delta C^{\mathrm{har}}_{\mathrm{qn-cl}}(T;L) = C^{\mathrm{har}}_{\mathrm{qn}}(T;L) - C^{\mathrm{har}}_{\mathrm{cl}}(L), $$

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

## Choose $C_V$ or $C_P$ before running simulations

The current Hessian calculation produces harmonic $C_V$. Kapil et al. report
classical/PIMD $C_P$, and found $C_P\approx C_V$ for empty MOF-5. That
observation does not automatically prove equality for every methane loading or
for this MLIP.

Two internally coherent routes are available:

1. **Fixed-cell $C_V$, recommended until MLIP stresses are validated.** Run
   loaded NVT MD at the chosen cell and calculate
   $C^{\mathrm{anh}}_{V,\mathrm{cl}}$. Combine it with the fixed-cell
   harmonic correction. Report the result as approximate $C_V$.
2. **One-bar $C_P$.** Run converged loaded NPT MD with a validated MLIP stress
   and compute $d\langle H\rangle/dT$. Add the harmonic quantum correction as
   the paper did and explicitly label the small $C_P/C_V$ mixing assumption.
   If needed, quantify $C_P-C_V$ from thermal expansion and compressibility.

Do not label an NVT energy-fluctuation result as $C_P$, and do not silently
mix per-framework, per-loaded-system, molar, volumetric, and mass-normalized
terms.

## Supporting Information checks relevant to this workflow

The Supporting Information adds several useful controls without changing the
hybrid approximation implemented here:

- Section S6.1 confirms that the loaded harmonic comparison used five local
  minima obtained by optimizing five different Monte Carlo snapshots of
  MOF-5 with 100 methane molecules. Their harmonic curves were similar. This
  supports comparing corrections from several minima, but does not establish
  that an arbitrary insertion, one minimum, or a different MLIP is sufficient.
- Section S6.2 compares flexible-cell and isotropic MTK sampling in Yaff with
  isotropic BZP sampling in i-PI for the 100-methane system. The resulting
  classical $C_P(T)$ curves were not significantly affected by cell-shape
  sampling for that rigid framework and force field. This is useful evidence
  for a barostat-sensitivity check, not a substitute for validating the MLIP
  stress and the LAMMPS NPT implementation used here.
- Section S3 derives direct thermodynamic and operator double-virial heat-
  capacity estimators for Suzuki–Chin PIMD. They require bead-level quantum
  estimators and additional derivative evaluations. They are reference
  methods from the paper and are intentionally not implemented here. They do
  not apply to the present one-bead classical MD analysis, which differentiates
  $⟨E_{\mathrm{tot}}+P_{\mathrm{ext}}V⟩$ across temperatures.
- Section S6.3 decomposes the paper's force-field result by reevaluating each
  trajectory snapshot for the host alone and guests alone, then defining the
  host–guest potential contribution as the total minus those two terms. This
  is a diagnostic decomposition, not part of the hybrid correction. Applying
  it to a many-body MLIP requires separate validation of host-only and
  guest-only predictions and must not be interpreted as a unique atomic or
  pairwise energy decomposition.
- Section S6.4 defines volumetric heat capacity as $C_P(T)$ divided by the
  corresponding molar volume. Equivalently, for this project's gravimetric
  output,

  $$ C_{P,\mathrm{vol}}(T) = \rho(T) C_{P,\mathrm{grav}}(T), \qquad \rho(T)=\frac{m_{\mathrm{cell}}}{\langle V\rangle_T}. $$

  The hybrid assembler records both normalizations using the production NPT
  mean volume. The paper's volumetric curve had a minimum near 175 K; that
  temperature is a result of its force field and loading, not a target that
  this MLIP calculation is required to reproduce.

## QuickFF and PET: main ideas

[Appendix Section S4, pp. S-24–S-29](../pdf/MOF-5_APPENDIX.pdf#page=24)
describes the paper's **QuickFF/MM3–MBIS** model. QuickFF fits an analytical
force field to first-principles geometries and Hessians of molecular building
blocks. Matching the Hessian helps reproduce vibrational frequencies, which
determine harmonic heat capacity. Once fitted, the force field evaluates
energies and forces without further electronic-structure calculations.

The complete model combines three ingredients:

- **QuickFF bonded terms:** bond stretching, angle bending, torsions,
  out-of-plane motion, and coupling between bonds and angles. Framework bonds
  and angles include anharmonic terms; methane bonds and angles are harmonic.
- **MBIS electrostatics:** interactions between fixed Gaussian charge
  distributions derived from electronic densities.
- **MM3 van der Waals terms:** empirical repulsion and dispersion parameters,
  including host–guest interactions. These are not fitted by QuickFF.

**PET** learns the energy as a many-body function of atomic species and
geometric environments, using attention and message passing. Forces and
Hessians follow by differentiation of this learned energy. PET-MAD and
PET-SOL are trained potentials using this architecture; this project uses
them without fitting to the heat-capacity results. See [PET.md](PET.md).

| Aspect | Paper's QuickFF/MM3–MBIS | Project's PET models |
| --- | --- | --- |
| Energy representation | Prescribed bonded and nonbonded formulas with fitted or assigned parameters. | Learned many-body function of atomic environments. |
| Bonding | Fixed chemical topology. | Geometric neighbor graph without a prescribed bond list. |
| Anharmonicity | Explicit anharmonic terms and nonbonded interactions. | Nonlinear learned energy surface. |
| Nonbonded physics | Explicit electrostatics and dispersion. | Finite-range message passing; this workflow adds no explicit long-range electrostatic or dispersion model. |
| New systems | Check parameter transfer and derive missing parameters. | Reuse pretrained weights, then validate the new environments. |

Neither model is automatically more accurate. QuickFF targets the local
structure and vibrations of particular building blocks, while pretrained PET
aims for broader chemical coverage. Their reference data also differ:
the paper combines B3LYP bonded references, MBIS charges from B3LYP/PBE
densities, and MM3 parameters; PET-MAD-1.5 uses
[r²SCAN training data](https://arxiv.org/abs/2603.02089). Both require a
separate treatment of nuclear quantum effects.

## Comparing heat-capacity results

Match loading, pressure, temperature, and mass normalization. Compare
**classical with classical** and **hybrid with hybrid** to assess the effect
of changing the potential. Comparing PET-hybrid with QuickFF-PIMD changes
both the potential and the treatment of nuclear motion. Agreement in that
comparison supports the combined prediction, but does not independently
validate either component. QuickFF results are a simulation reference;
experimental accuracy requires comparison with experiment.

A concise description for the report is:

> The QuickFF/MM3–MBIS reference combines system-specific bonded terms fitted
> to first-principles geometries and Hessians with explicit electrostatic and
> van der Waals interactions. PET instead uses a pretrained many-body energy
> model. Both descriptions include anharmonicity. We compare their
> heat capacities at matching loadings and distinguish changes in the
> potential from changes in the nuclear treatment. Agreement indicates
> consistency for the thermal response over the tested range, rather than
> identical underlying interactions.

Support any claim of quantitative agreement with numerical deviations and
uncertainties. The paper's methane results do not validate PET for CO₂ or H₂O.

## Speed and efficiency

**A PET speed advantage over QuickFF has not been demonstrated here.** An
optimized analytical force field would generally be expected to cost less
per evaluation; actual performance depends on hardware and implementation.
PET's practical advantage is reuse across systems without deriving a new
force field, provided its predictions are validated. Published
[PET-MAD speed benchmarks](https://doi.org/10.1038/s41467-025-65662-7)
compare neural potentials and do not establish superiority over QuickFF.

For context, the archived [924-atom PET-MAD run at 300 K](../../../output/md/production/pet-mad-1.5-s-40nn/100ch4/300K/rep01/md.lammps.log)
achieved **0.254 ns/day**, taking **47.2 hours for 500 ps**, excluding Hessians.
There is no matching QuickFF timing. A useful efficiency comparison would
measure total compute time to the same heat-capacity uncertainty.

The savings from using classical MD plus a harmonic quantum correction
instead of full PIMD arise from the **hybrid approximation**, which can be
used with either potential. Sparse PET Hessians accelerate the MLIP analysis;
they do not by themselves establish an advantage over QuickFF derivatives.
