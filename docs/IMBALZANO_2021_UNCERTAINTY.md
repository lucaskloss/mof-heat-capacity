# Uncertainty estimation for MLIP molecular dynamics

This document summarizes G. Imbalzano *et al.*, *Uncertainty estimation for
molecular dynamics and sampling*, J. Chem. Phys. **154**, 074102 (2021),
[doi:10.1063/5.0036522](https://doi.org/10.1063/5.0036522). The source is the
[supplied article PDF](074102_1_online.pdf).

The first part explains the paper on its own terms. The final sections translate
its ideas into a possible uncertainty workflow for the methane-loaded MOF-5
heat-capacity calculation. That translation is a project recommendation, not a
method demonstrated explicitly in the paper. The practical relationship to
metatrain's analytical LLPR and shallow-ensemble outputs is documented in
[`METATRAIN_LLPR.md`](METATRAIN_LLPR.md).

## Takeaway in one paragraph

A spread among MLIP predictions on individual structures is not yet an error bar
on a thermodynamic average. Different potentials also assign different
Boltzmann probabilities to those structures, so MLIP uncertainty changes both
the value of an observable on a frame and which frames should be sampled. The
paper handles both effects with a **calibrated committee** of models. One MD
trajectory is driven by the committee-mean potential; the energies of every
committee member are evaluated on saved frames; reweighting estimates what each
member would have predicted from its own equilibrium ensemble; and the spread
of those reweighted averages is reported as ML-model uncertainty. Exact
exponential reweighting becomes noisy for large systems, so the authors
recommend a linear, covariance-based cumulant expansion. Separately, an
uncertainty-weighted baseline potential can keep MD stable in extrapolative
regions and identify configurations for active learning.

## What uncertainty the paper does and does not quantify

The paper targets **epistemic ML uncertainty** caused by finite and incomplete
reference training data. Its estimator asks how much the predicted result would
change across plausible models trained to the same reference level.

This must be distinguished from other errors in an MD result:

| Source | Meaning | Covered by the paper's committee method? |
| --- | --- | --- |
| Finite-trajectory sampling | Correlated MD provides only a finite number of effective samples. | No; use autocorrelation-aware blocking and independent trajectories. |
| Initial-condition or metastability sensitivity | Different methane placements or seeds may remain in different regions of phase space. | Not by itself; test independent preparations and replicas. |
| MLIP training-data uncertainty | A finite training set permits multiple plausible fitted potentials. | Yes, if the committee is constructed and calibrated appropriately. |
| Reference-method bias | The DFT functional, basis, or other labels differ systematically from the exact potential-energy surface. | No; all committee members inherit this bias. |
| Model-class bias | All committee members may share the same inadequate descriptors, cutoff, architecture, or long-range treatment. | Usually not; committee agreement is not proof of correctness. |
| Numerical and protocol error | Timestep, thermostat/barostat, finite size, temperature spacing, equilibration, and derivative choices affect the result. | No. |
| Hybrid heat-capacity approximation | Classical anharmonic MD plus a harmonic quantum correction is approximate. | No. |
| Hessian/minimum uncertainty | Different quenched minima, sparse-Hessian settings, and MLIP Hessians change the quantum correction. | Not studied in this paper. |

The resulting committee band is therefore not automatically a total confidence
interval. It is one named component of the uncertainty budget.

## 1. Calibrated committee models

### Constructing the committee

Starting from $N$ structures and reference values, the training data are
subsampled without replacement into $M$ smaller sets. One model is trained on
each set. For a property $y(\mathcal A)$ of configuration $\mathcal A$, the
committee prediction and raw sample variance are

$$
\bar y(\mathcal A)
= \frac{1}{M}\sum_{i=1}^{M}y^{(i)}(\mathcal A),
$$

$$
\sigma^2(\mathcal A)
= \frac{1}{M-1}\sum_{i=1}^{M}
  \left|y^{(i)}(\mathcal A)-\bar y(\mathcal A)\right|^2.
$$

The mean is the best committee prediction. The spread is only a raw uncertainty
indicator until it has been calibrated against reference calculations.

### Why calibration is essential

Models trained from related subsets are correlated, and their raw spread is
usually too narrow. The paper assumes that the shape of the predictive
distribution is approximately correct but its width needs a global scale
factor $\alpha$. With many committee members, the maximum-likelihood estimate
is

$$
\alpha^2
= \frac{1}{N_{\mathrm{val}}}
  \sum_{\mathcal A\in\mathrm{val}}
  \frac{|y_{\mathrm{ref}}(\mathcal A)-\bar y(\mathcal A)|^2}
       {\sigma^2(\mathcal A)}.
$$

For a finite committee this expression is biased. Under the paper's Gaussian
assumptions, its corrected estimator is

$$
\alpha^2
= -\frac{1}{M}
  + \frac{M-3}{M-1}\frac{1}{N_{\mathrm{val}}}
  \sum_{\mathcal A\in\mathrm{val}}
  \frac{|y_{\mathrm{ref}}(\mathcal A)-\bar y(\mathcal A)|^2}
       {\sigma^2(\mathcal A)}.
$$

The members are then rescaled around their unchanged mean,

$$
y^{(i)}(\mathcal A)
\leftarrow
\bar y(\mathcal A)
+\alpha\left[y^{(i)}(\mathcal A)-\bar y(\mathcal A)\right].
$$

This rescaling preserves the central prediction and adjusts the committee
spread to match observed validation errors.

Important qualifications are:

- at least four members are required for the corrected estimator to be
  meaningful, and the authors recommend at least six when uncertainties are
  propagated through nonlinear calculations;
- the correction is unbiased only under an approximately Gaussian predictive
  distribution;
- a single, configuration-independent $\alpha$ assumes that one global scale
  factor calibrates all relevant regions of phase space;
- validation structures must be representative and sufficiently decorrelated;
- the paper suggests either new reference calculations on configurations from
  short committee MD runs or out-of-bag structures omitted from several
  training subsets.

## 2. Robust sampling with a fallback potential

An MLIP may produce extreme forces as soon as a trajectory reaches a local
environment absent from its training set. Active learning can label such a
configuration later, but it does not stop the current trajectory from failing.

The paper considers a delta-learning model

$$
V^{(i)}(\mathcal A)
=V_b(\mathcal A)+V_\delta^{(i)}(\mathcal A),
$$

where $V_b$ is a cheaper, less accurate, but robust baseline and the committee
learns the correction to a higher reference level. If $\sigma(\mathcal A)$ is
the calibrated uncertainty of the ML correction and $\sigma_b$ estimates the
baseline error, dynamics can use

$$
U(\mathcal A)
=V_b(\mathcal A)
+\lambda(\mathcal A)\bar V_\delta(\mathcal A),
\qquad
\lambda(\mathcal A)
=\frac{\sigma_b^2}{\sigma_b^2+\sigma^2(\mathcal A)}.
$$

In familiar regions, $\sigma\ll\sigma_b$, so $\lambda\approx1$ and the full ML
correction is active. In extrapolative regions, $\lambda\rightarrow0$ and the
simulation smoothly falls back to the baseline. Forces must include the
coordinate dependence of $\lambda$; treating it as a constant while
differentiating would not produce the force of $U$.

When the MLIP has atom-centered energy contributions, the same idea can be
applied locally so that only uncertain atomic environments lose the ML
correction. Low-$\lambda$ structures are natural candidates for active
learning.

This mechanism provides **stability, not high-level accuracy**, in an
extrapolative region. A production observable cannot silently mix large amounts
of baseline sampling into a result labeled as the target MLIP. Frequent
fallback means that the training set needs improvement or that the reported
potential must explicitly be $U$ rather than the high-level committee mean.

## 3. Why framewise uncertainty is insufficient

For an observable $a(\mathbf q)$ in the canonical ensemble,

$$
\langle a\rangle_V
=\frac{\int a(\mathbf q)e^{-\beta V(\mathbf q)}d\mathbf q}
       {\int e^{-\beta V(\mathbf q)}d\mathbf q},
\qquad
\beta=\frac{1}{k_BT}.
$$

Changing the potential from $\bar V$ to $V^{(i)}$ has two effects:

1. a model-dependent observable can change on the same configuration; and
2. the Boltzmann probability of that configuration changes.

Simply evaluating the variance of model predictions on frames sampled from
$\bar V$ captures, at most, the first effect. It omits the change in the
sampled ensemble. Running one independent MD trajectory per member would
capture both effects, but multiplies the sampling cost by $M$.

## 4. One-trajectory reweighting

### Exact expression

Run a trajectory with the committee-mean potential $\bar V$ and evaluate every
member energy on each saved frame. Define

$$
r^{(i)}(\mathbf q)
=\exp\left[-\beta\left(V^{(i)}(\mathbf q)-\bar V(\mathbf q)\right)\right].
$$

Then the equilibrium average that would be obtained under member $i$ is

$$
\langle a^{(j)}\rangle_{V^{(i)}}
=\frac{\left\langle r^{(i)}a^{(j)}\right\rangle_{\bar V}}
       {\left\langle r^{(i)}\right\rangle_{\bar V}}.
$$

Here $i$ indexes the potential models and $j$ optionally indexes a separate
committee of observable models. For an ordinary structural observable there
may be no observable model at all; the only uncertainty then comes from how
the potential changes the sampling.

This identity is exact under equilibrium sampling, ergodicity, and adequate
phase-space overlap. It can also reuse frames produced before an on-the-fly
model update, provided the old and new potential energies are available for
those frames.

### Why exact reweighting often fails in large systems

The variance of the exponential weights grows rapidly with the variance of

$$
h^{(i)}=\beta\left(V^{(i)}-\bar V\right),
$$

which generally grows with system size. A few frames can then dominate the
weighted average, giving a formally exact but statistically useless estimate.
This overlap problem is especially relevant to a large periodic MOF plus many
methane molecules.

### Cumulant expansion approximation

The authors recommend a first-order cumulant expansion (CEA),

$$
\langle a^{(j)}\rangle_{V^{(i)}}
\approx
\langle a^{(j)}\rangle_{\bar V}
-\beta\,\mathrm{Cov}_{\bar V}
\left(a^{(j)},V^{(i)}-\bar V\right).
$$

This is a linear-response expression. It is more statistically stable because
it replaces exponential weights by a covariance. Its derivation assumes that
$a$ and the potential difference behave as correlated Gaussian variables. It
should not be trusted when committee members are far apart, the trajectory
crosses qualitatively different phases, or the response is strongly nonlinear.

The key physical quantity is the covariance. Even modest energy disagreements
can cause a large uncertainty in an average when they correlate strongly with
the observable. Conversely, a large framewise energy spread need not matter
much if it is nearly uncorrelated with the observable of interest.

## 5. Separating uncertainty in the observable and in sampling

If there are $M'$ models for the observable as well as $M$ potential models,
the paper separates two contributions:

$$
\sigma_a^2
=\frac{1}{M'-1}\sum_j
\left|\langle a^{(j)}\rangle_{\bar V}-\bar{\bar a}\right|^2,
$$

which is uncertainty in predicting the observable on configurations, and

$$
\sigma_{aV}^2
\approx\frac{\beta^2}{M-1}\sum_i
\left|\mathrm{Cov}_{\bar V}
\left(a,V^{(i)}-\bar V\right)\right|^2,
$$

which is uncertainty caused by the potential changing phase-space sampling.
For large committees, the total model variance is approximately

$$
\widetilde\sigma^2\approx\sigma_a^2+\sigma_{aV}^2.
$$

The paper gives finite-$M,M'$ prefactors for small committees. If only a best
prediction and pointwise uncertainty are available, rather than individual
committee members and their correlations, exact reweighting is impossible.
The authors derive the conservative upper bound

$$
\sigma_{\langle a\rangle}
\leq
\langle\sigma_a\rangle
+\beta\left\langle
  |\langle a\rangle-a|\,\sigma_V
\right\rangle.
$$

Their liquid-gallium example shows that this bound can substantially
overestimate the uncertainty. Retaining member-by-member predictions is much
more informative than storing only a scalar uncertainty.

## 6. What the examples establish

| Example | Purpose | Main lesson |
| --- | --- | --- |
| Phe-Gly-Phe replica-exchange MD | Weighted baseline and active learning | At high temperature the peptide decomposed into environments absent from training. The uncertain ML correction switched off and the DFTB baseline prevented failure, but those regions had baseline rather than target accuracy. |
| Liquid water pair distributions | One-trajectory reweighting | Exact and CEA reweighting reproduced the result of separate member-driven trajectories. For a four-member committee, the unbiased calibration gave $\alpha=2.1$ rather than the biased $3.75$. |
| Methanesulfonic acid in phenol | Structural uncertainty with limited statistics | ML uncertainty in the pair distribution was comparable to block-estimated statistical uncertainty and varied strongly with distance. One scalar error bar would hide this structure. |
| Ice--water coexistence | Propagation through a derived property | Reweighting each model's order parameter and refitting gave an illustrative melting point of $290\pm5$ K. The authors stress that finite-size and sampling errors were not converged, so this is an ML uncertainty rather than a total error. |
| Acid deprotonation with metadynamics | Enhanced-sampling free energy | Uncertainty was small in the trained neutral basin and much larger in the poorly represented deprotonated basin, producing an asymmetric free-energy interval. |
| Liquid-gallium electronic DOS | Separate property-model and sampling errors | Direct DOS-model uncertainty often dominated, but the sampling contribution was sizable and sometimes larger. Their pointwise-only upper bound was much too conservative. |

The examples span structural averages, free energies, phase coexistence,
enhanced sampling, and a learned electronic observable. The paper does not
apply the method to heat capacity, NPT enthalpy derivatives, or MLIP Hessians.

## 7. Implications for this project's heat capacity

This repository estimates the classical loaded-system contribution as

$$
C_{P,\mathrm{cl}}(T)
=\frac{d\langle H\rangle}{dT},
\qquad
H=E_{\mathrm{tot}}+P_{\mathrm{ext}}V,
$$

and then adds a loaded-system harmonic quantum correction,

$$
C_P^{\mathrm{hybrid}}(T)
=C_{P,\mathrm{cl}}(T)
+\Delta C_{\mathrm{qn-cl}}^{\mathrm{har}}(T).
$$

The current autocorrelation, replica, temperature-derivative, and
minimum-to-minimum error estimates are not the MLIP uncertainty studied by
Imbalzano *et al.* The paper suggests an additional model-uncertainty layer.

### A defensible committee-MD adaptation

For each temperature and loading:

1. Construct a genuine committee whose members use the same reference method,
   target, descriptors, and training protocol but differ through training-data
   resampling and initialization. Preserve member identity across every
   temperature.
2. Calibrate the energy uncertainty on decorrelated, thermodynamically relevant
   structures with new reference energies. Validate forces separately for safe
   dynamics; accurate energy differences are essential for reweighting.
3. Run equilibrated NPT trajectories driven by the differentiable
   committee-mean energy and force. Use independent methane placements and
   velocity seeds to quantify sampling and metastability separately.
4. On every analysis frame, store the cell and volume plus the total potential
   energy $V^{(i)}$ from every calibrated member. Do not store only the mean and
   standard deviation.
5. Form the physical enthalpy for each member,

   $$
   H^{(i)}=K+V^{(i)}+P_{\mathrm{ext}}V_{\mathrm{cell}}.
   $$

   Thermostat and barostat extended-system energies are not part of this
   physical enthalpy. Classical momenta have the same distribution for every
   member, so the reweighting factor depends only on the potential-energy
   difference.
6. Estimate $\langle H^{(i)}\rangle_{V^{(i)}}$ from the mean-potential
   trajectory. The direct paper adaptation is either

   $$
   \frac{\left\langle
     e^{-\beta(V^{(i)}-\bar V)}H^{(i)}
   \right\rangle_{\bar V}}
   {\left\langle e^{-\beta(V^{(i)}-\bar V)}\right\rangle_{\bar V}}
   $$

   or, more robustly for this large system,

   $$
   \langle H^{(i)}\rangle_{\bar V}
   -\beta\,\mathrm{Cov}_{\bar V}
   \left(H^{(i)},V^{(i)}-\bar V\right).
   $$

7. Apply the same temperature fit or finite-difference operator separately to
   each member's reweighted enthalpy curve. The spread of the resulting
   $C_{P,\mathrm{cl}}^{(i)}(T)$ curves is the propagated MLIP uncertainty.
   Differentiate first and take the committee spread afterward. Treating
   pointwise enthalpy errors at different temperatures as independent would
   discard their model-to-model correlation and can misestimate the error in
   the derivative.
8. Check exact-weight overlap and agreement between direct reweighting, CEA,
   and a small number of explicitly member-driven trajectories. Poor overlap or
   disagreement means the one-trajectory estimate is not reliable; it calls for
   more training data or direct simulations, not a wider unvalidated band.

The paper derives the basic formula in the canonical ensemble and demonstrates
the same potential-difference reweighting logic in a constant-pressure,
interface-pinning application. Extending it to this project's flexible-cell NPT
implementation still requires validation, especially of stresses and cell
sampling.

### The harmonic correction needs its own treatment

The paper does not propagate MLIP uncertainty through geometry relaxation,
Hessians, frequencies, or the hybrid quantum correction. There are two distinct
questions here:

- **minimum/numerical sensitivity:** repeat the relaxation and Hessian over
  independently selected minima and converged Hessian settings, as the current
  workflow already intends;
- **MLIP model uncertainty:** relax and compute the correction with each
  calibrated committee member, ideally preserving the same member identity as
  in the classical curve.

If member-specific Hessians are feasible, form the complete hybrid curve for
each member before taking the spread,

$$
C_P^{(i)}(T)
=C_{P,\mathrm{cl}}^{(i)}(T)
+\Delta C_{\mathrm{qn-cl}}^{\mathrm{har},(i)}(T).
$$

This retains covariance between the classical and harmonic responses of one
potential. Adding their marginal variances in quadrature assumes independence
and should not be the default when paired committee results are available.

### Different MLIPs are not automatically a calibrated committee

Curves from PET-MAD, another PET checkpoint, and a different MLIP architecture
are scientifically useful, but their spread is best labeled **between-MLIP
sensitivity** or **model-choice spread**. It becomes a quantitative committee
uncertainty only if the ensemble has a defined resampling distribution,
independent members, relevant validation references, and a demonstrated
calibration factor.

Strong agreement among several MLIPs can still miss a shared reference-method
bias or shared absence of long-range physics. Strong disagreement identifies
model sensitivity but does not, by itself, attach a confidence level to any one
curve.

Metatrain's analytical LLPR output is also not, by itself, such a committee: it
contains a standard deviation but not signed member deviations or their
cross-frame correlations. Its optional `energy_ensemble` output is the closer
match to the member-resolved potentials required for the reweighting equations.
It remains a last-layer approximation rather than a committee of independently
retrained full models; see [`METATRAIN_LLPR.md`](METATRAIN_LLPR.md).

### If separate MLIP trajectories already exist

Analyze each trajectory directly under the model that generated it; no
reweighting is needed to recover that model's own ensemble average. Propagate
each model's enthalpy curve through the same temperature derivative and hybrid
correction, with its own correlation-aware sampling error. The distribution of
the resulting complete curves is then a between-MLIP sensitivity comparison.

The paper's calibrated model-uncertainty estimate cannot be reconstructed from
the final curves alone. Post-processing an existing trajectory is possible in
principle if every committee energy can still be evaluated on every saved
frame, the committee has suitable reference calibration, and the ensembles
overlap. For a trajectory generated by a potential $V_0$, the exact weight for
member $i$ becomes $\exp[-\beta(V^{(i)}-V_0)]$. The committee-mean trajectory
used in the paper is usually a better balanced reference than one arbitrary
member. Pooling multiple model-specific trajectories with multistate estimators
could improve overlap, but that is beyond the method developed in this article
and would need separate validation.

## 8. Recommended error-bar presentation

Report components separately before considering a combined band:

| Reported component | Suggested source |
| --- | --- |
| MD statistical standard error | Autocorrelation-aware blocks within runs plus independent replicas. |
| Preparation/metastability spread | Independent methane placements and equilibrated starting structures. |
| MLIP sampling/model uncertainty | Calibrated committee with CEA or validated direct reweighting. |
| Harmonic minimum spread | Corrections from independently quenched representative loaded minima. |
| Hessian numerical sensitivity | Convergence tests for precision, `hops`, zero-mode handling, cell size, and relaxation. |
| Between-MLIP sensitivity | Complete curves produced by independently developed MLIPs, clearly labeled as sensitivity rather than a calibrated confidence interval. |
| Unquantified systematic limitations | Reference electronic structure, finite size, NPT stress quality, and the classical-plus-harmonic hybrid approximation. |

A total statistical band may be formed by quadrature only for components that
are reasonably independent. Otherwise use paired resampling, retain committee
identity through the complete calculation, or show separate nested bands. No
combination of these statistical components should hide the unquantified
systematic limitations in the caption or provenance record.

## 9. Practical diagnostics before trusting the band

- Plot the calibrated committee spread along each trajectory and identify
  which local structures, methane contacts, or cell distortions cause peaks.
- Confirm that the validation set covers every temperature, loading, and
  relevant adsorption or diffusion regime used in the final curve.
- Compare the raw and calibrated spreads; a large or regime-dependent
  correction signals that one global $\alpha$ may be inadequate.
- Center $V^{(i)}-\bar V$ before exponentiation for numerical stability; a
  constant energy offset cancels in normalized reweighting and in the CEA
  covariance.
- Monitor the concentration of exact reweighting weights. If only a handful of
  frames dominate, report the overlap failure and rely on neither the nominal
  exact estimate nor its apparent precision.
- Estimate every covariance with correlation-aware blocks. A covariance from
  many strongly correlated frames is not automatically well converged.
- Compare CEA estimates against direct reweighting and selected direct
  member-driven runs at representative temperatures.
- Preserve the same committee member across the entire temperature derivative
  and, when possible, through the Hessian correction.
- Keep ML uncertainty separate from the existing sampling standard error until
  their covariance and combination rule are justified.

## Bottom line for MOF-5

The paper provides a strong framework for adding an **MLIP epistemic
uncertainty** component to trajectory-derived properties. Its most useful idea
for this project is not to place framewise committee error bars directly on the
heat capacity, but to propagate calibrated member energy differences through
the equilibrium sampling and then through the enthalpy--temperature derivative.
For a large methane-loaded MOF, the CEA is likely more stable than exponential
reweighting, but must be validated against overlap diagnostics and selected
member-driven trajectories. The final heat-capacity result should continue to
show sampling, MLIP, Hessian, between-model, and systematic uncertainties as
distinct quantities unless a justified joint propagation scheme combines them.
