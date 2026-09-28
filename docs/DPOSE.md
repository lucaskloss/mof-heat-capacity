# Kellner & Ceriotti (2024): uncertainty propagation for heat capacity

**Recommendation:** keep the existing LLPR committees and evaluate the paper's
Gaussian fluctuation approximation as an additional classical heat-capacity
estimator. It can avoid numerical differentiation of noisy CEA enthalpies and
requires only data already saved here. It is a promising comparison, but it is
not yet a validated replacement: an exploratory calculation on the saved
trajectories decreases the classical model spread in four of six cases at
300 K and increases it in the other two.

The distinction is important: **DPOSE is an ensemble construction and
propagation framework, not a general replacement for CEA.** The paper itself
uses CEA for equilibrium averages. Its specific heat-capacity improvement is
the Gaussian treatment of centered moments in Eqs. (20)–(21).

This note summarizes the supplied paper, checks the implementation at commit
`5bd6361`, and inspects existing analysis outputs on 2026-09-28. The NPT
extension and numerical comparison below are project-specific deductions;
the paper demonstrates the method for NVT liquid water. No production code,
checkpoints, or existing results were changed.

## 1. Paper and main ideas

Matthias Kellner and Michele Ceriotti, **“Uncertainty quantification by direct
propagation of shallow ensembles,”** *Machine Learning: Science and Technology*
**5**, 035006 (2024).
[Published article](https://doi.org/10.1088/2632-2153/ad594a),
[supplied PDF](DPSOE.pdf),
[authors' code and data](https://github.com/bananenpampe/DPOSE).
Page numbers below refer to the printed article; the supplied PDF has an
additional cover page.

The paper separates four choices: uncertainty architecture, training loss,
calibration, and propagation to derived quantities. Its proposed **direct
propagation of shallow ensembles (DPOSE)** combines:

- A shared neural-network representation with several final-layer readouts,
  making committee predictions inexpensive on the same structure.
- Training that accounts for both predicted mean and spread, principally
  through a negative log-likelihood loss. Some atomistic benchmarks benefit
  from the alternative continuous ranked probability score loss.
- Propagation of each persistent member through the entire property
  calculation, followed by the spread of the resulting properties.

For a derived quantity $z^{(i)}=f[y^{(i)}]$, Eq. (11) computes

$$\overline z=\frac{1}{M}\sum_i z^{(i)},\qquad \sigma_z^2=\frac{1}{M-1}\sum_i\left(z^{(i)}-\overline z\right)^2.$$

This retains correlations between predictions made by the same member. It
matters for sums of atomic energies, forces obtained by differentiation,
thermodynamic averages, and nonlinear properties. Independent scalar error
bars generally cannot retain those correlations. In particular, atomic model
errors need not be independent or decrease like sampling error as the system
size increases (§4.4).

Calibration remains necessary: the member spread should match reference
residuals on representative held-out structures. Ensemble calibration acts on
member deviations around the mean before propagation, rather than rescaling
the final property arbitrarily (Eq. (9)). The paper also demonstrates that
extrapolative uncertainties can remain overconfident (§4.6); successful
in-distribution calibration does not establish reliability everywhere.

Our LLPR ensembles use sampled, calibrated last-layer weights around an
existing PET model, rather than DPOSE's joint uncertainty-aware training.
They nevertheless provide exactly the persistent energy functions needed for
member-wise propagation. The paper's conclusion explicitly discusses
sampling shallow weights from an approximate loss Hessian and cites the
prediction-rigidity work underlying LLPR. **Retraining as DPOSE is unnecessary
to test its heat-capacity prescription.** See also
[LLPR.md](LLPR.md).

## 2. What goes wrong with heat capacity in the paper

Section 4.5, printed pp. 14–15, considers 256 liquid-water molecules at 300 K
in a 300 ps NVT simulation driven by the mean potential $\overline U$.
Writing $U$ for potential energy, the classical fluctuation expression is

$$C_V=\frac{\operatorname{Var}(U)}{k_{\mathrm B}T^2}+\frac{3N}{2}k_{\mathrm B}.\tag{Paper Eq. 19}$$

The kinetic term here assumes the degrees of freedom in the paper's setup;
constraints or removed center-of-mass motion require the corresponding
degree-of-freedom count.

For an observable $A$, the paper's CEA expression is

$$\langle A\rangle_i\approx\langle A\rangle_0-\beta\operatorname{Cov}_0(A,\Delta U_i),\qquad \Delta U_i=U_i-\overline U,\quad \beta=(k_{\mathrm B}T)^{-1}.\tag{Paper Eq. 12}$$

Subscript $0$ denotes the mean-potential trajectory; $i$ denotes the
equilibrium distribution of member $i$. Applying this expression naively to
the squared energy fluctuation is problematic. Even if $U_i$ and
$\Delta U_i$ are jointly Gaussian, the squared fluctuation is not Gaussian.
Approximating its reweighted average without consistently treating the
shifted mean can give a nonsensical variance.

The paper reports the following **per water molecule**, in units of
$k_{\mathrm B}$:

| Calculation | Heat capacity |
| --- | ---: |
| Mean-potential simulation | 14.51 |
| Naive CEA propagation of heat capacity | $-0.8\pm20.9$ |
| Corrected Gaussian centered-moment treatment | $14.64\pm0.47$ |

These values are an illustration of propagation failure and its correction,
not a quantitative prediction for MOF-5. The authors also emphasize that
classical water heat capacity omits important nuclear quantum effects and
that the corrected model uncertainty is comparable to sampling uncertainty.

### The specific remedy: Eq. (21)

Exponential tilting of a jointly Gaussian distribution shifts its mean while
leaving its covariance unchanged. Applying the general centered-moment
formula, Eq. (20), to the energy variance gives

$$\operatorname{Var}_i(U_i)\approx\operatorname{Var}_0(U_i).\tag{Paper Eq. 21}$$

Thus evaluate each member's potential along the existing trajectory,
subtract **that member's own trajectory mean**, compute its variance, and
then compute the committee spread of the resulting heat capacities. The
recipe uses neither exponential weights nor a first-order correction to the
squared fluctuation.

This is a closure assumption about the joint distribution of energy and
log reweighting factor. It is **not** an assertion that a changed potential
never changes fluctuations. Anharmonicity, distinct basins, structural
transitions, or non-Gaussian tails can invalidate it. An approximately
Gaussian marginal histogram alone is insufficient.

## 3. What the current code actually computes

The project uses loaded-system NPT trajectories and assembles

$$C_{P,\mathrm{hyb}}=C_{P,\mathrm{cl}}^{\mathrm{NPT}}+C_{V,\mathrm{har}}^{\mathrm{qn}}-C_{V,\mathrm{har}}^{\mathrm{cl}}.$$

The classical central value is the temperature derivative of mean enthalpy.
For uncertainty, the code constructs, on each saved trajectory,

$$H_i(t)=K(t)+U_i(t)+P_{\mathrm{ext}}\mathcal V(t),\qquad h_i^{\mathrm{CEA}}(T)=\langle H_i\rangle_0-\beta\operatorname{Cov}_0(H_i,\Delta U_i).$$

It averages replicas member by member, differentiates each member's enthalpy
curve, and finally takes the sample standard deviation:

$$C_{P,i}^{\mathrm{CEA}}(T)=\frac{d h_i^{\mathrm{CEA}}}{dT},\qquad \sigma_{\mathrm{cl}}^{\mathrm{CEA}}(T)=\operatorname{SD}_i\!\left[C_{P,i}^{\mathrm{CEA}}(T)\right].$$

| Step | Implementation |
| --- | --- |
| Persistent, centered LLPR weights | `load_llpr_energy_parameters` in [uncertainty.py](../mof_heat_capacity/analysis/uncertainty.py) |
| CEA enthalpy, exponential reweighting, overlap diagnostics | `committee_enthalpy_estimates` in [uncertainty.py](../mof_heat_capacity/analysis/uncertainty.py) |
| Replica averaging and member derivatives | `write_model_uncertainty_outputs` in [results.py](../mof_heat_capacity/analysis/results.py) |
| Central derivative, harmonic correction, uncertainty assembly | `run` in [hybrid.py](../mof_heat_capacity/analysis/hybrid.py) |

The derivative uses `np.gradient(..., edge_order=2)`: centered interior
differences and second-order one-sided endpoint differences. Model spread is
not divided by $\sqrt M$. Member identity across temperatures is checked via
the checkpoint hash.

**The current code does not make the paper's specific mistake of applying
CEA directly to a squared fluctuation.** Therefore the paper's dramatic
improvement cannot be assumed to repeat here. The existing estimator can
still suffer from inaccurate CEA enthalpies and, especially, statistical
noise in their covariance corrections amplified by differentiation.

The hybrid code preserves classical–harmonic model correlation by combining
deviations for the same LLPR member. It then combines model spread and
separately estimated sampling errors in quadrature. The displayed central
curve uses the central model, not the CEA committee mean. Current harmonic
members are evaluated at central-model minima; member-specific relaxation
uncertainty is omitted. Further background is in
[CEA.md](CEA.md).

## 4. Adapting the paper to this project's NPT calculation

The following is an extension of the paper's argument, not an NPT result
demonstrated in the paper. At fixed external pressure and composition, a
correctly sampled classical NPT distribution obeys

$$C_{P,i}=\frac{\operatorname{Var}_i(H_i)}{k_{\mathrm B}T^2}.$$

The probability ratio between member and central ensembles still depends on
$\exp(-\beta\Delta U_i)$: the $P_{\mathrm{ext}}\mathcal V$ contribution and
common phase-space measure cancel. If $(H_i,\Delta U_i)$ is adequately
described by a joint Gaussian, the counterpart of Eq. (21) is

$$C_{P,i}^{\mathrm{G},H}\approx\frac{\operatorname{Var}_0(K+U_i+P_{\mathrm{ext}}\mathcal V)}{k_{\mathrm B}T^2}.$$

Use physical enthalpy and the configured external pressure. Thermostat and
barostat auxiliary energies do not belong in this expression.

For separable quadratic kinetic degrees of freedom, integrating out momenta
provides a useful alternative. Define $Q_i=U_i+P_{\mathrm{ext}}\mathcal V$;
with $f$ active quadratic momentum degrees of freedom,

$$C_{P,i}^{\mathrm G}\approx\frac{f}{2}k_{\mathrm B}+\frac{\operatorname{Var}_0(Q_i)}{k_{\mathrm B}T^2},\qquad \sigma_{\mathrm{cl}}^{\mathrm G}=\operatorname{SD}_i\!\left[C_{P,i}^{\mathrm G}\right].$$

This version assumes joint Gaussian behavior of $(Q_i,\Delta U_i)$ and avoids
finite-trajectory noise from kinetic fluctuations and their empirical
cross-covariances. The kinetic term is identical for every member, so its
value does not affect the committee spread. Confirm $f$ before reporting
absolute heat capacities. Keeping volume inside $Q_i$ retains potential–volume
covariance; inserting the NVT potential-only formula into NPT data would
instead omit part of $C_P$.

These formulas give heat capacity for the full cell. Divide by its mass and
convert eV to J to obtain $J\,g^{-1}\,K^{-1}$. They require neither new LLPR
members nor new MD to perform an initial comparison. The saved
`model_uncertainty.npz` files already contain member potentials, kinetic
energy, volume, frame times, and the central potential.

For hybrid propagation retain the same member index:

$$C_{P,\mathrm{hyb},i}^{\mathrm G}=C_{P,i}^{\mathrm G}+\left(C_{V,\mathrm{har},i}^{\mathrm{qn}}-C_{V,\mathrm{har},i}^{\mathrm{cl}}\right),\qquad \sigma_{\mathrm{hyb}}^{\mathrm G}=\operatorname{SD}_i\!\left[C_{P,\mathrm{hyb},i}^{\mathrm G}\right].$$

Do not combine the two model standard deviations in quadrature: their
covariance can increase or decrease the final spread. A comparison that keeps
the current central derivative but substitutes Gaussian member deviations
should explicitly describe that choice, rather than implying the central
value and uncertainty came from the same estimator.

## 5. What the existing results show

The following is a read-only diagnostic from the available outputs, not a
converged validation study. At 300 K, each case has 64 members and one replica;
the member archive contains 401 frames spanning 100–500 ps at 1 ps spacing.
The configured pressure is 1 bar. The central enthalpy statistics use a denser
set of 8001 frames over the same production interval.

For the Gaussian diagnostic I computed the variance of each member's
$Q_i=U_i+P_{\mathrm{ext}}\mathcal V$ over those 401 frames using `ddof=0`,
divided by $k_{\mathrm B}T^2$ and cell mass, and took the member standard
deviation using `ddof=1`. No reweighting or fitting was performed.

All heat-capacity columns below are in $J\,g^{-1}\,K^{-1}$.

| Model | Methane molecules | Existing hybrid central value | Existing classical CEA model SD | Gaussian $Q_i$ model SD | Range of $\operatorname{Var}_0(\beta\Delta U_i)$ | Direct weight-effective frame count, range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PET-MAD | 50 | 0.929 | 0.112 | 0.093 | 3.76–306.54 | 1.00–20.19 |
| PET-MAD | 100 | 1.080 | 0.355 | 0.132 | 5.13–404.05 | 1.00–10.80 |
| PET-MAD | 150 | 1.165 | 0.094 | 0.121 | 4.79–505.20 | 1.00–12.81 |
| PET-SOL | 50 | 0.926 | 0.153 | 0.090 | 5.42–297.38 | 1.00–23.49 |
| PET-SOL | 100 | 1.069 | 0.159 | 0.129 | 6.19–491.54 | 1.00–20.58 |
| PET-SOL | 150 | 1.166 | 0.095 | 0.218 | 6.15–775.69 | 1.00–9.43 |

The hybrid central values are from the existing
`heat-capacity.exploratory-discard-imaginary.npz` archives: they use the
exploratory mode-discarding policy named in those files. The two SD columns
are **classical model uncertainties only**, not final hybrid or combined
uncertainties. The Gaussian column has not been combined with the harmonic
members or assigned a sampling confidence interval.

Sources under `output/post-processing/`:

- `trajectory-analysis/{model}/{loading}ch4/model_uncertainty_heat_capacity.npz`
- `trajectory-analysis/{model}/{loading}ch4/300K/rep01/model_uncertainty.npz`
- `trajectory-analysis/{model}/{loading}ch4/300K/rep01/summary.json`
- `harmonic-correction/{model}/{loading}ch4/heat-capacity.exploratory-discard-imaginary.npz`

These ignored, generated files can change when analyses are rerun.

Three observations follow:

1. **Ordinary exponential reweighting is poorly supported by the current
   frames.** Its Kish effective count is often close to one and is not
   corrected for time correlation. A smaller direct-reweighting band would
   not establish improved accuracy.
2. **The small-perturbation justification for CEA is absent here.** All
   members in these six 300 K cases have $\operatorname{Var}(\beta\Delta U_i)>1$.
   This is a warning, not a mathematical proof that the CEA mean is wrong:
   under exact joint Gaussianity, the mean-shift formula is exact even for
   large shifts. Conversely, these large shifts make untested Gaussian-tail
   assumptions consequential for the proposed method too.
3. **There is no universal reduction.** The Gaussian diagnostic substantially
   lowers PET-MAD/100's classical spread, but raises PET-SOL/150's from about
   0.095 to 0.218. The latter is more than twice as large. Smaller is not the
   acceptance criterion; convergence and predictive reliability are.

Across the saved 175–425 K grid, the largest classical CEA SD in each case
occurs at an endpoint, with maxima spanning 0.455–1.262. The displayed hybrid
archives contain 200–400 K, so these full-grid maxima are not all displayed
hybrid points. Endpoint sensitivity is nevertheless a useful diagnostic of
the derivative method.

For reproducibility, this minimal calculation reproduces PET-MAD/100's
Gaussian model SD from the repository root:

```python
import json
from pathlib import Path
import numpy as np
from mof_heat_capacity.analysis.statistics import KB_EV_PER_K, EV_TO_J, AMU_TO_G
from mof_heat_capacity.analysis.uncertainty import BAR_A3_TO_EV

run = Path("output/post-processing/trajectory-analysis/"
           "pet-mad-1.5-s-40nn/100ch4/300K/rep01")
summary = json.loads((run / "summary.json").read_text())
mass_g = summary["metadata"]["total_mass_amu"] * AMU_TO_G
temperature_K, pressure_bar = 300.0, 1.0  # verified for this saved run
with np.load(run / "model_uncertainty.npz") as data:
    q = (data["member_potential_eV"]
         + pressure_bar * data["volume_A3"][:, None] * BAR_A3_TO_EV)
cp_config_by_member = (
    np.var(q, axis=0, ddof=0)
    / (KB_EV_PER_K * temperature_K**2) * EV_TO_J / mass_g
)
print(cp_config_by_member.std(ddof=1))  # 0.1316988624 J g^-1 K^-1
# The common kinetic contribution cancels from this member standard deviation.
```

## 6. Why the present bands may be large

The code preserves member correlations across temperature correctly, but
each member's estimated enthalpy still has finite-trajectory error. For a
uniform grid spacing $\Delta T$, the interior derivative is

$$C_{P,i}(T_j)\approx\frac{h_i(T_{j+1})-h_i(T_{j-1})}{2\Delta T}.$$

At an endpoint its coefficients are $(-3,4,-1)/(2\Delta T)$.
For independent enthalpy errors of equal variance at different temperatures,
the endpoint derivative has 13 times the variance of an interior derivative.
This is a statement about sampling noise, not a universal relation between
committee spreads. It explains why derivative convergence needs particular
attention at boundaries.

The CEA covariance estimates currently use only 401 selected frames per
temperature. Their sampling errors vary across members, so part of the
observed committee spread can be finite-data noise. The separately reported
central-model MD standard error does not by itself quantify sampling error
in every member's CEA correction.

There is also a scale issue in the hybrid result. For PET-MAD/100 at 300 K,
the central classical contribution is about 2.762, while the final hybrid
value is about 1.080. Subtracting the classical harmonic contribution and
adding its quantum counterpart reduces the central value. A classical
model spread of 0.355 consequently looks much larger relative to the hybrid
value. Only combining the classical and harmonic **member deviations** can
determine how much of that uncertainty cancels as well.

Large error bars therefore do not, by themselves, establish a bug or an
overestimated uncertainty. Possible contributors include real LLPR spread,
CEA approximation error, derivative-amplified sampling noise, and incomplete
classical–harmonic cancellation.

## 7. Recommended validation and implementation path

1. **Add an optional Gaussian NPT fluctuation analysis alongside CEA.** Read
   the existing member archives and compute both physical-enthalpy and
   configurational-enthalpy versions. Compare their central estimates with
   the current enthalpy derivative. Their agreement should improve with
   sampling; a persistent discrepancy needs investigation before replacing
   an estimator. Keep the current output for comparison.
2. **Quantify sampling noise in the complete member calculation.** Use time
   blocks longer than the relevant correlation times, including those of
   squared fluctuations and covariance products. Within each trajectory,
   resample the same blocks for every member, then repeat the CEA,
   differentiation, Gaussian-variance, and member-spread calculations. Use
   independent resampling for independent temperature trajectories, and
   incorporate independent replicas when available. Compare frame strides
   and trajectory lengths. This distinguishes unstable propagation from a
   persistent model spread; do not subtract a guessed sampling variance.
3. **Check the Gaussian approximation.** Examine joint distributions of
   $Q_i$ and $\Delta U_i$, stationarity, tails, skewness, and basin changes.
   The first correction omitted from variance invariance involves
   $-\beta\kappa_0(Q_i,Q_i,\Delta U_i)$, where $\kappa$ is a joint cumulant;
   higher terms must also be negligible. This explains why normal-looking
   separate histograms alone do not validate Eq. (21).
4. **Benchmark selected persistent members with their own NPT trajectories.**
   Include members spanning representative deviations and problematic
   cases. This tests both the CEA enthalpy and Gaussian fluctuation estimates
   against sampling under the actual member potential. Such a benchmark
   needs consistent member energies, forces, and cell derivatives/stress;
   the current energy-head post-processing path alone does not supply an
   already validated member-MD workflow. This validates propagation within
   the LLPR ensemble, not calibration against the electronic-structure truth.
5. **Retain and assess LLPR calibration independently.** Check held-out
   reference residuals relevant to the loadings and temperatures of interest.
   Do not shrink the checkpoint spread to obtain attractive heat-capacity
   bars, and do not divide model SD by $\sqrt M$. More committee members
   improve estimation of the distribution, rather than reducing the
   distribution's physical uncertainty.
6. **Reassemble the hybrid uncertainty with matched members.** Preserve the
   checkpoint hash and member ordering when adding the existing harmonic
   corrections. Report classical, harmonic, sampling, and combined components
   separately. Shared representation bias, reference-method error, finite
   size, spectral treatment, and the hybrid approximation remain outside
   this statistical propagation calculation.

Smoothing or fitting the member enthalpy curves could also reduce derivative
noise, but this is a separate numerical choice, not the paper's Eq. (21).
Any fit should preserve member identity, be tested for temperature-grid and
fit bias, and avoid smoothing away a physical transition.

The most useful next change is therefore **an optional, validated Gaussian
NPT fluctuation estimator using the existing LLPR outputs**, with a direct
comparison to CEA and a block-based sampling analysis. The paper supplies a
sound reason to test this route; the saved-data comparison supplies evidence
that its effect is system-dependent rather than an automatic reduction of
the error bars.
