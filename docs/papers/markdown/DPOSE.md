# DPOSE and heat-capacity uncertainty

Matthias Kellner and Michele Ceriotti, [“Uncertainty quantification by direct
propagation of shallow ensembles,” *Machine Learning: Science and Technology*
5, 035006 (2024)](https://doi.org/10.1088/2632-2153/ad594a), describe DPOSE:
propagate each persistent ensemble member through a property calculation,
then calculate the spread of the resulting properties. The supplied paper is
also available as [`DPSOE.pdf`](DPSOE.pdf), and project background on the
members is in [LLPR.md](LLPR.md).

## How this differs from the previous CEA calculation

The previous analysis estimates each member's enthalpy on trajectories driven
by the central model using first-order cumulant expansion:

$$h_i^{\mathrm{CEA}}(T)=\langle H_i\rangle_0-\beta\operatorname{Cov}_0(H_i,\Delta U_i),\qquad \Delta U_i=U_i-U_0,\quad \beta=(k_{\mathrm B}T)^{-1}.$$

It differentiates each member's enthalpy curve with respect to temperature,
then takes the spread across member heat-capacity curves. This preserves
member identity across temperatures, but finite-trajectory noise in the CEA
covariance corrections can be amplified by differentiation, especially at
grid endpoints.

DPOSE is a general ensemble-propagation framework; it does not replace CEA in
all calculations. For heat capacity, the paper identifies a specific issue:
applying a first-order CEA correction directly to an energy **variance** can
fail because the squared fluctuation is not Gaussian, even when the energy
and reweighting difference are jointly Gaussian. Under a joint-Gaussian
approximation, exponential reweighting shifts the mean but leaves the
covariance unchanged. The member variance can therefore be estimated from
the member energies evaluated along the existing central-model trajectory,
without differentiating noisy CEA enthalpies or reweighting a squared
fluctuation. The final heat capacity is still propagated member by member,
and its uncertainty is the sample standard deviation across members, not the
standard error of their mean.

This can reduce noise from the CEA-plus-differentiation route. It does **not**
guarantee a more accurate result or smaller uncertainty: the Gaussian closure
may be wrong, and the ensemble spread depends on the model and sampled
system.

## NPT form used here

The paper demonstrates its heat-capacity treatment for NVT liquid water. This
project's extension to loaded MOF-5 uses NPT trajectories and is therefore a
project-specific approximation. At fixed pressure, composition, and a
correctly sampled classical NPT distribution,

$$C_{P,i}=\frac{\operatorname{Var}_i(H_i)}{k_{\mathrm B}T^2},\qquad H_i=K+U_i+P_{\mathrm{ext}}\mathcal V.$$

The reweighting difference is $\Delta U_i$ because the pressure-volume term
is common to the central and member ensembles. Under the joint-Gaussian
closure, the physical-enthalpy estimate is

$$C_{P,i}^{\mathrm{G},H}\approx\frac{\operatorname{Var}_0(K+U_i+P_{\mathrm{ext}}\mathcal V)}{k_{\mathrm B}T^2}.$$

The implementation uses an equivalent configurational form, integrating out
separable quadratic momenta. With $Q_i=U_i+P_{\mathrm{ext}}\mathcal V$ and
$f$ active momentum degrees of freedom,

$$C_{P,i}^{\mathrm{G}}\approx\frac{f}{2}k_{\mathrm B}+\frac{\operatorname{Var}_0(Q_i)}{k_{\mathrm B}T^2}.$$

Here $f=3N-3$ for the current unconstrained system initialized with zero
linear momentum; change this count if constraints or removed degrees of
freedom change. The kinetic term is common to members, so it does not affect
their spread. Keep $P\mathcal V$ in $Q_i$ for NPT data. Thermostat and
barostat auxiliary energies are not part of physical enthalpy. The reported
gravimetric value uses the cell mass and converts eV to joules.

For a hybrid property, preserve the same member index when adding the
member's harmonic correction:

$$C_{P,\mathrm{hyb},i}^{\mathrm{G}}=C_{P,i}^{\mathrm{G}}+\Delta C_{\mathrm{har},i},\qquad \sigma_{\mathrm{hyb}}^{\mathrm{G}}=\operatorname{SD}_i(C_{P,\mathrm{hyb},i}^{\mathrm{G}}).$$

This retains classical–harmonic covariance. Adding their separate standard
deviations in quadrature would lose that covariance.

## What the current plots show

The updated loading-comparison plots deliberately retain the original CEA
central hybrid values and CEA sampling errors. DPOSE supplies the classical
LLPR member spread; the same members' harmonic deviations are paired with it
to obtain the hybrid model spread. The plotted band combines that spread with
the retained sampling error. Thus the curve values do not change when the
uncertainty method changes.

`plot_uncertainty_components.py --uncertainty-method both` writes matching
CEA and DPOSE figures with common axes and colors. It reads the same MD
MD sampling and Hessian model errors for both methods, and checks that they
match exactly. Hessian sampling uncertainty is excluded from these plots. See
the [plotting command](../scripts/properties/README.md)
for reproducible figures and companion CSVs.

LLPR evaluation now defaults to every saved production frame (stride 1).
The [all-frame workflow](../scripts/properties/README.md#reevaluate-llpr-on-every-saved-frame)
shows how to rerun the CEA analysis, regenerate DPOSE from the same frames,
and plot both with `--require-all-frames`. The plots read refreshed classical
members while retaining the saved MD and Hessian errors, record actual frame
counts, and reject stale comparisons. The current example trajectories have
8,001 production coordinate frames, so a literal 10,000-frame calculation
requires a trajectory containing that many saved production frames.

Rerun into the same analysis directory to reuse cached LLPR energies. The
evaluator matches existing and interrupted rows by timestep, evaluates only
the missing frames, and atomically updates the archive in trajectory order.
Then regenerate DPOSE and the plots from the updated CEA aggregate.

The effect is case-dependent. In the saved 300 K diagnostic, the classical
model SDs (J g$^{-1}$ K$^{-1}$) were:

| Model | 50 CH4 CEA → Gaussian | 100 CH4 CEA → Gaussian | 150 CH4 CEA → Gaussian |
| --- | ---: | ---: | ---: |
| PET-MAD | 0.112 → 0.093 | 0.355 → 0.132 | 0.094 → 0.122 |
| PET-SOL | 0.153 → 0.090 | 0.159 → 0.129 | 0.095 → 0.219 |

These are classical member spreads, not final hybrid bands or confidence
intervals. They show that the Gaussian estimate can be smaller or larger;
they do not establish that either model is more accurate. The saved
trajectories also have large $\operatorname{Var}(\beta\Delta U_i)$ and
near-one direct-reweighting effective sample counts in some cases. This warns
that both first-order CEA and Gaussian-tail assumptions need validation.

## Run and validate the comparison

From the repository root, compare member-resolved CEA and Gaussian estimates
using saved analysis outputs:

```bash
python scripts/properties/compare_heat_capacity_estimators.py \\
  --model-label pet-mad-1.5-s-40nn --loading 100
```

It writes comparison CSV, NPZ, and PNG files beside the trajectory analysis
archives; it does not run MD. To plot the existing CEA central curves with
Gaussian NPT uncertainty, use:

```bash
python scripts/properties/plot_hybrid_by_loading.py \\
  --model-label pet-mad-1.5-s-40nn --loadings 0,50,100,150 \\
  --uncertainty-method dpose --output comparison.png
```

The Gaussian approximation is exploratory for this NPT system. Stronger
validation requires block resampling long enough to capture correlations in
the energy variances and CEA covariance terms, checks of stationarity and
joint tails of $(Q_i,\Delta U_i)$, and comparison with simulations under
selected individual members. Calibration against held-out reference data is
also needed to interpret LLPR spread as predictive uncertainty. Do not shrink
the spread merely to make bands smaller, or divide it by the number of
members.

See [CEA.md](CEA.md) for the existing CEA error budget and
[`scripts/properties/README.md`](../scripts/properties/README.md) for the
analysis workflow.
