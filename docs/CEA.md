# CEA uncertainty for MOF-5 heat capacity

This guide explains how the project uses the cumulant expansion approximation
(CEA) to estimate LLPR model uncertainty in methane-loaded MOF-5 heat
capacity. The method follows Imbalzano *et al.*, [J. Chem. Phys. **154**,
074102 (2021)](https://doi.org/10.1063/5.0036522); the source paper is
[available here](CEA.pdf). A related worked example is the Atomistic
Cookbook's [PET-MAD uncertainty recipe](https://atomistic-cookbook.org/examples/pet-mad-uq/pet-mad-uq.html#cumulant-expansion-approximation-cea).

## Project workflow at a glance

1. Run classical NPT MD with the central PET model at each temperature and
   replica.
2. Evaluate each supplied, persistent LLPR member on selected production
   frames. Each member supplies a potential energy as well as an enthalpy
   observable for those same frames.
3. At each temperature, estimate each member's mean enthalpy with CEA. Direct
   exponential reweighting is retained as a diagnostic.
4. Average replicas member by member, then finite-difference each member's
   full enthalpy-versus-temperature curve.
5. Calculate the LLPR spread across the resulting member heat-capacity curves.
   Add the matching member-resolved harmonic correction for the hybrid result.

LLPR changes the final PET energy readout while keeping the representation
fixed. Report the resulting spread as **LLPR model uncertainty**. It is a
predictive spread across members, not an error bar on their mean. See
[LLPR.md](LLPR.md) for supplied checkpoint details.

## CEA enthalpy at one temperature

For frame $t$ and member $i$, define the member-to-central energy difference
and the member enthalpy as

$$\Delta V_t^{(i)}=V_t^{(i)}-\bar V_t,\qquad H_t^{(i)}=K_t+V_t^{(i)}+P_{\mathrm{ext}}V_{\mathrm{cell},t}.$$

Here $\bar V_t$ is the potential that drove the MD, $K_t$ is the saved kinetic
energy, and the saved NPT cell volume supplies the pressure term. At fixed
pressure the pressure contribution stays in the enthalpy observable; the
reweighting difference is $\Delta V$.

The project uses the first-order CEA estimate

$$h_i^{\mathrm{CEA}}(T)=\overline{H^{(i)}}-\beta\,\operatorname{cov}(H^{(i)},\Delta V^{(i)}),\qquad \beta=(k_{\mathrm B}T)^{-1}.$$

The averages and covariance are ordinary trajectory averages. The covariance
correction accounts for the fact that a member changes which frames are more
probable: frames with higher member energy receive less weight. Direct
reweighting, used for comparison, is

$$h_i^{\mathrm{direct}}(T)=\frac{\sum_t e^{-\beta\Delta V_t^{(i)}}H_t^{(i)}}{\sum_t e^{-\beta\Delta V_t^{(i)}}}.$$

Direct reweighting can be dominated by a few frames in a large periodic
system, so CEA is the primary estimate. The analysis records both the direct
weight-effective sample count
$N_{\mathrm{eff}}=1/\sum_t\widetilde w_t^2$ and
$\operatorname{var}(\beta\Delta V^{(i)})$. Low $N_{\mathrm{eff}}$ signals poor
overlap for direct reweighting; variance not much smaller than one warns that
first-order CEA may be inaccurate. Neither diagnostic alone establishes
convergence.

## From enthalpy to classical heat capacity

At each temperature, average each member's CEA enthalpy across replicas while
preserving its identity. The same member index and checkpoint hash must be
used at every temperature. For member $i$, calculate

$$C_{P,\mathrm{cl}}^{(i)}(T_j)=\frac{\mathrm d h_i^{\mathrm{CEA}}}{\mathrm dT}(T_j).$$

The finite-difference stencil matches the main hybrid analysis: centered
finite differences at interior grid points and second-order one-sided
finite differences at the endpoints. On a uniform grid with spacing $\Delta T$,
an interior point is

$$C_{P,\mathrm{cl}}^{(i)}(T_j)\approx\frac{h_i(T_{j+1})-h_i(T_{j-1})}{2\Delta T}.$$

The classical LLPR heat-capacity uncertainty is the sample standard deviation
across the differentiated member curves:

$$\sigma_{\mathrm{MLIP,cl}}(T_j)=\operatorname{SD}_{i}\!\left[C_{P,\mathrm{cl}}^{(i)}(T_j)\right].$$

Differentiate each paired member curve before taking the spread. Differencing
independent scalar enthalpy error bars would lose the cross-temperature
correlations and does not reproduce the implemented uncertainty.

## Error budget for the final hybrid curve

The gravimetric hybrid estimate is

$$C_{P,\mathrm{hyb}}(T)=C_{P,\mathrm{cl}}^{\mathrm{NPT}}(T)+\Delta C_{\mathrm{har}}(T),\qquad \Delta C_{\mathrm{har}}=C_{V,\mathrm{har}}^{\mathrm{qn}}-C_{V,\mathrm{har}}^{\mathrm{cl}}.$$

The reported curve uses the central-model classical result and central-model
harmonic correction averaged over the selected minima. Member-resolved curves
are used to estimate the LLPR uncertainty around it.

The reported total has three components. The MD and harmonic terms are
sampling uncertainties; the LLPR term is model spread. Each section below
gives the corresponding saved output field.

### Sampling terms

For each temperature $T_k$ and replica $r$, let $e_{rk}$ be the
autocorrelation-aware standard error of its production enthalpy mean. The
combined enthalpy error is

$$e_{H,k}^2=\left(\frac{1}{R}\sqrt{\sum_{r=1}^{R}e_{rk}^2}\right)^2+\frac{s_{\bar H,k}^2}{R},\qquad s_{\bar H,k}=\operatorname{SD}_{r}(\bar H_{rk}).$$

The code draws 10,000 synthetic enthalpy curves from these mean and error
estimates, finite-differences every curve, and takes the standard deviation
at each temperature to obtain $\sigma_{\mathrm{MD}}$. Temperature points are
drawn independently, so correlations between errors from separate
isothermal runs are not represented. The harmonic sampling error is
$\sigma_{\mathrm{har}}=\operatorname{SD}_{r}(\Delta C_{\mathrm{har},r})/\sqrt R$.
With one minimum it is recorded as zero, meaning minimum-to-minimum error was
not estimated; it does not mean the correction is exact. The separate terms
are saved as `classical_anharmonic_cp_standard_error_J_per_gK` and
`harmonic_quantum_correction_standard_error_J_per_gK`.

The combined sampling-only error saved in
`approximate_cp_standard_error_J_per_gK` is

$$\sigma_{\mathrm{sampling,hyb}}(T)=\sqrt{\sigma_{\mathrm{MD}}^2(T)+\sigma_{\mathrm{har}}^2(T)}.$$

### LLPR terms and final total

For the hybrid LLPR term, center the classical heat-capacity and harmonic
correction values over members separately, add each member's two deviations,
then take the sample standard deviation:

$$d_i^{\mathrm{hyb}}(T)=\left[C_{P,\mathrm{cl}}^{(i)}-\overline C_{P,\mathrm{cl}}\right]+\left[\Delta C_{\mathrm{har}}^{(i)}-\overline{\Delta C}_{\mathrm{har}}\right],\qquad \sigma_{\mathrm{MLIP,hyb}}(T)=\operatorname{SD}_{i}[d_i^{\mathrm{hyb}}(T)].$$

This retains the covariance between the classical and harmonic effects of
each member. Their two separate standard deviations are diagnostic fields;
do **not** add them independently to the hybrid error. LLPR Hessians are
evaluated at the central-model minimum, so member-specific geometry
relaxation is excluded. The paired spread is saved as
`approximate_cp_model_standard_deviation_J_per_gK`; the separate diagnostic
spreads are `classical_anharmonic_cp_model_standard_deviation_J_per_gK` and
`harmonic_quantum_correction_model_standard_deviation_J_per_gK`.

The final gravimetric uncertainty is

$$\sigma_{\mathrm{total,hyb}}(T)=\sqrt{\sigma_{\mathrm{sampling,hyb}}^2(T)+\sigma_{\mathrm{MLIP,hyb}}^2(T)}.$$

This quadrature treats MD sampling, harmonic minimum sampling, and LLPR model
spread as independent. The final value is saved as
`approximate_cp_combined_standard_uncertainty_J_per_gK`. If no LLPR member
arrays are available, the combined field contains only the two sampling terms
and the model-spread field is unavailable. The total is an uncertainty
estimate, not a bound on shared model bias, reference-method error,
finite-size error, temperature-grid bias, or CEA truncation error.

For volumetric heat capacity, $C_{\mathrm{vol}}=\rho C_{\mathrm{grav}}$.
Density uncertainty is propagated as

$$\sigma_{\mathrm{sampling,vol}}^2=(\rho\,\sigma_{\mathrm{sampling,hyb}})^2+(C_{\mathrm{grav}}\,\sigma_{\rho})^2,$$

assuming zero covariance between heat capacity and density. The model term is
scaled by density, $\sigma_{\mathrm{MLIP,vol}}=\rho\sigma_{\mathrm{MLIP,hyb}}$,
and the two volumetric terms are combined in quadrature.

## Run and interpret

Use the supplied LLPR checkpoint, run trajectory analysis, compute loaded and
empty Hessians as required, then assemble the hybrid result. See
[LLPR.md](LLPR.md) for checkpoint locations and validation details and
[`scripts/properties/README.md`](../scripts/properties/README.md) for complete
commands and options. The standard sequence is:

```bash
./scripts/properties/submit_analysis.sh --model pet-mad --loading 50 \
  --replicas 1 --model-uncertainty
./scripts/properties/submit_heat_capacity.sh --model pet-mad --loading 50 \
  --source-temperatures 200,225,250,275,300,325,350,375,400 --replicas 1
./scripts/properties/submit_hybrid_analysis.sh --model pet-mad --loading 50 \
  --replicas 1 --model-uncertainty
```

Before interpreting a curve, check that the same LLPR checkpoint was used for
enthalpy and Hessian members, inspect $N_{\mathrm{eff}}$ and
$\operatorname{var}(\beta\Delta V)$ at every temperature, and check trajectory
stationarity, autocorrelation, replica agreement, and Hessian stability. CEA
is first order and can be inaccurate when member energy differences fluctuate
substantially. Its reported spread also excludes bias shared across members.
