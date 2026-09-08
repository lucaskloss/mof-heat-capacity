# LLPR shallow ensembles for CEA

The CEA trajectory analysis needs signed, member-resolved potential energies.
For PET-MAD and PET-SOL, this project obtains them from metatrain's last-layer
prediction-rigidity (LLPR) wrapper. LLPR is less expensive than training a full
deep committee, but it samples only the final linear readout and must be
reported as **LLPR ensemble uncertainty**.

The LLPR definition follows Bigi *et al.*,
[*A prediction rigidity formalism for low-cost uncertainties in trained neural
networks*](https://doi.org/10.1088/2632-2153/ad805f), and its implementation
here uses the
[metatrain LLPR wrapper](https://metatrain.readthedocs.io/en/stable/architectures/generated/llpr.html).
CEA follows Imbalzano *et al.* as summarized in
[IMBALZANO_2021_UNCERTAINTY.md](IMBALZANO_2021_UNCERTAINTY.md).

LLPR and CEA do two different jobs. LLPR constructs a calibrated distribution
of plausible potential-energy predictions. CEA propagates samples from that
distribution through an equilibrium average. Neither method estimates the
finite-duration error of the MD trajectory; that error is calculated
separately from autocorrelation-aware statistics and independent replicas.

## What LLPR calculates

Let the central PET energy readout be
$\bar V(\mathbf x)=\mathbf w_0^{\mathrm T}\mathbf f(\mathbf x)$, where
$\mathbf f$ contains the frozen last-layer features of structure $\mathbf x$.
If $\mathbf F$ collects the corresponding features from the covariance set,
the metatrain LLPR predictive variance has the form

$$\sigma_{\mathrm{LLPR}}^2(\mathbf x)=\alpha^2\mathbf f(\mathbf x)^{\mathrm T}\left(\mathbf F^{\mathrm T}\mathbf F+\lambda\mathbf I\right)^{-1}\mathbf f(\mathbf x).$$

$\lambda$ regularizes the feature covariance and $\alpha$ is a single scale
factor fitted to reference-energy residuals on the separate calibration set.
The default `absolute_residuals` calibration is robust to outliers and assumes
a Gaussian error distribution. The resulting `energy_uncertainty` is an
analytical **standard deviation in eV**, not a variance and not a standard
error of a trajectory mean.

For CEA, the scalar standard deviation is insufficient because reweighting
needs the sign and correlation of each energy perturbation from frame to
frame. The wrapper therefore samples persistent last-layer weights from the
same calibrated covariance and exports their predictions as
`energy_ensemble`:

$$V^{(i)}(\mathbf x)=\mathbf w_i^{\mathrm T}\mathbf f(\mathbf x),\qquad \mathbf w_i\sim\mathcal N\!\left(\mathbf w_0,\alpha^2\left(\mathbf F^{\mathrm T}\mathbf F+\lambda\mathbf I\right)^{-1}\right).$$

The finite-member standard deviation of `energy_ensemble` should approach
`energy_uncertainty` as the number of members increases. The export validator
records their RMS discrepancy as a convergence diagnostic. CEA uses
`energy_ensemble`; the analytical `energy_uncertainty` is archived for
validation but is not inserted directly into an enthalpy or heat-capacity
formula.

This construction varies only the last linear readout while keeping the PET
representation fixed. It can measure uncertainty associated with directions
that are weakly constrained in that feature space, after calibration, but it
cannot expose a bias common to the representation, missing chemistry, errors
in the reference labels, or all uncertainty that would be seen across fully
independently trained neural networks.

## Required data

LLPR construction needs two labeled datasets that are not included here:

- a covariance/training set representative of the structures used to fit the
  base PET representation; and
- a disjoint calibration/validation set with reference energies from the same
  electronic-structure definition.

The calibration set should cover the MOF-5 loadings, temperatures, adsorption
environments, and framework distortions that will be interpreted. MD energies
predicted by the base model are not reference labels and must not be used to
calibrate that model's uncertainty.

Both inputs may be extended XYZ files. `--energy-key` selects the reference
energy property and defaults to `energy`; the default units are eV and
angstrom.

## Build an ensemble

Run the command from the repository root on an Izar login node:

```bash
./scripts/setup/submit_llpr_ensemble.sh --model pet-mad \
  --training-set /path/to/pet-training-or-covariance.extxyz \
  --validation-set /path/to/mof-calibration.extxyz \
  --energy-key energy --members 32 --dry-run
```

Remove `--dry-run` to submit one GPU job. Build PET-MAD and PET-SOL ensembles
separately. The default outputs are:

```text
models/pet-mad-1.5-s-llpr-ensemble.pt
models/pet-sol-s-best-llpr-ensemble.pt
```

The command invokes metatrain's `llpr` architecture without `num_epochs`, so it
computes and calibrates analytical LLPR and samples shallow members without
additional gradient training. It refuses to overwrite an existing model. The
generated provenance file records the base and LLPR checkpoints, datasets,
hashes, options, software versions, and export validation results.

The post-export check requires `energy`, `energy_uncertainty`, and
`energy_ensemble`, verifies the requested member count, rejects non-finite
predictions, and requires the member mean to reproduce the central prediction.
This checks the export mechanics; it does not establish scientific calibration.

Treat the member count, covariance data, calibration split and method,
regularizer, and seed as convergence parameters. The default 32 members follow
the metatrain example and are only a starting point.

## Use the ensemble with CEA

After the ensemble passes independent calibration and coverage checks, submit
the existing 50-methane trajectories for post-processing:

```bash
./scripts/properties/submit_analysis.sh --model pet-mad --loading 50 \
  --replicas 1 --model-uncertainty \
  --uncertainty-model models/pet-mad-1.5-s-llpr-ensemble.pt
```

The same exported file is reused at every temperature so each member identity
is persistent across the enthalpy derivative. Inspect direct-reweighting
effective sample counts and `var(beta Delta V)` alongside the CEA curve. CEA
model spread does not replace autocorrelation-aware MD sampling errors or
independent replicas.

For every selected production frame $t$ at temperature $T$, the analysis reads
the trajectory-driving energy $\bar V_t$, kinetic energy $K_t$, and volume
$\mathcal V_t$, and evaluates every LLPR member energy $V_t^{(i)}$. It then
constructs

$$\Delta V_t^{(i)}=V_t^{(i)}-\bar V_t,\qquad H_t^{(i)}=K_t+V_t^{(i)}+P_{\mathrm{ext}}\mathcal V_t.$$

The first-order CEA enthalpy for member $i$ is calculated in
`committee_enthalpy_estimates` as

$$\left\langle H^{(i)}\right\rangle_i^{\mathrm{CEA}}=\left\langle H^{(i)}\right\rangle_{\bar V}-\beta\operatorname{cov}_{\bar V}\!\left(H^{(i)},\Delta V^{(i)}\right),\qquad \beta=(k_{\mathrm B}T)^{-1}.$$

The same function also computes normalized exponential reweighting as a
diagnostic. At each temperature, replicas are averaged member by member. Only
after each persistent member has a complete enthalpy curve does
`write_model_uncertainty_outputs` calculate

$$C_{P,\mathrm{cl}}^{(i)}(T)=\frac{\mathrm d\langle H^{(i)}\rangle_i^{\mathrm{CEA}}}{\mathrm dT},\qquad \sigma_{\mathrm{LLPR},C_P}(T)=\sqrt{\frac{1}{M-1}\sum_{i=1}^{M}\left[C_{P,\mathrm{cl}}^{(i)}(T)-\overline C_{P,\mathrm{cl}}(T)\right]^2}.$$

Thus the reported LLPR model error is the sample standard deviation of the
**final member-resolved heat-capacity curves**. It is not the framewise energy
spread, and it is not divided by $\sqrt M$: the members represent a predictive
distribution, rather than $M$ repeated measurements used to estimate its
mean.

## Where every reported uncertainty is calculated

| Reported quantity | Calculation | Code location | Main output |
| --- | --- | --- | --- |
| Framewise LLPR energy standard deviation | Calibrated last-layer feature covariance, equation above | metatrain during LLPR preparation and inference | `energy_uncertainty`; archived as `analytical_energy_uncertainty_eV` in each `model_uncertainty.npz` |
| Direct-reweighting overlap | $N_{\mathrm{eff}}=1/\sum_t\widetilde w_t^2$ for each member | [`committee_enthalpy_estimates`](../mof_heat_capacity/analysis/uncertainty.py) | `direct_effective_samples` |
| CEA validity diagnostic | $\operatorname{var}_t(\beta\Delta V_t^{(i)})$ | [`committee_enthalpy_estimates`](../mof_heat_capacity/analysis/uncertainty.py) | `dimensionless_delta_variance` |
| Classical LLPR model error | Sample standard deviation across member-resolved CEA $C_P$ curves | [`write_model_uncertainty_outputs`](../mof_heat_capacity/analysis/results.py) | `cea_cp_model_standard_deviation_J_per_gK` |
| MD sampling error | Correlated standard error $s/\sqrt{N_{\mathrm{eff}}}$ within a run, then within- and between-replica contributions | [`summarize_series`](../mof_heat_capacity/analysis/statistics.py) and [`_enthalpy_records`](../mof_heat_capacity/analysis/hybrid.py) | `classical_anharmonic_cp_standard_error_J_per_gK` |
| Harmonic sampling error | Standard error across independently quenched loaded minima | [`_harmonic_corrections`](../mof_heat_capacity/analysis/hybrid.py) | `harmonic_quantum_correction_standard_error_J_per_gK` |
| Combined hybrid uncertainty | Quadrature of MD, harmonic-minimum, and classical LLPR terms | [`run`](../mof_heat_capacity/analysis/hybrid.py) | `approximate_cp_combined_standard_uncertainty_J_per_gK` |

The component arrays are more informative than the combined band. The
quadrature result assumes independence. With only one MD replica there is no
between-replica estimate; with only one loaded minimum the stored harmonic
standard error is zero, which means "not estimated" rather than "known
exactly."

The calculation does not currently propagate LLPR members through geometry
optimization or PET-JAX Hessians. Consequently the LLPR term applies only to
the classical NPT contribution. The combined band also omits shared MLIP bias,
reference-method uncertainty, finite-size error, temperature-grid bias, and
other systematic convergence errors.

Before interpreting the result, validate normalized reference residuals and
coverage on a held-out test set, compare analytical LLPR uncertainty with the
finite-member spread, converge the number of members and frame stride, and
compare CEA with direct reweighting wherever direct weights retain useful
effective sample size.
