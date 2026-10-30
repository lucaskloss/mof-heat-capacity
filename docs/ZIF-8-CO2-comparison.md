# ZIF-8 + CO₂: experimental paper and comparison with the current simulations

Your existing ZIF-8 campaign uses **54 CO₂ molecules in a 2×2×2 supercell containing 96 ZIF-8 formula units**. Its loading is therefore **0.5625 mol CO₂/mol ZIF-8**, almost exactly the paper's **0.56** sample. Use that sample as the primary composition match; unlike the MOF-303/water comparison, interpolation between substantially different loadings is unnecessary.

The main qualification is thermodynamic: the experiment measures **heat capacity of a sealed system**, which the authors explicitly distinguish from $C_P$. Your existing hybrid result approximates $C_P$ of a periodic loaded crystal. The numerical comparison below is useful, but it is not a comparison of rigorously identical observables.

## Sources and saved data

Peter F. Rosen, Matthew S. Dickson, Jason J. Calvin, Nancy L. Ross, Tomislav Friščić, Alexandra Navrotsky, and Brian F. Woodfield, “Thermodynamic Evidence of Structural Transformations in CO₂-Loaded Metal–Organic Framework Zn(MeIm)₂ from Heat Capacity Measurements,” *Journal of the American Chemical Society* **142** (2020), 4833–4841. [Article DOI](https://doi.org/10.1021/jacs.9b13883); [supplied paper](ZIF-8-CO2.pdf).

I also retrieved the authors' publicly available [Supporting Information](https://doi.org/10.1021/jacs.9b13883.s001) from ACS Figshare. Files saved beside this note:

- [Supporting Information PDF](ZIF-8-CO2-supporting-information.pdf): original `ja9b13883_si_001.pdf`, downloaded from [Figshare file 21833136](https://ndownloader.figshare.com/files/21833136). Its MD5, `548742178ffc97d603cd0f3d66853b44`, matches the publisher's record.
- [Experimental data CSV](ZIF-8-CO2-X056-experimental.csv): all **92 temperature/heat-capacity pairs from SI Table 3, p. S3**, extracted and sorted by temperature. The CSV contains measured molar values, without smoothing or mass conversion. The relevant table was visually checked against the PDF.

The supplement is attributed to Rosen et al. and distributed by ACS Figshare under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The CSV is a reformatted extraction of that table. Numerical interpolations and simulation comparisons in this note are calculations made here. Main-paper references below use journal page numbers; SI pages are labeled S1–S7.

## What the paper establishes

The purpose is to identify changes in the loaded framework through low-temperature calorimetry, particularly possible gate opening caused by CO₂. It is not primarily a room-temperature heat-capacity correlation paper.

- **Material:** ZIF-8, the sodalite framework $\mathrm{Zn(MeIm)_2}$, where MeIm is 2-methylimidazolate. The authors report solvent-free material of approximately 98% purity (p. 4835).
- **Preparation:** samples were sealed in aluminum pans under CO₂. Buoyancy-corrected weighing before and after sealing determined the total gas amount. Nominal loading pressures were 207–689 kPa, but pressure alone did not reproducibly determine uptake.
- **Measurement:** thermal-relaxation calorimetry on a Quantum Design PPMS, nominally **1.8–300 K**. Aluminum DSC pans served as pressure containers; this was **not a conventional DSC heat-flow scan**. Contributions from the pan, copper, grease, and calorimeter addenda were subtracted (p. 4835).
- **Loadings:** 0.29, 0.56, 0.80, 0.87, and 0.99 mol CO₂ per mol ZIF-8, with a previous measurement of empty ZIF-8 as reference.
- **Broad anomaly:** samples at loading 0.56 and above exhibit a broad excess-heat-capacity feature spanning approximately **70–220 K**, with a shoulder suggesting overlapping processes. The authors associate the shoulder with gradual linker rotation/gate opening; the remaining feature may involve guest rearrangement, lattice expansion, or both. These microscopic assignments remain interpretations of the calorimetry and supporting literature (pp. 4837–4839).
- **Low-temperature feature:** all loaded samples show a smaller excess feature below approximately 70 K, attributed to changed vibrational modes arising from host–guest interactions.
- **Low-loading behavior:** the 0.29 sample differs substantially, including a feature near 190 K. The authors place the threshold for the broad higher-loading behavior somewhere between **0.29 and 0.56**, rather than establishing a precise critical loading.
- **Free-CO₂ control:** CO₂ in an otherwise empty pan exhibits a much larger sublimation feature near 210 K. Its behavior does not explain the observed loaded-framework anomalies simply as free-gas sublimation.

The authors use baseline subtraction and spline integration to quantify excess enthalpy and entropy. For the 0.56 sample, main-paper Table 1 reports approximately **4374 J/mol CO₂** and **46.29 J/(mol CO₂ K)**. Those are integrated excess quantities, not values of heat capacity. The footnote specifies integration to 230 K for this sample, despite the table's reference to 298.15 K; preserve that convention if reproducing their integrals.

## Loading convention: count formula units, not unit cells

The paper writes the composition as $\mathrm{ZIF\!\! -\!\! 8}\cdot X\mathrm{CO_2}$, with:

$$X=\frac{n_{\mathrm{CO_2}}}{n_{\mathrm{ZIF8}}}=\frac{N_{\mathrm{CO_2}}}{N_{\mathrm{formula\ units}}}.$$

One mole of ZIF-8 means one mole of **$\mathrm{ZnC_8H_{10}N_4}$ formula units**, each containing one Zn atom. It does not mean one mole of crystallographic cells, cages, or complete simulation supercells. $X=0.56$ is not 56% saturation, 56 wt%, or a relative pressure.

The [campaign metadata](../output/md/structures/zif8_2x2x2/54co2/300K/rep01/inputs.json) records `loading = 54` and `host_atoms = 2208`. Reading [the actual CIF](../input/zif8_2x2x2.cif) gives $\mathrm{Zn_{96}C_{768}H_{960}N_{384}}$, hence **96 formula units**:

$$X_{\mathrm{sim}}=\frac{54}{96}=0.5625.$$

The smaller [ZIF-8 CIF](../input/zif8.cif) contains 12 formula units and 276 atoms. Replicating it 2×2×2 gives 96 formula units and 2208 atoms. The loaded simulation contains 2370 atoms after adding 54 three-atom CO₂ molecules.

### Mass and molar conversions

Using ASE masses for the actual structures:

$$M_0=227.576\ \mathrm{g/mol\ ZIF8},\qquad M_{\mathrm{CO_2}}=44.009\ \mathrm{g/mol},\qquad M(X)=M_0+XM_{\mathrm{CO_2}}.$$

$$w=\frac{m_{\mathrm{CO_2}}}{m_{\mathrm{dry\ ZIF8}}}=\frac{XM_{\mathrm{CO_2}}}{M_0},\qquad x_{\mathrm{CO_2}}=\frac{w}{1+w},\qquad L\,[\mathrm{mmol/g\ dry\ ZIF8}]=\frac{1000X}{M_0}.$$

| Quantity | Your 54-CO₂ simulation |
|---|---:|
| $X$, mol CO₂/mol ZIF-8 | **0.5625** |
| CO₂/dry-framework mass, $w$ | 0.108777 g/g |
| CO₂ mass fraction of the loaded sample | 9.8105% |
| Uptake per dry framework mass | 2.47170 mmol/g |
| Dry supercell mass | 21847.296 u |
| Loaded supercell mass | 24223.782 u |
| Loaded molar mass per mole of ZIF-8 formula units | **252.33106 g/mol** |

The paper's sample map is:

| Paper's $X$ | Equivalent CO₂ per 96-formula-unit host | Nearest integer count | Initial sealing pressure, kPa | SI sample molar mass, g/mol |
|---:|---:|---:|---:|---:|
| 0.29 | 27.84 | 28 | 207 | 240.1 |
| **0.56** | **53.76** | **54** | **345** | **252.3** |
| 0.80 | 76.80 | 77 | 689 | 262.9 |
| 0.87 | 83.52 | 84 | 517 | 266.0 |
| 0.99 | 95.04 | 95 | 689 | 271.1 |

The SI reports 10.86 mg ZIF-8 and 1.18 mg CO₂ for the 0.56 sample. These rounded masses imply $X\approx0.56187$, even closer to your 0.5625. The reported mass standard uncertainty is 0.06 mg for each mass. Assuming independent mass uncertainties gives an illustrative $u(X)\approx0.029$, much larger than the 0.0025 difference between your loading and the printed label. This propagation is our estimate, not an author-reported loading error bar. There is no basis for treating the two-decimal loading label as an exact composition requiring a finely tuned correction.

If a different run uses **50** CO₂ in this same cell, its loading would instead be **$50/96=0.52083$**. Recompute the comparison for that run; the results below apply to the **54-CO₂ campaign found in this checkout**.

## Which heat capacity should be compared?

### Match total-system heat capacity and its mass basis

Use **Figure 1 and SI Table 3**, which give the heat capacity of the loaded sample per mole of ZIF-8 formula units, with its associated CO₂. Do not compare your total loaded-system heat capacity directly with Figures 3–6, where one or both reference contributions have been subtracted and some results are expressed per mole of CO₂.

Convert the SI's molar heat capacity to specific heat using its explicitly stated loaded molar mass:

$$c_{\mathrm{exp}}(T)=\frac{C_{m,\mathrm{exp}}(T)}{252.3\ \mathrm{g/mol}}.$$

This gives $\mathrm{J\,g^{-1}\,K^{-1}}$ **per gram of ZIF-8 plus CO₂**, matching the mass basis of your CSV's **`approximate_cp_J_per_gK`**. The denominator is not 227.576 g/mol, which would normalize by dry framework mass, and not 44.009 g/mol, which is the CO₂ molar mass.

Alternatively, put your simulation onto the paper's molar axis:

$$C_{m,\mathrm{sim}}(T)=c_{p,\mathrm{sim}}(T)\left(252.3310625\ \mathrm{g/mol}\right).$$

Each sample uses its own loaded molar mass. The very small difference between 252.3 and 252.33106 is consistent with the rounded experimental composition. For a whole-cell heat capacity in eV/K, conversion to J/(mol ZIF-8 K) is $96485.33212/96$ times the whole-cell value. Do not apply a whole-cell conversion to values already in J/g/K.

### Keep the thermodynamic conditions explicit

The authors state that **total CO₂ amount, not pressure, is held fixed**. Neglecting expansion of the sealed pan, they describe the result as approximately constant-volume heat capacity of the **whole system: crystal plus any external CO₂** (pp. 4835–4836). They cannot determine exact pressure or the fraction adsorbed at each temperature, and assume all CO₂ is sorbed for their normalization. They acknowledge likely free gas near room temperature.

Consequences for your simulation:

1. **345 kPa is a preparation pressure**, not a pressure maintained throughout the measurement. Do not set every MD temperature to 345 kPa and call that an exact experimental match. The 1.2 mPa pressure listed for the PPMS is outside the sealed pan, not the pressure experienced inside the ZIF-8 sample.
2. A fixed-number periodic cell matches the nominal total composition, but places every CO₂ molecule inside the crystal. It has no external pan headspace and no reservoir exchanging molecules with the pores. This mismatch may matter particularly near room temperature.
3. Your [hybrid implementation](../mof_heat_capacity/analysis/hybrid.py) estimates $C_P$ from an NPT enthalpy derivative plus a harmonic quantum correction. It is not automatically equal to the measured sealed-cell $C$:

$$C_P^{\mathrm{hybrid}}\approx C_P^{\mathrm{classical,NPT}}+\left(C_V^{\mathrm{quantum,harmonic}}-C_V^{\mathrm{classical,harmonic}}\right).$$

For a homogeneous equilibrium material at fixed composition, $C_P-C_V=TV\alpha_V^2/\kappa_T$, using consistent extensive or molar quantities. Thermal expansion and compressibility could help estimate the ensemble difference, but the sealed pan's composite system is more complicated. Fixed-volume MD of a periodic crystal is not a full model of a rigid pan containing a crystal that can itself expand and exchange CO₂ with headspace. Do not subtract an ideal-gas $R$ or $N k_B$ correction from the whole loaded framework.

Use the current hybrid $C_P$ curve for a **qualified numerical comparison**, and quantify these condition differences before interpreting a few-percent residual as force-field error. A purely harmonic $C_V$ curve also cannot, by itself, capture structural redistribution through the broad anomaly.

## Numerical comparison with the current output

The source is your [hybrid CSV](../output/post-processing/harmonic-correction/zif8_2x2x2/pet-mad-1.5-s-40nn/54co2/heat-capacity.fd-two-gpu.exploratory-discard-imaginary.csv), for `pet-mad-1.5-s-40nn`. The [accompanying metadata](../output/post-processing/harmonic-correction/zif8_2x2x2/pet-mad-1.5-s-40nn/54co2/heat-capacity.fd-two-gpu.exploratory-discard-imaginary.json) flags discarded imaginary/near-zero modes **and** at least one Hessian evaluated at a relaxation frame that failed the requested force threshold. Treat these values as exploratory.

For each simulation temperature, I linearly interpolated **in temperature only** between neighboring measured values in SI Table 3, then divided by 252.3. This is not a fitted loading model. The supplement contains points up to 302.71 K, so evaluation at 300 K is interpolation even though the nominal experimental range ends at 300 K.

| $T$, K | Experimental $C_m$, J/(mol ZIF-8 K) | Experimental $c$, J/g/K | Simulated hybrid $c_p$, J/g/K | Simulation combined standard uncertainty, J/g/K | Signed difference |
|---:|---:|---:|---:|---:|---:|
| 200 | 198.6693 | 0.78743 | 0.80505 | 0.15026 | +2.24% |
| 225 | 208.1862 | 0.82515 | 0.83038 | 0.18461 | +0.63% |
| 250 | 224.1389 | 0.88838 | 0.87307 | 0.11573 | −1.72% |
| 275 | 238.9575 | 0.94712 | 0.92645 | 0.41480 | −2.18% |
| 300 | 253.9802 | 1.00666 | 0.96654 | 0.15099 | −3.98% |

The signed difference is $100(c_{p,\mathrm{sim}}-c_{\mathrm{exp}})/c_{\mathrm{exp}}$. These are **interpolated experimental references**, not measurements taken exactly at these five temperatures. Printed digits support reproducibility and do not imply this level of experimental precision.

SI Table 3 gives approximately **2% standard uncertainty in molar heat capacity** in this temperature region and about **4 mK** in temperature. As a practical display, a 2% experimental reference band corresponds to roughly 0.016–0.020 J/g/K here; it is not a rigorously propagated interpolation confidence band. Keep possible loading, interpolation, and thermodynamic-condition errors separate. The main paper's 0.1% resolution used for excess-anomaly plots must not replace the SI's 2% measurement uncertainty.

The simulation column uses `approximate_cp_combined_standard_uncertainty_J_per_gK`, rather than the much smaller sampling error alone. It does not include systematic errors from discarded modes, incomplete relaxation, or the experimental/simulation condition mismatch. The apparent few-percent agreement is therefore encouraging as a numerical check, but does not establish quantitative model validation.

## A practical comparison workflow

1. **Use the correct sample.** Label the simulation $N_{\mathrm{CO_2}}=54$, $N_{\mathrm{ZIF8}}=96$, $X=0.5625$, and the experiment $X\approx0.56$. No saturation fraction is needed.
2. **Use a common observable and unit label.** Plot total loaded-sample experimental $c$ and simulated hybrid $c_p$ in J/g/K, and state the condition mismatch in the caption. Retain both uncertainty sources.
3. **Restrict the comparison to the shared temperature range.** The present reported grid overlaps at **200, 225, 250, 275, and 300 K**. Do not extrapolate experimental values to 325–400 K. The 175 K trajectory is a differentiation support point, not automatically a reported heat-capacity result.
4. **Separate high-temperature agreement from transition validation.** The existing 200–300 K comparison covers only the upper tail of the 70–220 K anomaly and the higher-temperature region. It cannot establish the main peak, the low-temperature shoulder, or the below-70 K behavior.
5. **Resolve the exploratory flags before final validation.** Establish stable, force-converged minima and analyze the problematic modes. Recompute uncertainties and residuals after changing the estimator or relaxing structures further.
6. **For a future study of the anomaly, improve temperature and structural sampling.** The current 25 K derivative grid averages over 50 K between neighboring enthalpy points at an interior temperature, which can smooth features. A finer grid and convergence checks would be needed across approximately 70–220 K, alongside linker-angle distributions, guest arrangements, cell dimensions, and sampling of multiple configurations. Classical MD with a harmonic correction has additional limitations at these temperatures; simply extending the current campaign is not sufficient to guarantee a quantitatively accurate quantum transition.

The current calculation reuses a spectrum from a **400 K source structure** across temperatures. That approximation deserves particular scrutiny if low-temperature configurations differ structurally from the source minimum. A single minimum's harmonic spectrum cannot represent all configuration changes responsible for an excess heat-capacity anomaly.

### If comparing excess heat capacity instead

The paper's baseline convention, per mole of ZIF-8, is:

$$C_{m,\mathrm{excess}}(T,X)=C_{m,\mathrm{loaded}}(T,X)-C_{m,\mathrm{empty}}(T)-X C_{m,\mathrm{CO_2,solid\ reference}}(T).$$

Divide this result by $X$ for the excess per mole of CO₂ used in Fig. 5B. The CO₂ baseline is a **solid-CO₂ lattice reference**, extrapolated above the sublimation region using their theoretical function; it is not an ideal-gas heat capacity or the measured empty-pan CO₂ transition curve. Exact reproduction requires the same empty-framework and solid-CO₂ reference functions, referenced in the main paper. The supplied SI table of loaded data alone does not provide those functions. Start with the total heat capacity comparison above rather than inventing an excess baseline.

## Reproduce the conversion and numerical table

Run the following from the repository root in an environment with ASE and NumPy. It reads the extracted experimental data and your existing result without changing either file. The factor 252.3 comes directly from SI Table 3.

```python
from pathlib import Path

import numpy as np
from ase.io import read

host = read("input/zif8_2x2x2.cif")
guest = read("input/co2.gro")
n_formula = host.get_chemical_symbols().count("Zn")
n_co2 = 54
X = n_co2 / n_formula
M0 = host.get_masses().sum() / n_formula
M_sim = M0 + X * guest.get_masses().sum()
print(f"formula units={n_formula}; X={X:.6f}; M_sim={M_sim:.6f} g/mol")

exp = np.genfromtxt(
    "docs/ZIF-8-CO2-X056-experimental.csv", delimiter=",", names=True
)
T_exp = exp["temperature_K"]
C_exp = exp["C_m_J_per_mol_ZIF8_K"]
assert np.all(np.diff(T_exp) > 0)

path = Path(
    "output/post-processing/harmonic-correction/zif8_2x2x2/"
    "pet-mad-1.5-s-40nn/54co2/"
    "heat-capacity.fd-two-gpu.exploratory-discard-imaginary.csv"
)
sim = np.genfromtxt(path, delimiter=",", names=True)
T = sim["temperature_K"]
keep = (T >= max(1.8, T_exp.min())) & (T <= min(300.0, T_exp.max()))
T = T[keep]
Cm_ref = np.interp(T, T_exp, C_exp)
c_ref = Cm_ref / 252.3
c_sim = sim["approximate_cp_J_per_gK"][keep]
u_sim = sim["approximate_cp_combined_standard_uncertainty_J_per_gK"][keep]
relative = 100 * (c_sim - c_ref) / c_ref
print("T_K, C_exp_J_per_molK, c_exp_J_per_gK, c_sim_J_per_gK, u_sim, difference_pct")
for row in zip(T, Cm_ref, c_ref, c_sim, u_sim, relative):
    print(", ".join(f"{value:.6f}" for value in row))
```

The experimental CSV is a source-data extraction; the comparison table is a snapshot of the current exploratory simulation output. Regenerate it after new simulations or analysis.
