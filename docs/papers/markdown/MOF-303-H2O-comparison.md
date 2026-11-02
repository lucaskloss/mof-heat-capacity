# MOF-303 + H₂O: experimental paper and comparison with 50-water simulations

For the **1,024-atom MOF-303 2×2×2 framework actually used in this repository**, 50 water molecules correspond to **0.07106 g H₂O/g dry MOF**, or **16.76% of the paper's saturated water loading** ($q=0.16761$). The relevant experimental curves are therefore the ones at **$q=0.1202$ and $q=0.3425$**. There is no measured curve at exactly your loading; comparison requires interpolation or the paper's global correlation.

## Source and scope

Ahmed E. Abu El-maaty, Ourida Saoudi, Ahmed S. Abdelrazik, and Rached Ben-Mansour, “Experimental investigation of specific heat and thermal stability of MOF-303 at different water vapor loading and various temperatures,” *International Communications in Heat and Mass Transfer* **165** (2025), 109088. [DOI](https://doi.org/10.1016/j.icheatmasstransfer.2025.109088); [supplied PDF](MOF-303-H2O.pdf).

Page references below refer to the supplied 14-page PDF. Experimental statements and coefficients come from that PDF; loading conversions, interpolated values, and simulation comparisons are calculations made here. The supplementary file and authors' raw DSC data were not supplied. The paper says data are available on request (p. 14).

## What the paper measures

- **Material:** solvothermally synthesized MOF-303, characterized using diffraction, nitrogen sorption, microscopy, TGA, and water sorption. The reported surface area is approximately $1250\ \mathrm{m^2/g}$ (p. 2).
- **Preparation:** saturate the material with water vapor, then partially dry separate samples using TGA. Heating to 150 °C defines the authors' dry reference. Seal each sample after preparation to retain its chosen total water content (pp. 3–6).
- **Measurement:** conventional DSC with sapphire calibration and empty-pan baseline subtraction. The reported specific heat is $c_p$, in $\mathrm{J\,g^{-1}\,K^{-1}}$, over **10–95 °C = 283.15–368.15 K**. Temperature increments in kelvin and Celsius have identical sizes, so $\mathrm{J/(g\,^{\circ}C)}$ and $\mathrm{J/(g\,K)}$ are numerically equivalent. Heating/cooling ramps use 10 °C/min.
- **Loading series:** seven water contents from dry to saturated. The preparation temperatures identify samples; they are **not** the temperatures at which those samples must be compared with MD. Each prepared sample is measured across the DSC temperature range.
- **Main result:** heat capacity generally increases with both temperature and water content. Dry MOF-303 increases from about **0.9175** at 10 °C to **1.1695** at 95 °C; saturated material increases from about **1.72** to **2.41**, in $\mathrm{J\,g^{-1}\,K^{-1}}$ (pp. 6–9).
- **Repeatability:** the authors report three measurements per temperature in Fig. 8, with most standard deviations below 5% of the mean. Sapphire validation is reported to have a maximum deviation of 2.38%. Neither statement is a universal uncertainty bar for every interpolated value.
- **Cycling:** repeated heating and cooling gives similar heat-flow traces during the reported short cycling experiment (Fig. 11). This supports repeatability over those cycles; it does not by itself establish long-term durability.
- **Models provided:** separate cubic temperature fits at each loading (Table 3) and a global temperature/loading correlation (Table 4).

## What “loading” means in this paper

They infer water mass from the mass lost on drying. A molecule count can then be inferred from water's molar mass, but an absolute count for a macroscopic powder is less useful than water mass relative to framework mass.

Define $m_d$ as dry MOF mass, $m_s$ as saturated sample mass including water, and $m$ as the partially hydrated sample mass. The precise reference masses in the text and Table 1 are:

$$m_d=7.3990\ \mathrm{mg},\qquad m_s=10.5358\ \mathrm{mg},\qquad m_{w,s}=m_s-m_d=3.1368\ \mathrm{mg}.$$

The paper uses two dimensionless quantities (Eqs. 1–2, p. 5):

$$q=\frac{m-m_d}{m_s-m_d},\qquad dQ=\frac{m-m_d}{m_s}.$$

Here **$q$ is the fraction of the saturated water amount**, with 0 = dry and 1 = saturated. It is neither relative humidity nor water mass fraction. Table 1 expresses $dQ$ as a percentage; its denominator is the **original saturated sample mass**, not the current hydrated mass.

For simulations, first calculate the usual dry-basis mass loading $w$ and then convert:

$$w=\frac{m_w}{m_d},\qquad w_s=\frac{m_s-m_d}{m_d}=0.423949\ \mathrm{g\ H_2O/g\ dry\ MOF},\qquad q=\frac{w}{w_s}.$$

$$dQ=\frac{w}{1+w_s}=0.297728q,\qquad x_w=\frac{m_w}{m_d+m_w}=\frac{w}{1+w}.$$

$x_w$ is the actual water mass fraction of the current sample. These denominators matter: the saturated sample has **42.39% water relative to dry mass** but **29.77% water relative to total saturated mass**. The paper's rounded “30% uptake” refers approximately to the latter.

### Converting every experimental sample to this simulation cell

The $q$ labels below are those printed in Table 1. Dry-basis loadings and equivalent molecule counts are recalculated from the listed sample masses, avoiding rounding of the water-mass column.

| TGA preparation | Paper's $q$ | $w$, g water/g dry MOF | Equivalent H₂O per 1,024-atom host |
|---|---:|---:|---:|
| 150 °C, dry | 0 | 0 | 0 |
| 100 °C | 0.0048 | 0.002054 | 1.45 |
| 80 °C | 0.1202 | 0.050939 | 35.84 |
| 60 °C | 0.3425 | 0.145222 | 102.19 |
| 40 °C | 0.7444 | 0.315610 | 222.08 |
| 30 °C | 0.8342 | 0.353669 | 248.86 |
| Saturated | 1 | 0.423949 | 298.32 |

Fractional counts express composition, not possible integer occupancies of one cell. If you later want simulations close to the two measured loadings bracketing your current run, use approximately **36 and 102 waters in the same supercell**. The inferred saturated count is an experimental mass-ratio mapping, not an independently established saturation capacity of your simulation model.

## Conversion for the existing 50-water campaign

The [300 K preparation metadata](../output/md/structures/mof303_2x2x2/50h2o/300K/rep01/inputs.json) records `mof303_2x2x2`, `host_atoms = 1024`, and `loading = 50`. Reading [the host CIF](../input/mof303_2x2x2.cif) with ASE gives:

- Composition: $\mathrm{Al_{64}C_{320}H_{192}N_{128}O_{320}}$, equivalent to 64 units of $\mathrm{AlC_5H_3N_2O_5}$.
- Dry cell mass: $M_h=12676.450464\ \mathrm{u}$.
- Water mass from the [H₂O template](../input/h2o.gro): $M_w=18.015\ \mathrm{u}$ per molecule.
- Fifty waters: $900.75\ \mathrm{u}$; total hydrated cell: $13577.200464\ \mathrm{u}$ and 1,174 atoms.

The atomic-mass-unit conversion cancels in the ratio:

$$w=\frac{N_wM_w}{M_h}=\frac{50(18.015)}{12676.450464}=0.07105696.$$

$$q=\frac{0.07105696}{0.42394918}=0.16760725.$$

| Convention | Your 50-water value |
|---|---:|
| Water/dry-framework mass | 0.07106 g/g, or 7.106% of dry mass |
| Water fraction of hydrated sample, $100x_w$ | 6.634% |
| Paper's saturation fraction, $100q$ | **16.761%** |
| Paper's degree of uptake, $100dQ$ | 4.990% |
| Waters per framework formula unit | 0.78125 |

These results depend on the cell. Fifty waters in the smaller [128-atom CIF](../input/mof303.cif) would give $q\approx1.341$, outside this paper's loading range. Recompute from the actual dry host whenever the supercell or framework composition changes.

## Obtaining an experimental reference at your loading

### Local interpolation between the neighboring loading curves

For an initial comparison, use the two neighboring Table 3 fits (Eqs. 6–7, p. 11). Their coefficients were also checked against the rendered Fig. 9. Let $t=T_K-273.15$ be the **numerical temperature in °C**. The resulting specific heats are in $\mathrm{J\,g^{-1}\,K^{-1}}$:

$$c_{0.1202}(t)=1.20482+0.00167t+6.27457\times10^{-5}t^2-4.6182\times10^{-7}t^3.$$

$$c_{0.3425}(t)=1.47328+0.00557t-2.9899\times10^{-5}t^2+8.20125\times10^{-8}t^3.$$

Interpolate at fixed temperature:

$$\alpha=\frac{q-0.1202}{0.3425-0.1202}=0.213258,\qquad c_{\mathrm{ref,local}}(T,q)=(1-\alpha)c_{0.1202}(t)+\alpha c_{0.3425}(t).$$

This is an explicit **linear interpolation assumption**, not a measured curve at $q=0.16761$. The paper suggests interpolation in Section 7.4.1 but does not establish that loading dependence is linear between these samples. Showing both neighboring curves alongside the interpolated one makes that assumption visible; they are not statistical confidence bounds.

### The authors' global correlation

Table 4 (p. 12) provides a direct alternative:

$$c_{\mathrm{ref,global}}(t,q)=0.91115+0.03069t^{0.52139}+0.59655q^{0.51417}+0.046t^{0.52139}q^{0.51417}.$$

Use **$q=0.16760725$**, not 16.76, 0.07106, or 0.06634. Use Celsius for $t$. The stated domain is $10\le t\le95$ and $0\le q\le1$. The adjusted $R^2$ is 0.93424; it is not an error estimate or “93.4% accuracy.”

| Temperature, K | Temperature, °C | Local interpolation | Global correlation |
|---:|---:|---:|---:|
| 283.15 | 10.00 | 1.2910 | 1.3122 |
| 298.15 | 25.00 | 1.3461 | 1.4120 |
| **300.00** | **26.85** | **1.3535** | **1.4220** |
| **325.00** | **51.85** | **1.4591** | **1.5336** |
| **350.00** | **76.85** | **1.5512** | **1.6211** |
| 368.15 | 95.00 | 1.5912 | 1.6763 |

All heat-capacity entries are calculated references in $\mathrm{J\,g^{-1}\,K^{-1}}$, not raw experimental measurements. The two methods differ by about 4.5–5.1% at 300–350 K. Report this sensitivity instead of treating either interpolation as exact. Fit-parameter covariances and raw replicate data would be needed for a defensible prediction interval.

## Matching the simulation observable and normalization

**Use the loaded system's total specific heat at fixed composition.** The experimental samples are sealed after their loading is prepared, so a fixed-number simulation is the appropriate first comparison. Do not follow an adsorption isotherm by changing the water count with temperature, and do not add an adsorption-enthalpy term for changing loading to your fixed-loading heat capacity.

The paper calls its measured quantity $c_p$. A fixed-cell harmonic calculation gives $C_V$, whereas NPT enthalpy derivatives give $C_P$. The repository's hybrid estimator combines them approximately:

$$C_P^{\mathrm{hybrid}}\approx C_P^{\mathrm{classical,NPT}}+\left(C_V^{\mathrm{quantum,harmonic}}-C_V^{\mathrm{classical,harmonic}}\right).$$

Use **`approximate_cp_J_per_gK`** from the hybrid CSV as the main simulation curve. The purely classical curve is a useful diagnostic, but classical high-frequency water vibrations can overcontribute compared with quantum statistics. A harmonic-only $C_V$ curve is not an equivalent experimental $C_P$ benchmark. The hybrid correction itself remains an approximation.

The implementation in [analysis/hybrid.py](../mof_heat_capacity/analysis/hybrid.py) divides by the **total hydrated cell mass**. Therefore its units are per gram of framework plus water. The natural working interpretation of the paper's DSC sample-specific heat is likewise per gram of the current hydrated sample. However, the main text does **not explicitly document the mass entered in the DSC normalization for each loading**. Treat that matching basis as an assumption to confirm with the authors or supplementary methods before claiming quantitative validation. The denominator of $dQ$ does not establish the denominator of $c_p$.

If conversion is needed, distinguish total-sample and dry-framework bases:

$$c_p^{\mathrm{per\ dry\ mass}}=(1+w)c_p^{\mathrm{per\ total\ mass}}=1.07105696\,c_p^{\mathrm{per\ total\ mass}}.$$

For a whole-cell result in eV/K, the total-mass conversion is:

$$c_p\,[\mathrm{J\,g^{-1}\,K^{-1}}]=\frac{96485.33212}{13577.200464}C_{P,\mathrm{cell}}\,[\mathrm{eV/K}].$$

Do not apply this conversion again to an output already labeled `J_per_gK`. Do not compare a water-only contribution or a loaded-minus-empty increment with the paper's total loaded-sample heat capacity.

## Practical comparison with your present results

1. Keep the actual 2×2×2 host and 50-water composition fixed; label plots with both $N_w=50$ and $q=0.16761$.
2. Restrict numerical comparisons to **283.15–368.15 K**. On the existing 25 K grid, **300, 325, and 350 K** are inside the measured range. The 275 and 375 K points are outside it. Convert the evaluation temperature to Celsius only when evaluating the paper's fits.
3. Plot `approximate_cp_J_per_gK` with its simulation uncertainty, the local interpolated reference, and the global reference. Optionally include the two measured-loading fits to show the interpolation interval. Label the reference curves as fits/interpolations.
4. At the same temperatures, report the signed relative discrepancy $100(c_{\mathrm{sim}}-c_{\mathrm{ref}})/c_{\mathrm{ref}}$. Keep experimental uncertainty, interpolation sensitivity, sampling uncertainty, and model uncertainty conceptually separate.
5. Before interpreting discrepancies as force-field error, check equilibration, enthalpy-derivative convergence, water configurations, framework stability, and the Hessian spectrum. Confirm the experimental mass normalization and resolve the DSC interpretation issues below for publication-level validation.

### Worked example from the output already in this checkout

The existing [50-water hybrid CSV](../output/post-processing/harmonic-correction/mof303_2x2x2/pet-mad-1.5-s-40nn/50h2o/heat-capacity.exploratory-discard-imaginary.csv) is for `pet-mad-1.5-s-40nn`. Its accompanying JSON explicitly sets `exploratory_discard_imaginary_modes = true`. The following is therefore a **preliminary comparison of an exploratory estimator**, under the total-sample-mass assumption above, not a validated simulation result.

| Temperature, K | Existing hybrid $c_p$ | Reported combined standard uncertainty | Local experimental reference | Signed discrepancy |
|---:|---:|---:|---:|---:|
| 300 | 1.1063 | 0.1329 | 1.3535 | −18.3% |
| 325 | 1.0226 | 0.1881 | 1.4591 | −29.9% |
| 350 | 1.2756 | 0.0821 | 1.5512 | −17.8% |

Heat capacities and uncertainties are in $\mathrm{J\,g^{-1}\,K^{-1}}$. The uncertainty column is `approximate_cp_combined_standard_uncertainty_J_per_gK`; it does not include experimental error, interpolation uncertainty, or systematic bias from discarding imaginary modes. These numbers are a snapshot of the files inspected for this note and should be regenerated after new analysis.

### Reproduce the conversion and reference values

Run this from the repository root in an environment with ASE. It prints references without changing simulation outputs.

```python
from ase.io import read

host = read("input/mof303_2x2x2.cif")
water = read("input/h2o.gro")
n_water = 50
w_sat = (10.5358 - 7.3990) / 7.3990
w = n_water * water.get_masses().sum() / host.get_masses().sum()
q = w / w_sat
alpha = (q - 0.1202) / (0.3425 - 0.1202)
assert 0 <= alpha <= 1, "Choose different loading brackets for this cell."
print(f"w = {w:.8f} g/g; q = {q:.8f}; saturation = {100*q:.3f}%")

for temperature_K in (300.0, 325.0, 350.0):
    t = temperature_K - 273.15
    assert 10 <= t <= 95
    low = 1.20482 + 0.00167*t + 6.27457e-5*t**2 - 4.6182e-7*t**3
    high = 1.47328 + 0.00557*t - 2.9899e-5*t**2 + 8.20125e-8*t**3
    local = (1-alpha)*low + alpha*high
    global_fit = (
        0.91115 + 0.03069*t**0.52139 + 0.59655*q**0.51417
        + 0.046*t**0.52139*q**0.51417
    )
    print(f"{temperature_K:.2f} K: local={local:.6f}, global={global_fit:.6f} J/g/K")
```

## Paper inconsistencies that affect interpretation

- **Loading-axis labels:** Section 7.2/Fig. 7 mix percentage-scale saturation with “kg/kg” language, including an implausible reference to approximately 100 kg/kg. Use the mass definitions in Eqs. 1–2 and Table 1, rather than interpreting those labels literally.
- **Rounding:** precise masses imply a saturated $dQ=29.7728\%$, whereas Table 1 prints 29.82%. Use the precise masses consistently; the small discrepancy does not change the choice of loading brackets.
- **A coefficient transcription problem:** Table 3 prints the cubic coefficient for the 83.4% curve as $1.12129\times10^{-7}$, whereas the Fig. 9 fit annotation on p. 10 gives $1.12129\times10^{-6}$. The printed Table 3 value conflicts with the plotted increasing curve. This fit is not used in the local interpolation above.
- **Other inconsistent descriptions:** the text mentions an intermediate $q=0.48$ although that is not a Table 1 loading; Section 7.3 attributes a heat capacity near 1.126 at 80 °C to the saturated sample, inconsistent with its roughly 2-plus curve. Use the explicitly identified data series and checked fits.
- **Chemical identity:** some prose incorrectly calls the MOF-303 linker triazole/triazolate; the synthesis section names pyrazoledicarboxylic acid. Use the actual CIF composition for mass conversion, as done here.
- **DSC interpretation:** p. 6 says sealing avoids continuous water loss, but its Fig. 5 discussion nevertheless attributes endothermic features to desorption. A sealed pan fixes total water, yet water can redistribute between adsorbed and vapor states internally. Such redistribution or thermal events can contribute to measured heat flow. A periodic filled crystal has no equivalent macroscopic pan headspace. The supplied text does not establish how completely these effects are separated from intrinsic fixed-loading sensible heat capacity, nor does sealing alone prove constant-pressure conditions inside the pan.

The loading conversion is well defined from the reported masses and your cell composition. The experimental heat-capacity interpretation and mass normalization have greater uncertainty; preserve those qualifications when using the paper to assess the simulation model.
