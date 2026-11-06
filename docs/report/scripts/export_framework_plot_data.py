"""Export MOF-303 and ZIF-8 totals and references for native PGFPlots."""

import json
import math
from pathlib import Path

import numpy as np

from export_mof5_plot_data import BASE, REPORT, ROOT, read_rows, source_record, write_rows


def main():
    destination = REPORT / "data"
    destination.mkdir(parents=True, exist_ok=True)
    inputs, outputs = [], []
    fields = ("temperature_K", "heat_capacity", "lower", "upper")

    def export(source, name, rows):
        for row in rows:
            if not all(math.isfinite(float(value)) for value in row.values()):
                raise ValueError(f"Nonfinite plot data: {source}")
        path = destination / name
        write_rows(path, tuple(rows[0]), rows)
        inputs.append(source_record(source))
        outputs.append(source_record(path))

    for host, short, loadings in (
        ("mof303_2x2x2", "mof303", ("50ch4", "50co2", "50h2o")),
        ("zif8_2x2x2", "zif8", ("54co2",)),
    ):
        directory = BASE / host / "pet-mad-1.5-s-40nn"
        for loading in loadings:
            stem = ("heat-capacity.fd-two-gpu.exploratory-discard-imaginary.csv"
                    if short == "zif8" else "heat-capacity.exploratory-discard-imaginary.csv")
            source = directory / loading / stem
            rows = []
            for row in read_rows(source):
                central = float(row["approximate_cp_J_per_gK"])
                spread = float(row["approximate_cp_combined_standard_uncertainty_J_per_gK"])
                if spread < 0:
                    raise ValueError(f"Negative uncertainty: {source}")
                rows.append(dict(zip(fields, (float(row["temperature_K"]), central,
                                              central-spread, central+spread))))
            export(source, f"{short}-total-{loading}.csv", rows)
        empty_references = (("co2", "empty"), ("ch4", "empty-ch4")) if short == "mof303" else (("co2", "empty"),)
        for guest, name in empty_references:
            source = directory / f"0{guest}/hessians/hessian.npz"
            if source.is_file():
                with np.load(source, allow_pickle=False) as data:
                    temperatures = data["temperatures_K"]
                    central = data["cv_J_per_gK"][0]
                    spread = data["llpr_cv_J_per_gK"][0].std(axis=0, ddof=1)
                rows = [dict(zip(fields, (t, c, c-s, c+s)))
                        for t, c, s in zip(temperatures, central, spread, strict=True)]
                export(source, f"{short}-total-{name}.csv", rows)

    references = (
        ("mof303/mof-303_h20_0pct.csv", "mof303-experiment-dry.csv", 0),
        ("mof303/mof-303_h20_12p02pct.csv", "mof303-experiment-water12.csv", 0),
        ("mof303/mof-303_h20_34p25pct.csv", "mof303-experiment-water34.csv", 0),
        ("zif8/zif-8_co2.csv", "zif8-experiment-co2.csv", 0.02),
    )
    for relative, name, fraction in references:
        source = ROOT / "output/reference_heat_capacity" / relative
        rows = [dict(temperature_K=float(row["temperature_K"]),
                     heat_capacity=float(row["heat_capacity_J_per_gK"]),
                     uncertainty=fraction * float(row["heat_capacity_J_per_gK"]))
                for row in read_rows(source) if 195 <= float(row["temperature_K"]) <= 405]
        if not rows:
            raise ValueError(f"No measured points in plotting range: {source}")
        export(source, name, rows)

    provenance_path = REPORT / "results-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["results-framework-totals.tex"] = {
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "description": "Native LaTeX total hybrid heat capacities with archived combined standard uncertainty. MOF-303 includes 50 CH4, 50 CO2, and 50 H2O, empty harmonic cV with sample LLPR spread, and dry/water experimental references. The methane panel uses its own 0ch4 empty-framework Hessian. ZIF-8 retains measured CO2 reference points and 2% standard uncertainty. Measurements are restricted to 195–405 K without extrapolation. All y-axes start at zero; negative band bounds are preserved in tables and clipped visually.",
        "inputs": inputs, "exported_data": outputs,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported {len(outputs)} MOF-303/ZIF-8 PGFPlots tables.")


if __name__ == "__main__":
    main()
