"""Export MOF-5 CO2/H2O total heat capacities for native LaTeX plots."""

import json
import math
from pathlib import Path

import numpy as np

from export_mof5_plot_data import BASE, REPORT, ROOT, read_rows, source_record, write_rows


def main():
    directory = BASE / "mof5/pet-mad-1.5-s-40nn"
    destination = REPORT / "figures/data"
    destination.mkdir(parents=True, exist_ok=True)
    fields = ("temperature_K", "heat_capacity", "lower", "upper")
    inputs, outputs = [], []
    for loading in ("50co2", "50h2o"):
        source = directory / loading / "heat-capacity.exploratory-discard-imaginary.csv"
        rows = []
        for row in read_rows(source):
            central = float(row["approximate_cp_J_per_gK"])
            spread = float(row["approximate_cp_combined_standard_uncertainty_J_per_gK"])
            if not all(math.isfinite(v) for v in (central, spread)) or spread < 0:
                raise ValueError(f"Invalid heat capacity or uncertainty: {source}")
            rows.append(dict(temperature_K=float(row["temperature_K"]),
                             heat_capacity=central, lower=central-spread, upper=central+spread))
        output = destination / f"mof5-additional-{loading}.csv"
        write_rows(output, fields, rows)
        inputs.append(source_record(source))
        outputs.append(source_record(output))

    source = directory / "0co2/hessians/hessian.npz"
    with np.load(source, allow_pickle=False) as data:
        temperatures = data["temperatures_K"]
        central = data["cv_J_per_gK"][0]
        spread = data["llpr_cv_J_per_gK"][0].std(axis=0, ddof=1)
    if not all(np.isfinite(values).all() for values in (temperatures, central, spread)):
        raise ValueError(f"Invalid empty-framework data: {source}")
    output = destination / "mof5-additional-empty.csv"
    write_rows(output, fields, [dict(temperature_K=t, heat_capacity=c, lower=c-s, upper=c+s)
                               for t, c, s in zip(temperatures, central, spread, strict=True)])
    inputs.append(source_record(source))
    outputs.append(source_record(output))
    provenance_path = REPORT / "results-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["results-mof5-additional.tex"] = {
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "description": "Native PGFPlots panels for 50 CO2 and 50 H2O: total hybrid cP and the archived empty harmonic cV reference. Bands preserve combined standard uncertainties for loaded systems and sample LLPR spread for the empty reference. Axes span 0 to 2 J/g/K; negative uncertainty bounds remain in the data and are clipped visually at zero.",
        "inputs": inputs, "exported_data": outputs,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported {len(outputs)} additional MOF-5 PGFPlots tables.")


if __name__ == "__main__":
    main()
