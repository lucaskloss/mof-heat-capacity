"""Export digitized empty-MOF measurements with explicit mass normalization."""

import csv
import json
import math
from pathlib import Path

from export_mof5_plot_data import REPORT, ROOT, source_record, write_rows


# Zn(C4H5N2)2: the empty framework, without adsorbed CO2.
ZIF8_MOLAR_MASS_G_PER_MOL = 227.576
REFERENCES = (
    ("mof5/exp_mof-5.csv", "mof5-experiment-empty.csv", None,
     "10.1016/j.micromeso.2013.11.015", "ming2014mof5",
     ("results-mof5-panels.tex", "results-mof5-additional.tex")),
    ("zif8/exp_zif-8.csv", "zif8-experiment-empty.csv", ZIF8_MOLAR_MASS_G_PER_MOL,
     "10.1016/j.jct.2019.05.008", "rosen2019zif8",
     ("results-framework-totals.tex",)),
)


def read_digitized_reference(path, molar_mass):
    """Interpret these supplied x/y files as K and the recorded source units."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if reader.fieldnames != ["x", "y"]:
            raise ValueError(f"Expected the supplied digitized x,y columns: {path}")
        values = [(float(row["x"]), float(row["y"])) for row in reader]
    if not values or not all(math.isfinite(v) for pair in values for v in pair):
        raise ValueError(f"Empty or nonfinite experimental data: {path}")
    if any(t <= 0 or c < 0 for t, c in values):
        raise ValueError(f"Invalid experimental temperature or heat capacity: {path}")
    if any(b[0] <= a[0] for a, b in zip(values, values[1:])):
        raise ValueError(f"Experimental temperatures must be strictly increasing: {path}")
    divisor = 1.0 if molar_mass is None else molar_mass
    if not math.isfinite(divisor) or divisor <= 0:
        raise ValueError("Molar mass must be finite and positive")
    rows = [dict(temperature_K=t, heat_capacity=c / divisor)
            for t, c in values if 195 <= t <= 405]
    if not rows:
        raise ValueError(f"No measured points in the displayed range: {path}")
    return rows


def main():
    destination = REPORT / "data"
    destination.mkdir(parents=True, exist_ok=True)
    provenance_path = REPORT / "results-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    inputs, outputs = [], []
    for relative, name, molar_mass, doi, citation, figures in REFERENCES:
        source = ROOT / "output/reference_heat_capacity" / relative
        rows = read_digitized_reference(source, molar_mass)
        output = destination / name
        write_rows(output, ("temperature_K", "heat_capacity"), rows)
        record = {
            **source_record(source), "doi": doi, "citation_key": citation,
            "loading": 0, "temperature_unit": "K",
            "input_heat_capacity_unit": "J/(g K)" if molar_mass is None else "J/(mol K)",
            "output_heat_capacity_unit": "J/(g K)", "mass_basis": "dry framework",
            "molar_mass_g_per_mol": molar_mass,
            "normalization": "unchanged" if molar_mass is None else "divide molar heat capacity by molar mass",
            "uncertainty": "Not supplied in the digitized CSV; no error bars inferred.",
        }
        table = source_record(output)
        inputs.append(record)
        outputs.append(table)
        for figure in figures:
            provenance[figure]["empty_experimental_reference"] = {**record, "exported_data": table}
        print(f"Exported {len(rows)} experimental points: {output.name}")
    provenance["empty-experimental-references"] = {
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "description": "User-supplied digitized empty MOF-5 powder and ZIF-8 SOD measurements. Source temperatures are preserved and restricted to 195–405 K without interpolation or extrapolation. MOF-5 is already mass-specific; ZIF-8 is converted from molar units using the dry framework molar mass, not the loaded-sample mass.",
        "inputs": inputs, "exported_data": outputs,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
