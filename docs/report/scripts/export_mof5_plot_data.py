"""Export archived MOF-5 curves for native PGFPlots figures, without plotting."""

import csv
import hashlib
import json
import math
from pathlib import Path


REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parents[1]
BASE = ROOT / "output/post-processing/harmonic-correction"
MODELS = {"pet-mad": "pet-mad-1.5-s-40nn", "pet-sol": "pet-sol-s-best"}


def read_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    temperatures = [float(row["temperature_K"]) for row in rows]
    if not temperatures or not all(math.isfinite(t) for t in temperatures):
        raise ValueError(f"Empty or invalid temperature grid: {path}")
    if any(b <= a for a, b in zip(temperatures, temperatures[1:])):
        raise ValueError(f"Temperature grid must be strictly increasing: {path}")
    return rows


def write_rows(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_record(path):
    return {"source": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    destination = REPORT / "figures/data"
    destination.mkdir(parents=True, exist_ok=True)
    uncertainty_path = BASE / "loading-comparisons/hybrid-heat-capacity-uncertainty-components-cea.csv"
    uncertainties = {}
    with uncertainty_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "cea":
                raise ValueError("Expected archived CEA uncertainties")
            key = (row["model"], int(row["loading_CH4"]), float(row["temperature_K"]))
            if key in uncertainties:
                raise ValueError(f"Duplicate CEA uncertainty: {key}")
            uncertainties[key] = row

    inputs = [source_record(uncertainty_path)]
    outputs = []
    for loading in (0, 50, 100, 150):
        reference = ROOT / f"output/reference_heat_capacity/mof5/{loading}ch4.csv"
        reference_rows = read_rows(reference)
        output = destination / f"mof5-reference-{loading}ch4.csv"
        write_rows(output, ("temperature_K", "reference_heat_capacity_J_per_gK"), reference_rows)
        inputs.append(source_record(reference))
        outputs.append(source_record(output))
        for short, model in MODELS.items():
            source = BASE / model / f"{loading}ch4/heat-capacity.exploratory-discard-imaginary.csv"
            rows = []
            for row in read_rows(source):
                temperature = float(row["temperature_K"])
                central = float(row["approximate_cp_J_per_gK"])
                if loading:
                    archived = uncertainties[(model, loading, temperature)]
                    if not math.isclose(central, float(archived["central_hybrid_cp_J_per_gK"]), abs_tol=1e-12):
                        raise ValueError(f"Central curve differs from CEA archive: {source}")
                    spread = float(archived["combined_standard_uncertainty_J_per_gK"])
                else:
                    spread = float(row["approximate_cp_combined_standard_uncertainty_J_per_gK"])
                if not all(math.isfinite(v) for v in (central, spread)) or spread < 0:
                    raise ValueError(f"Invalid heat capacity or uncertainty: {source}")
                rows.append(dict(temperature_K=temperature, heat_capacity=central,
                                 lower=central - spread, upper=central + spread))
            output = destination / f"mof5-{short}-{loading}ch4.csv"
            write_rows(output, ("temperature_K", "heat_capacity", "lower", "upper"), rows)
            inputs.append(source_record(source))
            outputs.append(source_record(output))

    provenance_path = REPORT / "results-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["results-mof5-panels.tex"] = {
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "description": "Native PGFPlots 1x3 panels per model for 50, 100, and 150 CH4. Each panel overlays loaded hybrid cP and empty harmonic cV with their matching digitized Kapil references. Distinct colors identify empty, 50, 100, and 150 CH4; solid results and dashed references share a color. Loaded bands preserve archived CEA combined standard uncertainties; empty bands preserve harmonic LLPR spread. All axes span 0 to 1.8 J/g/K.",
        "inputs": inputs,
        "exported_data": outputs,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported {len(outputs)} PGFPlots data tables to {destination}")


if __name__ == "__main__":
    main()
