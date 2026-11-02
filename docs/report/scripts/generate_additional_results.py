"""Plot archived non-methane heat capacities and experimental references."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mof-report-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[1]
BASE = ROOT / "output/post-processing/harmonic-correction"
MODEL = "pet-mad-1.5-s-40nn"
REFERENCES = ROOT / "output/reference_heat_capacity"
CAMPAIGNS = {
    "mof5": ("MOF-5", ("50co2", "50h2o")),
    "mof303_2x2x2": ("MOF-303", ("50co2", "50h2o")),
    "zif8_2x2x2": ("ZIF-8", ("54co2",)),
}
EXPERIMENTS = {
    "mof303_2x2x2": (
        ("mof303/mof-303_h20_0pct.csv", "Experiment: dry MOF-303", "#333333", "D"),
        ("mof303/mof-303_h20_12p02pct.csv", r"Experiment: H$_2$O, $q=12.02\%$", "#19876c", "^"),
        ("mof303/mof-303_h20_34p25pct.csv", r"Experiment: H$_2$O, $q=34.25\%$", "#8b4a9e", "v"),
    ),
    "zif8_2x2x2": (
        ("zif8/zif-8_co2.csv", r"Experiment: CO$_2$, $X=0.56$", "#19876c", "D"),
    ),
}
EXPERIMENT_NOTES = {
    "mof303_2x2x2": (
        "q is the fraction of saturated water loading; 50 H2O corresponds to q = 16.76%.\n"
        "Experimental points are digitized; dashed lines connect the supplied points."
    ),
    "zif8_2x2x2": (
        "X is mol CO2 per mol ZIF-8; simulation X = 0.5625.\n"
        "Experiment: sealed-cell C; simulation: approximate Cp. Experimental bars: 2% standard uncertainty."
    ),
}


def read_experiment(path):
    """Read references already expressed in K and J per gram of loaded sample."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"temperature_K", "heat_capacity_J_per_gK"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Experimental CSV requires explicit K and J/g/K columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Experimental CSV is empty: {path}")
    values = np.array([
        [float(row["temperature_K"]), float(row["heat_capacity_J_per_gK"])]
        for row in rows
    ])
    if not np.isfinite(values).all() or (np.diff(values[:, 0]) <= 0).any():
        raise ValueError(f"Experimental temperatures must be finite and strictly increasing: {path}")
    return values[:, 0], values[:, 1]


def plot_experiments(axis, host):
    """Overlay measured points within the displayed range, without extrapolation."""
    sources = []
    for relative_path, label, color, marker in EXPERIMENTS.get(host, ()):
        path = REFERENCES / relative_path
        temperatures, heat_capacity = read_experiment(path)
        visible = (temperatures >= 195) & (temperatures <= 405)
        temperatures, heat_capacity = temperatures[visible], heat_capacity[visible]
        if not len(temperatures):
            raise ValueError(f"No experimental points in the displayed range: {path}")
        style = dict(color=color, marker=marker, markersize=4, markerfacecolor="white",
                     linestyle="--", linewidth=1.1, label=label, zorder=4)
        if host.startswith("zif8"):
            axis.errorbar(temperatures, heat_capacity, yerr=0.02 * heat_capacity,
                          capsize=2, elinewidth=0.8, **style)
        else:
            axis.plot(temperatures, heat_capacity, **style)
        record = source_record(path)
        record.update({"label": label, "units": "temperature_K; heat_capacity_J_per_gK"})
        sources.append(record)
    return sources


def source_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"source": str(path.relative_to(ROOT)), "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", choices=tuple(CAMPAIGNS), default=list(CAMPAIGNS))
    args = parser.parse_args()
    provenance_path = REPORT / "results-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    for host in args.hosts:
        title, loadings = CAMPAIGNS[host]
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), layout="constrained",
                                 gridspec_kw={"width_ratios": (1.35, 1)})
        sources = []
        for loading in loadings:
            directory = BASE / host / MODEL / loading
            stem = ("heat-capacity.fd-two-gpu.exploratory-discard-imaginary"
                    if host.startswith("zif8") else
                    "heat-capacity.exploratory-discard-imaginary")
            path = directory / f"{stem}.csv"
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            values = lambda key: np.array([float(row[key]) for row in rows])
            temperature = values("temperature_K")
            hybrid = values("approximate_cp_J_per_gK")
            uncertainty = values("approximate_cp_combined_standard_uncertainty_J_per_gK")
            water = loading.endswith("h2o")
            color = "#2675a5" if water else "#c26024"
            label = loading[:-3] + (r" H$_2$O" if water else r" CO$_2$")
            simulation_label = "Simulation: " + label
            if host == "mof303_2x2x2" and water:
                simulation_label += r" ($q=16.76\%$)"
            elif host == "zif8_2x2x2":
                simulation_label += r" ($X=0.5625$)"
            axes[0].plot(temperature, hybrid, "o-", color=color, label=simulation_label, ms=4)
            axes[0].fill_between(temperature, hybrid - uncertainty,
                                 hybrid + uncertainty, color=color, alpha=0.17)
            axes[1].plot(temperature, values("classical_anharmonic_cp_J_per_gK"),
                         "o-", color=color, label=label + r": $c_{\rm cl}$", ms=4)
            axes[1].plot(temperature, values("harmonic_quantum_correction_J_per_gK"),
                         "s--", color=color, label=label + r": $\Delta c_{\rm har}$", ms=3)
            sources.extend([source_record(path), source_record(path.with_suffix(".json"))])
            sources.append(source_record(directory / "minima/400K/rep01/optimized.relax.json"))
            validation = directory / "ensemble-validation.fd-two-gpu.json"
            if validation.is_file():
                sources.append(source_record(validation))
            summaries = ROOT / "output/post-processing/trajectory-analysis" / host / MODEL / loading
            sources.extend(source_record(p) for p in sorted(summaries.glob("*K/rep01/summary.json")))

        empty = BASE / host / MODEL / "0co2/hessians/hessian.npz"
        if empty.is_file():
            with np.load(empty, allow_pickle=False) as data:
                temperatures = data["temperatures_K"]
                central = data["cv_J_per_gK"][0]
                spread = data["llpr_cv_J_per_gK"][0].std(axis=0, ddof=1)
            axes[0].plot(temperatures, central, ":", color="#555555",
                         label=r"Empty: harmonic $c_V$")
            axes[0].fill_between(temperatures, central - spread, central + spread,
                                 color="#555555", alpha=0.10)
            sources.append(source_record(empty))
        empty_relaxation = BASE / host / MODEL / "0co2/minima/optimized.relax.json"
        if empty_relaxation.is_file():
            sources.append(source_record(empty_relaxation))

        experimental_sources = plot_experiments(axes[0], host)
        sources.extend(experimental_sources)
        axes[0].set_title(title + ": exploratory heat capacity")
        axes[1].set_title("Classical term and harmonic correction")
        for ax in axes:
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel(r"Heat capacity (J g$^{-1}$ K$^{-1}$)")
            ax.set_xlim(195, 405)
            ax.grid(alpha=0.18)
            ax.legend(fontsize=8, frameon=False)
        if host in EXPERIMENT_NOTES:
            fig.supxlabel(EXPERIMENT_NOTES[host]
                          + "\nSimulation shading: combined standard uncertainty.", fontsize=8)
        axes[1].axhline(0, color="#777777", lw=0.6)
        filename = f"results-{host}-additional.pdf"
        fig.savefig(REPORT / "figures" / filename, bbox_inches="tight")
        preview = REPORT / "figures" / filename.replace(".pdf", ".png")
        fig.savefig(preview, dpi=180, bbox_inches="tight")
        plt.close(fig)
        provenance[filename] = {
            "generator": str(Path(__file__).resolve().relative_to(ROOT)),
            "description": "Archived central hybrid values and paired CEA+MD standard uncertainties; empty curves are harmonic cV with LLPR spread. Experimental references use their recorded sample loadings and total-sample mass normalization; dashed lines connect supplied experimental points without extrapolation. All mass normalizations are system-specific.",
            "inputs": sources,
        }
        if experimental_sources:
            provenance[filename]["experimental_reference"] = {
                "description": EXPERIMENT_NOTES[host],
                "doi": ("10.1016/j.icheatmasstransfer.2025.109088"
                        if host.startswith("mof303") else "10.1021/jacs.9b13883.s001"),
                "uncertainty": ("Not supplied with the digitized MOF-303 points."
                                if host.startswith("mof303") else
                                "2% standard uncertainty from Supporting Information Table 3."),
            }
        print(f"Saved {filename} and {preview.name}")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
