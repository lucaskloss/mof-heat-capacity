#!/usr/bin/env python3
"""Generate the compact results figure used by docs/report.tex."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURE_PATH = PROJECT_DIR / "docs" / "results_overview.pdf"


def model_label(run_name: str) -> str:
    if "pet-mad" in run_name:
        return "PET-MAD"
    if "pet-sol" in run_name:
        return "PET-SOL"
    raise ValueError(f"unrecognized model in run name: {run_name}")


def load_records() -> list[dict]:
    records = []
    for loading in (0, 100):
        analysis_dir = OUTPUT_DIR / f"analysis-{loading}ch4"
        for summary_path in sorted(analysis_dir.glob("*/summary.json")):
            summary = json.loads(summary_path.read_text())
            run_name = summary["run"]
            archive = OUTPUT_DIR / run_name / "heat-capacity-times-200ps-350ps-500ps.npz"
            with np.load(archive, allow_pickle=False) as data:
                temperatures = np.asarray(data["temperatures_K"], dtype=float)
                target = float(summary["target_temperature_K"])
                temperature_index = int(np.flatnonzero(temperatures == target)[0])
                frame_cv = np.asarray(data["cv_J_per_gK"], dtype=float)[:, temperature_index]
                zero_modes = np.count_nonzero(
                    np.asarray(data["frequencies_cm1"], dtype=float) == 0.0, axis=1
                )

            records.append(
                {
                    "loading": loading,
                    "model": model_label(run_name),
                    "temperature": target,
                    "density": summary["statistics"]["density_g_cm3"]["mean"],
                    "framework_rmsd": summary["structural_statistics"]["framework_rmsd_A"]["mean"],
                    "cv_mean": float(frame_cv.mean()),
                    "cv_std": float(frame_cv.std(ddof=1)),
                    "zero_modes": float(zero_modes.mean()),
                }
            )
    return records


def main() -> None:
    matplotlib_dir = OUTPUT_DIR / ".matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_records()
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True)
    colors = {"PET-MAD": "#1f77b4", "PET-SOL": "#d95f02"}
    markers = {0: "o", 100: "s"}
    loading_labels = {0: "pristine", 100: r"100 CH$_4$"}

    for loading in (0, 100):
        for model in ("PET-MAD", "PET-SOL"):
            selected = sorted(
                (item for item in records if item["loading"] == loading and item["model"] == model),
                key=lambda item: item["temperature"],
            )
            temperatures = [item["temperature"] for item in selected]
            label = f"{model}, {loading_labels[loading]}"
            style = {
                "color": colors[model],
                "marker": markers[loading],
                "linestyle": "-" if loading == 0 else "--",
                "linewidth": 1.4,
                "markersize": 5,
            }
            axes[0, 0].plot(temperatures, [item["density"] for item in selected], label=label, **style)
            axes[0, 1].plot(
                temperatures, [item["framework_rmsd"] for item in selected], **style
            )
            axes[1, 0].errorbar(
                temperatures,
                [item["cv_mean"] for item in selected],
                yerr=[item["cv_std"] for item in selected],
                capsize=2,
                label=label,
                **style,
            )
            axes[1, 1].plot(temperatures, [item["zero_modes"] for item in selected], **style)

    axes[0, 0].set_ylabel(r"Density (g cm$^{-3}$)")
    axes[0, 1].set_ylabel(r"Framework RMSD ($\AA$)")
    axes[1, 0].set_ylabel(r"Diagnostic harmonic $c_V$ (J g$^{-1}$ K$^{-1}$)")
    axes[1, 1].set_ylabel("Collapsed zero-frequency modes")
    axes[1, 0].set_xlabel("MD target temperature (K)")
    axes[1, 1].set_xlabel("MD target temperature (K)")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
