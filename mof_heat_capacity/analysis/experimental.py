"""Experimental heat-capacity reference data and provenance for loaded MOFs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import warnings

import numpy as np


@dataclass(frozen=True)
class ExperimentalCurve:
    path: Path
    label: str
    temperatures: np.ndarray
    heat_capacity: np.ndarray
    color: str
    marker: str
    relative_uncertainty: float | None
    doi: str

    def provenance(self) -> dict:
        return {
            "source": str(self.path.resolve()),
            "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "label": self.label,
            "temperature_unit": "K",
            "heat_capacity_unit": "J/(g K)",
            "mass_basis": "sample mass including guest (dry host for zero loading)",
            "relative_standard_uncertainty": self.relative_uncertainty,
            "doi": self.doi,
        }


def load_experimental_curves(root: Path, mof: str, guest: str) -> list[ExperimentalCurve]:
    """Select only measured compositions for the requested host and guest.

    Inputs are already in K and J/(g K); no unit conversion or interpolation
    is applied. Missing optional files are reported and skipped.
    """
    if mof in {"mof303", "mof303_2x2x2"} and guest == "h2o":
        doi = "10.1016/j.icheatmasstransfer.2025.109088"
        specifications = [
            ("mof303/mof-303_h20_0pct.csv", "Experiment: dry MOF-303", "#333333", "D", None),
            ("mof303/mof-303_h20_12p02pct.csv", r"Experiment: H$_2$O, $q=12.02\%$", "#19876c", "^", None),
            ("mof303/mof-303_h20_34p25pct.csv", r"Experiment: H$_2$O, $q=34.25\%$", "#8b4a9e", "v", None),
        ]
    elif mof in {"zif8", "zif8_2x2x2"} and guest == "co2":
        doi = "10.1021/jacs.9b13883.s001"
        specifications = [
            ("zif8/zif-8_co2.csv", r"Experiment: CO$_2$, $X=0.56$", "#19876c", "D", 0.02),
        ]
    else:
        return []

    curves = []
    for filename, label, color, marker, uncertainty in specifications:
        path = root / filename
        if not path.is_file():
            warnings.warn(f"Optional experimental reference not found: {path}", stacklevel=2)
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"temperature_K", "heat_capacity_J_per_gK"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Experimental CSV must specify temperature_K and heat_capacity_J_per_gK: {path}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"Experimental CSV has no data rows: {path}")
        try:
            values = np.asarray([
                [float(row["temperature_K"]), float(row["heat_capacity_J_per_gK"])]
                for row in rows
            ])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Experimental CSV contains invalid numerical data: {path}") from error
        if (
            not np.isfinite(values).all()
            or np.any(values[:, 0] <= 0)
            or np.any(np.diff(values[:, 0]) <= 0)
        ):
            raise ValueError(f"Experimental temperatures must be positive and strictly increasing, with finite heat capacities: {path}")
        curves.append(ExperimentalCurve(
            path, label, values[:, 0], values[:, 1], color, marker, uncertainty, doi
        ))
    return curves


def experimental_note(mof: str, guest: str, loading: int, total_mass_amu: float | None = None) -> str:
    """Distinguish experimental loading conventions and thermodynamic conditions."""
    if mof in {"mof303", "mof303_2x2x2"} and guest == "h2o":
        note = "Experimental q is the fraction of saturated water loading; digitized points."
        if total_mass_amu is not None:
            host_mass = total_mass_amu - loading * 18.015
            if host_mass <= 0:
                raise ValueError("Loaded mass must exceed the inserted water mass")
            q = (loading * 18.015 / host_mass) / ((10.5358 - 7.3990) / 7.3990)
            note += f" Simulation q = {100 * q:.2f}%."
        return note
    if mof in {"zif8", "zif8_2x2x2"} and guest == "co2":
        note = "Experimental X is mol CO2/mol ZIF-8; sealed-cell C versus simulated Cp."
        if total_mass_amu is not None:
            host_mass = total_mass_amu - loading * 44.009
            if host_mass <= 0:
                raise ValueError("Loaded mass must exceed the inserted CO2 mass")
            note += f" Simulation X = {loading / (host_mass / 227.576):.5g}."
        return note + "\nExperimental bars: 2% standard uncertainty."
    return ""
