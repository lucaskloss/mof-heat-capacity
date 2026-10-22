"""Parse thermodynamic time series written by LAMMPS."""

from __future__ import annotations

from pathlib import Path

import numpy as np


_COLUMN_MAP = {
    "Step": "md_step",
    "Time": "time_ps",
    "Temp": "temperature_K",
    "PotEng": "potential_energy_eV",
    "KinEng": "kinetic_energy_eV",
    "TotEng": "total_energy_eV",
    "Press": "pressure_bar",
    "Pxx": "pressure_xx_bar",
    "Pyy": "pressure_yy_bar",
    "Pzz": "pressure_zz_bar",
    "Pxy": "pressure_xy_bar",
    "Pxz": "pressure_xz_bar",
    "Pyz": "pressure_yz_bar",
    "Volume": "volume_A3",
    "Density": "density_g_cm3",
    "Lx": "cell_lx_A",
    "Ly": "cell_ly_A",
    "Lz": "cell_lz_A",
    "Xy": "cell_xy_A",
    "Xz": "cell_xz_A",
    "Yz": "cell_yz_A",
}


def read_lammps_thermo(log_path: Path) -> dict[str, np.ndarray]:
    """Read complete thermo rows and collapse overlapping restart records.

    A job can restart from its most recent periodic checkpoint, which may be
    earlier than the last thermo row written before the job reached its safe
    wall-time boundary. The appended log then repeats those steps. Keep the
    first occurrence so it stays aligned with the first corresponding dump
    frames, and reject repeated steps with inconsistent simulation times.
    """
    path = log_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LAMMPS log not found: {path}")

    rows: list[dict[str, float]] = []
    columns: list[str] | None = None
    with path.open(errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if fields and fields[0] == "Step":
                missing = sorted(set(_COLUMN_MAP).difference(fields))
                if missing:
                    raise ValueError(
                        f"LAMMPS thermo header in {path} is missing: "
                        + ", ".join(missing)
                    )
                columns = fields
                continue
            if columns is None or len(fields) != len(columns):
                continue
            try:
                values = [float(field) for field in fields]
            except ValueError:
                continue
            raw = dict(zip(columns, values, strict=True))
            rows.append(
                {target: raw[source] for source, target in _COLUMN_MAP.items()}
            )

    if not rows:
        raise ValueError(f"no complete LAMMPS thermo rows found in {path}")

    result = {
        name: np.asarray([row[name] for row in rows], dtype=float)
        for name in _COLUMN_MAP.values()
    }
    result["md_step"] = result["md_step"].astype(np.int64)
    first_index: dict[int, int] = {}
    for index, step in enumerate(result["md_step"]):
        step = int(step)
        previous = first_index.get(step)
        if previous is None:
            first_index[step] = index
            continue
        if not np.isclose(
            result["time_ps"][index],
            result["time_ps"][previous],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"LAMMPS thermo contains an inconsistent repeated step/time in {path}"
            )
    if len(first_index) != len(result["md_step"]):
        keep = np.fromiter(first_index.values(), dtype=int)
        result = {name: values[keep] for name, values in result.items()}
    if np.any(np.diff(result["md_step"]) < 0) or np.any(
        np.diff(result["time_ps"]) < 0.0
    ):
        raise ValueError(
            f"LAMMPS thermo steps or times go backwards outside a restart overlap in {path}"
        )
    return result
