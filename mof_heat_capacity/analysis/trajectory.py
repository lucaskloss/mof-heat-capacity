"""Extract thermodynamic and structural observables from ASE trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np


AMU_PER_ANGSTROM3_TO_G_PER_CM3 = 1.66053906660
EV_PER_ANGSTROM3_TO_BAR = 1.602176634e6


def _methane_centers_fractional(atoms, host_atoms: int) -> np.ndarray:
    """Return wrapped methane centers of mass for appended C/H/H/H/H groups."""
    symbols = np.asarray(atoms.get_chemical_symbols())
    guest_count = len(atoms) - host_atoms
    if guest_count == 0:
        return np.empty((0, 3), dtype=float)
    if guest_count % 5:
        raise ValueError(
            f"{guest_count} guest atoms cannot be divided into five-atom methane groups"
        )

    scaled = atoms.get_scaled_positions(wrap=False)
    masses = atoms.get_masses()
    centers = []
    for start in range(host_atoms, len(atoms), 5):
        group = slice(start, start + 5)
        group_symbols = symbols[group]
        carbon = np.flatnonzero(group_symbols == "C")
        if len(carbon) != 1 or np.count_nonzero(group_symbols == "H") != 4:
            raise ValueError(
                f"atoms {start}:{start + 5} are not one C followed by four H atoms"
            )
        anchor = scaled[start + int(carbon[0])]
        displacements = scaled[group] - anchor
        displacements -= np.rint(displacements)
        unwrapped_group = anchor + displacements
        center = np.average(unwrapped_group, axis=0, weights=masses[group])
        centers.append(center - np.floor(center))
    return np.asarray(centers)


def _initialize_framework_reference(atoms, host_atoms: int, bond_cutoff: float):
    from ase.neighborlist import neighbor_list

    framework = atoms[:host_atoms]
    reference_scaled = framework.get_scaled_positions(wrap=False)
    first, second, distances = neighbor_list(
        "ijd", framework, bond_cutoff, self_interaction=False
    )
    unique = first < second
    return (
        reference_scaled,
        np.asarray(first[unique], dtype=int),
        np.asarray(second[unique], dtype=int),
        np.asarray(distances[unique], dtype=float),
    )


def _framework_metrics(
    atoms,
    host_atoms: int,
    reference_scaled: np.ndarray,
    bond_first: np.ndarray,
    bond_second: np.ndarray,
    reference_bonds: np.ndarray,
) -> tuple[float, float, float]:
    from ase.geometry import find_mic

    current_scaled = atoms.get_scaled_positions(wrap=False)[:host_atoms]
    fractional_displacement = current_scaled - reference_scaled
    fractional_displacement -= np.rint(fractional_displacement)
    displacement = fractional_displacement @ atoms.cell.array
    displacement -= displacement.mean(axis=0)
    rmsd = float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))))

    if len(bond_first) == 0:
        return rmsd, float("nan"), float("nan")
    vectors = atoms.positions[bond_second] - atoms.positions[bond_first]
    _, distances = find_mic(vectors, atoms.cell, atoms.pbc)
    changes = np.asarray(distances) - reference_bonds
    return (
        rmsd,
        float(np.sqrt(np.mean(changes**2))),
        float(np.max(np.abs(changes))),
    )


def _distance_metrics(
    atoms, host_atoms: int, methane_centers_fractional: np.ndarray
) -> tuple[float, float, float]:
    from ase.geometry import get_distances

    if len(methane_centers_fractional) == 0:
        return float("nan"), float("nan"), float("nan")
    centers = methane_centers_fractional @ atoms.cell.array
    host = atoms.positions[:host_atoms]
    guest = atoms.positions[host_atoms:]
    _, host_guest = get_distances(host, guest, cell=atoms.cell, pbc=atoms.pbc)
    _, host_center = get_distances(host, centers, cell=atoms.cell, pbc=atoms.pbc)
    _, center_center = get_distances(
        centers, centers, cell=atoms.cell, pbc=atoms.pbc
    )
    upper = center_center[np.triu_indices(len(centers), k=1)]
    return (
        float(host_guest.min()),
        float(host_center.min()),
        float(upper.min()) if len(upper) else float("nan"),
    )


def _accumulate_rdf(
    atoms,
    host_atoms: int,
    methane_centers_fractional: np.ndarray,
    edges: np.ndarray,
    host_guest_counts: np.ndarray,
    guest_guest_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from ase.geometry import get_distances

    if len(methane_centers_fractional) == 0:
        return host_guest_counts, guest_guest_counts
    centers = methane_centers_fractional @ atoms.cell.array
    host = atoms.positions[:host_atoms]
    _, host_guest = get_distances(host, centers, cell=atoms.cell, pbc=atoms.pbc)
    _, guest_guest = get_distances(
        centers, centers, cell=atoms.cell, pbc=atoms.pbc
    )
    upper = guest_guest[np.triu_indices(len(centers), k=1)]
    host_guest_counts += np.histogram(host_guest, bins=edges)[0]
    guest_guest_counts += np.histogram(upper, bins=edges)[0]
    return host_guest_counts, guest_guest_counts


def read_trajectory_observables(
    trajectory_path: Path,
    *,
    frame_spacing_fs: float,
    production_start_ps: float,
    host_atoms: int,
    structural_stride: int,
    rdf_stride: int,
    rdf_bins: int,
    rdf_cutoff_A: float | None = None,
    framework_bond_cutoff_A: float = 2.2,
    thermodynamic_series: dict[str, np.ndarray] | None = None,
) -> dict:
    """Stream an ASE trajectory and return thermodynamic/structural arrays."""
    from ase.io import iread

    if frame_spacing_fs <= 0.0:
        raise ValueError("frame_spacing_fs must be positive")
    if structural_stride < 1 or rdf_stride < 1 or rdf_bins < 2:
        raise ValueError("strides and rdf_bins must be positive")
    if rdf_stride % structural_stride:
        raise ValueError("rdf_stride must be an integer multiple of structural_stride")

    series_names = (
        "frame",
        "md_step",
        "time_ps",
        "temperature_K",
        "total_energy_eV",
        "potential_energy_eV",
        "kinetic_energy_eV",
        "volume_A3",
        "density_g_cm3",
        "pressure_bar",
        "max_force_eV_A",
        "rms_force_eV_A",
        "cell_a_A",
        "cell_b_A",
        "cell_c_A",
        "cell_alpha_deg",
        "cell_beta_deg",
        "cell_gamma_deg",
        "pressure_xx_bar",
        "pressure_yy_bar",
        "pressure_zz_bar",
        "pressure_xy_bar",
        "pressure_xz_bar",
        "pressure_yz_bar",
        "cell_lx_A",
        "cell_ly_A",
        "cell_lz_A",
        "cell_xy_A",
        "cell_xz_A",
        "cell_yz_A",
    )
    series_lists = {name: [] for name in series_names}
    structural_lists = {
        name: []
        for name in (
            "frame",
            "time_ps",
            "framework_rmsd_A",
            "framework_bond_rms_change_A",
            "framework_bond_max_change_A",
            "minimum_host_guest_atom_distance_A",
            "minimum_host_methane_com_distance_A",
            "minimum_methane_com_distance_A",
        )
    }

    atom_count: int | None = None
    total_mass_amu: float | None = None
    chemical_formula = ""
    reference_scaled = None
    bond_first = bond_second = reference_bonds = None
    previous_centers = None
    unwrapped_centers = None
    methane_center_frames: list[np.ndarray] = []
    methane_center_cells: list[np.ndarray] = []
    rdf_edges = None
    rdf_host_guest = None
    rdf_guest_guest = None
    rdf_host_guest_ideal = None
    rdf_guest_guest_ideal = None
    rdf_frames = 0
    raw_frame_count = 0
    duplicate_thermo_frames = 0

    for raw_frame_index, atoms in enumerate(iread(str(trajectory_path), index=":")):
        raw_frame_count = raw_frame_index + 1
        if atom_count is None:
            atom_count = len(atoms)
            if host_atoms < 1 or host_atoms > atom_count:
                raise ValueError(
                    f"host_atoms={host_atoms} is outside 1..{atom_count}"
                )
            total_mass_amu = float(atoms.get_masses().sum())
            chemical_formula = atoms.get_chemical_formula()
            (
                reference_scaled,
                bond_first,
                bond_second,
                reference_bonds,
            ) = _initialize_framework_reference(
                atoms, host_atoms, framework_bond_cutoff_A
            )
            cutoff = rdf_cutoff_A or 0.5 * float(min(atoms.cell.lengths()))
            rdf_edges = np.linspace(0.0, cutoff, rdf_bins + 1)
            rdf_host_guest = np.zeros(rdf_bins, dtype=float)
            rdf_guest_guest = np.zeros(rdf_bins, dtype=float)
            rdf_host_guest_ideal = np.zeros(rdf_bins, dtype=float)
            rdf_guest_guest_ideal = np.zeros(rdf_bins, dtype=float)
        elif len(atoms) != atom_count or not np.isclose(
            atoms.get_masses().sum(), total_mass_amu, rtol=1e-12, atol=1e-10
        ):
            raise ValueError("atom count or total mass changes along the trajectory")

        if thermodynamic_series is not None:
            if raw_frame_index >= len(thermodynamic_series["time_ps"]):
                raise ValueError(
                    "trajectory has more frames than the LAMMPS thermo log"
                )
            if raw_frame_index > 0:
                repeated_step = (
                    thermodynamic_series["md_step"][raw_frame_index]
                    == thermodynamic_series["md_step"][raw_frame_index - 1]
                )
                repeated_time = np.isclose(
                    thermodynamic_series["time_ps"][raw_frame_index],
                    thermodynamic_series["time_ps"][raw_frame_index - 1],
                    rtol=0.0,
                    atol=1e-12,
                )
                if repeated_step != repeated_time:
                    raise ValueError(
                        "LAMMPS thermo contains an inconsistent repeated step/time"
                    )
                if repeated_step:
                    duplicate_thermo_frames += 1
                    continue

        frame_index = len(series_lists["frame"])
        time_ps = frame_index * frame_spacing_fs / 1000.0
        volume = float(atoms.get_volume())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        cell = atoms.cell.cellpar()
        force_norms = np.linalg.norm(forces, axis=1)

        if thermodynamic_series is not None:
            frame_values = {
                name: values[raw_frame_index]
                for name, values in thermodynamic_series.items()
            }
            frame_values.update({
                "frame": frame_index,
                "max_force_eV_A": float(force_norms.max()),
                "rms_force_eV_A": float(np.sqrt(np.mean(force_norms**2))),
                "cell_a_A": float(cell[0]),
                "cell_b_A": float(cell[1]),
                "cell_c_A": float(cell[2]),
                "cell_alpha_deg": float(cell[3]),
                "cell_beta_deg": float(cell[4]),
                "cell_gamma_deg": float(cell[5]),
            })
            time_ps = float(frame_values["time_ps"])
        else:
            potential = float(atoms.get_potential_energy())
            kinetic = float(atoms.get_kinetic_energy())
            stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
            pressure_tensor = -stress * EV_PER_ANGSTROM3_TO_BAR
            frame_values = {
                "frame": frame_index,
                "md_step": frame_index,
                "time_ps": time_ps,
                "temperature_K": float(atoms.get_temperature()),
                "total_energy_eV": potential + kinetic,
                "potential_energy_eV": potential,
                "kinetic_energy_eV": kinetic,
                "volume_A3": volume,
                "density_g_cm3": (
                    float(total_mass_amu) / volume
                    * AMU_PER_ANGSTROM3_TO_G_PER_CM3
                ),
                "pressure_bar": float(np.trace(pressure_tensor)) / 3.0,
                "max_force_eV_A": float(force_norms.max()),
                "rms_force_eV_A": float(np.sqrt(np.mean(force_norms**2))),
                "cell_a_A": float(cell[0]),
                "cell_b_A": float(cell[1]),
                "cell_c_A": float(cell[2]),
                "cell_alpha_deg": float(cell[3]),
                "cell_beta_deg": float(cell[4]),
                "cell_gamma_deg": float(cell[5]),
                "pressure_xx_bar": float(pressure_tensor[0, 0]),
                "pressure_yy_bar": float(pressure_tensor[1, 1]),
                "pressure_zz_bar": float(pressure_tensor[2, 2]),
                "pressure_xy_bar": float(pressure_tensor[0, 1]),
                "pressure_xz_bar": float(pressure_tensor[0, 2]),
                "pressure_yz_bar": float(pressure_tensor[1, 2]),
                "cell_lx_A": float(atoms.cell[0, 0]),
                "cell_ly_A": float(atoms.cell[1, 1]),
                "cell_lz_A": float(atoms.cell[2, 2]),
                "cell_xy_A": float(atoms.cell[1, 0]),
                "cell_xz_A": float(atoms.cell[2, 0]),
                "cell_yz_A": float(atoms.cell[2, 1]),
            }
        for name, value in frame_values.items():
            series_lists[name].append(value)

        if frame_index % structural_stride == 0:
            centers = _methane_centers_fractional(atoms, host_atoms)
            if previous_centers is None:
                unwrapped_centers = centers.copy()
            else:
                step = centers - previous_centers
                step -= np.rint(step)
                unwrapped_centers = unwrapped_centers + step
            previous_centers = centers
            methane_center_frames.append(unwrapped_centers @ atoms.cell.array)
            methane_center_cells.append(np.asarray(atoms.cell.array))

            framework = _framework_metrics(
                atoms,
                host_atoms,
                reference_scaled,
                bond_first,
                bond_second,
                reference_bonds,
            )
            distances = _distance_metrics(atoms, host_atoms, centers)
            structural_values = {
                "frame": frame_index,
                "time_ps": time_ps,
                "framework_rmsd_A": framework[0],
                "framework_bond_rms_change_A": framework[1],
                "framework_bond_max_change_A": framework[2],
                "minimum_host_guest_atom_distance_A": distances[0],
                "minimum_host_methane_com_distance_A": distances[1],
                "minimum_methane_com_distance_A": distances[2],
            }
            for name, value in structural_values.items():
                structural_lists[name].append(value)

            if frame_index % rdf_stride == 0 and time_ps >= production_start_ps:
                rdf_host_guest, rdf_guest_guest = _accumulate_rdf(
                    atoms,
                    host_atoms,
                    centers,
                    rdf_edges,
                    rdf_host_guest,
                    rdf_guest_guest,
                )
                shell_volume = 4.0 * np.pi / 3.0 * (
                    rdf_edges[1:] ** 3 - rdf_edges[:-1] ** 3
                )
                guest_molecules = len(centers)
                rdf_host_guest_ideal += (
                    host_atoms * guest_molecules * shell_volume / volume
                )
                rdf_guest_guest_ideal += (
                    guest_molecules
                    * (guest_molecules - 1)
                    / 2.0
                    * shell_volume
                    / volume
                )
                rdf_frames += 1

    if atom_count is None or total_mass_amu is None:
        raise ValueError(f"trajectory contains no frames: {trajectory_path}")
    if thermodynamic_series is not None and raw_frame_count != len(
        thermodynamic_series["time_ps"]
    ):
        raise ValueError(
            "trajectory and LAMMPS thermo log contain different frame counts"
        )

    rdf_centers = 0.5 * (rdf_edges[:-1] + rdf_edges[1:])
    host_guest_rdf = np.divide(
        rdf_host_guest,
        rdf_host_guest_ideal,
        out=np.full_like(rdf_host_guest, np.nan),
        where=rdf_host_guest_ideal > 0.0,
    )
    guest_guest_rdf = np.divide(
        rdf_guest_guest,
        rdf_guest_guest_ideal,
        out=np.full_like(rdf_guest_guest, np.nan),
        where=rdf_guest_guest_ideal > 0.0,
    )
    return {
        "metadata": {
            "trajectory": str(trajectory_path),
            "atom_count": atom_count,
            "host_atoms": host_atoms,
            "methane_molecules": (atom_count - host_atoms) // 5,
            "total_mass_amu": total_mass_amu,
            "chemical_formula": chemical_formula,
            "frames": len(series_lists["frame"]),
            "raw_frames": raw_frame_count,
            "duplicate_restart_frames_removed": duplicate_thermo_frames,
            "frame_spacing_fs": frame_spacing_fs,
            "structural_stride": structural_stride,
            "rdf_stride": rdf_stride,
            "framework_reference_bonds": len(reference_bonds),
            "rdf_frames": rdf_frames,
        },
        "series": {
            name: np.asarray(values) for name, values in series_lists.items()
        },
        "structural": {
            **{
                name: np.asarray(values)
                for name, values in structural_lists.items()
            },
            "methane_com_unwrapped_A": np.asarray(methane_center_frames),
            "methane_com_cells_A": np.asarray(methane_center_cells),
            "rdf_distance_A": rdf_centers,
            "host_methane_com_rdf": host_guest_rdf,
            "methane_com_rdf": guest_guest_rdf,
        },
    }


def methane_mean_squared_displacement(
    methane_centers_A: np.ndarray,
    times_ps: np.ndarray,
    *,
    production_start_ps: float,
    maximum_lags: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute methane COM MSD using multiple time origins."""
    centers = np.asarray(methane_centers_A, dtype=float)
    times = np.asarray(times_ps, dtype=float)
    if centers.ndim != 3 or centers.shape[0] != len(times):
        raise ValueError("methane center and time arrays are inconsistent")
    selected = np.flatnonzero(times >= production_start_ps)
    if len(selected) < 2 or centers.shape[1] == 0:
        return np.empty(0), np.empty(0)
    centers = centers[selected]
    times = times[selected]
    lags = np.unique(
        np.linspace(0, len(times) - 1, min(maximum_lags, len(times)), dtype=int)
    )
    msd = []
    lag_times = []
    for lag in lags:
        if lag == 0:
            msd.append(0.0)
            lag_times.append(0.0)
            continue
        displacement = centers[lag:] - centers[:-lag]
        msd.append(float(np.mean(np.sum(displacement**2, axis=2))))
        lag_times.append(float(np.mean(times[lag:] - times[:-lag])))
    return np.asarray(lag_times), np.asarray(msd)
