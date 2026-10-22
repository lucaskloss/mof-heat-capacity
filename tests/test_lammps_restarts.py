"""Tests for LAMMPS logs and dumps that overlap at safe restart points."""

from pathlib import Path
import tempfile
import unittest

from mof_heat_capacity.analysis.lammps import _COLUMN_MAP, read_lammps_thermo
from mof_heat_capacity.analysis.trajectory import read_trajectory_observables


def _write_thermo_log(path: Path, steps: list[int]) -> None:
    columns = list(_COLUMN_MAP)
    with path.open("w") as handle:
        handle.write(" ".join(columns) + "\n")
        for frame, step in enumerate(steps):
            values = {column: 0.0 for column in columns}
            values.update({"Step": float(step), "Time": step * 0.0005,
                           "Temp": 300.0, "PotEng": float(frame)})
            handle.write(" ".join(str(values[column]) for column in columns) + "\n")


def _write_dump(path: Path, steps: list[int]) -> None:
    with path.open("w") as handle:
        for frame, step in enumerate(steps):
            handle.write(
                "ITEM: TIMESTEP\n"
                f"{step}\n"
                "ITEM: NUMBER OF ATOMS\n"
                "4\n"
                "ITEM: BOX BOUNDS pp pp pp\n"
                "0 20\n0 20\n0 20\n"
                "ITEM: ATOMS id element xu yu zu vx vy vz fx fy fz\n"
            )
            positions = (
                (5.0, 5.0, 5.0),
                (10.0 + 0.01 * frame, 10.0, 10.0),
                (11.2 + 0.01 * frame, 10.0, 10.0),
                (9.4 + 0.01 * frame, 10.0, 10.0),
            )
            for atom_id, (element, position) in enumerate(
                zip(("Al", "C", "O", "O"), positions, strict=True), 1
            ):
                handle.write(
                    f"{atom_id} {element} {position[0]} {position[1]} {position[2]} "
                    "0 0 0 0 0 0\n"
                )


class LammpsRestartOverlapTest(unittest.TestCase):
    def test_thermo_log_keeps_first_occurrence_of_restart_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "md.lammps.log"
            _write_thermo_log(path, [0, 100, 200, 100, 200, 300])

            thermo = read_lammps_thermo(path)

        self.assertEqual(thermo["md_step"].tolist(), [0, 100, 200, 300])
        # Step 100's potential energy comes from its first occurrence.
        self.assertEqual(thermo["potential_energy_eV"][1], 1.0)

    def test_trajectory_analysis_skips_replayed_dump_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            thermo_path = Path(directory) / "md.lammps.log"
            dump_path = Path(directory) / "trajectory.lammpstrj"
            steps = [0, 100, 200, 100, 200, 300]
            _write_thermo_log(thermo_path, steps)
            _write_dump(dump_path, steps)

            result = read_trajectory_observables(
                dump_path,
                frame_spacing_fs=50.0,
                production_start_ps=0.0,
                host_atoms=1,
                guest="co2",
                structural_stride=1,
                rdf_stride=1,
                rdf_bins=10,
                thermodynamic_series=read_lammps_thermo(thermo_path),
            )

        self.assertEqual(result["metadata"]["raw_frames"], 6)
        self.assertEqual(result["metadata"]["frames"], 4)
        self.assertEqual(result["metadata"]["duplicate_trajectory_records_ignored"], 2)
        self.assertEqual(result["series"]["md_step"].tolist(), [0, 100, 200, 300])


if __name__ == "__main__":
    unittest.main()
