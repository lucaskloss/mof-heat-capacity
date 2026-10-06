"""Read the supplied CIFs by MOF label and prepare loaded LAMMPS campaigns."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase import io

from mof_heat_capacity import campaign
from mof_heat_capacity.config import load_run_config, loaded_config_path
from mof_heat_capacity.io import load_structure
from mof_heat_capacity.protocols import loaded
from mof_heat_capacity.simulation import production


PROJECT = Path(__file__).resolve().parents[1]
MODEL = "pet-mad-1.5-s-40nn"


class SuppliedMOFsTest(unittest.TestCase):
    def test_label_only_preparation_preserves_cells_and_lammps_element_mapping(self):
        expected = {
            "mgmof74": (162, {"C", "H", "O", "Mg"}, "6 1 8 12"),
            "mof303": (128, {"C", "H", "O", "N", "Al"}, "6 1 8 7 13"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(PROJECT / "input", root / "input")
            (root / "configs").mkdir()
            template = root / "configs/mof5_100ch4_hybrid_npt.toml"
            shutil.copy(PROJECT / "configs/mof5_100ch4_hybrid_npt.toml", template)
            (root / "models").mkdir()
            (root / "models/pet-mad-1.5-s_40nn_nostress.pt").touch()

            for mof, (host_count, elements, pair_coeff) in expected.items():
                source = load_structure(PROJECT / "input" / f"{mof}.cif")
                self.assertEqual(len(source), host_count)
                self.assertEqual(set(source.get_chemical_symbols()), elements)
                for guest in ("ch4", "co2", "h2o"):
                    with self.subTest(mof=mof, guest=guest):
                        argv = ["loaded", "--model", "pet-mad", "--mof", mof,
                                "--guest", guest, "--loading", "50", "--temperatures", "300",
                                "--output-root", str(root / "output")]
                        with patch.object(campaign, "PROJECT_DIR", root), \
                                patch.object(loaded, "PROJECT_DIR", root), \
                                patch.object(loaded, "CONFIG_TEMPLATE", template), \
                                patch("sys.argv", argv), redirect_stdout(StringIO()):
                            loaded.main()
                        path = loaded_config_path(root / "configs", MODEL, 50, 300, 1, mof, guest)
                        config = load_run_config(path)
                        self.assertEqual(config.initial_relaxation, mof == "mgmof74")
                        self.assertEqual(config.source_host, root / "input" / f"{mof}.cif")
                        self.assertEqual(config.host_atoms, host_count)
                        self.assertEqual(config.required_elements, frozenset(elements))
                        self.assertEqual(config.structure.suffix, ".extxyz")
                        atoms = load_structure(config.structure)
                        guest_atoms = 5 if guest == "ch4" else 3
                        self.assertEqual(len(atoms), host_count + 50 * guest_atoms)
                        self.assertEqual(atoms.get_chemical_symbols()[:host_count], source.get_chemical_symbols())
                        np.testing.assert_allclose(atoms.cell.array, source.cell.array, rtol=0, atol=1e-12)
                        np.testing.assert_allclose(atoms.positions[:host_count], source.positions, rtol=0, atol=1e-8)
                        self.assertTrue(config.structure.with_suffix(".pdb").is_file())
                        data = io.read(config.structure.with_suffix(".data"), format="lammps-data")
                        self.assertEqual(data.get_chemical_symbols(), atoms.get_chemical_symbols())
                        np.testing.assert_allclose(data.cell.array, source.cell.array, rtol=0, atol=1e-12)
                        self.assertEqual(config.md_steps * config.timestep_fs / 1000, 500)
                        self.assertEqual(config.equilibration_steps * config.timestep_fs / 1000, 100)

                        # Exercise the real MD input writer without executing LAMMPS.
                        with patch.object(production, "_validate_stress_capability"), \
                                patch.object(production, "_executable", return_value="lmp"), \
                                patch.object(production.subprocess, "run") as execute, \
                                redirect_stdout(StringIO()):
                            production.run_classical_npt(config)
                        self.assertEqual(execute.call_count, 1)
                        script = (config.output_dir / "md.lammps.in").read_text()
                        if mof == "mgmof74":
                            self.assertLess(script.index("minimize "), script.index("velocity all create"))
                            self.assertLess(script.index("reset_timestep 0"), script.index("fix loaded_npt"))
                            self.assertIn("min_modify norm inf dmax 0.05", script)
                            self.assertNotIn("box/relax", script)
                            self.assertIn("quit 1", script)
                            restart = config.output_dir / "md.restart.100000"
                            restart.touch()
                            with patch.object(production, "_validate_stress_capability"), \
                                    patch.object(production, "_executable", return_value="lmp"), \
                                    patch.object(production.subprocess, "run"), \
                                    redirect_stdout(StringIO()):
                                production.run_classical_npt(config, resume=True)
                            resumed = (config.output_dir / "md.lammps.in").read_text()
                            self.assertNotIn("minimize ", resumed)
                            self.assertNotIn("reset_timestep", resumed)
                            self.assertIn(f"read_restart {restart}", resumed)
                        else:
                            self.assertNotIn("minimize ", script)
                        self.assertIn(f"pair_coeff * * {pair_coeff}\n", script)
                        runtime_data = io.read(config.output_dir / "md.initial.data", format="lammps-data")
                        np.testing.assert_allclose(runtime_data.cell.array, source.cell.array, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
