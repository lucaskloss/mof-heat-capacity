"""Campaign isolation, guest analysis, and Slurm selection propagation."""

from contextlib import redirect_stdout
from io import StringIO
import json
import shlex
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms, io
from ase.calculators.singlepoint import SinglePointCalculator

from mof_heat_capacity import campaign
from mof_heat_capacity.analysis import results
from mof_heat_capacity.analysis import hybrid
from mof_heat_capacity.analysis.statistics import AMU_TO_G, EV_TO_J
from mof_heat_capacity.analysis.lammps import _COLUMN_MAP
from mof_heat_capacity.analysis.trajectory import (
    _guest_centers_fractional,
    read_trajectory_observables,
)
from mof_heat_capacity.config import LOADED_RUN_PATTERN, load_run_config, loaded_config_path
from mof_heat_capacity.io import lammps_species, write_classical_npt_lammps_input, write_lammps_data
from mof_heat_capacity.protocols import loaded


PROJECT = Path(__file__).resolve().parents[1]
MODEL = "pet-mad-1.5-s-40nn"


class CampaignSelectionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "configs").mkdir()
        (self.root / "input").mkdir()
        self.template = self.root / "configs" / "template.toml"
        shutil.copy(PROJECT / "configs/mof5_100ch4_hybrid_npt.toml", self.template)
        for guest in campaign.GUEST_SYMBOLS:
            shutil.copy(PROJECT / "input" / f"{guest}.gro", self.root / "input")
        self.host = self.root / "input" / "example.cif"
        io.write(self.host, Atoms("Al", positions=[[10, 10, 10]], cell=[20, 20, 20], pbc=True))

    def prepare(self, guest, model="pet-mad", *extra):
        argv = ["loaded", "--mof", "example", "--guest", guest, "--model", model,
                "--host", str(self.host), "--loading", "2", "--temperatures", "300",
                "--output-root", str(self.root / "output"), *extra]
        with patch.object(loaded, "PROJECT_DIR", self.root), \
                patch.object(loaded, "CONFIG_TEMPLATE", self.template), \
                patch("sys.argv", argv), redirect_stdout(StringIO()):
            loaded.main()
        return loaded_config_path(self.root / "configs", loaded.MODEL_PRESETS[model]["label"], 2, 300, 1, "example", guest)

    def test_all_guests_prepare_distinct_structures_and_correct_model_paths(self):
        paths = []
        for guest, symbols in campaign.GUEST_SYMBOLS.items():
            path = self.prepare(guest)
            config = load_run_config(path)
            atoms = io.read(config.structure)
            self.assertEqual(len(atoms), 1 + 2 * len(symbols))
            self.assertEqual(config.host_atoms, 1)
            self.assertEqual((config.mof, config.guest, config.loading), ("example", guest, 2))
            self.assertEqual(config.required_elements, frozenset({"Al", *symbols}))
            self.assertEqual(config.checkpoint, self.root / "models/pet-mad-1.5-s_40nn_nostress.ckpt")
            self.assertEqual(config.output_dir, self.root / f"output/md/production/example/{MODEL}/2{guest}/300K/rep01")
            paths.append(config.structure)
        self.assertEqual(len(set(paths)), 3)
        before = paths[1].read_bytes()
        config = load_run_config(self.prepare("co2", "pet-sol", "--skip-existing"))
        self.assertEqual(config.structure, paths[1])
        self.assertEqual(paths[1].read_bytes(), before)

    def test_reuse_rejects_changed_host_contents(self):
        path = self.prepare("h2o")
        before = path.read_bytes()
        self.host.write_text(self.host.read_text() + "\n# edited source\n")
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.prepare("h2o", "pet-mad", "--skip-existing")
        self.assertEqual(path.read_bytes(), before)

    def test_legacy_paths_and_names_remain_usable(self):
        self.assertEqual(campaign.system_directory(MODEL, 100), Path(MODEL) / "100ch4")
        legacy_path = self.root / "legacy.toml"
        legacy_path.write_text(self.template.read_text()
                              .replace('"REPLACE_WITH_STRUCTURE_PATH"', json.dumps(str(self.host)))
                              .replace('"REPLACE_WITH_OUTPUT_DIR"', json.dumps(str(self.root / "legacy-output"))))
        config = load_run_config(legacy_path)
        self.assertEqual((config.mof, config.guest, config.host_atoms), ("mof5", "ch4", 424))
        for mof in ("mof5", "uio-66", "mof_74"):
            for guest in campaign.GUEST_SYMBOLS:
                match = LOADED_RUN_PATTERN.fullmatch(f"{mof}-100{guest}-{MODEL}-npt-300K-rep01")
                self.assertEqual((match["mof"], match["guest"]), (mof, guest))

    def test_invalid_selection_and_wrong_molecule_are_rejected(self):
        for mof in ("../mof5", "bad/name", "MOF-5", ""):
            with self.assertRaises(ValueError):
                campaign.validate_selection(mof, "co2")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            campaign.validate_guest(io.read(self.root / "input/ch4.gro"), "h2o")
        with self.assertRaisesRegex(ValueError, "positive"):
            self.prepare("co2", "pet-mad", "--replicas", "0")

    def test_guest_com_across_periodic_boundary_and_streamed_counts(self):
        for guest in campaign.GUEST_SYMBOLS:
            molecule = io.read(self.root / "input" / f"{guest}.gro")
            molecule.positions += [19.9, 5, 5]
            host = Atoms("Al", positions=[[10, 10, 10]], cell=[20, 20, 20], pbc=True)
            atoms = host + molecule
            expected = molecule.get_center_of_mass() / 20 % 1
            atoms.wrap()
            np.testing.assert_allclose(_guest_centers_fractional(atoms, 1, guest)[0], expected)
            atoms.calc = SinglePointCalculator(atoms, energy=0, forces=np.zeros((len(atoms), 3)), stress=np.zeros(6))
            trajectory = self.root / f"{guest}.traj"
            io.write(trajectory, [atoms, atoms], format="traj")
            extracted = read_trajectory_observables(trajectory, frame_spacing_fs=1, production_start_ps=0,
                                                   host_atoms=1, guest=guest, structural_stride=1, rdf_stride=1, rdf_bins=5)
            self.assertEqual(extracted["metadata"]["guest_molecules"], 1)
            self.assertEqual(extracted["structural"]["guest_com_unwrapped_A"].shape, (2, 1, 3))
            if guest != "ch4":
                self.assertNotIn("methane_molecules", extracted["metadata"])

    def test_additional_mof_elements_have_consistent_lammps_types(self):
        atoms = Atoms("AlCOH", positions=np.zeros((4, 3)), cell=[20, 20, 20], pbc=True)
        species = lammps_species(atoms)
        self.assertEqual(species, ("C", "H", "O", "Al"))
        data = self.root / "structure.data"
        write_lammps_data(data, atoms)
        self.assertEqual(io.read(data, format="lammps-data").get_chemical_symbols(), atoms.get_chemical_symbols())
        output = self.root / "md.in"
        write_classical_npt_lammps_input(output, data_path=data, model_path=self.root / "model.pt", device="cuda",
            trajectory_path=self.root / "trajectory", thermo_path=self.root / "log", restart_prefix=self.root / "restart",
            final_data_path=self.root / "final.data", temperature_K=300, pressure_bar=1, timestep_fs=.5,
            thermostat_tau_fs=100, thermostat_chain_length=3, barostat_tau_fs=1000, steps=1000,
            equilibration_steps=200, output_stride=100, restart_stride=100, seed=2025, species=species)
        self.assertIn("pair_coeff * * 6 1 8 13\n", output.read_text())
        self.assertIn("dump_modify production element C H O Al ", output.read_text())

    def test_direct_analysis_groups_systems_before_aggregation(self):
        runs = [(Path("config"), SimpleNamespace(name=f"{mof}-2{guest}-{MODEL}-npt-300K-rep01",
                    mof=mof, guest=guest, loading=2), Path("trajectory"))
                for mof in ("mof5", "example") for guest in ("co2", "h2o")]
        with patch("sys.argv", ["results", "--analysis-dir", str(self.root / "analysis")]), \
                patch.object(results, "discover_runs", return_value=(runs, [])), \
                patch.object(results, "process_runs") as process:
            results.main()
        directories = [call.args[0].analysis_dir for call in process.call_args_list]
        self.assertEqual(len(set(directories)), 4)
        self.assertIn(self.root / f"analysis/example/{MODEL}/2co2", directories)

    def test_completed_water_analysis_uses_campaign_metadata(self):
        path = self.prepare("h2o")
        config = load_run_config(path)
        config.output_dir.mkdir(parents=True)
        frames = []
        for index in range(8):
            atoms = io.read(config.structure)
            atoms.calc = SinglePointCalculator(atoms, energy=float(index % 3),
                forces=np.zeros((len(atoms), 3)), stress=np.zeros(6))
            frames.append(atoms)
        io.write(config.output_dir / "trajectory.traj", frames, format="traj")
        path.write_text(path.read_text().replace('driver = "lammps"', 'driver = "ase"')
                        .replace('trajectory_file = "trajectory.lammpstrj"', 'trajectory_file = "trajectory.traj"'))
        analysis = self.root / "analysis"
        with patch("sys.argv", ["results", "--config-dir", str(self.root / "configs"),
                "--analysis-dir", str(analysis), "--discard-ps", "0", "--no-plots",
                "--structural-stride", "1", "--rdf-stride", "1"]), redirect_stdout(StringIO()):
            results.main()
        group = analysis / f"example/{MODEL}/2h2o"
        summary = json.loads((group / "300K/rep01/summary.json").read_text())
        self.assertEqual(summary["metadata"]["host_atoms"], 1)
        self.assertEqual(summary["metadata"]["guest_molecules"], 2)
        self.assertEqual((summary["mof"], summary["guest"]), ("example", "h2o"))
        self.assertTrue((group / "runs.csv").is_file())
        with np.load(group / "300K/rep01/structural_distributions.npz") as archive:
            self.assertIn("guest_com_msd_A2", archive.files)
            self.assertNotIn("methane_com_msd_A2", archive.files)

    def test_default_grid_padding_gives_centered_endpoint_derivatives(self):
        with patch("sys.argv", ["loaded"]):
            md_args = loaded.parse_args()
        temperatures = np.arange(175, 426, 25, dtype=float)
        self.assertEqual(loaded.temperatures(md_args.temperatures), temperatures.tolist())
        with patch("sys.argv", ["hybrid", "--hybrid-dir", str(self.root / "harmonic")]):
            args = hybrid.parse_args()
        records = (temperatures, temperatures**3, np.zeros(11), np.full(11, 1000), np.zeros(11), 100., [])
        with patch.object(hybrid, "_enthalpy_records", return_value=records) as enthalpy, \
                patch.object(hybrid, "_harmonic_corrections", return_value=(np.zeros(9), np.zeros(9), None, None, [])), \
                patch.object(hybrid, "plot_hybrid_heat_capacity"), redirect_stdout(StringIO()):
            output = hybrid.run(args)
        self.assertEqual(enthalpy.call_args.kwargs["temperatures"], temperatures.tolist())
        with np.load(output) as data:
            np.testing.assert_array_equal(data["temperatures_K"], np.arange(200, 401, 25))
            cp = data["classical_anharmonic_cp_J_per_gK"]
        conversion = EV_TO_J / (100 * AMU_TO_G)
        # A cubic distinguishes centered stencils from one-sided endpoint estimates.
        self.assertAlmostEqual(cp[0] / conversion, (225**3 - 175**3) / 50)
        self.assertAlmostEqual(cp[-1] / conversion, (425**3 - 375**3) / 50)
        self.assertEqual(campaign.default_report_temperatures([275, 300, 325]), [275, 300, 325])


class SlurmSelectionTest(unittest.TestCase):
    def test_dry_run_pipeline_and_property_commands_keep_host_and_guest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ("scripts", "mof_heat_capacity", "input"):
                shutil.copytree(PROJECT / folder, root / folder, ignore=shutil.ignore_patterns("__pycache__"))
            (root / "configs").mkdir()
            shutil.copy(PROJECT / "configs/mof5_100ch4_hybrid_npt.toml", root / "configs")
            (root / "models").mkdir()
            (root / "models/pet-mad-1.5-s_40nn_nostress.pt").touch()
            # A stand-in model provides capabilities only; no GPU or scheduler is used.
            (root / "metatomic").mkdir()
            (root / "metatomic/__init__.py").touch()
            (root / "metatomic/torch.py").write_text(
                "from types import SimpleNamespace\n"
                "def load_atomistic_model(path):\n"
                "    return SimpleNamespace(capabilities=lambda: SimpleNamespace(outputs={'stress'}, atomic_types=range(1,119)))\n"
            )
            host = root / "input/example.cif"
            io.write(host, Atoms("Al", positions=[[10, 10, 10]], cell=[20, 20, 20], pbc=True))
            env = {**os.environ, "MOF_OUTPUT_ROOT": str(root / "output"), "MOF_CAMPAIGN_PYTHON": sys.executable,
                   "MOF_ANALYSIS_PYTHON": sys.executable, "MOF_SLURM_OUTPUT_DIR": str(root / "output/slurm")}
            selection = ["--model", "pet-mad", "--mof", "example", "--guest", "CO2", "--loading", "2", "--dry-run"]

            def run(script, *extra):
                completed = subprocess.run(["bash", str(root / script), *selection, *extra], cwd=root,
                    env=env, text=True, capture_output=True)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                return completed.stdout

            output = run("scripts/md/submit_loaded_md.sh", "--temperatures", "300", "--host", str(host))
            planner = shlex.split(output.splitlines()[-1].removeprefix("DRY RUN: "))
            continuation = planner[planner.index("--wrap") + 1]
            self.assertIn("--mof example --guest co2", continuation)
            self.assertIn(f"--host {host}", continuation)
            self.assertIn(f"md/calibration/example/{MODEL}/2co2/", output)
            config = load_run_config(root / f"configs/example/{MODEL}/2co2/300K-rep01.toml")
            config.output_dir.mkdir(parents=True)
            write_lammps_data(config.output_dir / "md.final.data", io.read(config.structure))
            (config.output_dir / "trajectory.lammpstrj").touch()
            columns = list(_COLUMN_MAP)
            values = [1000000 if name == "Step" else 500 if name == "Time" else 1 for name in columns]
            (config.output_dir / "md.lammps.log").write_text(" ".join(columns) + "\n" + " ".join(map(str, values)) + "\n")
            output = run("scripts/properties/submit_analysis.sh", "--temperatures", "300", "--no-model-uncertainty")
            self.assertIn(f"trajectory-analysis/example/{MODEL}/2co2/300K/rep01", output)
            self.assertIn(f"trajectory-analysis/example/{MODEL}/2co2/summary", output)
            output = run("scripts/properties/submit_heat_capacity.sh", "--source-temperature", "300", "--host", str(host), "--no-model-uncertainty")
            self.assertIn(f"harmonic-correction/example/{MODEL}/2co2", output)
            self.assertIn(f"harmonic-correction/example/{MODEL}/0co2", output)

            output = run("scripts/md/submit_loaded_md.sh", "--host", str(host))
            self.assertTrue((root / f"configs/example/{MODEL}/2co2/175K-rep01.toml").is_file())
            self.assertTrue((root / f"configs/example/{MODEL}/2co2/425K-rep01.toml").is_file())
            output = run("scripts/properties/submit_hybrid_analysis.sh")
            self.assertIn("--temperatures 175\\,200\\,225\\,250\\,275\\,300\\,325\\,350\\,375\\,400\\,425", output)
            self.assertIn("--report-temperatures 200\\,225\\,250\\,275\\,300\\,325\\,350\\,375\\,400", output)
            output = run("scripts/properties/submit_hybrid_analysis.sh", "--temperatures", "275,300,325")
            self.assertIn("--mof example --guest co2", output)
            self.assertIn(f"harmonic-correction/example/{MODEL}/2co2", output)


if __name__ == "__main__":
    unittest.main()
