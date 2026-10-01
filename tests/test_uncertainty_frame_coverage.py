"""Regression checks for dense frame selection and stale uncertainty archives."""

import tempfile
import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mof_heat_capacity.analysis.results import parse_args
from mof_heat_capacity.analysis.uncertainty import _selected_thermodynamic_indices
from mof_heat_capacity.analysis.uncertainty_comparison import (
    archive_sha256,
    hybrid_uncertainty_components,
)
from scripts.properties.compare_heat_capacity_estimators import compare


class FrameCoverageTest(unittest.TestCase):
    def test_default_selects_all_ten_thousand_coordinate_frames(self):
        with patch("sys.argv", ["analysis", "--model-uncertainty"]):
            args = parse_args()
        self.assertEqual(args.uncertainty_stride, 1)
        dump_steps = np.arange(2000, 12000) * 100
        # Thermo includes earlier equilibration and a duplicate restart step.
        raw_steps = np.insert(np.arange(12000) * 100, 2501, 250000)
        selected, thermo = _selected_thermodynamic_indices(
            {"md_step": raw_steps}, dump_steps, np.ones(10000, dtype=bool),
            args.uncertainty_stride,
        )
        np.testing.assert_array_equal(selected, np.arange(10000))
        np.testing.assert_array_equal(raw_steps[thermo], dump_steps)

    def test_production_cutoff_is_preserved_at_stride_one(self):
        steps = np.arange(10001) * 100
        mask = np.arange(10001) >= 2000
        selected, thermo = _selected_thermodynamic_indices(
            {"md_step": steps}, steps, mask, 1,
        )
        self.assertEqual(len(selected), 8001)
        self.assertEqual(selected[0], 2000)
        np.testing.assert_array_equal(selected, thermo)


class RefreshedUncertaintyTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.hybrid = self.root / "hybrid.npz"
        self.ensemble = self.root / "cea.npz"
        self.variance = self.root / "gaussian.npz"
        self.cea = np.array([[2., 3., 4.], [4., 5., 6.]])
        self.gaussian = np.array([[3., 4., 5.], [4., 5., 6.]])
        self.harmonic = np.array([[.5, .5, .5], [-.5, -.5, -.5]])
        np.savez(
            self.hybrid, temperatures_K=[200, 300, 400],
            approximate_cp_J_per_gK=[1., 1.1, 1.2],
            classical_anharmonic_cp_standard_error_J_per_gK=[.1, .1, .1],
            harmonic_quantum_correction_standard_error_J_per_gK=[.2, .2, .2],
            classical_anharmonic_cp_model_standard_deviation_J_per_gK=[8., 8., 8.],
            harmonic_quantum_correction_model_standard_deviation_J_per_gK=[np.sqrt(.5)] * 3,
            approximate_cp_model_standard_deviation_J_per_gK=[9., 9., 9.],
            harmonic_quantum_correction_by_member_J_per_gK=self.harmonic,
            llpr_checkpoint_sha256="test-model",
        )
        self.write_ensemble()
        np.savez(
            self.variance, temperatures_K=[200, 300, 400],
            cea_cp_by_member_J_per_gK=self.cea,
            gaussian_npt_cp_by_member_J_per_gK=self.gaussian,
            selected_frame_count=[10000] * 3, production_frame_count=[10000] * 3,
            frame_stride=1, model_sha256="test-model",
            cea_source_sha256=archive_sha256(self.ensemble),
        )

    def write_ensemble(self, stride=1):
        np.savez(
            self.ensemble, temperatures_K=[200, 300, 400],
            cea_cp_by_member_J_per_gK=self.cea, model_sha256="test-model",
            selected_frame_count=[10000 if stride == 1 else 500] * 3,
            production_frame_count=[10000] * 3, frame_stride=stride,
            replica_counts=[1] * 3,
        )

    def read(self, method):
        return hybrid_uncertainty_components(
            self.hybrid, method=method, ensemble_path=self.ensemble,
            variance_path=self.variance, require_all_frames=True,
        )

    def test_fresh_members_keep_sampling_hessians_and_paired_covariance(self):
        cea, gaussian = self.read("cea"), self.read("dpose")
        # The fresh CEA spread replaces the old hybrid archive's SD of 8.
        np.testing.assert_allclose(cea["classical_model_standard_deviation_J_per_gK"], np.sqrt(2))
        np.testing.assert_allclose(cea["hybrid_model_standard_deviation_J_per_gK"], np.sqrt(.5))
        # Gaussian and harmonic deviations cancel member by member.
        np.testing.assert_allclose(gaussian["hybrid_model_standard_deviation_J_per_gK"], 0)
        np.testing.assert_allclose(gaussian["combined_standard_uncertainty_J_per_gK"], np.sqrt(.05))
        for field in (
            "central_hybrid_cp_J_per_gK", "md_sampling_standard_error_J_per_gK",
            "harmonic_sampling_standard_error_J_per_gK", "harmonic_model_standard_deviation_J_per_gK",
        ):
            np.testing.assert_array_equal(cea[field], gaussian[field])

    def test_all_frame_plot_rejects_sparse_archives(self):
        self.write_ensemble(stride=20)
        with self.assertRaisesRegex(ValueError, "does not cover all"):
            self.read("cea")

    def test_gaussian_must_be_regenerated_after_cea_changes(self):
        self.cea += .25
        self.write_ensemble()
        with self.assertRaisesRegex(ValueError, "stale"):
            self.read("dpose")


class GaussianArchiveCoverageTest(unittest.TestCase):
    def test_comparison_consumes_all_ten_thousand_frames_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = root / "model" / "100ch4"
            group.mkdir(parents=True)
            config = root / "run.toml"
            config.write_text("[md]\npressure_bar = 1.0\n")
            ensemble = group / "model_uncertainty_heat_capacity.npz"
            np.savez(
                ensemble, temperatures_K=[200, 300, 400], model_sha256="model",
                cea_cp_by_member_J_per_gK=np.ones((2, 3)),
                direct_cp_by_member_J_per_gK=np.ones((2, 3)),
                selected_frame_count=[10000] * 3, frame_stride=1,
            )
            values = np.linspace(-1, 1, 10000)
            for temperature in (200, 300, 400):
                run = group / f"{temperature}K" / "rep01"
                run.mkdir(parents=True)
                (run / "summary.json").write_text(json.dumps({
                    "config": str(config),
                    "metadata": {"total_mass_amu": 100, "atom_count": 10},
                    "model_uncertainty": {"sampled_frames": 10000, "frame_stride": 1},
                }))
                np.savez(
                    run / "model_uncertainty.npz", model_sha256="model",
                    central_potential_eV=values,
                    member_potential_eV=np.column_stack([values, 2 * values]),
                    volume_A3=np.ones(10000), frame_stride=1,
                    production_frame_count=10000,
                )
            args = types.SimpleNamespace(
                analysis_dir=root, model_label="model", loading=100,
                replicas=[1], output_dir=None, require_all_frames=True,
            )
            with patch("scripts.properties.compare_heat_capacity_estimators._plot_comparison"):
                _, output = compare(args)
            with np.load(output) as archive:
                np.testing.assert_array_equal(archive["selected_frame_count"], [10000] * 3)
                np.testing.assert_array_equal(archive["production_frame_count"], [10000] * 3)
                self.assertEqual(int(archive["frame_stride"]), 1)
                self.assertEqual(str(archive["cea_source_sha256"]), archive_sha256(ensemble))


if __name__ == "__main__":
    unittest.main()
