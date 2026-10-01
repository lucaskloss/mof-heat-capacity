"""Check that dense LLPR updates preserve cached predictions and resume holes."""

import contextlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import ase.io

from mof_heat_capacity.analysis import uncertainty


class CommitteeCacheTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model.pt"
        self.model.write_bytes(b"fixed-model")
        self.trajectory = self.root / "trajectory.extxyz"
        self.trajectory.write_text("fixed-trajectory")
        self.cache = self.root / "model_uncertainty.npz"
        self.progress = self.root / "model_uncertainty.progress.npz"
        self.steps = np.arange(9) * 100 + 200000
        self.thermo = {
            "md_step": self.steps,
            "time_ps": 100 + np.arange(9) * .05,
            "potential_energy_eV": 10 + np.arange(9, dtype=float),
            "kinetic_energy_eV": np.full(9, 2.),
            "volume_A3": np.full(9, 100.),
        }
        self.evaluated = []
        self.fail_after_batches = None
        self.batches = 0
        test = self

        class Calculator:
            def run_model(self, structures, _requested):
                if test.fail_after_batches is not None and test.batches >= test.fail_after_batches:
                    raise RuntimeError("interrupted")
                test.batches += 1
                frames = np.asarray(structures, dtype=int)
                test.evaluated.extend(frames.tolist())
                energy = 10 + frames.astype(float)
                deviation = .1 * (frames + 1)
                return {
                    "energy": energy,
                    "energy_ensemble": np.column_stack([energy + deviation, energy - deviation]),
                    "energy_uncertainty": deviation,
                }

        torch_module = types.ModuleType("metatomic.torch")
        torch_module.ModelOutput = lambda **_kwargs: None
        torch_module.load_atomistic_model = lambda _path: types.SimpleNamespace(
            capabilities=lambda: types.SimpleNamespace(outputs={"energy", "energy_ensemble", "energy_uncertainty"})
        )
        metatomic = types.ModuleType("metatomic")
        metatomic.torch = torch_module
        metatomic.__path__ = []
        calculator_module = types.ModuleType("metatomic_ase")
        calculator_module.MetatomicCalculator = lambda *_args, **_kwargs: Calculator()
        self.modules = {
            "metatomic": metatomic, "metatomic.torch": torch_module,
            "metatomic_ase": calculator_module,
        }

    def evaluate(self, stride=1):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", self.modules))
            self.reader_mock = stack.enter_context(patch("ase.io.iread", return_value=iter(range(9))))
            stack.enter_context(patch.object(uncertainty, "_use_vesin_neighbor_lists"))
            stack.enter_context(patch.object(uncertainty, "_system_values", side_effect=lambda value, *_args: value))
            stack.enter_context(patch.object(uncertainty, "_ensemble_values", side_effect=lambda value, *_args: value))
            return uncertainty.evaluate_trajectory_committee(
                self.trajectory, self.thermo, self.steps, np.ones(9, dtype=bool),
                model_path=self.model, device="cpu", temperature_K=300,
                pressure_bar=1, stride=stride, batch_size=2,
                checkpoint_path=self.progress, cache_path=self.cache,
            )

    def save_sparse(self, legacy=False):
        values = self.evaluate(stride=4)
        if legacy:
            values = {key: value for key, value in values.items() if key not in {
                "inference_model_path", "inference_model_sha256", "trajectory_path",
                "trajectory_size_bytes", "trajectory_mtime_ns", "frame_stride",
            }}
        uncertainty.write_committee_archive(self.cache, values)
        self.progress.unlink(missing_ok=True)
        self.evaluated.clear()
        self.batches = 0
        return values

    def test_legacy_sparse_rows_are_kept_and_only_holes_are_evaluated(self):
        sparse = self.save_sparse(legacy=True)
        dense = self.evaluate()
        self.assertEqual(self.evaluated, [1, 2, 3, 5, 6, 7])
        self.assertEqual(dense["reused_frame_count"], 3)
        self.assertEqual(dense["evaluated_frame_count"], 6)
        np.testing.assert_array_equal(dense["md_step"], self.steps)
        for field in ("central_potential_eV", "member_potential_eV", "analytical_energy_uncertainty_eV"):
            np.testing.assert_array_equal(dense[field][[0, 4, 8]], sparse[field])

    def test_interruption_preserves_sparse_file_and_resumes_nonprefix_progress(self):
        self.save_sparse()
        original_bytes = self.cache.read_bytes()
        self.fail_after_batches = 1
        with self.assertRaisesRegex(RuntimeError, "could not evaluate"):
            self.evaluate()
        self.assertEqual(self.cache.read_bytes(), original_bytes)
        with np.load(self.progress) as archive:
            np.testing.assert_array_equal(archive["frame"], [0, 1, 2, 4, 8])
        self.fail_after_batches = None
        self.evaluated.clear()
        result = self.evaluate()
        self.assertEqual(self.evaluated, [3, 5, 6, 7])
        self.assertEqual(result["reused_frame_count"], 5)
        uncertainty.write_committee_archive(self.cache, result)
        self.progress.unlink()
        self.evaluated.clear()
        complete = self.evaluate()
        self.reader_mock.assert_not_called()
        self.assertEqual(self.evaluated, [])
        self.assertEqual(complete["reused_frame_count"], 9)

    def test_changed_model_is_rejected_without_overwriting_cached_values(self):
        self.save_sparse()
        original_bytes = self.cache.read_bytes()
        self.model.write_bytes(b"different-model")
        with self.assertRaisesRegex(ValueError, "checkpoint hash changed"):
            self.evaluate()
        self.assertEqual(self.cache.read_bytes(), original_bytes)
        self.assertEqual(self.evaluated, [])

    def test_changed_thermo_is_rejected_before_inference(self):
        self.save_sparse()
        self.thermo["volume_A3"][4] += 1
        with self.assertRaisesRegex(ValueError, "volume_A3"):
            self.evaluate()
        self.assertEqual(self.evaluated, [])

    def test_changed_trajectory_is_rejected_before_inference(self):
        self.save_sparse()
        self.trajectory.write_text("changed-trajectory")
        with self.assertRaisesRegex(ValueError, "trajectory was modified"):
            self.evaluate()
        self.assertEqual(self.evaluated, [])


if __name__ == "__main__":
    unittest.main()
