"""CPU regression checks with a deterministic stand-in for PET Hessians."""

import contextlib
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io import write

from mof_heat_capacity.analysis import harmonic
from mof_heat_capacity.analysis.llpr_merge import merge_archives, read_archive


class LLPRDistributionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.trajectory = self.root / "minimum.extxyz"
        write(self.trajectory, Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]], cell=[5, 5, 5], pbc=True))
        self.config = types.SimpleNamespace(
            ad_backend="pet-jax", heat_shadow=False, heat_dtype="float64",
            heat_hops=3, heat_chunk_size=1, heat_remat=True, heat_frames="all",
            heat_temperatures="200:400:25", heat_start_frame=0,
            checkpoint=self.root / "model", jax_checkpoint=self.root / "jax-model",
            name="test", structure=self.trajectory,
        )
        random = np.random.default_rng(8)
        self.matrices = []
        for _ in range(65):
            a = random.normal(size=(6, 6))
            self.matrices.append(a @ a.T)
        self.evaluated = []
        def evaluate(_fn, params, *_args):
            self.evaluated.append(params)
            return self.matrices[params]
        def cv_curve(frequencies, temperatures, mass):
            positive = frequencies[frequencies > 1]
            return {t: float(np.sum(np.exp(-positive / t)) / mass) for t in temperatures}
        pet = types.ModuleType("sadmof.models.pet")
        pet.load_pet = lambda *_args, **_kwargs: (None, 0, {})
        observables = types.ModuleType("sadmof.observables")
        observables.cv_curve = cv_curve
        jax = types.ModuleType("jax")
        jax.config = types.SimpleNamespace(update=lambda *_args: None)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict("sys.modules", {
            "jax": jax, "sadmof": types.ModuleType("sadmof"),
            "sadmof.models": types.ModuleType("sadmof.models"),
            "sadmof.models.pet": pet, "sadmof.observables": observables,
        }))
        self.stack.enter_context(patch.object(harmonic, "ensure_jax_checkpoint", return_value=self.config.jax_checkpoint))
        self.stack.enter_context(patch.object(harmonic, "prepare_frame_hessian", return_value=(None, None, None, None)))
        self.stack.enter_context(patch.object(harmonic, "evaluate_frame_hessian", side_effect=evaluate))
        self.stack.enter_context(patch.object(harmonic, "load_llpr_energy_parameters", return_value={"sha256": "committee", "path": self.root / "llpr"}))
        self.stack.enter_context(patch.object(harmonic, "llpr_member_parameter_sets", return_value=range(1, 65)))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def run_archive(self, name, *options):
        path = self.root / name
        argv = ["harmonic", "--config", "unused", "--trajectory", str(self.trajectory),
                "--frame-indices", "0", "--output", str(path), *options]
        with patch("sys.argv", argv):
            args = harmonic.parse_args()
        harmonic.run_heat_capacity(self.config, args)
        return path

    def distributed(self):
        central = self.run_archive("central.npz")
        batches = [self.run_archive(f"batch-{i}.npz", "--llpr-checkpoint", "llpr",
                                   "--central-archive", str(central),
                                   "--llpr-member-range", f"{i * 8}:{(i + 1) * 8}") for i in range(8)]
        return central, batches

    def test_merge_matches_serial_and_computes_central_once(self):
        serial = read_archive(self.run_archive("serial.npz", "--llpr-checkpoint", "llpr"))
        self.evaluated.clear()
        central, batches = self.distributed()
        self.assertEqual(self.evaluated.count(0), 1)
        self.assertEqual(sorted(self.evaluated), list(range(65)))
        merged = read_archive(merge_archives(central, list(reversed(batches)), self.root / "merged.npz"))
        for key in (
            "frequencies_cm1", "cv_J_per_gK", "llpr_frequencies_cm1", "llpr_cv_J_per_gK",
            "llpr_frequency_standard_deviation_cm1", "llpr_cv_standard_deviation_J_per_gK",
            "llpr_hessian_member_rms_deviation_eV_per_A2",
            "llpr_hessian_rms_standard_deviation_eV_per_A2",
            "llpr_hessian_member_mean_eV_per_A2", "llpr_hessian_member_m2_eV2_per_A4",
        ):
            np.testing.assert_allclose(merged[key], serial[key], rtol=1e-12, atol=1e-12, err_msg=key)
        np.testing.assert_array_equal(merged["llpr_member_indices"], np.arange(64))

    def test_merge_rejects_missing_and_duplicate_members(self):
        central, batches = self.distributed()
        for bad in [batches[:-1], batches[:-1] + [batches[0]]]:
            with self.assertRaisesRegex(ValueError, "exactly once"):
                merge_archives(central, bad, self.root / "bad.npz")

    def test_worker_rejects_changed_geometry(self):
        central = self.run_archive("central.npz")
        self.trajectory.write_text(self.trajectory.read_text().replace("0.70000000", "0.80000000"))
        with self.assertRaisesRegex(ValueError, "different geometry"):
            self.run_archive("batch.npz", "--llpr-checkpoint", "llpr", "--central-archive",
                             str(central), "--llpr-member-range", "0:8")

    def test_batch_resumes_its_own_checkpoint(self):
        central = self.run_archive("central.npz")
        options = ("--llpr-checkpoint", "llpr", "--central-archive", str(central),
                   "--llpr-member-range", "0:8")
        expected = read_archive(self.run_archive("expected.npz", *options))
        original = harmonic.evaluate_frame_hessian
        def fail_after_four(*args):
            if args[1] == 5:
                raise RuntimeError("interrupted")
            return original(*args)
        with patch.object(harmonic, "evaluate_frame_hessian", side_effect=fail_after_four):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_archive("restart.npz", *options)
        self.assertTrue((self.root / ".restart.frame-0.checkpoint.npz").exists())
        self.evaluated.clear()
        actual = read_archive(self.run_archive("restart.npz", *options))
        self.assertEqual(self.evaluated, [5, 6, 7, 8])
        for key in ("llpr_frequencies_cm1", "llpr_hessian_member_mean_eV_per_A2",
                    "llpr_hessian_member_m2_eV2_per_A4"):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)
        self.assertFalse((self.root / ".restart.frame-0.checkpoint.npz").exists())


if __name__ == "__main__":
    unittest.main()
