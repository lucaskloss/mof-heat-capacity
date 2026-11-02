import csv
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mof_heat_capacity.analysis.experimental import (
    experimental_note, load_experimental_curves, plot_experimental_curves,
)


class ExperimentalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "zif8").mkdir()
        self.path = self.root / "zif8/zif-8_co2.csv"

    def write_rows(self, header, rows):
        with self.path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def test_mass_specific_data_are_not_converted_twice_or_extrapolated(self):
        self.write_rows(
            ["temperature_K", "heat_capacity_J_per_gK"],
            [(100, 0.4), (200, 0.8), (300, 1.0)],
        )
        curves = load_experimental_curves(self.root, "zif8_2x2x2", "co2")
        np.testing.assert_array_equal(curves[0].heat_capacity, [0.4, 0.8, 1.0])
        figure, axis = plt.subplots()
        self.addCleanup(plt.close, figure)
        plot_experimental_curves(axis, curves, [175, 225, 325, 400])
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), [200, 300])
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), [0.8, 1.0])
        bars = axis.containers[0].lines[2][0].get_segments()
        np.testing.assert_allclose(bars[0][:, 1], [0.784, 0.816])
        record = curves[0].provenance()
        self.assertEqual(record["heat_capacity_unit"], "J/(g K)")
        self.assertEqual(record["relative_standard_uncertainty"], 0.02)
        self.assertEqual(len(record["sha256"]), 64)

    def test_references_are_specific_to_host_and_guest(self):
        self.assertEqual(load_experimental_curves(self.root, "mof303_2x2x2", "co2"), [])
        self.assertEqual(load_experimental_curves(self.root, "zif8_2x2x2", "h2o"), [])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(load_experimental_curves(self.root, "zif8", "co2"), [])
        self.assertIn("not found", str(caught[0].message))

    def test_ambiguous_units_and_malformed_grids_are_rejected(self):
        for header, rows in (
            (["x", "y"], [(200, 200)]),
            (["temperature_K", "heat_capacity_J_per_gK"], []),
            (["temperature_K", "heat_capacity_J_per_gK"], [(200, 0.8), (200, 0.9)]),
            (["temperature_K", "heat_capacity_J_per_gK"], [(300, 1.0), (200, 0.8)]),
            (["temperature_K", "heat_capacity_J_per_gK"], [(200, "nan")]),
        ):
            with self.subTest(header=header, rows=rows):
                self.write_rows(header, rows)
                with self.assertRaises(ValueError):
                    load_experimental_curves(self.root, "zif8", "co2")

    def test_simulation_loading_notes_use_actual_cell_mass(self):
        note = experimental_note("mof303_2x2x2", "h2o", 50, 13577.200464)
        self.assertIn("16.76%", note)
        larger = experimental_note("mof303_2x2x2", "h2o", 50, 2 * 12676.450464 + 50 * 18.015)
        self.assertIn("8.38%", larger)
        note = experimental_note("zif8_2x2x2", "co2", 54, 24223.782)
        self.assertIn("0.5625", note)
        self.assertIn("sealed-cell", note)

    def test_hybrid_assembly_records_sources_without_changing_computed_arrays(self):
        from mof_heat_capacity.analysis import hybrid

        self.write_rows(
            ["temperature_K", "heat_capacity_J_per_gK"], [(200, 0.8), (300, 1.0)]
        )
        with patch("sys.argv", [
            "hybrid", "--mof", "zif8_2x2x2", "--guest", "co2", "--loading", "54",
            "--hybrid-dir", str(self.root / "hybrid"), "--reference-root", str(self.root),
            "--bootstrap-samples", "100",
        ]):
            args = hybrid.parse_args()
        t = np.arange(175, 426, 25, dtype=float)
        records = (t, 0.02 * t, np.zeros(11), np.full(11, 10000.), np.zeros(11), 24223.782, [])
        outputs = []
        with patch.object(hybrid, "_enthalpy_records", return_value=records), \
                patch.object(hybrid, "_harmonic_corrections", return_value=(
                    np.zeros(9), np.zeros(9), None, None, []
                )), patch.object(hybrid, "plot_hybrid_heat_capacity") as plot, redirect_stdout(StringIO()):
            for disabled in (False, True):
                args.no_experimental_reference = disabled
                output = hybrid.run(args)
                with np.load(output, allow_pickle=False) as data:
                    outputs.append({key: data[key].copy() for key in data.files})
                expected_count = 0 if disabled else 1
                self.assertEqual(len(plot.call_args.kwargs["experimental_curves"]), expected_count)
                metadata = json.loads(output.with_suffix(".json").read_text())
                self.assertEqual(len(metadata["experimental_references"]), expected_count)
        self.assertEqual(outputs[0].keys(), outputs[1].keys())
        for key in outputs[0]:
            np.testing.assert_array_equal(outputs[0][key], outputs[1][key], err_msg=key)


if __name__ == "__main__":
    unittest.main()
