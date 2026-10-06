"""Rigid guest packing under exact triclinic periodic separation constraints."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import unittest

from ase import Atoms, io
from ase.geometry import get_distances
import numpy as np

from mof_heat_capacity.structures.methane import _PeriodicNeighbors, insert_molecules


PROJECT = Path(__file__).resolve().parents[1]


class MoleculeInsertionTest(unittest.TestCase):
    def test_neighbor_index_matches_ase_for_skew_cells_and_periodic_images(self):
        random = np.random.default_rng(81)
        cells = [io.read(PROJECT / "input" / f"{mof}.cif").cell
                 for mof in ("mgmof74", "mof303")]
        cells.append(np.array([[12, 0, 0], [35, 10, 0], [19, 21, 8]]))
        for cell in cells:
            with self.subTest(cell=str(cell)):
                positions = (random.random((30, 3)) * 4 - 2) @ cell
                candidates = (random.random((100, 3)) * 6 - 3) @ cell
                host = Atoms("He" * len(positions), positions=positions, cell=cell, pbc=True)
                neighbors = _PeriodicNeighbors(host)
                neighbors.update(positions)
                actual = neighbors.tree.query(neighbors.wrap(candidates))[0]
                expected = get_distances(candidates, positions, cell=cell, pbc=True)[1].min(axis=1)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_mof303_methane_recovers_the_blocked_175K_seed(self):
        host = io.read(PROJECT / "input/mof303.cif")
        molecule = io.read(PROJECT / "input/ch4.gro")
        seed = 20267501
        with self.assertRaisesRegex(RuntimeError, "49/50"):
            insert_molecules(host, molecule, 50, 20000, 1.5, seed, method="random")
        with redirect_stdout(StringIO()):
            atoms = insert_molecules(host, molecule, 50, 20000, 1.5, seed)
        self.assertEqual(len(atoms), len(host) + 250)
        self.assertEqual(atoms.info["insertion_method"], "repacking")
        self.assertGreater(atoms.info["packing_rounds"], 0)
        np.testing.assert_array_equal(atoms.positions[:len(host)], host.positions)
        np.testing.assert_array_equal(atoms.cell.array, host.cell.array)
        reference = molecule.get_all_distances()
        for offset in range(len(host), len(atoms), 5):
            np.testing.assert_allclose(atoms[offset:offset + 5].get_all_distances(), reference, atol=1e-12)
            distances = get_distances(atoms.positions[offset:offset + 5], atoms.positions[:offset],
                                      cell=atoms.cell, pbc=True)[1]
            self.assertGreaterEqual(distances.min(), 1.5)

    def test_seed_is_reproducible_and_failure_keeps_the_host(self):
        host = Atoms("He", positions=[[4, 4, 4]], cell=[15, 15, 15], pbc=True)
        guest = io.read(PROJECT / "input/h2o.gro")
        first = insert_molecules(host, guest, 5, 100, 1.5, 26, method="repacking")
        second = insert_molecules(host, guest, 5, 100, 1.5, 26, method="repacking")
        np.testing.assert_array_equal(first.positions, second.positions)
        small = Atoms("He", positions=[[0, 0, 0]], cell=[1, 1, 1], pbc=True)
        with redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, "0/1"):
            insert_molecules(small, Atoms("He"), 1, 3, 2, 26, packing_restarts=2)
        self.assertEqual(len(small), 1)
        np.testing.assert_array_equal(small.positions, [[0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
