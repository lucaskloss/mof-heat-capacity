import unittest
from types import SimpleNamespace

from mof_heat_capacity.models import configure_metatomic_neighbors


class ModelNeighborCapacityTests(unittest.TestCase):
    def test_metatomic_atom_count_factor_is_removed_from_neighbor_capacity(self):
        calls = []

        def neighbor_list(*args, **kwargs):
            calls.append(kwargs["max_neighbors"])
            return "neighbors"

        backend = SimpleNamespace(
            HAS_NVALCHEMIOPS=True,
            nvalchemi_neighbor_list=neighbor_list,
        )
        configure_metatomic_neighbors(backend)

        positions = [None] * 2370
        result = backend.nvalchemi_neighbor_list(
            positions, 7.65, max_neighbors=len(positions) * 448
        )

        self.assertEqual(result, "neighbors")
        self.assertEqual(calls, [448])

    def test_small_capacity_is_only_cast_to_integer(self):
        calls = []

        def neighbor_list(*args, **kwargs):
            calls.append(kwargs["max_neighbors"])
            return None

        backend = SimpleNamespace(
            HAS_NVALCHEMIOPS=True,
            nvalchemi_neighbor_list=neighbor_list,
        )
        configure_metatomic_neighbors(backend)
        backend.nvalchemi_neighbor_list([None] * 10, 4.0, max_neighbors=80.9)

        self.assertEqual(calls, [80])


if __name__ == "__main__":
    unittest.main()
