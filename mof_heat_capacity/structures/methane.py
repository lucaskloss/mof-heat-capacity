"""Insert CH4, CO2, or H2O molecules into a periodic MOF structure with ASE."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from ase import Atoms, io
from ase.geometry import get_distances, minkowski_reduce
from scipy.spatial import cKDTree

from ..campaign import GUEST_SYMBOLS, host_path, structure_directory, validate_guest, validate_selection
from ..config import output_root
from ..io import write_lammps_data, write_structure_pdb


PROJECT_DIR = Path(__file__).resolve().parents[2]
INSERTION_METHODS = ("auto", "random", "repacking")


def parse_args() -> argparse.Namespace:
    """Parse structure and insertion options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mof", default="mof5")
    parser.add_argument("--guest", type=str.lower, choices=tuple(GUEST_SYMBOLS), default="ch4")
    parser.add_argument("--host", type=Path,
                        help="Periodic host structure (PDB, CIF, or GRO).")
    parser.add_argument("--molecule", type=Path,
                        help="Single guest molecule structure (GRO or PDB).")
    parser.add_argument("--output", type=Path,
                        help="Combined PDB output structure (default: md/structures/[<mof>/]<N><guest>/manual/seed<seed>).")
    parser.add_argument("--data-output", type=Path,
                        help="LAMMPS data output (default: md/structures/[<mof>/]<N><guest>/manual/seed<seed>).")
    parser.add_argument("--nmol", type=int, default=1,
                        help="Number of guest molecules to insert (default: 1).")
    parser.add_argument("--try", dest="tries", type=int, default=1000,
                        help="Placement attempts per molecule (default: 1000).")
    parser.add_argument("--min-distance", type=float, default=1.5,
                        help="Minimum periodic atom distance in Angstrom (default: 1.5).")
    parser.add_argument("--insertion-method", choices=INSERTION_METHODS, default="auto",
                        help="auto retries blocked random insertion with repacking (default: auto).")
    parser.add_argument("--packing-restarts", type=int, default=20,
                        help="Maximum rearrangement/restart rounds (default: 20).")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed (default: 2025).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and prepare without writing the output.")
    return parser.parse_args()


def random_rotation(random: np.random.Generator) -> np.ndarray:
    """Return a uniformly distributed three-dimensional rotation matrix."""
    quaternion = random.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def prepare_molecule(molecule: Atoms, rotation: np.ndarray) -> Atoms:
    """Center and randomly orient one molecule without changing its geometry."""
    prepared = molecule.copy()
    prepared.set_pbc(False)
    prepared.positions -= prepared.get_center_of_mass()
    prepared.positions = prepared.positions @ rotation.T
    return prepared


def has_no_overlap(candidate: Atoms, existing: Atoms, minimum: float) -> bool:
    """Check candidate distances against an existing periodic structure."""
    distances = get_distances(candidate.positions, existing.positions,
                              cell=existing.cell, pbc=existing.pbc)[1]
    return bool(np.min(distances) >= minimum)


class _PeriodicNeighbors:
    """Nearest atom distances using images of a Minkowski-reduced cell."""

    def __init__(self, host: Atoms):
        self.cell, _ = minkowski_reduce(host.cell, pbc=host.pbc)
        self.inverse = np.linalg.inv(self.cell)
        self.shifts = np.array(list(product((-1, 0, 1), repeat=3))) @ self.cell
        self.tree = None

    def wrap(self, positions: np.ndarray) -> np.ndarray:
        return ((positions @ self.inverse) % 1.0) @ self.cell

    def update(self, positions: np.ndarray) -> None:
        wrapped = self.wrap(positions)
        images = wrapped[None, :, :] + self.shifts[:, None, :]
        self.tree = cKDTree(images.reshape(-1, 3))

    def accepts(self, positions: np.ndarray, minimum: float) -> bool:
        distances, _ = self.tree.query(self.wrap(positions), k=1)
        return bool(np.all(distances >= minimum))


def insert_molecules(host: Atoms, molecule: Atoms, count: int, tries: int,
                     minimum: float, seed: int, *, method: str = "auto",
                     packing_restarts: int = 20) -> Atoms:
    """Pack rigid guests, revisiting earlier placements when insertion jams.

    Auto first tries the original sequential random algorithm. Repacking can
    remove and reinsert a random quarter of the guests, with a complete restart
    every fourth round. Neither the host nor the separation rule is changed.
    """
    if method not in INSERTION_METHODS:
        raise ValueError(f"unknown insertion method: {method}")
    if count < 0 or tries < 1 or minimum <= 0 or packing_restarts < 0:
        raise ValueError("invalid molecule count, placement attempts, distance, or restarts")
    if host.cell.volume <= 0 or not all(host.pbc):
        raise ValueError("insertion requires a nonzero, fully periodic host cell")
    random = np.random.default_rng(seed)
    neighbors = _PeriodicNeighbors(host)
    placements = []
    best_count = 0
    rounds = 0 if method == "random" else packing_restarts
    for restart in range(rounds + 1):
        positions = np.concatenate([host.positions, *placements], axis=0)
        neighbors.update(positions)
        while len(placements) < count:
            for _ in range(tries):
                candidate = prepare_molecule(molecule, random_rotation(random))
                candidate.positions += host.cell.cartesian_positions(random.random(3))
                if neighbors.accepts(candidate.positions, minimum):
                    placements.append(candidate.positions.copy())
                    neighbors.update(np.concatenate([host.positions, *placements], axis=0))
                    break
            else:
                break
        best_count = max(best_count, len(placements))
        if len(placements) == count:
            result = host.copy()
            for positions in placements:
                guest = molecule.copy()
                guest.positions = positions
                # Independently confirm the exact ASE minimum-image distances.
                if not has_no_overlap(guest, result, minimum):
                    raise RuntimeError("periodic separation validation failed; no structure was returned")
                result += guest
            result.info["insertion_method"] = "repacking" if restart or method == "repacking" else "random"
            result.info["packing_rounds"] = restart
            return result
        if restart == rounds:
            break
        print(f"Insertion blocked at {len(placements)}/{count} molecules; "
              f"repacking round {restart + 1}/{rounds} at {minimum:g} A separation", flush=True)
        if (restart + 1) % 4 == 0:
            placements = []
        else:
            remove_count = max(1, len(placements) // 4)
            if placements:
                removed = set(random.choice(len(placements), remove_count, replace=False))
                placements = [p for i, p in enumerate(placements) if i not in removed]
    raise RuntimeError(
        f"could only place {best_count}/{count} guest molecules at {minimum:g} A separation "
        f"with {tries} attempts per molecule and {rounds} repacking rounds; "
        "no partial structure was returned. Try more --packing-restarts or --try; "
        "the requested loading may not fit this unit cell at the specified separation."
    )


def main() -> None:
    """Read structures, insert guest molecules, and write the combined structure."""
    args = parse_args()
    validate_selection(args.mof, args.guest)
    args.host = host_path(args.mof, args.host)
    args.molecule = args.molecule or PROJECT_DIR / "input" / f"{args.guest}.gro"
    directory = output_root() / "md" / "structures" / structure_directory(args.nmol, args.mof, args.guest) / "manual" / f"seed{args.seed}"
    args.output = args.output or directory / "structure.pdb"
    args.data_output = args.data_output or directory / "structure.data"
    if not args.host.is_file() or not args.molecule.is_file():
        raise FileNotFoundError("host and molecule structure files must exist")
    if args.nmol < 1 or args.tries < 1 or args.min_distance <= 0.0:
        raise ValueError("--nmol, --try, and --min-distance must be positive")
    input_paths = {args.host.resolve(), args.molecule.resolve()}
    if args.output.resolve() in input_paths or args.data_output.resolve() in input_paths:
        raise ValueError("output files must differ from both input structures")

    host = io.read(args.host)
    molecule = io.read(args.molecule)
    if host.cell.volume <= 0.0 or not all(host.pbc):
        raise ValueError("host structure must have a non-zero periodic cell")
    validate_guest(molecule, args.guest)

    combined = insert_molecules(host, molecule, args.nmol, args.tries,
                                args.min_distance, args.seed, method=args.insertion_method,
                                packing_restarts=args.packing_restarts)
    print(f"Prepared {len(combined)} atoms ({args.nmol} {args.guest.upper()} molecules) in "
          f"{combined.cell.volume:.3f} A^3")
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_structure_pdb(args.output, combined)
        write_lammps_data(args.data_output, combined)
        print(f"Wrote host-plus-{args.guest.upper()} PDB structure: {args.output}")
        print(f"Wrote host-plus-{args.guest.upper()} LAMMPS data: {args.data_output}")


if __name__ == "__main__":
    main()
