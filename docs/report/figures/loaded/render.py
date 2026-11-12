"""Render guest-loaded counterparts of the report's empty framework figures."""

import argparse
import base64
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import chemiscope
import numpy as np
from ase import Atoms
from ase.io import read
from chemiscope.headless import ChemiscopeHeadless


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from mof_heat_capacity.structures.methane import insert_molecules


def atoms_from_structure(structure):
    return Atoms(
        symbols=structure["names"],
        positions=np.array([structure[axis] for axis in "xyz"]).T,
        cell=np.array(structure["cell"]).reshape(3, 3),
        pbc=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=(
        ROOT / "output/chemiscope/mof5-mof303-zif8-empty-and-loaded.json.gz"
    ))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    source = json.loads(gzip.decompress(raw))
    selections = [
        (0, 1, "mof5-50ch4", "MOF-5 + 50 CH4", "CH4", 50, [0, 0, 1]),
        (4, 7, "mof303-50h2o", "MOF-303 2x2x2 + 50 H2O", "H2O", 50, [1, 0, 0]),
        (8, None, "zif8-54co2", "ZIF-8 2x2x2 + 54 CO2", "CO2", 54, [0, 0, 1]),
    ]
    structures, figures = [], []
    template = ROOT / "input/co2.gro"
    for host_index, loaded_index, name, label, guest, count, direction in selections:
        host = atoms_from_structure(source["structures"][host_index])
        if loaded_index is None:
            atoms = insert_molecules(host, read(template), count, tries=1000,
                                     minimum=1.5, seed=2025)
        else:
            assert source["properties"]["Guest"]["values"][loaded_index] == guest
            assert source["properties"]["Loading (molecules)"]["values"][loaded_index] == count
            atoms = atoms_from_structure(source["structures"][loaded_index])
        # Validate the complete guest composition and unchanged host coordinates.
        expected = {"CH4": Counter(C=count, H=4 * count),
                    "H2O": Counter(H=2 * count, O=count),
                    "CO2": Counter(C=count, O=2 * count)}[guest]
        assert Counter(atoms[len(host):].get_chemical_symbols()) == expected
        assert atoms[:len(host)].get_chemical_symbols() == host.get_chemical_symbols()
        np.testing.assert_allclose(atoms.positions[:len(host)], host.positions)
        np.testing.assert_allclose(atoms.cell, host.cell)
        structures.append(atoms)
        figures.append({
            "file": f"{name}.png", "structure": label, "guest": guest,
            "molecules": count, "host_atoms": len(host), "atoms": len(atoms),
            "source_host_index": host_index, "source_loaded_index": loaded_index,
            "view_direction_cartesian": direction,
        })
    settings = {
        "target": "structure", "pinned": [0],
        "structure": [{
            "unitCell": True, "axes": "abc", "rotation": False,
            "supercell": [1, 1, 1], "keepOrientation": False,
            "environments": {"activated": False, "center": False},
        }],
    }
    dataset = chemiscope.create_input(
        structures=structures,
        metadata={"name": "Guest-loaded report frameworks", "description": (
            "Starting configurations before relaxation: MOF-5 + 50 CH4, "
            "MOF-303 2x2x2 + 50 H2O, and ZIF-8 2x2x2 + 54 CO2. "
            "Guests inserted with the project 1.5 angstrom minimum-distance rule."
        )},
        properties={"Structure": {"target": "structure", "values": [
            figure["structure"] for figure in figures
        ]}},
        settings=settings,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "loaded.json.gz").write_bytes(
        gzip.compress(json.dumps(dataset).encode(), mtime=0)
    )
    viewer = ChemiscopeHeadless(dataset, mode="structure", width=1100, height=1100)
    try:
        viewer._run_sync(viewer._page.add_style_tag(content=(
            "#chemiscope-meta,#chemiscope-map,#chemiscope-info{display:none!important}"
        )), timeout=60)
        for index, figure in enumerate(figures):
            viewer._run_sync(viewer._page.evaluate(
                "async (index) => { await window.visualizer.structure.show({structure:index}); }",
                index,
            ), timeout=120)
            viewer._run_sync(viewer._page.evaluate(
                "(direction) => { const grid=window.visualizer.structure;"
                "const cell=grid._cellsData.get(grid.active);"
                "cell.viewer._viewAlong(direction); }", figure["view_direction_cartesian"],
            ), timeout=60)
            uri = viewer._run_sync(viewer._page.evaluate(
                "async () => { await new Promise(r => requestAnimationFrame(r));"
                "return window.visualizer.structure.exportActivePNG(); }"
            ), timeout=120)
            (args.output / figure["file"]).write_bytes(base64.b64decode(uri.split(",", 1)[1]))
            print(f"Saved {figure['file']} ({figure['atoms']} atoms)", flush=True)
    finally:
        viewer.close()
    (args.output / "manifest.json").write_text(json.dumps({
        "source": str(args.input.resolve().relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "chemiscope_version": chemiscope.__version__, "settings": settings,
        "zif8_insertion": {"template": "input/co2.gro",
                           "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
                           "seed": 2025, "minimum_distance_angstrom": 1.5,
                           "tries": 1000, "method": "auto", "packing_restarts": 20},
        "figures": figures,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
