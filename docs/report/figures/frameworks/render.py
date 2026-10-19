"""Export the project's three empty structures using Chemiscope's renderer."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import chemiscope
from chemiscope.headless import ChemiscopeHeadless


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = ROOT / "output/chemiscope/mof5-mof303-zif8-empty-and-loaded.json.gz"
SELECTION = [(0, "mof5-empty"), (4, "mof303-empty"), (8, "zif8-empty")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    raw = args.input.read_bytes()
    dataset = json.loads(gzip.decompress(raw))
    settings = {
        "target": "structure",
        "pinned": [0],
        "structure": [{
            "unitCell": True,
            "axes": "abc",
            "rotation": False,
            "supercell": [1, 1, 1],
            "keepOrientation": False,
            "environments": {"activated": False, "center": False},
        }],
    }
    dataset["settings"] = settings
    args.output.mkdir(parents=True, exist_ok=True)
    viewer = ChemiscopeHeadless(dataset, mode="structure", width=1100, height=1100)
    manifest = {
        "source": str(args.input.resolve().relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "chemiscope_version": chemiscope.__version__,
        "description": dataset["meta"]["description"],
        "settings": settings,
        "figures": [],
    }

    try:
        # Hide unused panels so the structure canvas occupies the viewport.
        viewer._run_sync(viewer._page.add_style_tag(content=(
            "#chemiscope-meta,#chemiscope-map,#chemiscope-info"
            "{display:none!important}"
        )), timeout=60)
        for index, name in SELECTION:
            assert dataset["properties"]["Guest"]["values"][index] == "None"
            viewer._run_sync(viewer._page.evaluate(
                "async (index) => { await window.visualizer.structure.show({structure:index}); }",
                index,
            ), timeout=120)
            direction = [1, 0, 0] if index == 4 else [0, 0, 1]
            viewer._run_sync(viewer._page.evaluate(
                "(direction) => { const grid=window.visualizer.structure;"
                "const cell=grid._cellsData.get(grid.active);"
                "cell.viewer._viewAlong(direction); }", direction,
            ), timeout=60)
            # Software WebGL exports can exceed the widget's 10-second default.
            uri = viewer._run_sync(viewer._page.evaluate(
                "async () => {"
                "await new Promise(r => requestAnimationFrame(r));"
                "return window.visualizer.structure.exportActivePNG(); }"
            ), timeout=120)
            output = args.output / f"{name}.png"
            output.write_bytes(base64.b64decode(uri.split(",", 1)[1]))
            manifest["figures"].append({
                "file": output.name,
                "structure_index": index,
                "structure": dataset["properties"]["Structure"]["values"][index],
                "atoms": dataset["structures"][index]["size"],
                "view_direction_cartesian": direction,
            })
            print(f"Saved {output.name}", flush=True)
    finally:
        viewer.close()

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
