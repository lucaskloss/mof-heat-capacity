"""Render the actual guest insertion templates with Chemiscope (software WebGL)."""

import base64
import hashlib
import json
from pathlib import Path

import chemiscope
from ase.io import read
from chemiscope.headless import ChemiscopeHeadless


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
GUESTS = [('ch4', [1, 1.414, 0]), ('co2', [0, 0, 1]), ('h2o', [0, 0, 1])]


def main():
    structures, sources = [], []
    for name, direction in GUESTS:
        source = ROOT / 'input' / f'{name}.gro'
        atoms = read(source)
        atoms.set_pbc(False)
        atoms.set_cell([0, 0, 0])
        atoms.positions -= atoms.get_center_of_mass()
        structures.append(atoms)
        sources.append({
            'file': f'{name}.png',
            'source': str(source.relative_to(ROOT)),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'atoms': len(atoms),
            'formula': atoms.get_chemical_formula(),
            'view_direction_cartesian': direction,
        })
    settings = {
        'target': 'structure', 'pinned': [0],
        'structure': [{
            'unitCell': False, 'axes': 'off', 'rotation': False,
            'supercell': [1, 1, 1], 'keepOrientation': False,
            'environments': {'activated': False, 'center': False},
        }],
    }
    dataset = chemiscope.create_input(
        structures=structures,
        metadata={'name': 'Guest insertion templates',
                  'description': 'Isolated CH4, CO2, and H2O templates used by the project insertion routine.'},
        settings=settings,
    )
    (OUTPUT / 'guests.json').write_text(json.dumps(dataset, indent=2) + '\n')
    viewer = ChemiscopeHeadless(dataset, mode='structure', width=1100, height=1100)
    try:
        viewer._run_sync(viewer._page.add_style_tag(content=(
            '#chemiscope-meta,#chemiscope-map,#chemiscope-info{display:none!important}'
        )), timeout=60)
        for index, (name, direction) in enumerate(GUESTS):
            viewer._run_sync(viewer._page.evaluate(
                'async (index) => { await window.visualizer.structure.show({structure:index}); }', index,
            ), timeout=120)
            viewer._run_sync(viewer._page.evaluate(
                '(direction) => { const grid=window.visualizer.structure;'
                'const cell=grid._cellsData.get(grid.active);'
                'cell.viewer._viewAlong(direction); }', direction,
            ), timeout=60)
            uri = viewer._run_sync(viewer._page.evaluate(
                'async () => { await new Promise(r => requestAnimationFrame(r));'
                'return window.visualizer.structure.exportActivePNG(); }'
            ), timeout=120)
            (OUTPUT / f'{name}.png').write_bytes(base64.b64decode(uri.split(',', 1)[1]))
            print(f'Saved {name}.png', flush=True)
    finally:
        viewer.close()
    (OUTPUT / 'manifest.json').write_text(json.dumps({
        'chemiscope_version': chemiscope.__version__, 'settings': settings,
        'figures': sources,
        'description': 'Template geometry preserved; centered and rendered without periodic cells. Each image uses independent automatic zoom.',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
