# Empty framework figures

These images are exported by Chemiscope from the project dataset
`output/chemiscope/mof5-mof303-zif8-empty-and-loaded.json.gz`.
They use empty structures only:

| Image | Dataset index | Structure | Atoms |
| --- | --- | --- | --- |
| `mof5-empty.png` | 0 | MOF-5 | 424 |
| `mof303-empty.png` | 4 | MOF-303, 2 × 2 × 2 supercell | 1024 |
| `zif8-empty.png` | 8 | ZIF-8, 2 × 2 × 2 supercell | 2208 |

No coordinates, cells, or chemical identities are changed. The dataset
identifies these as starting configurations, rather than MD snapshots or
optimized harmonic references. No additional supercell replication is
applied by the viewer. Images use element colors, bonds, periodic-cell
outlines, and crystallographic axes, with independent automatic zoom. MOF-303 is viewed along Cartesian x;
MOF-5 and ZIF-8 along Cartesian z. The report trims white image margins
using LaTeX, leaving the original Chemiscope exports intact.

To regenerate, use Chemiscope 1.1.0 with Playwright and its Chromium browser
installed, then run from the repository root:

```bash
python docs/report/figures/frameworks/render.py
```

The renderer uses software WebGL and does not require a GPU allocation.
`manifest.json` records the input hash, selected structures, and settings.
The original source dataset remains unchanged. The Python renderer and
images are kept together so the report figures can be reproduced.
