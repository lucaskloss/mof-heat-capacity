# Guest-loaded framework figures

These Chemiscope figures show the same host cells and Cartesian viewing
directions as `../frameworks/`, with guests inside the pores:

| Image | Host atoms | Guest molecules | Total atoms | View |
| --- | --- | --- | --- | --- |
| `mof5-50ch4.png` | 424 | 50 CH4 | 674 | z |
| `mof303-50h2o.png` | 1024 | 50 H2O | 1174 | x |
| `zif8-54co2.png` | 2208 | 54 CO2 | 2370 | z |

MOF-303 and ZIF-8 use the same 2 × 2 × 2 supercells as the empty figures.
MOF-5/CH4 and MOF-303/H2O retain the insertion configurations at indices 1
and 7 of `output/chemiscope/mof5-mof303-zif8-empty-and-loaded.json.gz`.
For ZIF-8, the renderer inserts exactly 54 CO2 molecules into the empty
host at index 8 using `input/co2.gro` and the project's insertion routine
(seed 2025, minimum periodic atom separation 1.5 Å, automatic repacking).
Host coordinates and cells are preserved and guest compositions are checked.
All three are starting configurations before relaxation, not MD snapshots.

To regenerate with Chemiscope 1.1.0, ASE, NumPy, SciPy, Playwright, and its
Chromium browser installed, run from the repository root:

```bash
python docs/report/figures/loaded/render.py
```

The renderer uses software WebGL, element colors, bonds, cell outlines,
crystallographic axes, and independent automatic zoom, matching the empty
figures. The report trims white margins in LaTeX. `manifest.json` records
source hashes, insertion parameters, atom counts, and viewing directions.
`loaded.json.gz` preserves the three complete rendered structures and can
also be opened in Chemiscope for interactive inspection.
