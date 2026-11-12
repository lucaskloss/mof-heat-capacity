# Guest molecule figures

Chemiscope renders the actual insertion templates `input/ch4.gro`,
`input/co2.gro`, and `input/h2o.gro` with the same element colors as the
framework figures. Each molecule is centered without changing its internal
geometry, and displayed without a periodic cell or axes. Views use independent
automatic zoom, so the panels are not a common length scale.

Regenerate from the repository root with Chemiscope 1.1.0, Playwright, and
Playwright's Chromium headless shell installed:

```bash
python docs/report/figures/guests/render.py
```

If dependencies or browsers are installed in temporary directories, set
`PYTHONPATH` and `PLAYWRIGHT_BROWSERS_PATH` accordingly. Rendering uses software
WebGL and does not require a GPU allocation. `guests.json` is the viewable
Chemiscope dataset; `manifest.json` records source hashes and view settings.
