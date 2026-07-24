# MOF-5 heat-capacity project

This repository contains a small PET-MAD molecular-dynamics workflow for
periodic MOF-5. The current code is an integration and stability smoke test;
automatic-differentiation Hessians, phonon eigenvalues, and heat-capacity
analysis are planned but are not implemented yet.

## Layout

- `mof-md.py` is the command-line entry point.
- `workflow_io.py` validates structures and writes the four input files.
- `lammps_runner.py` launches i-PI and metatomic-enabled LAMMPS.
- `model_utils.py` exports the bundled PET-MAD checkpoint when needed.
- `data/` contains the structure and fixed i-PI/LAMMPS inputs.
- `output/` contains generated properties, trajectories, logs, and restarts.
- `models/` contains supplied model checkpoints and exported model files.

Use the metatensor `lammps-metatomic` package. Do not substitute ordinary
LAMMPS: the workflow requires both `pair_style metatomic` and `fix ipi`.

The default run uses `data/mof5.cif`, CUDA model evaluation, one bead, and a
short NVT smoke test. Run `python mof-md.py` from this directory. If the
structure or simulation settings change, regenerate the four files in `data/`
with `python mof-md.py --prepare-inputs --rerun`.

Keep the generated output out of source control. Do not add exported weights,
trajectories, restart files, logs, or caches unless explicitly requested.

## Code organization example

Keep each logical phase in its own visibly separated block. Prefer this style:

```python
"""Demonstrate the preferred organization of a small Python program."""

from pathlib import Path


DEFAULT_LIMIT = 10
DEFAULT_OUTPUT = Path("results.txt")


def transform(value: int) -> int:
    """Transform one value."""
    return value * 2


def process_values(values: list[int]) -> list[int]:
    """Validate and transform a sequence of values."""
    results = []

    for value in values:
        if value < 0:
            continue

        results.append(transform(value))

    return results


def write_results(path: Path, values: list[int]) -> None:
    """Write results, preserving a useful error message for callers."""
    try:
        path.write_text("\n".join(str(value) for value in values) + "\n")

    except OSError as error:
        raise RuntimeError(f"Could not write results to {path}") from error


def main() -> None:
    """Run the complete workflow."""
    values = list(range(-2, DEFAULT_LIMIT))

    try:
        results = process_values(values)

        if not results:
            raise ValueError("No valid values were provided")

        write_results(DEFAULT_OUTPUT, results)

    except (OSError, RuntimeError, ValueError) as error:
        print(f"Workflow failed: {error}")
        return

    print(f"Wrote {len(results)} results to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
```

Use blank lines between imports, constants, functions, conditionals, loops,
validation, file preparation, and runtime execution. Keep functions focused on
one responsibility and use descriptive names for paths and simulation settings.
