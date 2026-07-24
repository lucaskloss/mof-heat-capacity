# MOF-5 heat-capacity project

This repository contains a small PET-MAD molecular-dynamics workflow for
periodic MOF-5. The current code is an integration and stability smoke test;
automatic-differentiation Hessians, phonon eigenvalues, and heat-capacity
analysis are planned but are not implemented yet.

## Layout

- `run.py` is the ASE/metatomic command-line entry point.
- `ase_md.py` contains the socket-free ASE/metatomic MD implementation.
- `workflow_io.py` validates periodic structures.
- `model_utils.py` exports the bundled PET-MAD checkpoint when needed.
- `data/` contains the input crystal structure.
- `output/` contains generated properties, trajectories, logs, and restarts.
- `models/` contains supplied model checkpoints and exported model files.

The active workflow uses `metatomic-ase` and does not require LAMMPS, i-PI, or
the Unix-socket force-client path.

The default run uses `data/mof5.cif`, CUDA model evaluation, one bead, and a
short NVT smoke test. Run `python run.py` from this directory.

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
