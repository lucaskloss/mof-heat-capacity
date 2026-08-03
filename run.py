"""Run configuration-driven ASE molecular dynamics with a metatomic MLIP."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ase import units
from ase.io import Trajectory
from ase.md import Langevin, MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from utils.model_utils import ensure_exported_model
from utils.workflow_config import RunConfig, load_run_config
from utils.workflow_io import load_structure


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "mof5_pet_mad.toml"


def parse_args() -> argparse.Namespace:
    """Parse a run specification and lightweight MD overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="Override model.device, e.g. cuda or cpu")
    parser.add_argument("--steps", type=int, help="Override md.steps")
    parser.add_argument("--output-dir", type=Path, help="Override run.output_dir")
    parser.add_argument("--prefix", help="Override md.prefix")
    parser.add_argument(
        "--rerun", action="store_true", help="Overwrite an existing trajectory"
    )
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def run_md(
    config: RunConfig,
    *,
    device: str | None = None,
    steps: int | None = None,
    output_dir: Path | None = None,
    prefix: str | None = None,
    rerun: bool = False,
) -> Path:
    """Run one NVT trajectory described by ``config`` and return its path."""
    from metatomic_ase import MetatomicCalculator
    import metatomic_ase._neighbors as metatomic_neighbors

    active_steps = config.md_steps if steps is None else steps
    active_device = config.device if device is None else device
    active_output = (
        config.output_dir if output_dir is None else output_dir.expanduser().resolve()
    )
    active_prefix = config.md_prefix if prefix is None else Path(prefix).name
    if active_steps < 1:
        raise ValueError("steps must be positive")

    active_output.mkdir(parents=True, exist_ok=True)
    trajectory_path = active_output / f"{active_prefix}.traj"
    log_path = active_output / f"{active_prefix}.log"
    if trajectory_path.exists() and not rerun:
        print(f"Reusing existing trajectory: {trajectory_path}")
        return trajectory_path

    atoms = load_structure(config.structure, required_elements=config.required_elements)
    model_path = _prepare_metatomic_model(config)
    _patch_nvalchemi_max_neighbors(metatomic_neighbors)
    # Prefer the installed nvalchemi-toolkit-ops GPU neighbor backend when it
    # is available. It avoids Vesin's NVRTC JIT path, which is not accepted by
    # Izar's V100/CUDA runtime. Set MOF_USE_NVALCHEMIOPS=0 to force Vesin for
    # diagnostics on a different CUDA stack.
    if os.environ.get("MOF_USE_NVALCHEMIOPS", "1").lower() in {"0", "false", "no"}:
        metatomic_neighbors.HAS_NVALCHEMIOPS = False
    atoms.calc = MetatomicCalculator(
        model_path, device=active_device, check_consistency=True
    )

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(
        f"Run {config.name}: {len(atoms)} atoms; energy={energy:.6f} eV; "
        f"max force={abs(forces).max():.6f} eV/A"
    )
    MaxwellBoltzmannDistribution(atoms, temperature_K=config.temperature_K)
    Stationary(atoms)
    dynamics = Langevin(
        atoms,
        timestep=config.timestep_fs * units.fs,
        temperature_K=config.temperature_K,
        friction=0.01 / units.fs,
    )
    trajectory = Trajectory(trajectory_path, "w", atoms)
    dynamics.attach(trajectory.write, interval=1)
    dynamics.attach(
        MDLogger(dynamics, atoms, log_path, header=True, stress=False, peratom=False),
        interval=1,
    )
    dynamics.run(active_steps)
    trajectory.close()
    print(f"Completed ASE MD. Trajectory: {trajectory_path}")
    return trajectory_path


def _patch_nvalchemi_max_neighbors(metatomic_neighbors) -> None:
    """Normalize nvalchemi's neighbor capacity for older Torch releases.

    metatomic-ase computes ``max_neighbors`` from a floating-point cutoff.
    nvalchemi 0.3/0.4 passes that value to ``torch.full`` as a tensor shape,
    while Torch 2.5 requires integer dimensions. Keep the backend enabled and
    normalize only this argument at the integration boundary.
    """
    if not metatomic_neighbors.HAS_NVALCHEMIOPS:
        return
    if getattr(metatomic_neighbors.nvalchemi_neighbor_list, "_mof5_patched", False):
        return

    original = metatomic_neighbors.nvalchemi_neighbor_list

    def neighbor_list_with_integer_capacity(*args, **kwargs):
        if "max_neighbors" in kwargs:
            kwargs["max_neighbors"] = int(kwargs["max_neighbors"])
        return original(*args, **kwargs)

    neighbor_list_with_integer_capacity._mof5_patched = True
    metatomic_neighbors.nvalchemi_neighbor_list = neighbor_list_with_integer_capacity


def _prepare_metatomic_model(config: RunConfig) -> Path:
    """Return an exported model, exporting a PET-MAD checkpoint only when needed."""
    if config.exported_model.is_file():
        return config.exported_model
    if config.checkpoint is None:
        raise FileNotFoundError(
            f"exported metatomic model not found: {config.exported_model}; "
            "model.checkpoint is required to create it"
        )
    return ensure_exported_model(config.checkpoint, config.exported_model)


def main() -> None:
    """Load a TOML specification and run its MD stage."""
    args = parse_args()
    config = load_run_config(args.config)
    if args.print_config:
        print(config)
        return
    run_md(
        config,
        device=args.device,
        steps=args.steps,
        output_dir=args.output_dir,
        prefix=args.prefix,
        rerun=args.rerun,
    )


if __name__ == "__main__":
    main()
