"""Production classical-NPT execution backend."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from ..config import RunConfig
from ..io import (
    load_structure,
    write_classical_npt_lammps_input,
    write_lammps_data,
)
from ..models import ensure_exported_model


def _model_path(config: RunConfig) -> Path:
    if config.exported_model.is_file():
        return config.exported_model
    if config.checkpoint is None:
        raise FileNotFoundError(
            f"exported metatomic model not found: {config.exported_model}"
        )
    return ensure_exported_model(config.checkpoint, config.exported_model)


def _validate_npt_model(config: RunConfig) -> None:
    if not config.stress_validated:
        raise ValueError(
            "NPT requires model.stress_validated=true after force, energy, and "
            "virial/stress validation for the intended MOF/loading state"
        )


def _validate_stress_capability(model_path: Path) -> None:
    """Confirm that the exported metatomic model can provide cell stress."""
    import metatomic.torch as metatomic_torch

    model = metatomic_torch.load_atomistic_model(str(model_path))
    outputs = set(model.capabilities().outputs)
    if not outputs.intersection({"stress", "non_conservative_stress"}):
        raise ValueError(
            f"NPT requires an exported model with a stress output: {model_path}"
        )


def _executable(command: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        raise FileNotFoundError(f"required executable is unavailable: {command}")
    return executable


def _latest_lammps_restart(prefix: Path) -> Path | None:
    candidates = []
    for path in prefix.parent.glob(f"{prefix.name}.*"):
        suffix = path.name.removeprefix(f"{prefix.name}.")
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    return max(candidates, default=(0, None))[1]


def run_classical_npt(
    config: RunConfig,
    *,
    steps: int | None = None,
    output_dir: Path | None = None,
    prefix: str | None = None,
    rerun: bool = False,
    resume: bool = False,
) -> Path:
    """Run fully flexible 1-bar NPT with LAMMPS MTTK/NHC dynamics."""
    _validate_npt_model(config)
    active_steps = config.md_steps if steps is None else steps
    active_output = config.output_dir if output_dir is None else output_dir.resolve()
    active_prefix = config.md_prefix if prefix is None else Path(prefix).name
    if active_steps < 1:
        raise ValueError("steps must be positive")
    active_output.mkdir(parents=True, exist_ok=True)

    trajectory_path = active_output / config.md_trajectory_file
    if trajectory_path.exists() and not rerun and not resume:
        print(f"Reusing existing trajectory: {trajectory_path}")
        return trajectory_path
    if rerun and resume:
        raise ValueError("--rerun and --resume are mutually exclusive")

    atoms = load_structure(config.structure, required_elements=config.required_elements)
    data_path = active_output / f"{active_prefix}.initial.data"
    input_path = active_output / f"{active_prefix}.lammps.in"
    thermo_path = active_output / f"{active_prefix}.lammps.log"
    restart_prefix = active_output / f"{active_prefix}.restart"
    final_data_path = active_output / f"{active_prefix}.final.data"
    restart_path = _latest_lammps_restart(restart_prefix) if resume else None
    if resume and restart_path is None:
        raise FileNotFoundError(f"no numeric restart found for {restart_prefix}")
    if restart_path is None:
        write_lammps_data(data_path, atoms)

    model_path = _model_path(config)
    _validate_stress_capability(model_path)
    write_classical_npt_lammps_input(
        input_path,
        data_path=data_path,
        model_path=model_path,
        device=config.device,
        trajectory_path=trajectory_path,
        thermo_path=thermo_path,
        restart_prefix=restart_prefix,
        final_data_path=final_data_path,
        temperature_K=config.temperature_K,
        pressure_bar=config.pressure_bar,
        timestep_fs=config.timestep_fs,
        thermostat_tau_fs=config.thermostat_tau_fs,
        thermostat_chain_length=config.thermostat_chain_length,
        barostat_tau_fs=config.barostat_tau_fs,
        steps=active_steps,
        output_stride=config.output_stride,
        restart_stride=config.restart_stride,
        seed=config.random_seed,
        restart_path=restart_path,
    )
    print(
        f"Run {config.name}: LAMMPS flexible NPT, {config.temperature_K:g} K, "
        f"{config.pressure_bar:g} bar, {active_steps} steps"
    )
    subprocess.run(
        [_executable(config.lammps_command), "-in", str(input_path)],
        cwd=active_output,
        check=True,
    )
    return trajectory_path
