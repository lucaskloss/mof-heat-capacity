"""TOML run specifications for reusable MD and harmonic-Cv workflows."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tomllib


PROJECT_DIR = Path(__file__).resolve().parents[1]
LOADED_RUN_PATTERN = re.compile(
    r"^mof5-(?P<loading>[1-9][0-9]*)ch4-(?P<model>.+)-npt-"
    r"(?P<temperature>[1-9][0-9]*)K-rep(?P<replica>[0-9]+)$"
)


def output_root() -> Path:
    """Return the configured output root, defaulting to the repository output tree."""
    configured = os.environ.get("MOF_OUTPUT_ROOT")
    if not configured:
        return PROJECT_DIR / "output"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


@dataclass(frozen=True)
class RunConfig:
    """Resolved paths and settings for one structure/model workflow."""

    name: str
    structure: Path
    required_elements: frozenset[str] | None
    md_backend: str
    ad_backend: str
    checkpoint: Path | None
    exported_model: Path
    jax_checkpoint: Path | None
    device: str
    stress_validated: bool
    md_driver: str
    md_ensemble: str
    temperature_K: float
    md_steps: int
    timestep_fs: float
    md_prefix: str
    md_trajectory_file: str
    equilibration_steps: int
    output_stride: int
    restart_stride: int
    random_seed: int
    thermostat: str
    thermostat_tau_fs: float
    thermostat_chain_length: int
    pressure_bar: float
    barostat: str
    barostat_tau_fs: float
    lammps_command: str
    heat_frames: str
    heat_start_frame: int
    heat_temperatures: str
    heat_dtype: str
    heat_chunk_size: int
    heat_hops: int
    heat_remat: bool
    heat_shadow: bool
    output_dir: Path


def load_run_config(path: Path) -> RunConfig:
    """Load a run TOML file and resolve all paths relative to that file."""
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"run configuration not found: {config_path}")

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    base = config_path.parent
    run = _table(data, "run")
    structure = _table(data, "structure")
    model = _table(data, "model")
    md = _table(data, "md")
    execution = _table(data, "execution")
    heat = _table(data, "heat_capacity")

    md_driver = str(md.get("driver", "ase"))
    md_prefix = str(md.get("prefix", config_path.stem))
    default_trajectory = (
        f"{md_prefix}.traj"
        if md_driver == "ase"
        else (
            f"{md_prefix}.lammpstrj"
            if md_driver == "lammps"
            else f"{md_prefix}.centroid.extxyz"
        )
    )

    required = structure.get("required_elements")
    required_elements = frozenset(str(item) for item in required) if required else None
    config = RunConfig(
        name=str(run.get("name", config_path.stem)),
        structure=_resolve(base, structure["path"]),
        required_elements=required_elements,
        md_backend=str(model.get("md_backend", "metatomic")),
        ad_backend=str(model.get("ad_backend", "pet-jax")),
        checkpoint=_optional_resolve(base, model.get("checkpoint")),
        exported_model=_resolve(base, model["exported_model"]),
        jax_checkpoint=_optional_resolve(base, model.get("jax_checkpoint")),
        device=str(model.get("device", "cuda")),
        stress_validated=bool(model.get("stress_validated", False)),
        md_driver=md_driver,
        md_ensemble=str(md.get("ensemble", "nvt")),
        temperature_K=float(md.get("temperature_K", 300.0)),
        md_steps=int(md.get("steps", 10)),
        timestep_fs=float(md.get("timestep_fs", 0.25)),
        md_prefix=md_prefix,
        md_trajectory_file=Path(
            str(md.get("trajectory_file", default_trajectory))
        ).name,
        equilibration_steps=int(md.get("equilibration_steps", 0)),
        output_stride=int(md.get("output_stride", 1)),
        restart_stride=int(md.get("restart_stride", 10000)),
        random_seed=int(md.get("seed", 2025)),
        thermostat=str(md.get("thermostat", "langevin")),
        thermostat_tau_fs=float(md.get("thermostat_tau_fs", 100.0)),
        thermostat_chain_length=int(md.get("thermostat_chain_length", 3)),
        pressure_bar=float(md.get("pressure_bar", 1.0)),
        barostat=str(md.get("barostat", "none")),
        barostat_tau_fs=float(md.get("barostat_tau_fs", 1000.0)),
        lammps_command=str(execution.get("lammps_command", "lmp")),
        heat_frames=str(heat.get("frames", "1")),
        heat_start_frame=int(heat.get("start_frame", 0)),
        heat_temperatures=str(heat.get("temperatures_K", "100:500:10")),
        heat_dtype=str(heat.get("dtype", "float32")),
        heat_chunk_size=int(heat.get("chunk_size", 1)),
        heat_hops=int(heat.get("hops", 3)),
        heat_remat=bool(heat.get("remat", True)),
        heat_shadow=bool(heat.get("shadow", False)),
        output_dir=_resolve(base, run.get("output_dir", "../output")),
    )
    _validate(config)
    return config


def classical_output_directories(config: RunConfig) -> tuple[Path, ...]:
    """Return concise, declared, and historical production directories."""
    candidates = [config.output_dir]
    match = LOADED_RUN_PATTERN.fullmatch(config.name)
    if match:
        root = output_root() / "md" / "production"
        historical_root = output_root() / "classical" / "production"
        candidates.extend(
            (
                root
                / match.group("model")
                / f"{match.group('loading')}ch4"
                / f"{match.group('temperature')}K"
                / f"rep{int(match.group('replica')):02d}",
                root
                / match.group("model")
                / f"{match.group('loading')}ch4"
                / config.name,
                historical_root
                / match.group("model")
                / f"{match.group('loading')}ch4"
                / config.name,
                historical_root / f"{match.group('loading')}ch4" / config.name,
            )
        )
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def find_classical_output_file(config: RunConfig, *filenames: str) -> Path | None:
    """Find any named output in the concise, declared, or historical directory."""
    for directory in classical_output_directories(config):
        for filename in filenames:
            path = directory / filename
            if path.is_file():
                return path
    return None


def run_output_parts(config: RunConfig) -> tuple[str, str] | None:
    """Return the contextual temperature/replica path for a loaded run."""
    match = LOADED_RUN_PATTERN.fullmatch(config.name)
    if match is None:
        return None
    return (
        f"{match.group('temperature')}K",
        f"rep{int(match.group('replica')):02d}",
    )


def loaded_config_path(
    configs_dir: Path,
    model: str,
    loading: int,
    temperature: int,
    replica: int,
) -> Path:
    """Return the concise path for one generated loaded-run configuration."""
    return (
        configs_dir
        / model
        / f"{loading}ch4"
        / f"{temperature}K-rep{replica:02d}.toml"
    )


def find_loaded_config(
    configs_dir: Path,
    model: str,
    loading: int,
    temperature: int,
    replica: int,
) -> Path:
    """Find a concise generated config, falling back to its historical flat path."""
    concise = loaded_config_path(configs_dir, model, loading, temperature, replica)
    if concise.is_file():
        return concise
    name = f"mof5-{loading}ch4-{model}-npt-{temperature}K-rep{replica:02d}.toml"
    return configs_dir / name


def _table(data: dict, name: str) -> dict:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _resolve(base: Path, value: str) -> Path:
    return (base / value).expanduser().resolve()


def _optional_resolve(base: Path, value: str | None) -> Path | None:
    return _resolve(base, value) if value else None


def _validate(config: RunConfig) -> None:
    if config.md_backend != "metatomic":
        raise ValueError(f"unsupported MD backend {config.md_backend!r}; choose 'metatomic'")
    if config.ad_backend == "pet-jax" and (
        config.checkpoint is None or config.jax_checkpoint is None
    ):
        raise ValueError("pet-jax AD requires model.checkpoint and model.jax_checkpoint")
    if config.ad_backend not in {"pet-jax", "none"}:
        raise ValueError("unsupported AD backend; choose 'pet-jax' or 'none'")
    if config.md_driver not in {"ase", "lammps"}:
        raise ValueError("md.driver must be 'ase' or 'lammps'")
    if config.temperature_K <= 0.0 or config.md_steps < 1 or config.timestep_fs <= 0.0:
        raise ValueError("MD temperature, steps, and timestep must be positive")
    if not 0 <= config.equilibration_steps < config.md_steps:
        raise ValueError("md.equilibration_steps must be in [0, md.steps)")
    if config.output_stride < 1 or config.restart_stride < 1:
        raise ValueError("MD output and restart strides must be positive")
    if config.random_seed <= 0:
        raise ValueError("md.seed must be positive")
    if config.thermostat_tau_fs <= 0.0 or config.thermostat_chain_length < 1:
        raise ValueError("thermostat time and chain length must be positive")
    if config.md_ensemble == "npt-flexible" and (
        config.pressure_bar <= 0.0 or config.barostat_tau_fs <= 0.0
    ):
        raise ValueError("NPT pressure and barostat time must be positive")
    if config.md_driver == "lammps" and config.md_ensemble != "npt-flexible":
        raise ValueError("the LAMMPS production driver requires ensemble='npt-flexible'")
    if config.md_driver == "lammps" and (
        config.thermostat != "nose-hoover-chain" or config.barostat != "mttk"
    ):
        raise ValueError(
            "the LAMMPS production driver requires thermostat='nose-hoover-chain' "
            "and barostat='mttk'"
        )
    if config.heat_start_frame < 0:
        raise ValueError("heat_capacity.start_frame must be non-negative")
    if config.heat_dtype not in {"float32", "float64"}:
        raise ValueError("heat_capacity.dtype must be 'float32' or 'float64'")
    if config.heat_chunk_size < 1 or config.heat_hops < 0:
        raise ValueError("heat_capacity.chunk_size must be positive and hops non-negative")
