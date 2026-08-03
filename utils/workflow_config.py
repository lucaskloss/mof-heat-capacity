"""TOML run specifications for reusable MD and harmonic-Cv workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


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
    temperature_K: float
    md_steps: int
    timestep_fs: float
    md_prefix: str
    heat_frames: str
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
    heat = _table(data, "heat_capacity")

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
        temperature_K=float(md.get("temperature_K", 300.0)),
        md_steps=int(md.get("steps", 10)),
        timestep_fs=float(md.get("timestep_fs", 0.25)),
        md_prefix=str(md.get("prefix", config_path.stem)),
        heat_frames=str(heat.get("frames", "1")),
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
    if config.temperature_K <= 0.0 or config.md_steps < 1 or config.timestep_fs <= 0.0:
        raise ValueError("MD temperature, steps, and timestep must be positive")
    if config.heat_dtype not in {"float32", "float64"}:
        raise ValueError("heat_capacity.dtype must be 'float32' or 'float64'")
    if config.heat_chunk_size < 1 or config.heat_hops < 0:
        raise ValueError("heat_capacity.chunk_size must be positive and hops non-negative")
