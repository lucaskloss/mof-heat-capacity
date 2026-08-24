"""Prepare independent loaded structures and classical NPT configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

from ase import io

from ..io import write_lammps_data, write_structure_pdb
from ..structures.methane import insert_molecules

PROJECT_DIR = Path(__file__).resolve().parents[2]


DEFAULT_REPLICAS = 1
CONFIG_TEMPLATE = PROJECT_DIR / "configs" / "mof5_100ch4_hybrid_npt.toml"
MODEL_PRESETS = {
    "pet-mad": {
        "label": "pet-mad-1.5-s-40nn",
        "checkpoint": "../models/pet-mad-1.5-s_40nn_nostress.ckpt",
        "exported_model": "../models/pet-mad-1.5-s_40nn_nostress.pt",
        "jax_checkpoint": "../models/pet-mad-1.5-s_40nn_jax",
    },
    "pet-sol": {
        "label": "pet-sol-s-best",
        "checkpoint": "../models/pet_sol-s-best_nostress.ckpt",
        "exported_model": "../models/pet_sol-s-best_nostress.pt",
        "jax_checkpoint": "../models/pet_sol-s-best_nostress_jax",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_PRESETS), default="pet-mad")
    parser.add_argument(
        "--temperatures",
        default=(
            "100,200,300,400,500"
        ),
        help="Classical enthalpy grid (default: 100 to 500 K in 100 K steps)",
    )
    parser.add_argument("--replicas", type=int)
    parser.add_argument(
        "--loading",
        type=int,
        default=100,
        help="positive methane molecule count (default: 100)",
    )
    parser.add_argument("--host", type=Path, default=PROJECT_DIR / "input" / "mof5.pdb")
    parser.add_argument("--methane", type=Path, default=PROJECT_DIR / "input" / "ch4.gro")
    parser.add_argument("--tries", type=int, default=20000)
    parser.add_argument("--min-distance", type=float, default=1.5)
    parser.add_argument("--seed-base", type=int, default=20250000)
    parser.add_argument("--checkpoint")
    parser.add_argument("--exported-model")
    parser.add_argument("--jax-checkpoint")
    parser.add_argument(
        "--stress-validated",
        action="store_true",
        help="Mark a custom model-path override validated for NPT virials/stresses",
    )
    parser.add_argument("--configs-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep existing configurations/structures and create only missing runs",
    )
    return parser.parse_args()


def temperatures(specification: str) -> list[int]:
    values = [int(item.strip()) for item in specification.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("--temperatures must contain positive comma-separated integers")
    return list(dict.fromkeys(values))


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"template does not contain exactly one {old!r}")
    return text.replace(old, new)


def render_config(
    template: str,
    *,
    temperature: int,
    replica: int,
    seed: int,
    loading: int,
    model_label: str,
    structure_path: Path,
    args: argparse.Namespace,
) -> tuple[str, str]:
    base_name = (
        f"mof5-{loading}ch4-{model_label}-npt-"
        f"{temperature}K-rep{replica:02d}"
    )
    template_name = "mof5-100ch4-hybrid-npt-300K-rep01"
    trajectory_suffix = ".lammpstrj"
    rendered = template
    rendered = replace_once(
        rendered,
        f'output_dir = "../output/classical/production/100ch4/{template_name}"',
        f'output_dir = "../output/classical/production/{loading}ch4/{base_name}"',
    )
    rendered = replace_once(
        rendered,
        f'trajectory_file = "{template_name}{trajectory_suffix}"',
        f'trajectory_file = "{base_name}{trajectory_suffix}"',
    )
    rendered = rendered.replace(template_name, base_name)
    rendered = replace_once(
        rendered,
        'path = "../output/structures/mof5-100ch4-seed2025.pdb"',
        f'path = "../{structure_path.relative_to(PROJECT_DIR)}"',
    )
    rendered = replace_once(rendered, "temperature_K = 300.0", f"temperature_K = {temperature}.0")
    rendered = replace_once(rendered, "seed = 202501", f"seed = {seed}")
    rendered = replace_once(
        rendered,
        'checkpoint = "../models/REPLACE_WITH_STRESS_VALIDATED.ckpt"',
        f'checkpoint = "{args.checkpoint}"',
    )
    rendered = replace_once(
        rendered,
        'exported_model = "../models/REPLACE_WITH_STRESS_VALIDATED.pt"',
        f'exported_model = "{args.exported_model}"',
    )
    rendered = replace_once(
        rendered,
        'jax_checkpoint = "../models/REPLACE_WITH_STRESS_VALIDATED_jax"',
        f'jax_checkpoint = "{args.jax_checkpoint}"',
    )
    if args.stress_validated:
        rendered = replace_once(
            rendered, "stress_validated = false", "stress_validated = true"
        )
    return base_name, rendered


def prepare_structure(
    path: Path,
    data_path: Path,
    *,
    host,
    methane,
    loading: int,
    tries: int,
    minimum: float,
    seed: int,
) -> None:
    combined = (
        host.copy()
        if loading == 0
        else insert_molecules(host, methane, loading, tries, minimum, seed)
    )
    combined.set_pbc(host.pbc)
    write_structure_pdb(path, combined)
    write_lammps_data(data_path, combined)


def main() -> None:
    args = parse_args()
    if args.force and args.skip_existing:
        raise ValueError("--force and --skip-existing are mutually exclusive")
    if args.loading == 0:
        raise ValueError(
            "empty MOF-5 needs only direct relaxation and a Hessian; "
            "do not prepare an MD campaign"
        )
    preset = MODEL_PRESETS[args.model]
    custom_paths = any(
        value is not None
        for value in (args.checkpoint, args.exported_model, args.jax_checkpoint)
    )
    if custom_paths and not all(
        value is not None
        for value in (args.checkpoint, args.exported_model, args.jax_checkpoint)
    ):
        raise ValueError(
            "override --checkpoint, --exported-model, and --jax-checkpoint together"
        )
    args.checkpoint = args.checkpoint or preset["checkpoint"]
    args.exported_model = args.exported_model or preset["exported_model"]
    args.jax_checkpoint = args.jax_checkpoint or preset["jax_checkpoint"]
    if not custom_paths:
        args.stress_validated = True
    selected_temperatures = temperatures(args.temperatures)
    replicas = args.replicas or DEFAULT_REPLICAS
    if replicas < 1 or args.tries < 1 or args.min_distance <= 0:
        raise ValueError(
            "replicas, tries, and minimum distance must be positive"
        )
    template = CONFIG_TEMPLATE.read_text()
    if not args.host.is_file():
        raise FileNotFoundError("host input structure must exist")
    if args.loading > 0 and not args.methane.is_file():
        raise FileNotFoundError("methane input structure must exist for nonzero loading")
    host = methane = None
    if not args.dry_run and not args.configs_only:
        host = io.read(args.host)
        methane = io.read(args.methane) if args.loading > 0 else None

    count = 0
    for temperature in selected_temperatures:
        for replica in range(1, replicas + 1):
            seed = args.seed_base + temperature * 100 + replica
            structure_stem = (
                f"mof5-{args.loading}ch4-npt-"
                f"{temperature}K-rep{replica:02d}-seed{seed}"
            )
            structure_path = PROJECT_DIR / "output" / "structures" / f"{structure_stem}.pdb"
            data_path = structure_path.with_suffix(".data")
            name, config_text = render_config(
                template,
                temperature=temperature,
                replica=replica,
                seed=seed,
                loading=args.loading,
                model_label=str(preset["label"]),
                structure_path=structure_path,
                args=args,
            )
            config_path = PROJECT_DIR / "configs" / f"{name}.toml"
            print(f"{config_path.relative_to(PROJECT_DIR)} -> {structure_path.relative_to(PROJECT_DIR)}")
            if args.dry_run:
                count += 1
                continue
            if config_path.exists() and args.skip_existing:
                print(f"Keeping existing configuration: {config_path}")
                continue
            if config_path.exists() and not args.force:
                raise FileExistsError(f"configuration exists; use --force: {config_path}")
            if args.configs_only and not structure_path.is_file():
                raise FileNotFoundError(
                    "cannot reuse missing structure: "
                    f"{structure_path}; prepare structures before using --configs-only"
                )
            if not args.configs_only:
                if structure_path.exists() and not args.force:
                    raise FileExistsError(f"structure exists; use --force: {structure_path}")
                prepare_structure(
                    structure_path,
                    data_path,
                    host=host,
                    methane=methane,
                    loading=args.loading,
                    tries=args.tries,
                    minimum=args.min_distance,
                    seed=seed,
                )
            config_path.write_text(config_text)
            count += 1
    print(f"Prepared {count} loaded classical-NPT run specifications")
    if custom_paths and not args.stress_validated:
        print(
            "NPT remains blocked: provide a stress-capable model and rerun with "
            "--stress-validated only after validating its virials."
        )


if __name__ == "__main__":
    main()
