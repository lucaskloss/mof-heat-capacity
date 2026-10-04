"""Prepare independent loaded structures and classical NPT configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib

from ase import io

from ..campaign import (
    DEFAULT_MD_TEMPERATURES, GUEST_SYMBOLS, host_path, structure_directory, system_directory,
    validate_guest, validate_selection,
)
from ..config import loaded_config_path, output_root
from ..io import load_structure, write_lammps_data, write_structure_pdb
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
        default=",".join(map(str, DEFAULT_MD_TEMPERATURES)),
        help="Classical enthalpy grid (default: 175 to 425 K in 25 K steps)",
    )
    parser.add_argument("--replicas", type=int)
    parser.add_argument(
        "--loading",
        type=int,
        default=100,
        help="positive guest molecule count (default: 100)",
    )
    parser.add_argument("--mof", default="mof5", help="MOF label; defaults to input/<mof>.pdb or .cif")
    parser.add_argument("--guest", type=str.lower, choices=tuple(GUEST_SYMBOLS), default="ch4")
    parser.add_argument("--host", type=Path, help="Periodic host structure override")
    parser.add_argument("--molecule", "--methane", dest="molecule", type=Path, help="Single guest molecule override (--methane is a compatibility alias)")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=output_root(),
        help="Root for generated structures and MD outputs",
    )
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
    output_root: Path,
    structure_path: Path,
    args: argparse.Namespace,
) -> tuple[str, str]:
    base_name = (
        f"{args.mof}-{loading}{args.guest}-{model_label}-npt-"
        f"{temperature}K-rep{replica:02d}"
    )
    template_name = "mof5-100ch4-hybrid-npt-300K-rep01"
    rendered = template
    output_dir = (
        output_root / "md" / "production"
        / system_directory(model_label, loading, args.mof, args.guest)
        / f"{temperature}K" / f"rep{replica:02d}"
    )
    rendered = replace_once(
        rendered, 'output_dir = "REPLACE_WITH_OUTPUT_DIR"',
        f"output_dir = {json.dumps(str(output_dir))}",
    )
    rendered = rendered.replace(template_name, base_name)
    rendered = replace_once(
        rendered,
        'path = "REPLACE_WITH_STRUCTURE_PATH"',
        f"path = {json.dumps(str(structure_path))}",
    )
    rendered = replace_once(rendered, "temperature_K = 300.0", f"temperature_K = {temperature}.0")
    rendered = replace_once(rendered, "seed = 202501", f"seed = {seed}")
    rendered = replace_once(
        rendered,
        'checkpoint = "../models/REPLACE_WITH_STRESS_VALIDATED.ckpt"',
        f"checkpoint = {json.dumps(args.checkpoint)}",
    )
    rendered = replace_once(
        rendered,
        'exported_model = "../models/REPLACE_WITH_STRESS_VALIDATED.pt"',
        f"exported_model = {json.dumps(args.exported_model)}",
    )
    rendered = replace_once(
        rendered,
        'jax_checkpoint = "../models/REPLACE_WITH_STRESS_VALIDATED_jax"',
        f"jax_checkpoint = {json.dumps(args.jax_checkpoint)}",
    )
    if args.stress_validated:
        rendered = replace_once(
            rendered, "stress_validated = false", "stress_validated = true"
        )
    rendered = replace_once(
        rendered, 'required_elements = ["C", "H", "O", "Zn"]',
        "required_elements = " + json.dumps(args.required_elements),
    )
    rendered += "\n[campaign]\n" + "\n".join(
        f"{key} = {json.dumps(value)}" for key, value in {
            "mof": args.mof, "guest": args.guest, "loading": loading,
            "host_atoms": args.host_atoms, "host": str(args.host),
            "molecule": str(args.molecule),
            "host_sha256": hashlib.sha256(args.host.read_bytes()).hexdigest(),
            "molecule_sha256": hashlib.sha256(args.molecule.read_bytes()).hexdigest(),
        }.items()
    ) + "\n"
    return base_name, rendered


def prepare_structure(
    path: Path,
    data_path: Path,
    *,
    host,
    molecule,
    loading: int,
    tries: int,
    minimum: float,
    seed: int,
) -> None:
    combined = (
        host.copy()
        if loading == 0
        else insert_molecules(host, molecule, loading, tries, minimum, seed)
    )
    combined.set_pbc(host.pbc)
    write_structure_pdb(path, combined)
    write_lammps_data(data_path, combined)


def update_output_paths(config_path: Path, rendered_config: str) -> bool:
    """Update only generated-data paths in an existing campaign config."""
    existing = config_path.read_text()
    old = tomllib.loads(existing).get("campaign")
    new = tomllib.loads(rendered_config)["campaign"]
    if old is None and (
        Path(new["host"]) != PROJECT_DIR / "input" / "mof5.pdb"
        or Path(new["molecule"]) != PROJECT_DIR / "input" / "ch4.gro"
        or (new["mof"], new["guest"]) != ("mof5", "ch4")
    ):
        raise ValueError(f"cannot reuse a legacy config with different inputs: {config_path}; choose a new --mof label")
    if old is not None and old != new:
        raise ValueError(f"campaign input identity changed: {config_path}; choose a new --mof label or use --force")
    output_dir = re.search(r"^output_dir = .*$", rendered_config, flags=re.MULTILINE)
    structure_path = re.search(r"^path = .*$", rendered_config, flags=re.MULTILINE)
    if output_dir is None or structure_path is None:
        raise ValueError("rendered configuration is missing generated-data paths")
    updated = re.sub(r"^output_dir = .*$", output_dir.group(), existing, flags=re.MULTILINE)
    updated = re.sub(r"^path = .*$", structure_path.group(), updated, flags=re.MULTILINE)
    if updated == existing:
        return False
    config_path.write_text(updated)
    return True


def main() -> None:
    args = parse_args()
    validate_selection(args.mof, args.guest)
    args.host = host_path(args.mof, args.host)
    args.molecule = (args.molecule or PROJECT_DIR / "input" / f"{args.guest}.gro").expanduser().resolve()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_DIR / output_root).resolve()
    if args.force and args.skip_existing:
        raise ValueError("--force and --skip-existing are mutually exclusive")
    if args.loading <= 0:
        raise ValueError(
            "loading must be positive; an empty MOF needs only direct relaxation and a Hessian; "
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
    replicas = DEFAULT_REPLICAS if args.replicas is None else args.replicas
    if replicas < 1 or args.tries < 1 or args.min_distance <= 0:
        raise ValueError(
            "replicas, tries, and minimum distance must be positive"
        )
    template = CONFIG_TEMPLATE.read_text()
    host = load_structure(args.host)
    if not args.molecule.is_file():
        raise FileNotFoundError(f"guest input structure must exist: {args.molecule}")
    molecule = io.read(args.molecule)
    validate_guest(molecule, args.guest)
    args.host_atoms = len(host)
    args.required_elements = sorted(set(host.get_chemical_symbols()) | set(molecule.get_chemical_symbols()))

    count = 0
    for temperature in selected_temperatures:
        for replica in range(1, replicas + 1):
            seed = args.seed_base + temperature * 100 + replica
            structure_dir = (
                output_root / "md" / "structures" / structure_directory(args.loading, args.mof, args.guest)
                / f"{temperature}K" / f"rep{replica:02d}"
            )
            structure_path = structure_dir / "structure.pdb"
            identity_path = structure_dir / "inputs.json"
            identity = {
                "mof": args.mof, "guest": args.guest, "loading": args.loading,
                "host": str(args.host), "molecule": str(args.molecule),
                "host_sha256": hashlib.sha256(args.host.read_bytes()).hexdigest(),
                "molecule_sha256": hashlib.sha256(args.molecule.read_bytes()).hexdigest(),
                "host_atoms": len(host), "seed": seed,
                "min_distance_A": args.min_distance,
            }
            if identity_path.is_file() and not args.force:
                if json.loads(identity_path.read_text()) != identity:
                    raise ValueError(f"insertion inputs changed: {identity_path}; select a new --mof label or use --force")
            data_path = structure_dir / "structure.data"
            name, config_text = render_config(
                template,
                temperature=temperature,
                replica=replica,
                seed=seed,
                loading=args.loading,
                model_label=str(preset["label"]),
                output_root=output_root,
                structure_path=structure_path,
                args=args,
            )
            config_path = loaded_config_path(
                PROJECT_DIR / "configs",
                str(preset["label"]),
                args.loading,
                temperature,
                replica,
                args.mof,
                args.guest,
            )
            historical_config_path = PROJECT_DIR / "configs" / f"{name}.toml"
            # Resolve model paths against the template, independent of nesting depth.
            for model_path in (args.checkpoint, args.exported_model, args.jax_checkpoint):
                resolved = (CONFIG_TEMPLATE.parent / model_path).expanduser().resolve()
                relative = os.path.relpath(resolved, config_path.parent)
                config_text = config_text.replace(json.dumps(model_path), json.dumps(relative))
            print(f"{config_path.relative_to(PROJECT_DIR)} -> {structure_path}")
            if args.dry_run:
                count += 1
                continue
            existing_config = config_path if config_path.exists() else historical_config_path
            keep_config = existing_config.is_file() and args.skip_existing
            if keep_config:
                if update_output_paths(existing_config, config_text):
                    print(f"Updated output paths: {existing_config}")
                print(f"Keeping existing configuration: {existing_config}")
                if structure_path.is_file() and data_path.is_file():
                    continue
            if config_path.exists() and not args.force and not keep_config:
                raise FileExistsError(f"configuration exists; use --force: {config_path}")
            if args.configs_only and not structure_path.is_file():
                raise FileNotFoundError(
                    "cannot reuse missing structure: "
                    f"{structure_path}; prepare structures before using --configs-only"
                )
            if not args.configs_only:
                if structure_path.exists() and data_path.exists() and args.skip_existing:
                    print(f"Reusing existing structure: {structure_path}")
                else:
                    if structure_path.exists() and not args.force and not args.skip_existing:
                        raise FileExistsError(
                            f"structure exists; use --force: {structure_path}"
                        )
                    prepare_structure(
                        structure_path,
                        data_path,
                        host=host,
                        molecule=molecule,
                        loading=args.loading,
                        tries=args.tries,
                        minimum=args.min_distance,
                        seed=seed,
                    )
                identity_path.write_text(json.dumps(identity, indent=2) + "\n")
            if not keep_config:
                config_path.parent.mkdir(parents=True, exist_ok=True)
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
