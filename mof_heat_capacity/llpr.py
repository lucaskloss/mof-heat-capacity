"""Build and validate a metatrain LLPR shallow energy ensemble."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PRESETS = {
    "pet-mad": {
        "checkpoint": PROJECT_DIR / "models" / "pet-mad-1.5-s_40nn_nostress.ckpt",
        "output": PROJECT_DIR / "models" / "pet-mad-1.5-s-llpr-ensemble.pt",
        "label": "pet-mad-1.5-s-40nn",
    },
    "pet-sol": {
        "checkpoint": PROJECT_DIR / "models" / "pet_sol-s-best_nostress.ckpt",
        "output": PROJECT_DIR / "models" / "pet-sol-s-best-llpr-ensemble.pt",
        "label": "pet-sol-s-best",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), required=True)
    parser.add_argument(
        "--training-set",
        type=Path,
        required=True,
        help="Labeled structures used to construct the LLPR feature covariance",
    )
    parser.add_argument(
        "--validation-set",
        type=Path,
        required=True,
        help="Disjoint labeled structures used to calibrate the LLPR scale",
    )
    parser.add_argument(
        "--energy-key",
        default="energy",
        help="Energy property name in both structure files (default: energy)",
    )
    parser.add_argument("--energy-unit", default="eV")
    parser.add_argument("--length-unit", default="angstrom")
    parser.add_argument("--members", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Data-loader worker processes (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--calibration-method",
        choices=("absolute_residuals", "squared_residuals", "crps"),
        default="absolute_residuals",
    )
    parser.add_argument(
        "--regularizer",
        type=float,
        help="LLPR covariance regularizer; omit for metatrain auto-selection",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--base-precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--validation-frames",
        type=int,
        default=8,
        help="Number of validation structures used for export checks (default: 8)",
    )
    parser.add_argument(
        "--ensemble-mean-tolerance-eV",
        type=float,
        default=1e-4,
        help="Maximum allowed ensemble-mean/central residual (default: 1e-4 eV)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the generated metatrain command",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def validate_labeled_dataset(path: Path, energy_key: str, description: str) -> None:
    """Check that ASE can read a structure and expose the requested energy."""
    from ase.io import read

    try:
        atoms = read(path, index=0)
    except Exception as error:
        raise ValueError(f"cannot read {description}: {path}") from error
    calculator_results = getattr(getattr(atoms, "calc", None), "results", {})
    if energy_key not in atoms.info and energy_key not in calculator_results:
        available = sorted(set(atoms.info).union(calculator_results))
        raise ValueError(
            f"{description} does not expose energy key {energy_key!r} on its first "
            f"structure; available properties: {', '.join(available) or 'none'}"
        )


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    preset = MODEL_PRESETS[args.model]
    checkpoint = (args.checkpoint or preset["checkpoint"]).expanduser().resolve()
    output = (args.output or preset["output"]).expanduser().resolve()
    training_set = args.training_set.expanduser().resolve()
    validation_set = args.validation_set.expanduser().resolve()
    work_dir = (
        args.work_dir
        or PROJECT_DIR / "output" / "model-preparation" / "llpr" / preset["label"]
    ).expanduser().resolve()

    for description, path in (
        ("PET checkpoint", checkpoint),
        ("LLPR covariance/training set", training_set),
        ("LLPR calibration/validation set", validation_set),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} is missing: {path}")
    if training_set == validation_set:
        raise ValueError("training and validation sets must be separate files")
    validate_labeled_dataset(training_set, args.energy_key, "training set")
    validate_labeled_dataset(validation_set, args.energy_key, "validation set")
    if output.suffix != ".pt":
        raise ValueError("--output must end in .pt")
    if (
        args.members < 2
        or args.batch_size < 1
        or args.num_workers < 0
        or args.validation_frames < 1
    ):
        raise ValueError(
            "members must be at least two; batch size and validation frames must "
            "be positive; workers must be non-negative"
        )
    if args.regularizer is not None and args.regularizer <= 0.0:
        raise ValueError("--regularizer must be positive")
    if args.ensemble_mean_tolerance_eV <= 0.0:
        raise ValueError("--ensemble-mean-tolerance-eV must be positive")
    if not args.energy_key or not args.energy_unit or not args.length_unit:
        raise ValueError("energy key and units must not be empty")
    if output.is_file():
        raise FileExistsError(
            f"refusing to replace existing LLPR model: {output}; choose a new --output"
        )
    return {
        "checkpoint": checkpoint,
        "output": output,
        "training_set": training_set,
        "validation_set": validation_set,
        "work_dir": work_dir,
    }


def metatrain_options(args: argparse.Namespace, paths: dict[str, Path]) -> dict:
    training = {
        "model_checkpoint": str(paths["checkpoint"]),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "regularizer": args.regularizer,
        "calibration_method": args.calibration_method,
    }
    def dataset(path: Path) -> dict:
        return {
            "systems": {
                "read_from": str(path),
                "length_unit": args.length_unit,
            },
            "targets": {
                "energy": {
                    "key": args.energy_key,
                    "unit": args.energy_unit,
                }
            },
        }
    return {
        "device": args.device,
        "base_precision": args.base_precision,
        "seed": args.seed,
        "architecture": {
            "name": "llpr",
            "model": {"num_ensemble_members": {"energy": args.members}},
            "training": training,
        },
        "training_set": dataset(paths["training_set"]),
        "validation_set": dataset(paths["validation_set"]),
        "test_set": 0.0,
    }


def find_mtt() -> Path:
    alongside_python = Path(os.path.realpath(os.sys.executable)).with_name("mtt")
    if alongside_python.is_file():
        return alongside_python
    executable = shutil.which("mtt")
    if executable is None:
        raise RuntimeError(
            "metatrain is unavailable; install the environment from environment.yml"
        )
    return Path(executable).resolve()


def system_values(output, frame_count: int, name: str) -> np.ndarray:
    values = output[0].values.detach().cpu().numpy()
    values = np.asarray(values, dtype=float).reshape(frame_count, -1)
    if values.shape[1] != 1:
        raise ValueError(f"{name} must provide one system value per structure")
    return values[:, 0]


def validate_export(
    model_path: Path,
    validation_set: Path,
    *,
    device: str,
    expected_members: int,
    frame_limit: int,
    tolerance_eV: float,
) -> dict[str, float | int | list[str]]:
    from ase.io import iread
    import metatomic.torch as metatomic_torch
    from metatomic.torch import ModelOutput
    from metatomic_ase import MetatomicCalculator

    model = metatomic_torch.load_atomistic_model(str(model_path))
    outputs = sorted(model.capabilities().outputs)
    required = {"energy", "energy_uncertainty", "energy_ensemble"}
    missing = sorted(required.difference(outputs))
    if missing:
        raise ValueError("LLPR export is missing output(s): " + ", ".join(missing))
    del model

    structures = []
    for atoms in iread(str(validation_set), index=":"):
        structures.append(atoms)
        if len(structures) == frame_limit:
            break
    if not structures:
        raise ValueError(f"validation set contains no readable structures: {validation_set}")

    calculator = MetatomicCalculator(str(model_path), device=device)
    requested = {
        "energy": ModelOutput(sample_kind="system"),
        "energy_uncertainty": ModelOutput(sample_kind="system"),
        "energy_ensemble": ModelOutput(sample_kind="system"),
    }
    predictions = calculator.run_model(structures, requested)
    central = system_values(predictions["energy"], len(structures), "energy")
    uncertainty = system_values(
        predictions["energy_uncertainty"], len(structures), "energy_uncertainty"
    )
    members = predictions["energy_ensemble"][0].values.detach().cpu().numpy()
    members = np.asarray(members, dtype=float).reshape(len(structures), -1)
    if members.shape[1] != expected_members:
        raise ValueError(
            f"energy_ensemble has {members.shape[1]} members, expected {expected_members}"
        )
    if not all(np.all(np.isfinite(values)) for values in (central, uncertainty, members)):
        raise ValueError("LLPR export produced non-finite energy values")
    if np.any(uncertainty < 0.0):
        raise ValueError("LLPR export produced negative analytical uncertainties")
    residual = members.mean(axis=1) - central
    finite_ensemble_uncertainty = members.std(axis=1, ddof=1)
    maximum_residual = float(np.max(np.abs(residual)))
    if maximum_residual > tolerance_eV:
        raise ValueError(
            "LLPR ensemble mean does not reproduce the central energy: maximum "
            f"residual {maximum_residual:.6g} eV exceeds {tolerance_eV:.6g} eV"
        )
    return {
        "checked_frames": len(structures),
        "member_count": members.shape[1],
        "capability_outputs": outputs,
        "ensemble_mean_residual_rms_eV": float(np.sqrt(np.mean(residual**2))),
        "ensemble_mean_residual_max_abs_eV": maximum_residual,
        "minimum_analytical_uncertainty_eV": float(np.min(uncertainty)),
        "maximum_analytical_uncertainty_eV": float(np.max(uncertainty)),
        "ensemble_vs_analytical_uncertainty_rms_eV": float(
            np.sqrt(np.mean((finite_ensemble_uncertainty - uncertainty) ** 2))
        ),
    }


def main() -> None:
    args = parse_args()
    paths = resolve_inputs(args)
    options = metatrain_options(args, paths)
    mtt = find_mtt()
    candidate = paths["work_dir"] / f"{paths['output'].stem}.candidate.pt"
    candidate_checkpoint = candidate.with_suffix(".ckpt")
    llpr_checkpoint = paths["work_dir"] / f"{paths['output'].stem}.ckpt"
    options_path = paths["work_dir"] / "options-llpr.json"
    command = [str(mtt), "train", str(options_path), "-o", str(candidate)]

    if args.dry_run:
        print(json.dumps(options, indent=2))
        print("Command:", " ".join(command))
        return

    if candidate.exists() or candidate_checkpoint.exists():
        raise FileExistsError(
            f"candidate output already exists below {paths['work_dir']}; "
            "inspect or move it first"
        )
    if llpr_checkpoint.exists():
        raise FileExistsError(
            f"refusing to replace existing LLPR checkpoint: {llpr_checkpoint}"
        )
    paths["work_dir"].mkdir(parents=True, exist_ok=True)
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    options_path.write_text(json.dumps(options, indent=2) + "\n")
    subprocess.run(command, cwd=paths["work_dir"], check=True)
    validation = validate_export(
        candidate,
        paths["validation_set"],
        device=args.device,
        expected_members=args.members,
        frame_limit=args.validation_frames,
        tolerance_eV=args.ensemble_mean_tolerance_eV,
    )
    if not candidate_checkpoint.is_file():
        raise FileNotFoundError(
            f"metatrain did not write the expected LLPR checkpoint: {candidate_checkpoint}"
        )
    os.replace(candidate_checkpoint, llpr_checkpoint)
    os.replace(candidate, paths["output"])
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "llpr_checkpoint": str(llpr_checkpoint),
        "llpr_checkpoint_sha256": file_sha256(llpr_checkpoint),
        "training_set": str(paths["training_set"]),
        "training_set_sha256": file_sha256(paths["training_set"]),
        "validation_set": str(paths["validation_set"]),
        "validation_set_sha256": file_sha256(paths["validation_set"]),
        "output": str(paths["output"]),
        "output_sha256": file_sha256(paths["output"]),
        "options": options,
        "validation": validation,
        "software": {
            name: package_version(name)
            for name in (
                "ase",
                "metatrain",
                "metatomic-ase",
                "metatomic-torch",
                "torch",
            )
        },
    }
    provenance_path = paths["output"].with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"Validated LLPR ensemble: {paths['output']}")
    print(f"Provenance: {provenance_path}")


if __name__ == "__main__":
    main()
