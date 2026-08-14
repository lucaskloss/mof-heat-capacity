"""PET checkpoint export helpers."""

import fcntl
import os
from pathlib import Path
from uuid import uuid4

def ensure_exported_model(checkpoint_path: Path, exported_path: Path) -> Path:
    """Export a PET checkpoint to metatomic's portable format once."""
    try:
        import upet
    except ImportError as error:
        raise RuntimeError(
            "exporting a PET checkpoint requires the 'upet' package; activate "
            "the project environment or provide an existing exported model"
        ) from error
    checkpoint_path = checkpoint_path.expanduser().resolve()
    exported_path = exported_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PET checkpoint not found: {checkpoint_path}")
    if exported_path.is_file():
        return exported_path

    exported_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = exported_path.with_suffix(f"{exported_path.suffix}.lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if exported_path.is_file():
            return exported_path

        temporary_path = exported_path.with_name(
            f".{exported_path.stem}.{uuid4().hex}.tmp{exported_path.suffix}"
        )
        print(f"Exporting {checkpoint_path.name} to {exported_path}")
        try:
            upet.save_upet(
                checkpoint_path=str(checkpoint_path), output=str(temporary_path)
            )
            os.replace(temporary_path, exported_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return exported_path
