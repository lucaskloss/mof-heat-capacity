"""PET-MAD checkpoint export helpers."""

from pathlib import Path

import upet


def ensure_exported_model(checkpoint_path: Path, exported_path: Path) -> Path:
    """Export a bundled PET-MAD checkpoint to metatomic's portable format."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    exported_path = exported_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PET-MAD checkpoint not found: {checkpoint_path}")
    if exported_path.exists():
        return exported_path

    exported_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {checkpoint_path.name} to {exported_path}")
    upet.save_upet(checkpoint_path=str(checkpoint_path), output=str(exported_path))
    return exported_path
