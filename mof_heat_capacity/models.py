"""PET checkpoint export helpers."""

import fcntl
import os
from pathlib import Path
from uuid import uuid4


def configure_metatomic_neighbors(metatomic_neighbors) -> None:
    """Configure the metatomic ASE neighbor backend for the Izar stack.

    metatomic-ase computes ``max_neighbors`` from a floating-point cutoff.
    nvalchemi 0.3/0.4 passes that value to ``torch.full`` as a tensor shape,
    while Torch 2.5 requires integer dimensions. In the large-system cell-list
    path, metatomic-ase also multiplies the per-atom neighbor capacity by the
    atom count. Undo that multiplication: nvalchemi allocates a matrix with
    shape ``(atoms, max_neighbors)``, so retaining it causes quadratic memory
    growth and can exhaust a GPU for large periodic cells.
    """
    if metatomic_neighbors.HAS_NVALCHEMIOPS:
        neighbor_list = metatomic_neighbors.nvalchemi_neighbor_list
        if not getattr(neighbor_list, "_mof5_patched", False):
            original = neighbor_list

            def neighbor_list_with_integer_capacity(*args, **kwargs):
                if "max_neighbors" in kwargs:
                    capacity = int(kwargs["max_neighbors"])
                    positions = args[0] if args else kwargs.get("positions")
                    if positions is not None:
                        atom_count = len(positions)
                        # metatomic-ase supplies N * max(128, cutoff**3),
                        # although nvalchemi interprets this as neighbors per
                        # atom. Recover the intended per-atom bound for the
                        # >=2000-atom cell-list path.
                        if atom_count > 0 and capacity >= atom_count * 128:
                            capacity = max(128, (capacity + atom_count - 1) // atom_count)
                    kwargs["max_neighbors"] = capacity
                return original(*args, **kwargs)

            neighbor_list_with_integer_capacity._mof5_patched = True
            metatomic_neighbors.nvalchemi_neighbor_list = (
                neighbor_list_with_integer_capacity
            )

    # nvalchemi avoids Vesin's NVRTC JIT path, which is not accepted by Izar's
    # V100/CUDA runtime. Retain an opt-out for diagnostics on other CUDA stacks.
    if os.environ.get("MOF_USE_NVALCHEMIOPS", "1").lower() in {"0", "false", "no"}:
        metatomic_neighbors.HAS_NVALCHEMIOPS = False


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
