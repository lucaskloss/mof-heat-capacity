"""Shared identities, input selection, and directories for loaded campaigns."""

from collections import Counter
from pathlib import Path
import re


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MD_TEMPERATURES = tuple(range(175, 426, 25))
DEFAULT_REPORT_TEMPERATURES = tuple(range(200, 401, 25))
GUEST_SYMBOLS = {
    "ch4": ("C", "H", "H", "H", "H"),
    "co2": ("C", "O", "O"),
    "h2o": ("O", "H", "H"),
}


def default_report_temperatures(temperatures: list[int]) -> list[int]:
    """Use the padding for derivatives and report the default 200–400 K range."""
    if tuple(temperatures) == DEFAULT_MD_TEMPERATURES:
        return list(DEFAULT_REPORT_TEMPERATURES)
    return list(temperatures)


def validate_selection(mof: str, guest: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", mof):
        raise ValueError("--mof must be a lowercase label starting with a letter (letters, digits, _ and -)")
    if guest not in GUEST_SYMBOLS:
        raise ValueError("--guest must be ch4, co2, or h2o")


def system_directory(model: str, loading: int, mof: str = "mof5", guest: str = "ch4") -> Path:
    """Keep established MOF-5/CH4 paths; qualify all other systems by MOF."""
    validate_selection(mof, guest)
    path = Path(model) / f"{loading}{guest}"
    return path if (mof, guest) == ("mof5", "ch4") else Path(mof) / path


def structure_directory(loading: int, mof: str = "mof5", guest: str = "ch4") -> Path:
    validate_selection(mof, guest)
    path = Path(f"{loading}{guest}")
    return path if (mof, guest) == ("mof5", "ch4") else Path(mof) / path


def host_path(mof: str, override: Path | None = None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    for suffix in (".pdb", ".cif", ".gro", ".extxyz"):
        path = PROJECT_DIR / "input" / f"{mof}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"no input/{mof} structure found; supply --host PATH")


def validate_guest(molecule, guest: str) -> None:
    if Counter(molecule.get_chemical_symbols()) != Counter(GUEST_SYMBOLS[guest]):
        raise ValueError(f"--molecule must contain exactly one {guest.upper()} molecule")
