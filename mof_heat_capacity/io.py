"""Load structures and generate LAMMPS inputs for loaded-MOF workflows."""

from pathlib import Path

from ase import Atoms, io
from ase.data import atomic_numbers

MOF5_SPECIES = ("C", "H", "O", "Zn")


def load_structure(
    structure_path: Path, *, required_elements: frozenset[str] | None = None
) -> Atoms:
    """Read a non-empty periodic structure, optionally checking its elements."""
    structure = io.read(structure_path)
    if structure.cell.volume <= 0.0:
        raise ValueError("structure must define a non-zero periodic cell")
    if not all(structure.pbc):
        raise ValueError("structure must be periodic in all three directions")
    if len(structure) == 0:
        raise ValueError("structure contains no atoms")

    if required_elements is not None:
        elements = set(structure.get_chemical_symbols())
        missing = sorted(required_elements.difference(elements))
        if missing:
            raise ValueError("structure is missing required elements: " + ", ".join(missing))

    return structure

def load_mof5_structure(structure_path: Path) -> Atoms:
    """Read and validate a periodic MOF-5 structure with the expected elements."""
    structure = io.read(structure_path)
    if structure.cell.volume <= 0.0:
        raise ValueError("MOF-5 structure must define a non-zero periodic cell")
    if not all(structure.pbc):
        raise ValueError("MOF-5 structure must be periodic in all three directions")

    unknown = sorted(set(structure.get_chemical_symbols()).difference(MOF5_SPECIES))
    if unknown:
        raise ValueError(f"MOF-5 structure contains unsupported elements: {', '.join(unknown)}")
    missing = sorted(set(MOF5_SPECIES).difference(structure.get_chemical_symbols()))
    if missing:
        raise ValueError(f"MOF-5 structure is missing expected elements: {', '.join(missing)}")
    if len(structure) == 0:
        raise ValueError("MOF-5 structure contains no atoms")
    return structure


def _export_structure(structure: Atoms) -> Atoms:
    """Strip input-format labels and CIF metadata from a structure export."""
    return Atoms(
        symbols=structure.get_chemical_symbols(),
        positions=structure.positions,
        cell=structure.cell,
        pbc=structure.pbc,
    )


def write_structure_extxyz(output_path: Path, structure: Atoms) -> None:
    """Write a periodic structure without PDB coordinate/cell rounding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.write(output_path, _export_structure(structure), format="extxyz")


def write_structure_pdb(output_path: Path, structure: Atoms) -> None:
    """Write a clean ASE-readable periodic PDB structure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = _export_structure(structure)
    io.write(output_path, clean, format="proteindatabank")
    lines = output_path.read_text().splitlines()
    # Remove ASE's single-frame wrappers for broad PDB-reader compatibility.
    lines = [line for line in lines if line not in {"MODEL     1", "ENDMDL"}]
    if not lines or lines[-1] != "END":
        lines.append("END")
    output_path.write_text("\n".join(lines) + "\n")


def lammps_species(structure: Atoms) -> tuple[str, ...]:
    """Keep the legacy element order, followed by additional host elements."""
    elements = set(structure.get_chemical_symbols())
    legacy = tuple(symbol for symbol in MOF5_SPECIES if symbol in elements)
    additional = tuple(sorted(elements.difference(MOF5_SPECIES), key=atomic_numbers.__getitem__))
    return legacy + additional


def write_lammps_data(output_path: Path, structure: Atoms) -> None:
    """Write atomic data with a stable element order shared with the MD input."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.write(output_path, structure, format="lammps-data", atom_style="atomic", masses=True,
             specorder=lammps_species(structure))


def write_classical_npt_lammps_input(
    input_path: Path,
    *,
    data_path: Path,
    model_path: Path,
    device: str,
    trajectory_path: Path,
    thermo_path: Path,
    restart_prefix: Path,
    final_data_path: Path,
    temperature_K: float,
    pressure_bar: float,
    timestep_fs: float,
    thermostat_tau_fs: float,
    thermostat_chain_length: int,
    barostat_tau_fs: float,
    steps: int,
    equilibration_steps: int,
    output_stride: int,
    restart_stride: int,
    seed: int,
    atom_count: int = 0,
    restart_path: Path | None = None,
    species: tuple[str, ...] = MOF5_SPECIES,
    initial_relaxation: bool = False,
    relaxation_force_tolerance: float = 0.05,
    relaxation_max_iterations: int = 2000,
    relaxation_max_evaluations: int = 20000,
    relaxation_max_displacement: float = 0.05,
) -> None:
    """Write fully flexible LAMMPS NPT with explicit MTTK/NHC controls."""
    species_numbers = " ".join(str(atomic_numbers[symbol]) for symbol in species)
    timestep_ps = timestep_fs / 1000.0
    thermostat_tau_ps = thermostat_tau_fs / 1000.0
    barostat_tau_ps = barostat_tau_fs / 1000.0
    read_command = (
        f"read_restart {restart_path}" if restart_path else f"read_data {data_path}"
    )
    velocity_command = (
        ""
        if restart_path
        else (
            f"velocity all create {temperature_K:g} {seed} mom yes rot yes "
            "dist gaussian\n"
        )
    )
    append = "yes" if restart_path else "no"
    log_command = (
        f"log {thermo_path} append\n" if restart_path else f"log {thermo_path}\n"
    )
    equilibration_command = ""
    relaxation_command = ""
    if initial_relaxation and restart_path is None:
        relaxation_log = input_path.with_suffix(".relaxation.log")
        relaxed_data = input_path.with_suffix(".relaxed.data")
        relaxed_dump = input_path.with_suffix(".relaxed.lammpstrj")
        relaxation_command = (
            f"log {relaxation_log}\n"
            "thermo 100\n"
            "thermo_style custom step pe fnorm fmax vol lx ly lz xy xz yz\n"
            "thermo_modify flush yes\n"
            "min_style fire\n"
            f"min_modify norm inf dmax {relaxation_max_displacement:g}\n"
            f"minimize 0.0 {relaxation_force_tolerance:g} "
            f"{relaxation_max_iterations} {relaxation_max_evaluations}\n"
            f"write_data {relaxed_data}\n"
            f"write_dump all custom {relaxed_dump} id type xu yu zu fx fy fz modify sort id\n"
            "variable relaxation_force equal fmax\n"
            f'if "${{relaxation_force}} > {relaxation_force_tolerance:g}" then '
            '"print \'Initial relaxation did not converge; stopping before NPT\'" "quit 1"\n'
            "variable relaxation_force delete\n"
            "reset_timestep 0\n"
            # FIRE adapts its timestep; restore the configured MD timestep.
            f"timestep {timestep_ps:.12g}\n"
        )
    if equilibration_steps:
        equilibration_command = (
            "variable current_step equal step\n"
            f"if \"${{current_step}} < {equilibration_steps}\" then \"run {equilibration_steps} upto\"\n"
            "variable current_step delete\n"
        )
    input_path.write_text(
        "units metal\n"
        "atom_style atomic\n"
        "boundary p p p\n"
        f"{read_command}\n"
        "change_box all triclinic\n"
        f"pair_style metatomic {model_path} device {device}\n"
        f"pair_coeff * * {species_numbers}\n"
        "neighbor 2.0 bin\n"
        f"neigh_modify delay 0 every 1 check yes one {max(10000, 128 * atom_count)} "
        f"page {max(1000000, 1280 * atom_count)}\n"
        f"timestep {timestep_ps:.12g}\n"
        f"{relaxation_command}"
        f"{velocity_command}"
        f"fix loaded_npt all npt temp {temperature_K:g} {temperature_K:g} "
        f"{thermostat_tau_ps:.12g} tri {pressure_bar:g} {pressure_bar:g} "
        f"{barostat_tau_ps:.12g} tchain {thermostat_chain_length} "
        f"pchain {thermostat_chain_length} mtk yes\n"
        f"thermo {output_stride}\n"
        "thermo_style custom step time temp pe ke etotal press pxx pyy pzz "
        "pxy pxz pyz vol density lx ly lz xy xz yz\n"
        "thermo_modify flush yes\n"
        f"{log_command}"
        f"{equilibration_command}"
        f"dump production all custom {output_stride} {trajectory_path} "
        "id element xu yu zu vx vy vz fx fy fz\n"
        f"dump_modify production element {' '.join(species)} sort id first yes "
        f"append {append}\n"
        f"restart {restart_stride} {restart_prefix}.*\n"
        f"run {steps} upto\n"
        f"write_restart {restart_prefix}.final\n"
        f"write_data {final_data_path}\n"
    )
