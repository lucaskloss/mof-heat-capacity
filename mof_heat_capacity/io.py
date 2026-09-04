"""Load structures and generate LAMMPS inputs for the MOF-5 workflow."""

from pathlib import Path

from ase import Atoms, io

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


def write_structure_pdb(output_path: Path, structure: Atoms) -> None:
    """Write a clean ASE-readable periodic PDB structure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip source-format residue/atom-name arrays (e.g. methane H1..H4).
    clean = Atoms(
        symbols=structure.get_chemical_symbols(),
        positions=structure.positions,
        cell=structure.cell,
        pbc=structure.pbc,
    )
    io.write(output_path, clean, format="proteindatabank")
    lines = output_path.read_text().splitlines()
    # Remove ASE's single-frame wrappers for broad PDB-reader compatibility.
    lines = [line for line in lines if line not in {"MODEL     1", "ENDMDL"}]
    if not lines or lines[-1] != "END":
        lines.append("END")
    output_path.write_text("\n".join(lines) + "\n")


def write_lammps_data(output_path: Path, structure: Atoms) -> None:
    """Write topology-free LAMMPS atomic data with a fixed C/H/O/Zn type order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.write(output_path, structure, format="lammps-data", atom_style="atomic", masses=True,
             specorder=MOF5_SPECIES)


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
    restart_path: Path | None = None,
) -> None:
    """Write fully flexible LAMMPS NPT with explicit MTTK/NHC controls."""
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
        "pair_coeff * * 6 1 8 30\n"
        "neighbor 2.0 bin\n"
        "neigh_modify delay 0 every 1 check yes one 10000 page 1000000\n"
        f"timestep {timestep_ps:.12g}\n"
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
        "dump_modify production element C H O Zn sort id first yes "
        f"append {append}\n"
        f"restart {restart_stride} {restart_prefix}.*\n"
        f"run {steps} upto\n"
        f"write_restart {restart_prefix}.final\n"
        f"write_data {final_data_path}\n"
    )
