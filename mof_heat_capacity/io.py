"""Generate structure, LAMMPS, and i-PI inputs for the MOF-5 workflows."""

from pathlib import Path
import xml.etree.ElementTree as ET

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
    """Write the ASE-readable periodic structure in i-PI's PDB input format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip source-format residue/atom-name arrays (e.g. methane H1..H4).
    # i-PI interprets the PDB atom-name field as an element and cannot assign
    # masses to those residue-specific labels.
    clean = Atoms(
        symbols=structure.get_chemical_symbols(),
        positions=structure.positions,
        cell=structure.cell,
        pbc=structure.pbc,
    )
    io.write(output_path, clean, format="proteindatabank")
    lines = output_path.read_text().splitlines()
    # i-PI's PDB reader expects CRYST1 followed directly by ATOM records and
    # recognizes END as the only terminator (ASE emits MODEL/ENDMDL).
    lines = [line for line in lines if line not in {"MODEL     1", "ENDMDL"}]
    if not lines or lines[-1] != "END":
        lines.append("END")
    output_path.write_text("\n".join(lines) + "\n")


def build_ipi_pimd_xml(
    structure_path: Path,
    *,
    temperature: float,
    pressure_bar: float,
    steps: int,
    timestep_fs: float,
    output_prefix: Path,
    output_stride: int,
    restart_stride: int,
    socket_name: str,
    seed: int,
    beads: int,
    force_beads: int,
    thermostat_tau_fs: float,
    barostat_tau_fs: float,
    cell_thermostat_tau_fs: float,
    splitting: str,
    fd_epsilon_bohr: float,
) -> str:
    """Build the paper's 64-bead Suzuki-Chin isotropic-NPT protocol."""
    if temperature <= 0.0 or pressure_bar <= 0.0 or timestep_fs <= 0.0:
        raise ValueError("temperature, pressure, and timestep must be positive")
    if steps < 1 or output_stride < 1 or restart_stride < 1:
        raise ValueError("steps and output/restart strides must be positive")
    if beads < 2 or beads % 2 or force_beads != beads:
        raise ValueError("Suzuki-Chin with one MLIP requires all even-numbered beads")
    if splitting != "baoab":
        raise ValueError("the paper protocol uses BAOAB splitting")

    root = ET.Element(
        "simulation", verbosity="medium", safe_stride=str(restart_stride)
    )
    output = ET.SubElement(root, "output", prefix=str(output_prefix))
    properties = ET.SubElement(
        output, "properties", filename="properties", stride=str(output_stride)
    )
    properties.text = (
        "[ step, time{picosecond}, conserved{electronvolt}, "
        "potential_tdsc{electronvolt}, kinetic_tdsc{electronvolt}, "
        "temperature{kelvin}, pressure_tdsc{bar}, density{g/cm3}, "
        "volume{angstrom3}, cell_h{angstrom} ]"
    )
    trajectory = ET.SubElement(
        output,
        "trajectory",
        filename="centroid",
        stride=str(output_stride),
        format="ase",
        cell_units="angstrom",
    )
    trajectory.text = "x_centroid{angstrom}"
    ET.SubElement(
        output,
        "checkpoint",
        filename="restart",
        stride=str(restart_stride),
        overwrite="true",
    )
    ET.SubElement(root, "total_steps").text = str(steps)
    prng = ET.SubElement(root, "prng")
    ET.SubElement(prng, "seed").text = str(seed)
    forcefield = ET.SubElement(root, "ffsocket", name="lmp", mode="unix", pbc="true")
    ET.SubElement(forcefield, "address").text = socket_name
    ET.SubElement(forcefield, "latency").text = "1e-4"
    system = ET.SubElement(root, "system")
    initialize = ET.SubElement(system, "initialize", nbeads=str(beads))
    structure = ET.SubElement(initialize, "file", mode="pdb", units="angstrom")
    structure.text = str(structure_path)
    ET.SubElement(initialize, "velocities", mode="thermal", units="kelvin").text = (
        f"{temperature:g}"
    )
    forces = ET.SubElement(system, "forces")
    ET.SubElement(
        forces,
        "force",
        forcefield="lmp",
        nbeads=str(force_beads),
        fd_epsilon=f"{fd_epsilon_bohr:g}",
        name="mlip",
    ).text = "lmp"
    ET.SubElement(system, "normal_modes", propagator="exact")
    ensemble = ET.SubElement(system, "ensemble")
    ET.SubElement(ensemble, "temperature", units="kelvin").text = f"{temperature:g}"
    ET.SubElement(ensemble, "pressure", units="bar").text = f"{pressure_bar:g}"
    motion = ET.SubElement(system, "motion", mode="dynamics")
    dynamics = ET.SubElement(
        motion, "dynamics", mode="scnpt", splitting=splitting
    )
    thermostat = ET.SubElement(dynamics, "thermostat", mode="pile_l")
    ET.SubElement(thermostat, "tau", units="femtosecond").text = (
        f"{thermostat_tau_fs:g}"
    )
    ET.SubElement(thermostat, "pile_lambda").text = "1.0"
    barostat = ET.SubElement(dynamics, "barostat", mode="sc-isotropic")
    cell_thermostat = ET.SubElement(barostat, "thermostat", mode="langevin")
    ET.SubElement(cell_thermostat, "tau", units="femtosecond").text = (
        f"{cell_thermostat_tau_fs:g}"
    )
    ET.SubElement(barostat, "tau", units="femtosecond").text = (
        f"{barostat_tau_fs:g}"
    )
    ET.SubElement(dynamics, "timestep", units="femtosecond").text = f"{timestep_fs:g}"
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def build_input_xml(
    structure_path: Path,
    *,
    temperature: float,
    steps: int,
    timestep_fs: float,
    output_prefix: Path,
    output_stride: int,
    socket_name: str,
    seed: int,
) -> str:
    """Build the legacy one-bead NVT smoke-test input."""
    if temperature <= 0.0 or timestep_fs <= 0.0:
        raise ValueError("temperature and timestep_fs must be positive")
    if steps < 1 or output_stride < 1:
        raise ValueError("steps and output_stride must be positive")

    root = ET.Element("simulation", verbosity="medium", safe_stride="100")
    output = ET.SubElement(root, "output", prefix=str(output_prefix))
    properties = ET.SubElement(
        output, "properties", filename="out", stride=str(output_stride)
    )
    properties.text = (
        "[ step, time{picosecond}, conserved, potential, kinetic_cv, temperature ]"
    )
    trajectory = ET.SubElement(
        output, "trajectory", filename="pos", stride=str(output_stride), format="pdb"
    )
    trajectory.text = "positions{angstrom}"
    ET.SubElement(root, "total_steps").text = str(steps)
    prng = ET.SubElement(root, "prng")
    ET.SubElement(prng, "seed").text = str(seed)
    forcefield = ET.SubElement(root, "ffsocket", name="lmp", mode="unix", pbc="true")
    ET.SubElement(forcefield, "address").text = socket_name
    ET.SubElement(forcefield, "latency").text = "1e-4"
    system = ET.SubElement(root, "system")
    initialize = ET.SubElement(system, "initialize", nbeads="1")
    structure = ET.SubElement(initialize, "file", mode="pdb", units="angstrom")
    structure.text = str(structure_path)
    ET.SubElement(initialize, "velocities", mode="thermal", units="kelvin").text = (
        f"{temperature:g}"
    )
    forces = ET.SubElement(system, "forces")
    ET.SubElement(forces, "force", forcefield="lmp").text = "lmp"
    ensemble = ET.SubElement(system, "ensemble")
    ET.SubElement(ensemble, "temperature", units="kelvin").text = f"{temperature:g}"
    motion = ET.SubElement(system, "motion", mode="dynamics")
    dynamics = ET.SubElement(motion, "dynamics", mode="nvt")
    thermostat = ET.SubElement(dynamics, "thermostat", mode="langevin")
    ET.SubElement(thermostat, "tau", units="femtosecond").text = "100.0"
    ET.SubElement(dynamics, "timestep", units="femtosecond").text = f"{timestep_fs:g}"
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def write_lammps_data(output_path: Path, structure: Atoms) -> None:
    """Write topology-free LAMMPS atomic data with a fixed C/H/O/Zn type order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.write(output_path, structure, format="lammps-data", atom_style="atomic", masses=True,
             specorder=MOF5_SPECIES)


def write_lammps_input(
    input_path: Path,
    *,
    data_path: Path,
    model_path: Path,
    device: str,
    socket_name: str,
    seed: int,
) -> None:
    """Write the metatomic LAMMPS force-client input for PET-MAD."""
    input_path.write_text(
        "units metal\n"
        "atom_style atomic\n"
        f"read_data {data_path}\n"
        f"pair_style metatomic {model_path} device {device}\n"
        "pair_coeff * * 6 1 8 30\n"
        "neighbor 2.0 bin\n"
        "neigh_modify delay 0 every 1 check yes one 10000 page 1000000\n"
        f"fix ipi all ipi {socket_name} {seed} unix\n"
        "run 100000000\n"
    )


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
        f"fix paper_npt all npt temp {temperature_K:g} {temperature_K:g} "
        f"{thermostat_tau_ps:.12g} tri {pressure_bar:g} {pressure_bar:g} "
        f"{barostat_tau_ps:.12g} tchain {thermostat_chain_length} "
        f"pchain {thermostat_chain_length} mtk yes\n"
        f"thermo {output_stride}\n"
        "thermo_style custom step time temp pe ke etotal press pxx pyy pzz "
        "pxy pxz pyz vol density lx ly lz xy xz yz\n"
        "thermo_modify flush yes\n"
        f"{log_command}"
        f"dump production all custom {output_stride} {trajectory_path} "
        "id element xu yu zu vx vy vz fx fy fz\n"
        "dump_modify production element C H O Zn sort id first yes "
        f"append {append}\n"
        f"restart {restart_stride} {restart_prefix}.*\n"
        f"run {steps} upto\n"
        f"write_restart {restart_prefix}.final\n"
        f"write_data {final_data_path}\n"
    )
