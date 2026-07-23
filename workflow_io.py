"""Generate i-PI and LAMMPS inputs for a classical MOF-5 MD smoke test."""

from pathlib import Path
import xml.etree.ElementTree as ET

from ase import Atoms, io

MOF5_SPECIES = ("C", "H", "O", "Zn")


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
    io.write(output_path, structure, format="proteindatabank")


def build_input_xml(structure_path: Path, *, temperature: float, steps: int, timestep_fs: float,
                    output_prefix: Path, output_stride: int, socket_name: str, seed: int) -> str:
    """Build an NVT, one-bead i-PI input for lightweight classical MD."""
    if temperature <= 0.0 or timestep_fs <= 0.0:
        raise ValueError("temperature and timestep_fs must be positive")
    if steps < 1 or output_stride < 1:
        raise ValueError("steps and output_stride must be positive")

    root = ET.Element("simulation", verbosity="medium", safe_stride="100")
    output = ET.SubElement(root, "output", prefix=str(output_prefix))
    properties = ET.SubElement(output, "properties", filename="out", stride=str(output_stride))
    properties.text = "[ step, time{picosecond}, conserved, potential, kinetic_cv, temperature ]"
    trajectory = ET.SubElement(output, "trajectory", filename="pos", stride=str(output_stride), format="pdb")
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
    ET.SubElement(initialize, "velocities", mode="thermal", units="kelvin").text = f"{temperature:g}"
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


def write_lammps_input(input_path: Path, *, data_path: Path, model_path: Path, device: str,
                       socket_name: str, seed: int) -> None:
    """Write the metatomic LAMMPS force-client input for PET-MAD."""
    input_path.write_text(
        "units metal\n"
        "atom_style atomic\n"
        f"read_data {data_path}\n"
        f"pair_style metatomic {model_path} device {device}\n"
        "pair_coeff * * 6 1 8 30\n"
        "neighbor 2.0 bin\n"
        "neigh_modify delay 0 every 1 check yes\n"
        f"fix ipi all ipi {socket_name} {seed} unix\n"
        "run 100000000\n"
    )
