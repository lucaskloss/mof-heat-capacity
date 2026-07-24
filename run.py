"""Run a short PET-MAD/i-PI/LAMMPS NVT MD simulation for periodic MOF-5."""

from __future__ import annotations

import argparse
from pathlib import Path

from lammps_runner import run_lammps_socket
from model_utils import ensure_exported_model
from workflow_io import (
    build_input_xml,
    load_mof5_structure,
    write_lammps_data,
    write_lammps_input,
    write_structure_pdb,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "models" / "pet-mad-1.5-s_40nn_nostress.ckpt"
DEFAULT_EXPORTED_MODEL = SCRIPT_DIR / "models" / "pet-mad-1.5-s_40nn_nostress.pt"
DEFAULT_STRUCTURE = SCRIPT_DIR / "data" / "mof5.cif"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, default=DEFAULT_STRUCTURE,
                        help="periodic MOF-5 CIF, PDB, or ASE-readable file")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--exported-model", type=Path, default=DEFAULT_EXPORTED_MODEL)
    parser.add_argument(
        "--device", default="cuda", help="metatomic device, e.g. cpu or cuda"
    )
    parser.add_argument("--lammps", default="lmp", help="LAMMPS executable or name on PATH")
    parser.add_argument("--ipi", default="i-pi", help="i-PI executable or name on PATH")
    parser.add_argument("--clients", type=int, default=1, help="number of LAMMPS i-PI force clients")
    parser.add_argument("--temperature", type=float, default=300.0, help="NVT temperature in K")
    parser.add_argument(
        "--steps", type=int, default=10, help="number of i-PI steps; default is a smoke test"
    )
    parser.add_argument("--timestep-fs", type=float, default=0.25, help="integration timestep in fs")
    parser.add_argument("--stride", type=int, default=1, help="properties and position-output stride")
    parser.add_argument("--seed", type=int, default=32342)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR / "data",
                        help="directory for generated PDB, XML, and LAMMPS input files")
    parser.add_argument("--prepare-inputs", action="store_true",
                        help="regenerate the four input files in --input-dir before running")
    parser.add_argument(
        "--prefix", default="mof5-smoke", help="output filename stem inside --output-dir"
    )
    parser.add_argument("--rerun", action="store_true", help="rerun even if <prefix>.out exists")
    return parser.parse_args()


def main() -> None:
    """Prepare and run a small NVT MD integration check for MOF-5."""
    args = parse_args()
    structure_path = args.structure.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    prefix_name = Path(args.prefix).name
    if not prefix_name or prefix_name in {".", ".."}:
        raise ValueError("--prefix must contain a valid output filename stem")
    if not structure_path.is_file():
        raise FileNotFoundError(f"MOF-5 structure not found: {structure_path}")

    output_prefix = output_dir / prefix_name
    output_file = Path(f"{output_prefix}.out")
    if output_file.exists() and not args.rerun:
        print(f"Reusing existing output: {output_file}")
        return

    structure = load_mof5_structure(structure_path)
    print(
        f"Loaded {len(structure)} atoms, volume {structure.cell.volume:.3f} A^3 "
        f"from {structure_path}"
    )
    model_path = ensure_exported_model(args.checkpoint, args.exported_model)
    socket_name = "mof5-static"
    prepared_structure = input_dir / "mof5.pdb"
    xml_file = input_dir / "input.xml"
    data_file = input_dir / "mof5.data"
    lammps_input = input_dir / "in.lmp"
    input_files = (prepared_structure, xml_file, data_file, lammps_input)
    if args.prepare_inputs:
        write_structure_pdb(prepared_structure, structure)
        xml_file.write_text(
            build_input_xml(
                prepared_structure,
                temperature=args.temperature,
                steps=args.steps,
                timestep_fs=args.timestep_fs,
                output_prefix=output_prefix,
                output_stride=args.stride,
                socket_name=socket_name,
                seed=args.seed,
            )
            + "\n"
        )
        write_lammps_data(data_file, structure)
        write_lammps_input(
            lammps_input,
            data_path=data_file.resolve(),
            model_path=model_path,
            device=args.device,
            socket_name=socket_name,
            seed=args.seed,
        )
    elif not all(path.is_file() for path in input_files):
        missing = ", ".join(str(path) for path in input_files if not path.is_file())
        raise FileNotFoundError(f"Missing input file(s): {missing}. Run with --prepare-inputs first")
    # i-PI 3.3 backs up the properties file at startup even for a new run.
    output_file.touch(exist_ok=True)
    print(f"Running {args.steps}-step MOF-5 NVT smoke test with {args.clients} LAMMPS client(s)")
    run_lammps_socket(
        xml_file,
        lammps_input,
        socket_name=socket_name,
        clients=args.clients,
        ipi_command=args.ipi,
        lammps_command=args.lammps,
        runtime_dir=output_dir,
    )
    print(f"Completed MD. Properties: {output_file}")


if __name__ == "__main__":
    main()
