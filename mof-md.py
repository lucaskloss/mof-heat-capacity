"""Run a short PET-MAD/i-PI/LAMMPS NVT MD simulation for periodic MOF-5."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from lammps_runner import run_lammps_socket
from model_utils import ensure_exported_model
from workflow_io import (build_input_xml, load_mof5_structure, write_lammps_data, write_lammps_input,
                         write_structure_pdb)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "models" / "pet-mad-1.5-s_40nn_nostress.ckpt"
DEFAULT_EXPORTED_MODEL = SCRIPT_DIR / "models" / "pet-mad-1.5-s_40nn_nostress.pt"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True, help="periodic MOF-5 CIF, PDB, or ASE-readable file")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--exported-model", type=Path, default=DEFAULT_EXPORTED_MODEL)
    parser.add_argument("--device", default="cpu", help="metatomic device, e.g. cpu or cuda")
    parser.add_argument("--lammps", default="lmp", help="LAMMPS executable or name on PATH")
    parser.add_argument("--ipi", default="i-pi", help="i-PI executable or name on PATH")
    parser.add_argument("--clients", type=int, default=1, help="number of LAMMPS i-PI force clients")
    parser.add_argument("--temperature", type=float, default=300.0, help="NVT temperature in K")
    parser.add_argument("--steps", type=int, default=10, help="number of i-PI steps; default is a smoke test")
    parser.add_argument("--timestep-fs", type=float, default=0.25, help="integration timestep in fs")
    parser.add_argument("--stride", type=int, default=1, help="properties and position-output stride")
    parser.add_argument("--seed", type=int, default=32342)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--prefix", default="mof5-smoke", help="output filename stem inside --output-dir")
    parser.add_argument("--rerun", action="store_true", help="rerun even if <prefix>.out exists")
    return parser.parse_args()


def main() -> None:
    """Prepare and run a small NVT MD integration check for MOF-5."""
    args = parse_args()
    structure_path = args.structure.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"Loaded {len(structure)} atoms, volume {structure.cell.volume:.3f} A^3 from {structure_path}")
    model_path = ensure_exported_model(args.checkpoint, args.exported_model)
    socket_name = f"mof5-{os.getpid()}-{time.time_ns()}"
    prepared_structure = Path(f"{output_prefix}.pdb")
    xml_file = Path(f"{output_prefix}.xml")
    data_file = Path(f"{output_prefix}.data")
    lammps_input = Path(f"{output_prefix}.lmp")
    write_structure_pdb(prepared_structure, structure)
    xml_file.write_text(
        build_input_xml(prepared_structure, temperature=args.temperature, steps=args.steps,
                        timestep_fs=args.timestep_fs, output_prefix=output_prefix,
                        output_stride=args.stride, socket_name=socket_name, seed=args.seed)
        + "\n"
    )
    write_lammps_data(data_file, structure)
    write_lammps_input(lammps_input, data_path=data_file.resolve(), model_path=model_path, device=args.device,
                       socket_name=socket_name, seed=args.seed)
    print(f"Running {args.steps}-step MOF-5 NVT smoke test with {args.clients} LAMMPS client(s)")
    run_lammps_socket(xml_file, lammps_input, socket_name=socket_name, clients=args.clients,
                      ipi_command=args.ipi, lammps_command=args.lammps)
    print(f"Completed MD. Properties: {output_file}")


if __name__ == "__main__":
    main()
