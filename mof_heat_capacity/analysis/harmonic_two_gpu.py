"""Experimental two-device, streamed exact-AD Hessian; separate from harmonic.py."""
from pathlib import Path
import argparse
import inspect
import hashlib
import os
import json
import time
import numpy as np


def sharded_hvp(energy, devices, *, model_position_count=None, finite_difference_step=None):
    """Shard positions and HVP output across devices, replicate model/graph inputs.

    XLA partitions intermediate operations from these boundary constraints.
    This does not guarantee that every intermediate is distributed.
    """
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    if len(devices) != 2:
        raise ValueError('Exactly two devices are required')
    mesh = Mesh(np.asarray(devices), ('atoms',))
    split = NamedSharding(mesh, P('atoms', None))
    replicated = NamedSharding(mesh, P())

    def hvp(params, positions, cell, graph, direction):
        def scalar(p):
            # Padding is only for device placement. PET sees its original
            # array and graph, including its existing sentinel indices.
            model_positions = p if model_position_count is None else p[:model_position_count]
            result = energy(params, model_positions, cell, graph)
            return result[0] if isinstance(result, tuple) else result
        gradient = jax.grad(jax.checkpoint(scalar))
        return jax.jvp(gradient, (positions,), (direction,))[1]

    if finite_difference_step is not None:
        if finite_difference_step <= 0:
            raise ValueError('Finite-difference step must be positive')
        def gradient(params, positions, cell, graph):
            def scalar(p):
                p = p if model_position_count is None else p[:model_position_count]
                result = energy(params, p, cell, graph)
                return result[0] if isinstance(result, tuple) else result
            return jax.grad(jax.checkpoint(scalar))(positions)
        compiled_gradient = jax.jit(gradient,
            in_shardings=(replicated, split, replicated, replicated), out_shardings=split)
        def difference(params, positions, cell, graph, direction):
            # Separate calls keep both reverse-mode tapes from coexisting.
            h = finite_difference_step
            plus = jax.block_until_ready(compiled_gradient(params, positions + h * direction, cell, graph))
            minus = jax.block_until_ready(compiled_gradient(params, positions - h * direction, cell, graph))
            return (plus - minus) / (2 * h)
        return difference, replicated, split

    compiled = jax.jit(hvp,
        in_shardings=(replicated, split, replicated, replicated, split),
        out_shardings=split)
    return compiled, replicated, split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--structure', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New .npy file for the real-atom Hessian (eV/A^2)')
    parser.add_argument('--derivative', choices=('ad', 'finite-difference'), default='ad')
    parser.add_argument('--fd-step', type=float, default=1e-4, help='Explicit finite-difference displacement in A')
    parser.add_argument('--selection-mode', choices=('fixed', 'dynamic'), default='fixed')
    parser.add_argument('--columns', type=int, help='Diagnostic: compute only this many columns')
    parser.add_argument('--llpr-checkpoint', type=Path)
    parser.add_argument('--member', type=int, help='Zero-based LLPR member; omit for central model')
    args = parser.parse_args()
    import jax
    jax.config.update('jax_enable_x64', True)
    from ase.io import read
    from sadmof.models.pet import atoms_to_inputs, get_energy_fn, load_pet
    from ..config import load_run_config
    from .harmonic import ensure_jax_checkpoint, llpr_member_parameter_sets
    from .uncertainty import load_llpr_energy_parameters
    devices = jax.devices('gpu')
    if len(devices) != 2:
        raise RuntimeError(f'Request exactly two visible GPUs, found {devices}')
    if args.output.suffix != '.npy':
        raise ValueError('--output must end in .npy')
    metadata_path = args.output.with_suffix('.json')
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError('Choose a new output path; existing files are preserved')
    config = load_run_config(args.config)
    atoms = read(args.structure, index=-1)
    checkpoint = ensure_jax_checkpoint(config.checkpoint, config.jax_checkpoint)
    model, params, metadata = load_pet(checkpoint, dtype='float64')
    llpr = None
    if args.member is not None:
        if args.llpr_checkpoint is None or args.member < 0:
            raise ValueError('--member requires --llpr-checkpoint and a nonnegative index')
        llpr = load_llpr_energy_parameters(args.llpr_checkpoint)
        members = list(llpr_member_parameter_sets(params, llpr))
        params = members[args.member]
    elif args.llpr_checkpoint is not None:
        raise ValueError('Specify --member when supplying an LLPR checkpoint')
    # Keep raw neighbor/probe selection off the GPUs.
    with jax.default_device(jax.devices('cpu')[0]):
        positions, cell, graph = atoms_to_inputs(atoms, model, metadata, float_dtype=np.float64)
    model_position_count = len(positions)
    transport_padding = model_position_count % 2
    positions = np.pad(np.asarray(jax.device_get(positions)),
                       ((0, transport_padding), (0, 0)))
    # Support the installed legacy SADMOF API and the newer sibling checkout.
    if 'metadata' in inspect.signature(get_energy_fn).parameters:
        energy = get_energy_fn(model, metadata, no_shadow=True)
    else:
        energy = get_energy_fn(model, no_shadow=True)
    if args.selection_mode == 'fixed':
        from .pet_fixed_selection import prepare_fixed_selection
        energy, graph = prepare_fixed_selection(
            model, metadata, positions[:model_position_count], cell, graph, devices)
        # The fixed graph also has a masked extra atom when needed.
        model_position_count = len(positions)
    compute, replicated, split = sharded_hvp(
        energy, devices, model_position_count=model_position_count,
        finite_difference_step=args.fd_step if args.derivative == 'finite-difference' else None)
    params = jax.device_put(params, replicated)
    positions = jax.device_put(positions, split)
    cell = jax.device_put(cell, replicated)
    graph = jax.device_put(graph, replicated)
    size = 3 * len(atoms)
    columns = size if args.columns is None else args.columns
    if not 1 <= columns <= size:
        raise ValueError('--columns must be between 1 and 3*N')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(args.output, mode='w+', dtype=np.float64, shape=(size, columns))
    record = dict(structure_sha256=hashlib.sha256(args.structure.read_bytes()).hexdigest(),
                  llpr_sha256=llpr['sha256'] if llpr else None, xla_flags=os.environ.get('XLA_FLAGS', ''), jax_version=jax.__version__, implementation='experimental-two-device-streamed-hvp',
                  structure=str(args.structure.resolve()), config=str(args.config.resolve()),
                  atom_count=len(atoms), model_position_count=model_position_count,
                  transport_padding=transport_padding, columns_requested=columns, columns_completed=0,
                  complete=False, full_hessian=columns == size, dtype='float64',
                  no_shadow=True, selection_mode=args.selection_mode, derivative=args.derivative,
                  fd_step_A=args.fd_step if args.derivative == 'finite-difference' else None, member=args.member,
                  devices=[str(d) for d in devices])
    metadata_path.write_text(json.dumps(record, indent=2))
    record['column_seconds'] = []
    for column in range(columns):
        started = time.monotonic()
        direction = np.zeros(positions.shape, dtype=np.float64)
        direction.flat[column] = 1
        values = compute(params, positions, cell, graph, jax.device_put(direction, split))
        host = np.asarray(jax.device_get(values))[:len(atoms)].reshape(-1)
        if not np.isfinite(host).all():
            raise FloatingPointError(f'Nonfinite Hessian column {column}')
        matrix[:, column] = host
        matrix.flush()
        record['column_seconds'].append(time.monotonic() - started)
        record['columns_completed'] = column + 1
        metadata_path.write_text(json.dumps(record, indent=2))
        print(f'Column {column + 1}/{columns}', flush=True)
    record['complete'] = True
    record['device_memory_stats'] = [d.memory_stats() for d in devices]
    metadata_path.write_text(json.dumps(record, indent=2, default=str))


if __name__ == '__main__':
    main()
