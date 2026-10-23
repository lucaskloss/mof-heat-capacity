"""Fixed-geometry PET no-shadow graph preparation for experimental Hessians."""
import numpy as np


def prepare_fixed_selection(model, metadata, positions, cell, graph, devices=None):
    """Select once on CPU; keep cutoffs constant as in PET no_shadow AD.

    Valid only for derivatives at this exact geometry, not geometry optimization
    or MD. Padding adds masked rows without changing any original graph index.
    """
    import jax
    import jax.numpy as jnp
    from petjax.select import _selection, _pack_selected_to_flat, truncate
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    with jax.default_device(jax.devices('cpu')[0]):
        pos = jax.device_put(np.asarray(positions), jax.devices('cpu')[0])
        cell = jax.device_put(np.asarray(cell), jax.devices('cpu')[0])
        structure = {k: jax.device_put(np.asarray(graph[k]), jax.devices('cpu')[0])
                     for k in ('species', 'atom_mask', 'centers', 'others',
                               'cell_shifts', 'reverse', 'pair_mask', 'k_sel_sizer')}
        structure.update(positions=pos, cell=cell)
        config = metadata['config']
        probes = model.get_probes()
        args = (structure, probes, config['cutoff_width'], config['num_neighbors_adaptive'])
        _, _, selected = _selection(*args, no_shadow=True)
        n = len(pos)
        k = len(structure['k_sel_sizer'])
        _, ids, _, overflow = _pack_selected_to_flat(selected, structure['centers'], n, k)
        packed, truncated_overflow = truncate(*args, no_shadow=True)
        if bool(overflow) or bool(truncated_overflow):
            raise ValueError('PET selected-neighbor capacity overflow')
        fixed = {key: np.asarray(value) for key, value in packed.items() if key != 'R_ij'}
        fixed['cell_shifts'] = np.asarray(structure['cell_shifts'][ids])
        if n % 2:
            fixed['species'] = np.concatenate([fixed['species'], fixed['species'][-1:]])
            fixed['atom_mask'] = np.concatenate([fixed['atom_mask'], np.zeros(1, dtype=bool)])
            for key in ('centers', 'neighbors', 'reverse', 'pair_mask', 'pair_cutoffs', 'cell_shifts'):
                # Copy the existing masked sentinel row, maintaining valid indices.
                fixed[key] = np.concatenate([fixed[key], fixed[key][-k:]], axis=0)
            fixed['pair_mask'][-k:] = False
        fixed = {key: jnp.asarray(value) for key, value in fixed.items()}
    edge_sharding = atom_sharding = None
    if devices is not None:
        mesh = Mesh(np.asarray(devices), ('atoms',))
        edge_sharding = NamedSharding(mesh, P('atoms', None, None))
        atom_sharding = NamedSharding(mesh, P('atoms'))

    def energy(params, p, c, g):
        displacement = p[g['neighbors']] - p[g['centers']] + g['cell_shifts'] @ c
        if edge_sharding is not None:
            rows = len(g['species'])
            displacement = jax.lax.with_sharding_constraint(
                displacement.reshape(rows, k, 3), edge_sharding).reshape(-1, 3)
        inputs = {key: value for key, value in g.items() if key != 'cell_shifts'}
        per_atom = model.apply(params, R_ij=displacement, **inputs)
        if atom_sharding is not None:
            per_atom = jax.lax.with_sharding_constraint(per_atom, atom_sharding)
        return jnp.sum(per_atom)

    return energy, fixed
