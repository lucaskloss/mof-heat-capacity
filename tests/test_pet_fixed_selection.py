"""Opt-in actual PET regression: MOF_TEST_PET_CHECKPOINT=/path/to/PET-JAX."""
import os
import unittest
import numpy as np


@unittest.skipUnless(os.environ.get('MOF_TEST_PET_CHECKPOINT'), 'requires PET-JAX checkpoint')
class FixedSelectionPETTest(unittest.TestCase):
    def test_energy_gradient_and_hvp_match_original_no_shadow(self):
        import jax
        import jax.numpy as jnp
        from ase import Atoms
        from sadmof.models.pet import load_pet, atoms_to_inputs, get_energy_fn
        from mof_heat_capacity.analysis.pet_fixed_selection import prepare_fixed_selection
        jax.config.update('jax_enable_x64', True)
        with jax.default_device(jax.devices('cpu')[0]):
            model, params, metadata = load_pet(os.environ['MOF_TEST_PET_CHECKPOINT'], dtype='float64')
            atoms = Atoms('CO2', positions=[[0,0,0], [1.17,0,0], [-1.17,0,0]],
                          cell=[15,15,15], pbc=True)
            positions, cell, graph = atoms_to_inputs(atoms, model, metadata)
            original = get_energy_fn(model, metadata, no_shadow=True)
            fixed, packed = prepare_fixed_selection(model, metadata, positions, cell, graph)
            n = len(positions)
            padded = jnp.pad(positions, ((0, n % 2), (0,0)))
            reference = lambda p: original(params, p[:n], cell, graph)[0]
            candidate = lambda p: fixed(params, p, cell, packed)
            direction = jnp.asarray(np.random.default_rng(42).normal(size=padded.shape))
            for fn in (lambda f: f(padded),
                       lambda f: jax.grad(f)(padded),
                       lambda f: jax.jvp(jax.grad(f), (padded,), (direction,))[1]):
                expected = np.asarray(jax.jit(lambda p: fn(reference))(padded))
                actual = np.asarray(jax.jit(lambda p: fn(candidate))(padded))
                np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
