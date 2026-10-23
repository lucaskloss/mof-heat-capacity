"""Run in a fresh process with XLA_FLAGS=--xla_force_host_platform_device_count=2."""
import unittest
import numpy as np

class TwoDeviceHessianTest(unittest.TestCase):
    def test_coupled_quadratic(self):
        import jax
        import jax.numpy as jnp
        from mof_heat_capacity.analysis.harmonic_two_gpu import sharded_hvp
        jax.config.update('jax_enable_x64', True)
        devices = jax.devices('cpu')
        if len(devices) < 2:
            self.skipTest('requires two virtual CPU devices')
        def energy(params, positions, cell, graph):
            x = positions.reshape(-1)
            return 0.5 * x @ params @ x
        rng = np.random.default_rng(4)
        a = rng.normal(size=(12,12))
        a = a.T @ a
        f, replicated, split = sharded_hvp(energy, devices[:2])
        p = jax.device_put(a, replicated)
        x = jax.device_put(np.ones((4,3)), split)
        cell = jax.device_put(np.eye(3), replicated)
        columns = []
        for v in np.eye(12):
            result = f(p, x, cell, {}, jax.device_put(v.reshape(4,3), split))
            self.assertEqual(len(result.addressable_shards), 2)
            columns.append(np.asarray(result).reshape(-1))
        np.testing.assert_allclose(np.array(columns).T, a, atol=1e-11)

    def test_odd_model_positions_preserve_graph_and_ignore_transport_row(self):
        import jax
        from mof_heat_capacity.analysis.harmonic_two_gpu import sharded_hvp
        jax.config.update('jax_enable_x64', True)
        devices = jax.devices('cpu')
        if len(devices) < 2:
            self.skipTest('requires two virtual CPU devices')
        # Five model positions, including a sentinel row referenced by graph.
        def energy(params, positions, cell, graph):
            assert positions.shape == (5, 3)
            x = positions[graph['indices']].reshape(-1)
            return 0.5 * x @ params @ x
        rng = np.random.default_rng(8)
        a = rng.normal(size=(15, 15))
        a = a.T @ a
        f, replicated, split = sharded_hvp(energy, devices[:2], model_position_count=5)
        params = jax.device_put(a, replicated)
        x = jax.device_put(np.ones((6, 3)), split)
        cell = jax.device_put(np.eye(3), replicated)
        graph = jax.device_put({'indices': np.arange(5)}, replicated)
        columns = []
        for v in np.eye(18):
            result = f(params, x, cell, graph, jax.device_put(v.reshape(6, 3), split))
            self.assertEqual(len(result.addressable_shards), 2)
            columns.append(np.asarray(result).reshape(-1))
        expected = np.zeros((18, 18))
        expected[:15, :15] = a
        np.testing.assert_allclose(np.array(columns).T, expected, atol=1e-11)

    def test_finite_difference_fallback(self):
        import jax
        import jax.numpy as jnp
        from mof_heat_capacity.analysis.harmonic_two_gpu import sharded_hvp
        jax.config.update('jax_enable_x64', True)
        devices = jax.devices('cpu')
        if len(devices) < 2:
            self.skipTest('requires two virtual CPU devices')
        def energy(params, p, cell, graph):
            return jnp.sum(p**2) + 0.25 * jnp.sum(p)**2
        f, replicated, split = sharded_hvp(energy, devices[:2], finite_difference_step=1e-4)
        p = jax.device_put(np.ones((4,3)), split)
        v = np.random.default_rng(5).normal(size=(4,3))
        actual = np.asarray(f({}, p, jax.device_put(np.eye(3), replicated), {}, jax.device_put(v, split)))
        np.testing.assert_allclose(actual, 2*v + 0.5*v.sum(), atol=1e-9)
