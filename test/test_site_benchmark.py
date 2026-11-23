"""Independent ordering and derivative checks for benchmark-only site maps."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from test.site_benchmark_variants import site_mapper

jax.config.update("jax_enable_x64", True)


class TestSiteBatching(unittest.TestCase):
    def test_order_remainders_and_shared_gradients(self):
        # An analytic model makes incorrect axis pairing or shared-gradient
        # summation visible without depending on the codon model's eigensolver.
        def model(a, b, c, d, e, eta, theta, omega, pi, lp, pm, pmi, pmul, counts):
            return a * omega ** 2 + theta * jnp.dot(counts, jnp.arange(1, 4))

        for name in ("sequential", "batch16", "checkpoint16"):
            for n in (1, 15, 16, 17, 35):
                with self.subTest(variant=name, n=n):
                    omega = jnp.linspace(0.1, 1.2, n)
                    counts = jnp.arange(3*n).reshape(3, n)
                    mapper = site_mapper(model, name, "per-site", n, force=True)
                    def losses(a, w, theta):
                        return mapper(a, 1., 1., 1., 1., 1., theta, w,
                                      None, None, None, None, None, counts)
                    expected = 2 * omega**2 + 0.7 * (np.arange(1, 4) @ counts)
                    np.testing.assert_allclose(losses(2., omega, 0.7), expected)
                    grads = jax.grad(lambda a,w,t: jnp.sum(losses(a,w,t)), argnums=(0,1,2))(2.,omega,0.7)
                    np.testing.assert_allclose(grads[0], jnp.sum(omega**2))
                    np.testing.assert_allclose(grads[1], 4*omega)
                    np.testing.assert_allclose(grads[2], jnp.sum(np.arange(1,4) @ counts))

    def test_scalar_and_small_policy(self):
        def model(*args):
            return args[0]*args[7] + jnp.sum(args[-1])
        for mode,n in (("scalar",513),("per-site",512),("per-site",513)):
            with self.subTest(mode=mode,n=n):
                omega = 0.3 if mode == "scalar" else jnp.linspace(0.1,1,n)
                args = (2.,1.,1.,1.,1.,1.,1.,omega,None,None,None,None,None,jnp.ones((3,n)))
                expected = 2*omega + jnp.full(n,3.)
                np.testing.assert_allclose(site_mapper(model,"batch16",mode,n)(*args),expected)
