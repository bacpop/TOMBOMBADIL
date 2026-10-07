"""Behavioural checks for PERF-04 optimizer compilation alternatives."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from test.jit_benchmark_variants import make_variant, TRACES

jax.config.update("jax_enable_x64", True)


class TestCompiledMapUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = make_variant("baseline")
        cls.candidate = make_variant("update")

    def test_history_convergence_and_final_evaluation(self):
        for n_iter, convergence in ((0,None),(1,None),(5,None),
                (20,{"enabled":True,"tol":1e9,"patience":2,"check_every":2,"min_steps":4})):
            with self.subTest(n_iter=n_iter):
                p={"x":jnp.array([.3,-.7])}
                fn=lambda p:-jnp.sum((p["x"]-jnp.array([.8,-.2]))**2)
                callbacks=[]
                results=[]
                for module in (self.baseline,self.candidate):
                    records=[]
                    result=module._optimize_params(fn,p,optax.adam(.1),n_iter=n_iter,
                        verbose=False,convergence=convergence,
                        progress_callback=lambda s,n,o:records.append((s,n,o)))
                    results.append(result);callbacks.append(records)
                for field in ("n_steps","converged"):
                    self.assertEqual(results[0][field],results[1][field])
                np.testing.assert_allclose(results[1]["params"]["x"],results[0]["params"]["x"],rtol=1e-6,atol=1e-8)
                np.testing.assert_allclose(results[1]["objective_history"],results[0]["objective_history"],rtol=1e-6,atol=1e-8)
                np.testing.assert_allclose(callbacks[1],callbacks[0],rtol=1e-6,atol=1e-8)
                self.assertEqual(len(callbacks[1]),results[1]["n_steps"]+1)

    def test_replicates_reset_state_and_reuse_kernel(self):
        p={"x":jnp.array([.3,-.7])}
        fn=lambda p:-jnp.sum((p["x"]-jnp.array([.8,-.2]))**2)
        # Identical starts should produce identical results in every replicate;
        # inherited momentum or schedule counters would make them differ.
        original=self.candidate._perturb_params
        try:
            self.candidate._perturb_params=lambda p:dict(p)
            TRACES.clear()
            result=self.candidate._run_map_replicates(fn,p,{"x":"vec"},3,n_iter=5,
                                                      convergence=None,progress=False)
        finally:
            self.candidate._perturb_params=original
        for params in result[0][1:]:
            np.testing.assert_array_equal(params["x"],result[0][0]["x"])
        self.assertEqual(TRACES.get("update"),1)

    def test_new_same_shape_objective_is_not_stale(self):
        p={"x":jnp.array([.3,-.7])}
        results=[]
        for target in (.8,-1.1):
            fn=lambda p: -jnp.sum((p["x"]-target)**2)
            result=self.candidate._optimize_params(fn,p,optax.adam(.1),n_iter=1,
                                                  verbose=False)
            results.append(result["params"]["x"])
        self.assertGreater(float(results[0][0]),float(p["x"][0]))
        self.assertLess(float(results[1][0]),float(p["x"][0]))
