"""Analytic invariants for PERF-05 distribution and experimental block Adam."""
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from test.parallel_benchmark_variants import Evaluation, VARIANTS, alternating, optimizer_module, baseline_module

jax.config.update("jax_enable_x64", True)


class QuadraticModel:
    """Independent separable oracle with a shared prior and site priors."""
    @staticmethod
    def prepare_likelihood_transforms(X, pi):
        return None, None, None, None

    @staticmethod
    def make_ordered_loss_fn(pi, a, b, c, d, X, mask, **options):
        return lambda p: -(p["omega"]-X[0])**2-(p["alpha"]-X[1])**2

    @staticmethod
    def prior_log_likelihood(p, n, **options):
        if options["prior_mode"]=="none": return jnp.array(0.)
        return -.3*p["alpha"]**2-.2*jnp.sum(p["omega"]**2)/(n if options["aggregate"]=="mean" else 1)

    @classmethod
    def make_log_density_fn(cls, pi, a, b, c, d, X, mask, **options):
        losses=cls.make_ordered_loss_fn(pi,a,b,c,d,X,mask,**options)
        def fn(p):
            w=jnp.ones(X.shape[1]) if options["include_invariant"] else jnp.asarray(mask)
            denom=1 if options["aggregate"]=="sum" else jnp.maximum(jnp.sum(w),1)
            return jnp.sum(losses(p)*w)/denom+cls.prior_log_likelihood(p,X.shape[1],**options)
        return fn


class TestParallelEvaluation(unittest.TestCase):
    def test_scalar_shared_gradient_reduces_once(self):
        X=np.zeros((61,65));X[0]=np.linspace(.1,.9,65);X[1]=.4
        mask=(np.arange(65)%3!=0).astype(float)
        p={"alpha":jnp.array(.7),"omega":jnp.array(.3)}
        opts=dict(omega_mode="scalar",aggregate="mean",include_invariant=False,
            prior_mode="current",estimate_eta=False,eigen_jitter=True,omega_floor=True)
        expected=Evaluation(QuadraticModel,X,None,mask,opts).value_grad(p)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                actual=Evaluation(QuadraticModel,X,None,mask,opts,variant,len(jax.devices())).value_grad(p)
                for x,y in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
                    np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)

    def test_padding_reductions_and_prior_once(self):
        for variant in VARIANTS:
            for n, aggregate, masked in ((1,"sum",False),(63,"mean",False),
                    (65,"sum",True),(257,"mean",True)):
                with self.subTest(variant=variant,n=n):
                    X=np.zeros((61,n));X[0]=np.linspace(.1,.9,n);X[1]=np.linspace(.2,.8,n)
                    mask=np.zeros(n) if n==65 else np.arange(n)%2
                    p={"alpha":jnp.array(.7),"omega":jnp.linspace(.4,.8,n)}
                    options=dict(omega_mode="per-site",aggregate=aggregate,include_invariant=not masked,
                        prior_mode="current",estimate_eta=False,eigen_jitter=True,omega_floor=True)
                    expected=Evaluation(QuadraticModel,X,None,mask,options)
                    actual=Evaluation(QuadraticModel,X,None,mask,options,variant,len(jax.devices()))
                    for x,y in zip(jax.tree.leaves(actual.value_grad(p)),jax.tree.leaves(expected.value_grad(p))):
                        np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(actual.losses(p),expected.losses(p),rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(actual.value(p),expected.value(p),rtol=1e-12,atol=1e-12)

    def test_block_state_freezing_and_schedule_counters(self):
        p={"alpha":jnp.array(.7),"eta":jnp.array(1.),"omega":jnp.array([.3,.4])}
        fn=lambda p: -(p["alpha"]-.1)**2-jnp.sum((p["omega"]-.9)**2)
        e=SimpleNamespace(value=fn,variant="baseline",options={"estimate_eta":False})
        records=[]
        result=alternating(e,p,rounds=(1,5),max_updates=12,
            callback=lambda i,p,s,c: records.append((i,dict(p),tuple(s),c)))
        self.assertEqual(result["block_updates"],[2,10])
        self.assertEqual(result["cycles"],2)
        previous=p
        for index,params,states,counts in records:
            np.testing.assert_array_equal(params["eta"],p["eta"])
            frozen="omega" if index==0 else "alpha"
            np.testing.assert_array_equal(params[frozen],previous[frozen])
            for i,state in enumerate(states):
                self.assertEqual(int(state[0].count),counts[i])
            previous=params
        self.assertGreater(float(result["params"]["omega"][0]),float(p["omega"][0]))

    def test_alternating_parallel_matches_same_serial_schedule(self):
        n=7; X=np.zeros((61,n));X[0]=.9;X[1]=.1
        p={"alpha":jnp.array(.7),"omega":jnp.full(n,.3)}
        opts=dict(omega_mode="per-site",aggregate="sum",include_invariant=True,
            prior_mode="current",estimate_eta=False,eigen_jitter=True,omega_floor=True)
        serial=Evaluation(QuadraticModel,X,None,np.ones(n),opts)
        parallel=Evaluation(QuadraticModel,X,None,np.ones(n),opts,"shard64",len(jax.devices()))
        for rounds in ((1,5),(5,20)):
            with self.subTest(rounds=rounds):
                a=alternating(serial,p,rounds=rounds,max_updates=2*sum(rounds))
                b=alternating(parallel,p,rounds=rounds,max_updates=2*sum(rounds))
                for x,y in zip(jax.tree.leaves((a["params"],a["states"])),jax.tree.leaves((b["params"],b["states"]))):
                    np.testing.assert_allclose(x,y,rtol=1e-12,atol=1e-12)

    def test_joint_optimizer_seam_preserves_python_behaviour(self):
        import optax
        fn=lambda p:-jnp.sum((p["x"]-.8)**2)
        e=SimpleNamespace(value_grad=jax.value_and_grad(fn))
        baseline=baseline_module(); candidate=optimizer_module(e)
        for n in (0,1,5,20):
            with self.subTest(n=n):
                results=[]; callbacks=[]
                for module in (baseline,candidate):
                    records=[]
                    results.append(module._optimize_params(fn,{"x":jnp.array([.3,-.7])},optax.adam(.1),
                        n_iter=n,verbose=False,convergence={"enabled":True,"tol":1e9,"patience":2,"check_every":2,"min_steps":4},
                        progress_callback=lambda *x:records.append(x)))
                    callbacks.append(records)
                self.assertEqual(results[0]["n_steps"],results[1]["n_steps"])
                self.assertEqual(results[0]["converged"],results[1]["converged"])
                np.testing.assert_allclose(results[0]["objective_history"],results[1]["objective_history"],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(callbacks[0],callbacks[1],rtol=1e-12,atol=1e-12)


if __name__=="__main__":
    unittest.main()
