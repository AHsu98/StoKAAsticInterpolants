from diffrax import (
    diffeqsolve, ODETerm, Dopri5, ControlTerm, 
    Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree, Heun,
    WeaklyDiagonalControlTerm,
)
import jax
import equinox as eqx
import jax.numpy as jnp


class StochasticSampler():
    def __init__(
        self,
        gamma,
        b_model,
        eta_model,
        base_epsilon_func = lambda t: t*(1-t)
    ):
        self.vmapped_b_model = jax.vmap(b_model)
        self.vmapped_eta_model = jax.vmap(eta_model)
        self.base_epsilon_func = base_epsilon_func
        self.gamma = gamma
        
    def sample_trajectory(
        self,
        X0,
        eps = 0.1,
        solver = Heun(),
        saveat = SaveAt(dense=True),
        dt0 = 1e-2,
        max_steps = 20000
    ):
        def epsilon(t):
            return self.base_epsilon_func(t)*eps

        @eqx.filter_jit
        def dX_t(t,x,args):
            t_vec = jnp.ones((len(x),1))*t
            return self.vmapped_b_model(jnp.hstack([t_vec,x])) - (epsilon(t)/(self.gamma(t)+1e-12))*self.vmapped_eta_model(jnp.hstack([t_vec,x]))

        t0, t1 = 0.0, 1.0
        diffusion = lambda t, x, args: jnp.sqrt(2*epsilon(t))
        brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-6, shape=X0.shape, key=jax.random.PRNGKey(103))
        terms = MultiTerm(ODETerm(dX_t), WeaklyDiagonalControlTerm(diffusion, brownian_motion))
        
        sol = diffeqsolve(terms, solver, t0, t1, dt0=dt0, y0=X0, saveat=saveat,max_steps = max_steps)
        X = sol.evaluate(1.0)
        return X,sol