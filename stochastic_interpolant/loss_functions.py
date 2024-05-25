import jax
from jax import grad,vmap
import jax.numpy as jnp

def get_trig_interpolants():
    def I(t,x,y):
        return jnp.cos(jnp.pi*t/2)*x + jnp.sin(jnp.pi*t/2)*y

    def It(t,x,y):
        return -0.5*jnp.pi*jnp.sin(0.5*jnp.pi*t)*x + 0.5*jnp.pi*jnp.cos(0.5*jnp.pi*t)*y

    return I,It

def get_linear_interpolants():
    def I(t,x,y):
        return (1-t)*x + t*y

    def It(t,x,y):
        return y-x 
    return I,It

def vectorized_gamma_grad(gamma_func_single):
    return vmap(vmap(grad(gamma_func_single)))

def root_prod_gamma(t):
    return jnp.sqrt(2*t*(1-t)+1e-8)

root_prod_gammadot = vectorized_gamma_grad(root_prod_gamma)

def zero_gamma(t):
    return 0*t
def zero_gammadot(t):
    return 0*t

def loss_pieces(I,It,gamma,gammadot):
    def input_values(t,x,y,z):
        return jnp.hstack([t.reshape(-1,1),I(t,x,y)+gamma(t)*z])
    def v_term(t,x,y,z):
        return (It(t,x,y) + gammadot(t)*z)
    return input_values,v_term

def get_loss_functions(I,It,gamma,gammadot):
    get_func_inputs,v_term = loss_pieces(I,It,gamma,gammadot)

    def loss_b(b_model, t, x, y, z):
        tI = get_func_inputs(t,x,y,z)
        v = v_term(t,x,y,z)
        bhat = jax.vmap(b_model)(tI)  # vectorise the model over a batch of data
        bnorm2 = jnp.mean(jnp.sum(bhat**2,axis=1))
        dot_term = jnp.mean(jnp.sum(v*bhat,axis=1))
        return bnorm2 - 2*dot_term

    def loss_denoise(eta_model,t,x,y,z):
        tI = get_func_inputs(t,x,y,z)
        etahat = jax.vmap(eta_model)(tI)  # vectorise the model over a batch of data
        etanorm2 = jnp.mean(jnp.sum(etahat**2,axis=1))
        dot_term = jnp.mean(jnp.sum(etahat*z,axis=1))
        return etanorm2 - 2*dot_term
    return loss_b,loss_denoise