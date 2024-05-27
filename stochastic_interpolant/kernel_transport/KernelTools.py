import jax.numpy as jnp
from jax import jit,grad,hessian,jacfwd
import numpy as np
import matplotlib.pyplot as plt
import jax
from tqdm.auto import tqdm


def get_gaussianRBF(gamma):
    def f(x,y):
        return jnp.exp(-jnp.sum((x-y)**2)/(2*gamma**2))
    return f

def get_anisotropic_gaussianRBF(gamma,A):
    def f(x,y):
        diff = x-y
        return jnp.exp(-jnp.dot(diff,A@diff)/gamma)
    return f


def get_matern_five_half(rho):
    random_constant = 1.2345678e-100
    def k(x,y):
        d=jnp.sqrt(jnp.sum((x-y+random_constant)**2))
        return (1+jnp.sqrt(5)*d/rho+jnp.sqrt(5)*d**2/(rho**2))*jnp.exp(-jnp.sqrt(5)*d/rho)
    return k

def linear_kernel(x,y):
    return jnp.dot(x,y)

def get_poly_kernel(deg,c=1):
    def k(x,y):
        return (jnp.dot(x,y)+c)**deg
    return k

def get_anisotropic_poly_kernel(deg,A,c=1,shift = 0.):
    def k(x,y):
        return (jnp.dot(x-shift,A@(y-shift))+c)**deg
    return k

def get_shift_scale(X,scaling = 'diagonal',eps = 1e-7):
    cov = jnp.cov(X.T)
    if scaling == 'diagonal':
        A = jnp.linalg.inv(diagpart(cov))
    elif scaling == 'full':
        A = jnp.linalg.inv(cov + eps * jnp.eye(len(cov)))
    else:
        raise NotImplementedError("Only diagonal and full scaling are available")
    shift = jnp.mean(X,axis=0)
    return shift,A


def get_centered_scaled_poly_kernel(deg,X_train,c = 1,scaling = 'diagonal',eps = 1e-7):
    shift,A = get_shift_scale(X_train,scaling,eps)
    return get_anisotropic_poly_kernel(deg,A,c,shift=shift)

def get_centered_scaled_linear_kernel(X_train,c = 1,scaling = 'diagonal',eps = 1e-7):
    shift,A = get_shift_scale(X_train,scaling,eps)
    def k(x,y):
        return jnp.dot(x-shift,A@(y-shift)) + c
    return k

def get_sum_of_kernels(kernels,coefficients):
    def k(x,y):
        return jnp.sum(jnp.array([c * ki(x,y) for c,ki in zip(coefficients,kernels)]))
    return k

def diagpart(M):
    return jnp.diag(jnp.diag(M))

def vectorize_kfunc(k):
    return jax.vmap(jax.vmap(k, in_axes=(None,0)), in_axes=(0,None))

def op_k_apply(k,L_op,R_op):
    return R_op(L_op(k,0),1)

def make_block(k,L_op,R_op):
    return vectorize_kfunc(op_k_apply(k,L_op,R_op))

def get_kernel_block_ops(k,ops_left,ops_right,output_dim=1):
    def k_super(t_left,t_right):
        I = jnp.eye(output_dim)
        blocks = (
            [
                [jnp.kron(make_block(k,L_op,R_op)(t_left,t_right),I) for R_op in ops_right]
                for L_op in ops_left
            ]
        )
        return jnp.block(blocks)
    return k_super