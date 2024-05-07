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

def eval_k(k,index):
    return k

def diff_k(k,index):
    return grad(k,index)

def diff2_k(k,index):
    return grad(grad(k,index),index)


def get_selected_grad(k,index,selected_index):
    gradf = grad(k,index)
    def selgrad(x1,x2):
        return gradf(x1,x2)[selected_index]
    return selgrad


def dx_k(k,index):
    return get_selected_grad(k,index,1)

def dxx_k(k,index):
    return get_selected_grad(get_selected_grad(k,index,1),index,1)


def dt_k(k,index):
    return get_selected_grad(k,index,0)


from functools import partial

class InducedRKHS():
    """
    Still have to go back and allow for multiple operator sets
        For example, points on boundary only need evaluation, not the rest of the operators if we know boundary conditions
    This only does 1 dimensional output for now. 
    """
    def __init__(
            self,
            basis_points,
            operators,
            kernel_function,
            ) -> None:
        self.basis_points = basis_points
        self.operators = operators
        self.k = kernel_function
        self.get_all_op_kernel_matrix = jit(get_kernel_block_ops(self.k,self.operators,self.operators))
        self.get_eval_op_kernel_matrix = jit(get_kernel_block_ops(self.k,[eval_k],self.operators))
        self.kmat = self.get_all_op_kernel_matrix(self.basis_points,self.basis_points)
        self.num_params = len(basis_points) * len(operators)
    
    @partial(jax.jit, static_argnames=['self'])
    def evaluate_all_ops(self,eval_points,params):
        return self.get_all_op_kernel_matrix(eval_points,self.basis_points)@params
    
    @partial(jax.jit, static_argnames=['self'])
    def point_evaluate(self,eval_points,params):
        return self.get_eval_op_kernel_matrix(eval_points,self.basis_points)@params
    
    def evaluate_operators(self,operators,eval_points,params):
        return get_kernel_block_ops(self.k,operators,self.operators)(eval_points,self.basis_points)@params
    
    def get_fitted_params(self,X,y,lam = 1e-4):
        K = self.get_eval_op_kernel_matrix(X,self.basis_points)
        coeffs = jax.scipy.linalg.solve(K.T@K + lam * (self.kmat+1e-4 * diagpart(self.kmat)),K.T@y)
        return coeffs
    

def analyze_hessian(H,g):
    H_vals,H_vecs = jnp.linalg.eigh(H)
    g_H = H_vecs.T@g
    gh_energy = g_H**2
    plt.figure(figsize = (12,7))
    plt.subplot(2,2,1)
    plt.title("gradient components in positive eig directions")
    plt.scatter(H_vals[H_vals>0],gh_energy[H_vals>0],c='black',s = 6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("lam")
    plt.ylabel("gradient component squared")

    plt.subplot(2,2,2)

    plt.title("gradient components in negative eig directions")
    plt.scatter(-1 * H_vals[H_vals<0],gh_energy[H_vals<0],c='black',s = 6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("-1 * lam")
    plt.ylabel("gradient component squared")
    plt.subplot(2,2,3)

    plt.title("Hessian weighted gradient \n components in positive eig directions")
    keep_inds = H_vals>0
    plt.scatter(H_vals[keep_inds],H_vals[keep_inds] * gh_energy[keep_inds],c='black',s = 6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("lam")
    plt.ylabel("gradient component squared")

    plt.subplot(2,2,4)

    plt.title("Hessian weighted gradient \n components in negative eig directions")
    keep_inds = H_vals<0
    plt.scatter(-1 * H_vals[keep_inds],-1 * H_vals[keep_inds] * gh_energy[keep_inds],c='black',s = 6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("-1 * lam")
    plt.ylabel("gradient component squared")
    plt.tight_layout()
    plt.show()
    print("Most negative eigenvalue ",jnp.min(H_vals))