import jax
from stochastic_interpolant.kernel_transport.KernelTools import vectorize_kfunc
from stochastic_interpolant.kernel_transport.rpc import lazy_pivoted_partial_cholesky
from stochastic_interpolant.loss_functions import loss_pieces
from typing import Callable
import jax.numpy as jnp
from tqdm.auto import tqdm


def kernel_transport_loader(train_loader,I,It,gamma,gammadot):
    get_func_inputs,v_term = loss_pieces(I,It,gamma,gammadot)
    while True:
        t,x,y,z = next(train_loader)
        input_var = get_func_inputs(t,x,y,z)
        v_val = v_term(t,x,y,z)
        yield input_var,v_val

class KernelModel():
    weights: jax.Array
    k: Callable
    kvec: Callable
    kvv: Callable
    num_anchors: int
    anchors: jax.Array
    anchors_fitted: bool
    output_size: int

    def __init__(
            self,
            input_size,
            output_size,
            kernel,
            anchors = None,
            num_anchors = 2500
        ):
        self.num_anchors = num_anchors
        self.anchors_fitted = False
        self.output_size = output_size
        self.weights = jnp.zeros((num_anchors,output_size))            
        self.k = kernel
        self.kvec = jax.vmap(self.k,in_axes=(0,None))
        self.kvv = vectorize_kfunc(self.k)
        if anchors is None:
            self.anchors = jnp.zeros((input_size))
        else:
            self.anchors = anchors
            self.anchors_fitted = True
            self.kmat = self.kvv(self.anchors,self.anchors)

    def get_set_rpc_anchors(self,X,seed):
        F,chosen_anchors,d,trace_history = lazy_pivoted_partial_cholesky(
            X,
            self.k,
            self.num_anchors,
            tol = 1e-6,
            seed = seed
            )
        self.num_anchors = len(chosen_anchors)
        self.anchors = X[chosen_anchors]
        self.anchors_fitted = True
        self.kmat = self.kvv(self.anchors,self.anchors)
        self.weights = self.weights[:self.num_anchors]
        return F,chosen_anchors,d,trace_history
    
    def accumulate_system(
            self,
            Xy_sampler,
            num_batches = 1000,
            ):
        if self.anchors_fitted is False:
            raise("No kernel centers selected yet")
        
        AtA = jnp.zeros((self.num_anchors,self.num_anchors))
        Atb = jnp.zeros((self.num_anchors,self.output_size))
        for i in tqdm(range(num_batches)):
            X,y = next(Xy_sampler)
            Kpart = self.kvv(X,self.anchors)
            AtA += Kpart.T@Kpart/num_batches
            Atb += Kpart.T@y/num_batches
        return AtA,Atb

    def __call__(self, x):
        return jnp.dot(self.kvec(self.anchors,x),self.weights)
