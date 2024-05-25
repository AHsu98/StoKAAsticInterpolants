import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

def rpc_pivot(remaining,d,A,key):
    return jax.random.choice(key,remaining,p = d[remaining]/jnp.sum(d[remaining]))

def pivoted_partial_cholesky(A,k_pivots,pivoting_strategy = rpc_pivot,tol = 1e-5,seed = 203):
    N = len(A)
    k_pivots = jnp.minimum(N,k_pivots)
    d = jnp.diag(A)
    F = jnp.zeros((N,k_pivots))
    remaining_pivots = list(range(N))
    chosen_pivots = []
    trace_history = [jnp.sum(d)]
    key = jax.random.PRNGKey(seed)
    subkeys,_ = jax.random.split(key,k_pivots)

    for i in range(k_pivots):
        pivot = pivoting_strategy(remaining_pivots,d,A,subkeys[i])
        chosen_pivots.append(pivot)
        remaining_pivots.remove(pivot)
        g = A[pivot]
        g = g - F[:,:i]@F[pivot,:i].T
        F[:,i] = g/jnp.sqrt(g[pivot])
        d = d - F[:,i]**2
        d = jnp.maximum(d,0)
        trace_history.append(jnp.sum(d))

    return F,chosen_pivots,d,jnp.array(trace_history)

@jax.jit
def set_to_array(input_set):
    return jnp.array(list(input_set))


@jax.jit
def sample_pivot(key,values,probs):
    return jax.random.choice(key,values,p = probs)


def lazy_pivoted_partial_cholesky(X,k,k_pivots,tol = 1e-5,seed = 203):
    N = len(X)
    all_inds = jnp.arange(N)
    k_pivots = int(jnp.minimum(N,k_pivots))

    k_diag = jax.vmap(k,in_axes=(0,0))
    k_vec = jax.vmap(k,in_axes=(0,None))
    d = k_diag(X,X)
    F = jnp.zeros((N,k_pivots))
    
    pivot_remaining = jnp.ones(N).astype(bool)
    chosen_pivots = []

    trace_history = [jnp.sum(d)]

    key = jax.random.PRNGKey(seed)
    subkeys = jax.random.split(key,k_pivots)

    @jax.jit
    def process_pivot(F,d,pivot,i):
        g = k_vec(X,X[pivot])
        g = g - F@F[pivot]
        #The better way to do this is to iterative grow F (but batched), the cost of slicing is
        #bigger than the cost of just multiplying by the whole thing, since this is jittable
        F = F.at[:,i].set(g/jnp.sqrt(g[pivot]))
        d = d - F[:,i]**2
        d = jnp.maximum(d,0)
        return g,F,d

    #TODO: We can jit everything together in this for loop
    #jax.lax.fori should work here here, assign instead of append
    for i in tqdm(range(k_pivots)):
        probs = d/jnp.sum(d)
        pivot = sample_pivot(subkeys[i],all_inds,probs)
        g,F,d = process_pivot(F,d,pivot,i)
        pivot_remaining = pivot_remaining.at[pivot].set(False)
        chosen_pivots.append(pivot)
        trace_history.append(jnp.sum(d))
