import jax.numpy as jnp


def rpc_pivot(remaining,d,A,key):
    return jnp.random.choice(remaining,p = d[remaining]/jnp.sum(d[remaining]))

def pivoted_partial_cholesky(A,k,pivoting_strategy = rpc_pivot,tol = 1e-5,seed = 203):
    N = len(A)
    k = jnp.minimum(N,k)
    d = jnp.diag(A)
    F = jnp.zeros((N,k))
    remaining_pivots = list(range(N))
    chosen_pivots = []
    trace_history = [jnp.sum(d)]
    key = jax.random.PRNGKey(seed)
    subkeys,_ = jax.random.split(key,k)

    for i in range(k):
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