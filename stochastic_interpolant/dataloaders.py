import jax.numpy as jnp
import jax
import ott
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn, solve
import numpy as np
from abc import ABC,abstractmethod

def self_stack(x):
    return jnp.vstack([x,x])

class Sampler(ABC):
    @abstractmethod
    def sample_batch(self,batch_size,key):
        while False:
            yield None

class DatasetSampler(Sampler):
    def __init__(self,data,replace = True):
        self.data = data
    def sample_batch(self,batch_size,key):
        return jax.random.choice(key,self.data,(batch_size,))

class GaussianReferenceSampler(Sampler):
    def __init__(self,shape):
        self.shape = shape
    def sample_batch(self,batch_size,key):
        return jax.random.normal(key=key,shape = (batch_size,) + self.shape)

class IndependenceCouplingSampler(Sampler):
    def __init__(self,ReferenceSampler,TargetSampler):
        self.ReferenceSampler = ReferenceSampler
        self.TargetSampler = TargetSampler
    
    def sample_batch(self,batch_size,key):
        rkey,tkey = jax.random.split(key)
        return self.ReferenceSampler.sample_batch(batch_size,rkey),self.TargetSampler.sample_batch(batch_size,rkey)
    
def build_base_trainloader(batch_size,input_key,couplingSampler):
    key,subkey = jax.random.split(input_key)
    while True:
        key, iter_subkey = jax.random.split(key)
        data_key,normal_key,t_key = jax.random.split(iter_subkey,3)
        ref_batch,target_batch = couplingSampler.sample_batch(batch_size,data_key)
        t_vals = jax.random.uniform(t_key,(batch_size,1))
        z = jax.random.normal(normal_key,ref_batch.shape)
        yield t_vals,ref_batch,target_batch,z

def build_trainloader(batch_size,input_key,couplingSampler,antithetic = True):
    if antithetic is True:
        assert batch_size%2==0
        base_trainloader = build_base_trainloader(batch_size//2,input_key,couplingSampler)
        while True:
            t_vals,ref_batch,target_batch,z = next(base_trainloader)
            yield self_stack(t_vals),self_stack(ref_batch),self_stack(target_batch),jnp.vstack([z,-z])
    else:
        base_trainloader = build_base_trainloader(batch_size,input_key,couplingSampler)
        while True:
            yield next(base_trainloader)

def testloader_factory(batch_size,input_key,couplingSampler,num_batches = 100):
    def get_testloader():
        base_loader = build_base_trainloader(batch_size//2,input_key,couplingSampler)
        for i in range(num_batches):
            t_vals,ref_batch,target_batch,z = next(base_loader)
            yield self_stack(t_vals),self_stack(ref_batch),self_stack(target_batch),jnp.vstack([z,-z])
    return get_testloader

def sample_from_row(key,probs):
    return jax.random.choice(key,jnp.arange(len(probs)),p = probs)

mat_sample = jax.jit(jax.vmap(sample_from_row,in_axes=(0,0)))

def solve_sample_ot(key,ref,target):
    geom = pointcloud.PointCloud(ref, target)
    ot = solve_fn(geom)
    ot_keys = jax.random.split(key,len(ot.matrix))
    return target[mat_sample(ot_keys,ot.matrix)]



class OTCouplingSampler(Sampler):
    def __init__(self, ReferenceSampler, TargetSampler):
        self.ReferenceSampler = ReferenceSampler
        self.TargetSampler = TargetSampler
    
    def sample_batch(self, batch_size, key):
        rkey,tkey, ot_key = jax.random.split(key,3)
        ref_batch = self.ReferenceSampler.sample_batch(batch_size, rkey)
        int_target_batch = self.TargetSampler.sample_batch(batch_size,tkey)

        target_batch = solve_sample_ot(ot_key,ref_batch,int_target_batch)
        return ref_batch, target_batch

solve_fn = jax.jit(solve)

def OT_plan(xs, xt, batch_size):
    "This is the old way we were doing it"
    geom = pointcloud.PointCloud(xs, xt)
    # ot_prob = linear_problem.LinearProblem(geom)
    ot = solve_fn(geom)
    return ot.matrix@xt*batch_size

class OldOTCouplingSampler(Sampler):
    """This oversmooothes the data, wrong way of doing this"""
    def __init__(self, ReferenceSampler, TargetSampler):
        self.ReferenceSampler = ReferenceSampler
        self.TargetSampler = TargetSampler
    
    def sample_batch(self, batch_size, key):
        rkey, _ = jax.random.split(key)
        ref_batch = self.ReferenceSampler.sample_batch(batch_size, rkey)
        int_target_batch = self.TargetSampler.sample_batch(batch_size,rkey)
        target_batch = OT_plan(ref_batch, int_target_batch, batch_size)
        return ref_batch, target_batch
