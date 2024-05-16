import jax.numpy as jnp
import jax
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
