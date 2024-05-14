import jax.numpy as jnp
import jax
from abc import ABC


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
        return jax.random.choice(ref_key,self.data,(batch_size,))

class GaussianReferenceSampler(Sampler):
    def __init__(self,shape):
        self.shape = shape
    def sample_batch(self,batch_size,key):
        return jax.random.normal(key=key,shape = (batch_size,) + self.shape)


def build_base_trainloader(batch_size,input_key,reference_sampler,target_sampler):
    key,subkey = jax.random.split(input_key)
    while True:
        key, iter_subkey = jax.random.split(key)
        ref_key,target_key,normal_key,t_key = jax.random.split(iter_subkey,4)

        ref_batch = reference_sampler.sample(batch_size,key = ref_key)
        target_batch = target_sampler.sample(batch_size,key = ref_key)
        t_vals = jax.random.uniform(t_key,(batch_size,1))
        z = jax.random.normal(normal_key,ref_batch.shape)
        yield t_vals,ref_batch,target_batch,z

def build_trainloader(batch_size,input_key,reference_sampler,target_sampler,antithetic = True):
    if antithetic is True:
        assert batch_size%2==0
        base_trainloader = build_base_trainloader(batch_size//2,input_key,reference_sampler,target_sampler)
        while True:
            t_vals,ref_batch,target_batch,z = next(base_trainloader)
            yield self_stack(t_vals),self_stack(ref_batch),self_stack(target_batch),jnp.vstack([z,-z])
    else:
        base_trainloader = build_base_trainloader(batch_size,input_key,reference_sampler,target_sampler)
        while True:
            yield next(base_trainloader)

def testloader_factory(batch_size,input_key,reference_sampler,target_sampler,num_batches = 100):
    def get_testloader():
        base_loader = build_base_trainloader(batch_size//2,input_key,reference_sampler,target_sampler)
        for i in range(num_batches):
            t_vals,ref_batch,target_batch,z = next(base_trainloader)
            yield self_stack(t_vals),self_stack(ref_batch),self_stack(target_batch),jnp.vstack([z,-z])
    return get_testloader
