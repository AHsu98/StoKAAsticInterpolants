import jax
import equinox as eqx

class NeuralNetwork(eqx.Module):
    layers: list
    extra_bias: jax.Array
    residual_beta: float

    def __init__(self,input_size,output_size,layer_sizes, key,residual_beta = 0.05):
        keys = jax.random.split(key, len(layer_sizes)+2)
        # These contain trainable parameters.
        self.residual_beta = residual_beta
        self.layers = (
            [eqx.nn.Linear(input_size,layer_sizes[0],key = keys[0])] + 
            [eqx.nn.Linear(layer_in_size,layer_out_size,key = layer_key) for layer_in_size,layer_out_size,layer_key in zip(layer_sizes[:-1],layer_sizes[1:],keys[1:-1])] + 
            [eqx.nn.Linear(layer_sizes[-1],output_size,key = keys[-1])]
        )
        self.extra_bias = jax.numpy.zeros(output_size)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = (1-self.residual_beta) * jax.nn.elu(layer(x)) + self.residual_beta
        return self.layers[-1](x) + self.extra_bias
