import e3x
import math

from flax.experimental.nnx import softmax

from flax import linen as nn
from jax import numpy as jnp



# TODO:
# - where TensorDense layer instead of Dense layer?
# - how incorporate regularization techniques (dropout)?
# - how to incorporate training optimization techniques (residual connections, dropout, layer norm)?
class E3EqMulHeadAttBlock(nn.Module):

    num_features: int
    num_heads: int

    @nn.compact
    def __call__(self, x, weight_mask):
        # Compute size of each head
        #head_size = self.num_features // self.num_heads

        # Compute queries, keys and values
        Q = e3x.nn.Dense(features=self.num_features, use_bias=False)(x)
        K = e3x.nn.Dense(features=self.num_features, use_bias=False)(x)
        V = e3x.nn.Dense(features=self.num_features, use_bias=False)(x)
        
        # Distribute queries and keys over the heads
        Q = jnp.reshape(Q, (*Q.shape[:-1], -1, self.num_heads))
        K = jnp.reshape(K, (*K.shape[:-1], -1, self.num_heads))

        # Normalize queries
        depths = math.prod(Q.shape[-4:-1])
        Q /= jnp.sqrt(depths).astype(Q.dtype)

        # Compute attentation weights
        weight_mask = jnp.expand_dims(weight_mask, axis=-1)
        dotprod = jnp.einsum("...abplfh, ...cbplfh -> ...ach", Q, K) + weight_mask
        weights = softmax(dotprod, axis=-2)

        # Compute update per node (MO)
        V = jnp.einsum("...abh, ...bcplf -> ...acplfh", weights, V)
        V = jnp.reshape(V, (*V.shape[:-2], -1))
        V = e3x.nn.Dense(features=self.num_features, use_bias=False)(x)
        
        # Skip connection, update nodes (MOs) ??
        x = e3x.nn.add(x, V)
        #x = x + V
        
        # Feed forward for processing attentation updates
        x = e3x.nn.TensorDense()(x)
        x = e3x.nn.relu(x)
        
        return x



class SimpleEqAttentionHead(nn.Module):

    num_features: int

    @nn.compact
    def __call__(self, x, weight_mask):
        # Compute queries, keys and values
        Q = e3x.nn.Dense(features=self.num_features, use_bias=False)(x)
        K = e3x.nn.Dense(features=self.num_features, use_bias=False)(x)
        V = e3x.nn.Dense(features=self.num_features, use_bias=False)(x)

        # Compute attentation weights
        dotprod = jnp.einsum("...abplf, ...cbplf -> ...ac", Q, K) + weight_mask
        weights = softmax(dotprod, axis=-2)
        # Compute update per node (MO)
        V = jnp.einsum("...ab, ...bcplf -> ...acplf", weights, V)

        # Skip connection, update nodes (MOs) ??
        x = e3x.nn.add(x, V)
        #x = x + V

        #x = e3x.nn.Dense(features=2*self.num_features)(x)
        #x = e3x.nn.relu(x)
        #x = e3x.nn.Dense(features=self.num_features)(x)

        x = e3x.nn.TensorDense()(x)
        x = e3x.nn.relu(x)
        #x = e3x.nn.TensorDense()(x)
        #x = e3x.nn.relu(x)

        return V



class CoeffNet(nn.Module):

    num_features: int
    num_heads: int
    num_blocks: int

    @nn.compact
    def __call__(self, x, weight_mask=0):
        if self.num_features % self.num_heads != 0:
            print()
            print("*********************** WARNING *************************")
            print("*                                                       *")
            print("*   Number of heads does not match number of features   *")
            print("*                                                       *")
            print("*********************************************************")
            print()
            return x
        x = e3x.nn.TensorDense(features=self.num_features, max_degree=2, use_bias=False)(x)
        #for _ in range(self.num_blocks):
        #    x = E3EqMulHeadAttBlock(num_features=self.num_features, num_heads=self.num_heads)(x, pad_mask)
        x = SimpleEqAttentionHead(num_features=self.num_features)(x, weight_mask)
        x = SimpleEqAttentionHead(num_features=self.num_features)(x, weight_mask)
        #x = e3x.nn.TensorDense(features=1, include_pseudotensors=False, use_bias=False)(x)
        x = e3x.nn.TensorDense(features=1, include_pseudotensors=False, max_degree=0, use_bias=False)(x)
        x = jnp.squeeze(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = jnp.sum(x, axis=-1)
        return x
