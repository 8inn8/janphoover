from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import haiku.initializers as hki
import einops


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


class Time2Vec(hk.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        self.k = kernel_size

    def __call__(self, inputs, **kwargs):
        init = hki.RandomUniform()
        bias = hk.get_parameter("wb", (inputs.shape[1],), init=init) * inputs + hk.get_parameter("bb", shape=(inputs.shape[1]), init=init)
        dp = jnp.dot(inputs, hk.get_parameter("wa", shape=(1, inputs.shape[1], self.k), init=init)) + hk.get_parameter("ba", shape=(1, inputs.shape[1], self.k), init=init)
        wgts = jnp.sin(dp)

        ret = jnp.concatenate([jnp.expand_dims(bias, -1), wgts], axis=-1)
        return einops.rearrange(ret, "t b c -> t (b c)")


class AttentionBlock(hk.Module):
    def __init__(self, num_heads, head_size, ff_dim=None, dropout=0):
        super().__init__()
        if ff_dim is None:
            ff_dim = head_size

        self.num_heads = num_heads
        self.head_size = head_size
        self.ff_dim = ff_dim
        self.dropout = dropout

    def __call__(self, inputs, is_training=True):
        dropout = self.dropout if is_training else 0.0

        x = hk.MultiHeadAttention(num_heads=self.num_heads, key_size=self.head_size, w_init_scale=1.0)()
