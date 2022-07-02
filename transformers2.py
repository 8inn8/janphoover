import functools

import jax
import jax.numpy as jnp
import haiku as hk
import pandas as pd
import numpy as np
import jax.random as jr
import jax.nn as jnn


@functools.partial(jax.jit, static_argnums=(1, 2))
def get_angles(pos, i, D):
    angle_rates = 1 / jnp.power(10000, (2 * (i // 2)) / jnp.float32(D))
    return pos * angle_rates


@jax.jit
def positional_encoding(D, position=20):
    angle_rads = get_angles(jnp.arange(position)[:, jnp.newaxis], jnp.arange(D)[jnp.newaxis, :], D)
    a1 = jnp.sin(angle_rads[:, 0::2])
    a2 = jnp.cos(angle_rads[:, 1::2])
    angle_rads = angle_rads.at[:, 0::2].set(a1)
    angle_rads = angle_rads.at[:, 1::2].set(a2)

    return angle_rads[jnp.newaxis, ...]


def create_look_ahead_mask(size):
    mask = jnp.ones((size, size))
    mask = jnp.triu(mask, k=1)
    return mask


def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

class TransformerLayer(hk.Module):
    def __init__(self, dropout, num_heads, key_size, hidden_mlp_dim, dd):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.key_size = key_size
        self.hidden_mlp_dim = hidden_mlp_dim
        self.dd = dd

    def __call__(self, x, look_ahead_mask, is_training: bool = True):

        mha, mha_w = hk.MultiHeadAttention(self.num_heads, self.key_size, w_init_scale=0.02)(x, look_ahead_mask, x)
        mha = hk.dropout(hk.next_rng_key(), self.dropout, mha)
        mha = layer_norm(mha + x)

        initializer = hk.initializers.VarianceScaling(0.02)
        mlp = jnn.gelu(hk.Linear(self.hidden_mlp_dim, w_init=initializer)(mha_w))
        mlp = hk.Linear(self.dd, w_init=initializer)(mlp)
        mlp = hk.dropout(hk.next_rng_key(), self.dropout, mlp)

        output = layer_norm(mlp + mha)
        return output


class Transformer(hk.Module)




