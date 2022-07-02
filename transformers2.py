import functools
from typing import Optional

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


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)

class TransformerLayer(hk.Module):
    def __init__(self, dd, hh, hidden_mlp_dim, dropout, name):
        super(TransformerLayer, self).__init__(name)
        self.dropout = dropout
        self.num_heads = hh
        self.key_size = dd
        self.hidden_mlp_dim = hidden_mlp_dim

    def __call__(self, x, look_ahead_mask, is_training: bool = True):
        dropout = self.dropout if is_training else 0.0
        mha, mha_w = hk.MultiHeadAttention(self.num_heads, self.key_size, w_init_scale=0.02)(x, look_ahead_mask, x)
        mha = hk.dropout(hk.next_rng_key(), dropout, mha)
        mha = layer_norm(mha + x)

        initializer = hk.initializers.VarianceScaling(0.02)
        mlp = jnn.gelu(hk.Linear(self.hidden_mlp_dim, w_init=initializer)(mha_w))
        mlp = hk.Linear(self.key_size, w_init=initializer)(mlp)
        mlp = hk.dropout(hk.next_rng_key(), dropout, mlp)

        output = layer_norm(mlp + mha)
        return output


class Transformer(hk.Module):
        def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate):
            super(Transformer, self).__init__()
            self.sqrt_D = jnp.sqrt(D)
            self.num_layers = num_layers
            self.dropout = dropout_rate
            self.H = H
            self.D = D
            self.hidden_mlp = hidden_mlp_dim
            self.inp_features = inp_features
            self.out_features = out_features

        def __call__(self, x, mask, is_training: bool=True):
            B, S, _ = x.shape
            D, H = self.D, self.H
            dropout = self.dropout if is_training else 0.0
            initializer = hk.initializers.VarianceScaling(0.02)
            x = hk.Linear(D,w_init=initializer, name='inp_linear')(x)
            x = x * self.sqrt_D
            x = x + positional_encoding(D)[:, :S, :]
            x = hk.dropout(hk.next_rng_key(), dropout, x)

            for i in range(self.num_layers):
                x, _ = TransformerLayer(D, H, self.hidden_mlp, dropout=dropout, name=f'h{i}_mha')(x=x, look_ahead_mask=mask)
            return layer_norm(hk.Linear(self.out_features, w_init=initializer)(x), name='output_ln')


