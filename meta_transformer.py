import logging
import pickle
from typing import Optional, Mapping, Any

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import haiku.initializers as hki
import einops
import functools as ft

import numpy as np
import optax
import pandas as pd

def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=1e-6, name=name)(x)

class Time2Vec(hk.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        self.k = kernel_size

    def __call__(self, inputs):
        ii1 = inputs.shape[1]
        init = hki.RandomUniform(0, 0.01)
        bias = hk.get_parameter('wb', shape=(ii1,), init=init) * inputs + hk.get_parameter('bb', shape=(ii1,), init=init)
        wa = hk.get_parameter('wa', shape=(1, ii1, self.k), init=init)
        ba = hk.get_parameter('ba', shape=(1, ii1, self.k), init=init)
        dp = jnp.dot(inputs, wa) + ba
        weights = jnp.sin(dp)

        ret = jnp.concatenate([jnp.expand_dims(bias, axis=-1), weights], -1)
        ret = einops.rearrange(ret, "t b c -> t (b c)")
        return ret


class AttentionBlock(hk.Module):
    def __init__(self, num_heads, head_size, ff_dim=None, dropout=0.0):
        super().__init__()
        if ff_dim is None:
            self.ff_dim = head_size
        else:
            self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_size = head_size

    def __call__(self, inputs, is_training=True):
        dropout = self.dropout if is_training else 0
        out_features = inputs.shape[-1]

        x = hk.MultiHeadAttention(num_heads=self.num_heads, key_size=self.head_size)(inputs, inputs, inputs)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = layer_norm(x)

        x = hk.Conv1D(output_channels=self.ff_dim, kernel_shape=1, padding="same")(x)
        x = jnn.gelu(x)
        x = hk.Conv1D(output_channels=out_features, kernel_shape=1, padding="same")(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.gelu(x)

        return layer_norm(x + inputs)



class TransformerThunk(hk.Module):
    def __init__(self, time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0):
        super().__init__()
        self.time2vec_dim = time2vec_dim
        
    
        
     

    