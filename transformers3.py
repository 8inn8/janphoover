from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import haiku as hk
import optax
import functools as ft


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


def transformer_encoder(x0, head_size, num_heads, ff_dim, dropout=0.1, is_training=True):
    dropout = dropout if is_training else 0.0

    x = layer_norm(x0, name='enc_ln1')
    x = hk.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, w_init_scale=0.02, name='enc_head')(x, x)
    x = hk.dropout(hk.next_rng_key(), dropout, x)
    res = x + x0

    # Feed Forward Part
    x = layer_norm(res)
    x = hk.Conv1D(output_channels=ff_dim, kernel_shape=1, stride=1)(x)
    x = jnn.gelu(x)
    x = hk.dropout(hk.next_rng_key(), dropout, x)
    x = hk.Conv1D(filters=x0.shape[-1], kernel_shape=1)(x)
    return x + res


def global_pooling(x):
    return jnp.mean(x.reshape(x.shape[0], x.shape[1], -1), axis=2)


def transformer(x, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0.1, mlp_dropout=0.2, is_training=True)
    dropout = dropout if is_training else 0.0

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, is_training)

    x = global_pooling(x)
    for dim in mlp_units:
        x = hk.Linear(dim)(x)
        x = jnn.gelu(x)
        x = hk.dropout(hk.next_rng_key(), rate=dropout, x=x)
    outputs =