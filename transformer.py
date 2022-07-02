from typing import Optional, Mapping

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import jax.nn as jnn


class SelfAttention(hk.MultiHeadAttention):
    def __call__(self, query: jnp.ndarray, key: Optional[jnp.ndarray] = None,
                 value: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        key = key if key is not None else query
        value = value if value is not None else query

        seq_len = query.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.float32))
        mask = mask * causal_mask if mask is not None else causal_mask

        return super().__call__(query, key, value, mask)


class DenseBlock(hk.Module):
    def __init__(self, init_scale: float, widening_factor: int = 4, name: Optional[str] = None):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
        x = jnn.gelu(x)
        return hk.Linear(hiddens, w_init=initializer)(x)


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


class Transformer(hk.Module):
    def __init__(self, num_heads: int, num_layers: int, dropout_rate: float, name: Optional[str] = None):
        super().__init__(name=name)
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

    def __call__(self, h: jnp.ndarray, mask: Optional[jnp.ndarray], is_training: bool) -> jnp.ndarray:
        init_scale = 2. / self._num_layers
        dropout_rate = self._dropout_rate if is_training else 0
        if mask is not None:
            mask = mask[:, None, None, :]
        for i in range(self._num_layers):
            h_norm = layer_norm(h, name=f'h{i}_ln_1')
            h_attn = SelfAttention(num_heads=self._num_heads, key_size=64, w_init_scale=init_scale, name=f'h{i}_attn')(h_norm, mask=mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn
            h_norm = layer_norm(h, name=f'h{i}_ln_2')
            h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense
        h = layer_norm(h, name='ln_f')

        return h


def embeddings(data: Mapping[str, jnp.ndarray], vocab_size: int, d_model: int):
    tokens = data['obs']
    input_mask = jnp.greater(tokens, 0)
    seq_length = tokens.shape[1]

    # Embed the input tokens and positions
    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
    token_embs = token_embedding_map(tokens)
    positional_embeddings = hk.get_parameter('pos_embs', [seq_length, d_model], init=embed_init)
    input_embeddings = token_embs + positional_embeddings
    return input_embeddings, input_mask


def build_forward_fn(vocab_size: int, d_model: int, num_heads: int, num_layers: int, dropout_rate:float):
    def forward_fn(data: Mapping[str, jnp.ndarray], is_training: bool = True) -> jnp.ndarray:
        input_embeddings, input_mask = embeddings(data, vocab_size, d_model)
        transformer = Transformer(num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_rate)
        output_embeddings = transformer(input_embeddings, input_mask, is_training)

        return hk.Linear(vocab_size)(output_embeddings)

    return forward_fn

