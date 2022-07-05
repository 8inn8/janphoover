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
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


class Time2Vec(hk.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        self.k = kernel_size

    def __call__(self, inputs, **kwargs):
        init = hki.RandomUniform()
        ii1 = inputs.shape[1]
        bias = hk.get_parameter("wb", (ii1,), init=init) * inputs + hk.get_parameter("bb", shape=(ii1,), init=init)
        dp = jnp.dot(inputs, hk.get_parameter("wa", shape=(1, ii1, self.k), init=init)) + hk.get_parameter("ba", shape=(1, ii1, self.k), init=init)
        wgts = jnp.sin(dp)

        ret = jnp.concatenate([jnp.expand_dims(bias, -1), wgts], axis=-1)
        return einops.rearrange(ret, "t b c -> t (b c)")


class AttentionBlock(hk.Module):
    def __init__(self, num_heads, head_size, ff_dim=None, dropout=0.0):
        super().__init__()
        if ff_dim is None:
            ff_dim = head_size

        self.num_heads = num_heads
        self.head_size = head_size
        self.ff_dim = ff_dim
        self.dropout = dropout

    def __call__(self, inputs, is_training=True):
        dropout = self.dropout if is_training else 0.0
        out_features = inputs.shape[-1]

        x = hk.MultiHeadAttention(num_heads=self.num_heads, key_size=self.head_size, w_init_scale=1.0)(inputs, inputs, inputs)
        x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = layer_norm(x)

        init = hki.VarianceScaling(0.02)
        x = hk.Conv1D(output_channels=self.ff_dim, kernel_shape=1, stride=1, w_init=init)(x)
        x = jnn.gelu(x)
        x = hk.Conv1D(output_channels=out_features, kernel_shape=1, stride=1, w_init=init)(x)
        x = hk.dropout(hk.next_rng_key(), self.dropout, x)

        return layer_norm(inputs + x)


class TimeDistributed(hk.Module):
    def __init__(self, module, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        self.module = module

    def __call__(self, x):
        module = self.module
        if len(x.shape) <= 2:
            return module(x)

        x_reshape = einops.rearrange(x, "b c h -> (b c) h")

        y = module(x_reshape)


        return jnp.where(self.batch_first, jnp.reshape(y, newshape=(x.shape[0], -1, y.shape[-1])), jnp.reshape(y, newshape=(-1, x.shape[1], y.shape[-1])))


class Transformer(hk.Module):
    def __init__(self, num_layers, time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.ff_dim = head_size if ff_dim is None else ff_dim
        self.dropout = dropout
        self.time2vec_dim = time2vec_dim
        self.num_heads = num_heads
        self.head_size = head_size

    def __call__(self, inputs, is_training=True):
        time2vec = Time2Vec(kernel_size=self.time2vec_dim)
        time_embedding = TimeDistributed(time2vec)(inputs)
        x = jnp.concatenate([inputs, time_embedding], -1)
        for i in range(self.num_layers):
            x = AttentionBlock(self.num_heads, self.head_size, self.ff_dim, self.dropout)(x, is_training)
        x = einops.rearrange(x, 't c b -> t (c b)')
        init = hki.VarianceScaling(0.02)
        return hk.Linear(1, w_init=init)(x)


def build_forward_fn(num_layers, time2vec_dim, num_heads, head_size, ff_dim=None, dropout=0.5):
    def forward_fn(x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        tr = Transformer(num_layers, time2vec_dim, num_heads, head_size, ff_dim, dropout)
        return tr(x, is_training)

    return forward_fn


with open('./data/params.pkl') as fm:
    params = pickle.load(fm)

train_ds = pd.read_csv('./data/sales_train.csv')
test_ds = pd.read_csv('./data/test.csv')

monthly_data = train_ds.pivot_table(index = ['shop_id','item_id'], values = ['item_cnt_day'], columns = ['date_block_num'], fill_value = 0, aggfunc='sum')

monthly_data.reset_index(inplace = True)

train_data = monthly_data.drop(columns= ['shop_id','item_id'], level=0)
train_data.fillna(0,inplace = True)

x_train = np.expand_dims(train_data.values[:,:-1],axis = 2)
y_train = train_data.values[:,-1:]

test_rows = monthly_data.merge(test_ds, on = ['item_id','shop_id'], how = 'right')

x_test = test_rows.drop(test_rows.columns[:5], axis=1).drop('ID', axis=1)
x_test.fillna(0,inplace = True)
x_test = np.expand_dims(x_test,axis = 2)

num_layers = 16
head_size = 64
num_heads = 4
time2vec_dim = 8

transformer_network = build_forward_fn(num_layers=num_layers, time2vec_dim=time2vec_dim, num_heads=num_heads, head_size=head_size, dropout=0)

hk_net = hk.without_apply_rng(hk.transform(transformer_network))
fn = jax.jit(hk_net.apply, static_argnums=2)

result = np.array(fn(params, x_test, is_training=False))

result = pd.DataFrame(result)
result.to_csv('./data/submission.csv', index=True)

