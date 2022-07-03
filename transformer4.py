import logging
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


@ft.partial(jax.jit, static_argnums=(0, 5))
def lm_loss_fn(forward_fn, params, rng, x, y, is_training: bool = True) -> jnp.ndarray:
    y_pred = forward_fn(params, rng, x, is_training)
    loss = jnp.mean((y - y_pred) ** 2)
    return loss


class GradientUpdater:
    def __init__(self, net_init, loss_fn, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    def init(self, master_rng, x):
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, x)
        opt_state = self._opt.init(params)
        out = dict(step=jnp.array(0), rng=out_rng, opt_state=opt_state, params=params)
        return out

    def update(self, state: Mapping[str, Any], x: jnp.ndarray, y: jnp.ndarray):
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, x, y)

        updates, opt_state = self._opt.update(g, state['opt_state'], params)
        params = optax.apply_updates(params, updates)

        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params
        }

        metrics = {
            'step': state['step'],
            'loss': loss,
        }

        return new_state, metrics


def load_dataset(filename='./data/sales_train.csv'):
    train_ds = pd.read_csv(filename)

    monthly_data = train_ds.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
    monthly_data.reset_index(inplace=True)
    train_data = monthly_data.drop(columns=['shop_id', 'item_id'], level=0)
    train_data.fillna(0, inplace=True)
    x_train = np.expand_dims(train_data.values[:, :-1], axis=2).astype(np.float32)
    y_train = train_data.values[:, -1:].astype(np.float32)
    return jnp.array(x_train), jnp.array(y_train)


def get_generator(x, y, rng_key, batch_size):
    def batch_generator():
        n = x.shape[0]
        key = rng_key
        while True:
            key = jax.random.split(key)
            perm = jax.random.choice(rng_key, n, shape=(batch_size,))
            yield jnp.array(x[perm, :]), jnp.array(y[perm])
    return batch_generator()


def map_xy(x, y, half_batch):
    x0 = x[:half_batch, :]
    y0 = y[:half_batch]
    x1 = x[half_batch:, :]
    y1 = y[half_batch:]
    # print("Super shape.......", x.shape, x0.shape)
    xw = jnp.zeros((2, half_batch, x.shape[1]))
    yz = jnp.zeros((2, half_batch, 1))


    xw = xw.at[0, :, :].set(x0)
    yz = yz.at[0, :].set(y0)
    xw = xw.at[1, :, :].set(x1)
    yz = yz.at[1, :].set(y1)

    return xw, yz


def main():
    max_steps = 20000
    num_layers = 50
    head_size = 256
    num_heads = 4
    time2vec_dim = 1
    #ff_dim = 4
    dropout = 0.2

    grad_clip_value = 0.2
    learning_rate = 0.01
    batch_size = 128
    half_batch = batch_size // 2

    x, y = load_dataset()
    train_dataset = get_generator(x, y, jax.random.PRNGKey(64444), batch_size)

    forward_fn = build_forward_fn(num_layers, time2vec_dim, num_heads, head_size, dropout=dropout)

    forward_fn = hk.transform(forward_fn)

    forward_apply = jax.jit(forward_fn.apply, static_argnums=3)
    loss_fn = ft.partial(lm_loss_fn, forward_apply)

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_value),
        optax.radam(learning_rate=learning_rate)
    )

    #optimizer_wrapped = optax.lookahead(optimizer, sync_period=8, slow_step_size=0.8, reset_state=False)

    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

    logging.info('Initializing parameters...')
    rng = jax.random.PRNGKey(888)
    a = next(train_dataset)
    w, z = a
    #q1, q2 = map_xy(w, z, half_batch)
    state = updater.init(rng, w)

    fn_update = updater.update

    logging.info('Starting train loop ++++++++...')
    for i, (w, z) in zip(range(max_steps), train_dataset):
        logging.info(f'Step {i} computing forward-backward pass')

        #xs = jnp.arange(jax.local_device_count())
        #x_batched, y_batched = map_xy(w, z, half_batch)
        #jax.pmap(lambda j: h(x_batched[j, :, :], y_batched[j, :]))(xs)
        #state, metrics = jax.pmap(h)(w, z)
        h = ft.partial(fn_update, state)
        state, metrics = h(w, z)
        logging.info(f'At step {i} the loss is {metrics}')


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
