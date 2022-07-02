import logging
from typing import Optional, Mapping, Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import haiku as hk
import numpy as np
import optax
import functools as ft

import pandas as pd


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


def transformer_encoder(x0, head_size, num_heads, ff_dim, dropout=0.1, is_training=True):
    dropout = dropout if is_training else 0.0
    out_features = x0.shape[-1]

    x = layer_norm(x0, name='enc_ln1')
    x = hk.MultiHeadAttention(key_size=head_size, num_heads=num_heads, w_init_scale=0.02, name='enc_head')(x, x)
    x = hk.dropout(hk.next_rng_key(), dropout, x)
    res = x + x0

    # Feed Forward Part
    initializer = hk.initializers.VarianceScaling(0.02)
    x = layer_norm(res)
    x = hk.Conv1D(output_channels=ff_dim, kernel_shape=1, stride=1, padding='same', w_init=initializer)(x)
    x = jnn.gelu(x)
    x = hk.dropout(hk.next_rng_key(), dropout, x)
    x = hk.Conv1D(output_channels=out_features, kernel_shape=1, stride=1, padding='same', w_init=initializer)(x)
    return x + res


def global_pooling(x):
    return jnp.average(x, axis=(-2, -1)).values


def transformer(x, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0.1, mlp_dropout=0.2, is_training=True):
    dropout = dropout if is_training else 0.0

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, is_training)

    x = global_pooling(x)
    initializer = hk.initializers.VarianceScaling(0.02)
    for dim in mlp_units:
        x = hk.Linear(dim, w_init=initializer)(x)
        x = jnn.gelu(x)
        x = hk.dropout(hk.next_rng_key(), rate=dropout, x=x)
    outputs = hk.Linear(1, w_init=initializer)(x)

    return outputs


def build_forward_fn(head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout):
    def forward_fn(data: Mapping[str, jnp.ndarray], is_training: bool = True) -> jnp.ndarray:
        obs = data['obs']
        return transformer(obs, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=dropout, mlp_dropout=mlp_dropout, is_training=True)

    return forward_fn


@ft.partial(jax.jit, static_argnums=(0, 4))
def lm_loss_fn(forward_fn, params, rng, data: Mapping[str, jnp.ndarray], is_training: bool = True) -> jnp.ndarray:
    y_pred = forward_fn(params, rng, data, is_training)
    y = data['target']

    return -jnp.mean((y - y_pred) ** 2, axis=-1)


class GradientUpdater:
    def __init__(self, net_init, loss_fn, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    def init(self, master_rng, data):
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        out = dict(step=jnp.array(0), rng=out_rng, opt_state=opt_state, params=params)
        return out

    def update(self, state: Mapping[str, Any], data: Mapping[str, jnp.ndarray]):
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn(params, rng, data))

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
            yield x[perm, :], y[perm]
    return batch_generator()


def main():
    max_steps = 200
    head_size = 64
    num_heads = 4
    ff_dim = 4
    num_transformer_blocks = 1
    mlp_units = 1
    dropout = 0.2
    mlp_dropout = 0.5

    grad_clip_value = 1.0
    learning_rate = 0.01
    batch_size = 256

    x, y = load_dataset()
    train_dataset = get_generator(x, y, jax.random.PRNGKey(64444), batch_size)

    forward_fn = build_forward_fn(head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout)

    forward_fn = hk.transform(forward_fn)
    forward_apply = jax.jit(forward_fn.apply, static_argnums=3)
    loss_fn = ft.partial(lm_loss_fn, forward_apply)

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_value),
        optax.radam(learning_rate=learning_rate)
    )

    optimizer_wrapped = optax.lookahead(optimizer, sync_period=8, slow_step_size=0.8, reset_state=False)

    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer_wrapped)

    logging.info('Initializing parameters...')
    rng = jax.random.PRNGKey(888)
    a = next(train_dataset)
    w, z = a
    state = updater.init(rng, {'obs': w, 'target': z})

    logging.info('Starting train loop ++++++++...')
    for i, (w, z) in zip(range(max_steps), train_dataset):
        logging.info(f'Step {i} computing forward-backward pass')
        state, metrics = updater.update(state, {'obs': w, 'target': z})
        logging.info(f'At step {i} the loss is {metrics}')


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
