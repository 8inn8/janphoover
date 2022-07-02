import functools
import logging
from typing import Optional, Mapping, Any

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pandas as pd
import numpy as np
import jax.random as jr
import jax.nn as jnn


def get_angles(pos: jnp.ndarray, i: jnp.ndarray, D: int):
    angle_rates = 1 / jnp.power(10000, (2 * (i // 2)) / jnp.float32(D))
    return pos * angle_rates


@functools.partial(jax.jit, static_argnums=(0, 1))
def positional_encoding(D: int, position: int =20):
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
        def __init__(self, num_layers, D:int, H:int, S:int, hidden_mlp_dim, out_features, dropout_rate):
            super(Transformer, self).__init__()
            self.sqrt_D = jnp.array([jnp.sqrt(D)])
            self.num_layers = num_layers
            self.dropout = dropout_rate
            self.H = H
            self.D = D
            self.hidden_mlp = hidden_mlp_dim
            self.out_features = out_features
            self.mask = create_look_ahead_mask(S)

        def __call__(self, x, is_training: bool=True):
            B, S, D = x.shape
            H = self.H
            mask = self.mask
            dropout = self.dropout if is_training else 0.0
            initializer = hk.initializers.VarianceScaling(0.5)
            x = hk.Linear(D, w_init=initializer, name='inp_linear')(x)
            x = x * self.sqrt_D
            x = x + positional_encoding(D)[:, :S, :]
            x = hk.dropout(hk.next_rng_key(), dropout, x)

            for i in range(self.num_layers):
                x, _ = TransformerLayer(D, H, self.hidden_mlp, dropout=dropout, name=f'h{i}_mha')(x=x, look_ahead_mask=mask)
            return layer_norm(hk.Linear(self.out_features, w_init=initializer)(x), name='output_ln')


def build_forward_fn(d_model: int, num_heads: int, num_layers: int, hidden_mlp_dim: int, out_features: int, dropout_rate: float):
    def forward_fn(data: Mapping[str, jnp.ndarray], is_training: bool = True) -> jnp.ndarray:
        obs = data['obs']
        B, S, D = obs.shape
        transformer = Transformer(num_layers=num_layers, D=D, H=num_heads, S=S, hidden_mlp_dim=hidden_mlp_dim, out_features=out_features, dropout_rate=dropout_rate)
        output_embeddings = transformer(data['obs'], is_training)

        return output_embeddings
    return forward_fn


#@functools.partial(jax.jit, static_argnums=(0, 1, 5))
def lm_loss_fn(forward_fn, params, rng, data: Mapping[str, jnp.ndarray], is_training: bool = True) -> jnp.ndarray:
    y_pred = forward_fn(params, rng, data, is_training)
    y = data['target']

    return -jnp.mean((y - y_pred) ** 2)


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
    d_model = 128
    D = 256
    H = 1
    num_layers = 1
    hidden_mlp_dim = 32
    out_features = 1
    dropout_rate = 0.5
    grad_clip_value = 1.0
    learning_rate = 0.01
    batch_size = 256
    x, y = load_dataset()
    train_dataset = get_generator(x, y, jax.random.PRNGKey(64444), batch_size)

    forward_fn = build_forward_fn(d_model=d_model, num_heads=H, num_layers=num_layers, hidden_mlp_dim=hidden_mlp_dim,
                                  out_features=out_features, dropout_rate=dropout_rate)

    forward_fn = hk.transform(forward_fn)
    forward_apply = forward_fn.apply
    loss_fn = functools.partial(lm_loss_fn, forward_apply)

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
    print(w.shape)
    state = updater.init(rng, {'obs': w, 'target': z})

    logging.info('Starting train loop...')
    for i, (w, z) in zip(range(max_steps), train_dataset):
        logging.info(f'Step {i} computing forward-backward pass')
        state, metrics = updater.update(state, {'obs': w, 'target': z})
        logging.info(f'At step {i} the loss is {metrics}')


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()

