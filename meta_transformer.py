import logging
from mimetypes import init
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
import datetime as dt
from itertools import product

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=1e-6, name=name)(x)

class Time2Vec(hk.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        self.k = kernel_size

    def __call__(self, inputs):
        ii1 = inputs.shape[1]
        init = hki.RandomUniform(0, 1.0 / ii1)
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

        x = hk.MultiHeadAttention(num_heads=self.num_heads, key_size=self.head_size, w_init_scale=1.0)(inputs, inputs, inputs)
        x = hk.BatchNorm(False, False, decay_rate=0.99, eps=1e-6)(x, is_training)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = layer_norm(x)

        x = hk.Conv1D(output_channels=self.ff_dim, kernel_shape=1, padding="same")(x)
        x = hk.BatchNorm(False, False, decay_rate=0.99, eps=1e-6)(x, is_training)
        x = jnn.gelu(x, approximate=False)
        x = hk.Conv1D(output_channels=out_features, kernel_shape=1, padding="same")(x)
        x = hk.BatchNorm(False, False, decay_rate=0.99, eps=1e-6)(x, is_training)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.gelu(x, approximate=False)

        return layer_norm(x + inputs)


class TimeDistributed(hk.Module):
    def __init__(self, module, batch_first=True):
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


class TransformerThunk(hk.Module):
    def __init__(self, time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0):
        super().__init__()
        self.time2vec_dim = time2vec_dim
        if ff_dim is None:
            self.ff_dim = head_size
        else:
            self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers

    def __call__(self, inputs, is_training=True):
        dropout = self.dropout if is_training else 0.0
        time2vec = Time2Vec(kernel_size=self.time2vec_dim)
        time_embedding = TimeDistributed(time2vec)(inputs)

        x = jnp.concatenate([inputs, time_embedding], axis=-1)
        
        w_init = hki.VarianceScaling(2.0, mode='fan_in', distribution='truncated_normal')
        for i in range(self.num_layers):
            x = AttentionBlock(num_heads=self.num_heads, head_size=self.head_size, ff_dim=self.ff_dim, dropout=self.dropout)(x, is_training)
        x = jnp.mean(x, axis=-1)
        #x = einops.rearrange(x, 'b h c -> b (h c)')
        x = hk.Linear(256, w_init=w_init, b_init=hki.Constant(1e-6))(x)
        x = jnn.gelu(x, approximate=False)
        x = hk.Linear(128, w_init=w_init, b_init=hki.Constant(1e-6))(x)
        x = jnn.gelu(x, approximate=False)
        x = hk.Linear(64, w_init=w_init, b_init=hki.Constant(1e-6))(x)
        x = jnn.gelu(x, approximate=False)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        return hk.Linear(1, w_init=w_init, b_init=hki.Constant(1e-6))(x)

def build_forward_fn(num_layers, time2vec_dim, num_heads, head_size, ff_dim=None, dropout=0.5):
    def forward_fn(x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        tr = TransformerThunk(time2vec_dim, num_heads, head_size, ff_dim, num_layers, dropout)
        return tr(x, is_training)

    return forward_fn
        
     
@ft.partial(jax.jit, static_argnums=(0, 6))
def lm_loss_fn(forward_fn, params, state, rng, x, y, is_training: bool = True) -> jnp.ndarray:
    y_pred, state = forward_fn(params, state, rng, x, is_training)

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    return jnp.sqrt(jnp.mean((jnp.square(y - y_pred)))) + 1e-5 * l2_loss, state


class GradientUpdater:
    def __init__(self, net_init, loss_fn, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    def init(self, master_rng, x):
        out_rng, init_rng = jax.random.split(master_rng)
        params, state = self._net_init(init_rng, x)
        opt_state = self._opt.init(params)
        return jnp.array(0), out_rng, params, state, opt_state

    def update(self, num_steps, rng, params, state, opt_state, x:jnp.ndarray, y: jnp.ndarray):
        rng, new_rng = jax.random.split(rng)

        (loss, state), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(params, state, rng, x, y)

        #loss = jax.lax.pmean(loss, axis_name='j')

        grads = jax.lax.pmean(grads, axis_name='j')

        updates, opt_state = self._opt.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        metrics = {
            'step': num_steps,
            'loss': loss,
        }

        return num_steps + 1, new_rng, params, state, opt_state, metrics

def load(f1='./data/sales_train.csv', f2='./data/test.csv'):
    train = pd.read_csv(f1)
    test = pd.read_csv(f2)
    
    trn = train.copy()
    trn = trn.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum') 

    tst = test.copy()
    tst = tst.pivot_table(index = ['shop_id','item_id'],fill_value = 0)

    combo = pd.merge(tst, trn, how = 'left', on = ['shop_id','item_id']).fillna(0) #train test combined
    combo = combo.drop(columns = ['ID'])

    arrc = np.array(combo.values[:, :-1])
    train_data = arrc.reshape(arrc.shape[0], arrc.shape[1], 1)
    y = combo.values[:, -1:].clip(0, 50)

    arrt = np.array(combo.values[:, 1:])
    test_data = arrt.reshape(arrt.shape[0], arrt.shape[1], 1)

    return jnp.array(train_data), jnp.array(y), jnp.array(test_data), test


def load_dataset(f1='./data/sales_train.csv', f2='./data/test.csv'):
    train_ds = pd.read_csv(f1)
    test_ds = pd.read_csv(f2)
    monthly_data = train_ds.pivot_table(index = ['shop_id','item_id'], values = ['item_cnt_day'], columns = ['date_block_num'], fill_value = 0, aggfunc='sum')
    monthly_data.reset_index(inplace = True)
    train_data = monthly_data.drop(columns= ['shop_id','item_id'], level=0)
    train_data.fillna(0,inplace = True)

    y_train = train_data.values[:,-1:].clip(0, 20)

    sc = StandardScaler()
    x_train = np.expand_dims(sc.fit_transform(train_data.values[:,:-1]), axis=2)

    test_rows = monthly_data.merge(test_ds, on = ['item_id','shop_id'], how = 'right')
    x_test = test_rows.drop(test_rows.columns[:5], axis=1).drop('ID', axis=1)
    x_test.fillna(0,inplace = True)

    x_test = sc.transform(x_test)
    x_test = np.expand_dims(x_test,axis = 2)

    return jnp.array(x_train), jnp.array(y_train), jnp.array(x_test), test_ds

def get_generator_parallel(x, y, rng_key, batch_size, num_devices):
    def batch_generator():
        n = x.shape[0]
        key = rng_key
        kk = batch_size // num_devices
        while True:
            key, k1 = jax.random.split(key)
            perm = jax.random.choice(k1, n, shape=(batch_size,))
            
            yield x[perm, :].reshape(num_devices, kk, *x.shape[1:]), y[perm].reshape(num_devices, kk, *y.shape[1:])
    return batch_generator()

def replicate_tree(t, num_devices):
    return jax.tree_map(lambda x: jnp.array([x] * num_devices), t)

def main():
    max_steps = 10000
    num_heads = 8
    head_size = 128
    num_layers = 2
    dropout_rate = 0.2
    grad_clip_value = 1.0
    learning_rate = 0.001
    time2vec_dim = 15
    batch_size = 256
    
    num_devices = jax.local_device_count()

    print("Num devices :::: ", num_devices)

    x, y, x_test, test_ds = load()

    print("Examples :::: ", x.shape)
    print("Examples :::: ", y.shape)
    print("Testing Examples :::: ", x_test.shape)

    rng1, rng = jr.split(jax.random.PRNGKey(111))
    train_dataset = get_generator_parallel(x, y, rng1, batch_size, num_devices)

    forward_fn = build_forward_fn(num_layers, time2vec_dim, num_heads, head_size, dropout=dropout_rate)

    forward_fn = hk.transform_with_state(forward_fn)

    forward_apply = forward_fn.apply
    loss_fn = ft.partial(lm_loss_fn, forward_apply)


    scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=4000, decay_rate=0.99)

    optimizer = optax.chain(
        optax.adaptive_grad_clip(grad_clip_value),
        #optax.sgd(learning_rate=learning_rate, momentum=0.95, nesterov=True),
        #optax.scale_by_radam(),
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0)
    )

    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

    logging.info('Initializing parameters...')
    rng1, rng = jr.split(rng)
    a = next(train_dataset)
    w, z = a
    num_steps, rng, params, state, opt_state = updater.init(rng1, w[0, :, :, :])

    params_multi_device = params
    opt_state_multi_device = opt_state
    num_steps_replicated = num_steps
    rng_replicated = rng
    state_multi_device = state

    fn_update = jax.pmap(updater.update, axis_name='j', in_axes=(None, None, None, None, None, 0, 0), out_axes=(None, None, None, None, None, 0))

    logging.info('Starting train loop ++++++++...')
    for i, (w, z) in zip(range(max_steps), train_dataset):
        if (i + 1) % 100 == 0:
            logging.info(f'Step {i} computing forward-backward pass')
        num_steps_replicated, rng_replicated, params_multi_device, state_multi_device, opt_state_multi_device, metrics = \
            fn_update(num_steps_replicated, rng_replicated, params_multi_device, state_multi_device, opt_state_multi_device, w, z)

        if (i + 1) % 100 == 0:
            logging.info(f'At step {i} the loss is {metrics}')
    
    # Test part of the model
    forward_apply = jax.jit(forward_apply, static_argnames=['is_training'])
    params_reduced = params_multi_device # Reduce parameters for single device
    state_reduced = state_multi_device
    N = x_test.shape[0]
    result = jnp.zeros((N,))
    rng = rng_replicated

    count = N // 100
    for i in range(count):
        if i % 200 == 0:
            print('Computing ', i * 100)
        (rng,) = jr.split(rng, 1)
        a, b = i * 100, (i + 1) * 100
        eli = x_test[a:b, :, :]
        fa, _ = forward_apply(params_reduced, state_reduced, rng,  eli, is_training=False)
        result = result.at[a:b].set(fa[:, 0])

    result = np.array(result)

    output = pd.DataFrame({'ID': test_ds['ID'], 'item_cnt_month': result.clip(0, 20).ravel()})
    output.to_csv('./data/submission1.csv', index=False)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
