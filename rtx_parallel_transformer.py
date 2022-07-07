import logging
import pickle
from turtle import forward
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
        init = hki.VarianceScaling(0.01)
        init1 = hki.Constant(0.0)
        ii1 = inputs.shape[1]
        bias = hk.get_parameter("wb", (ii1,), init=init1) * inputs + hk.get_parameter("bb", shape=(ii1,), init=init1)
        dp = jnp.dot(inputs, hk.get_parameter("wa", shape=(1, ii1, self.k), init=init)) + hk.get_parameter("ba", shape=(1, ii1, self.k), init=init1)
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

        init = hki.VarianceScaling(0.01)
        x = hk.Conv1D(output_channels=self.ff_dim, kernel_shape=1, stride=1, w_init=init)(x)
        x = jnn.gelu(x)
        x = hk.Conv1D(output_channels=out_features, kernel_shape=1, stride=1, w_init=init)(x)
        x = hk.dropout(hk.next_rng_key(), self.dropout, x)

        return layer_norm(inputs + x)


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
        print("Embedding shape ::: ", time_embedding.shape)
        x = jnp.concatenate([inputs, time_embedding], -1)
        for i in range(self.num_layers):
            x = AttentionBlock(self.num_heads, self.head_size, self.ff_dim, self.dropout)(x, is_training)
        #x = einops.rearrange(x, 't c b -> t (c b)')
        x = jnp.mean(x, axis=(-2,-1))
        init = hki.VarianceScaling(0.01)
        ln1 = hk.Linear(1, w_init=init)(x)
        return jnn.sigmoid(ln1)

def build_forward_fn(num_layers, time2vec_dim, num_heads, head_size, ff_dim=None, dropout=0.5):
    def forward_fn(x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        tr = Transformer(num_layers, time2vec_dim, num_heads, head_size, ff_dim, dropout)
        return tr(x, is_training)

    return forward_fn


@ft.partial(jax.jit, static_argnums=(0, 5))
def lm_loss_fn(forward_fn, params, rng, x, y, is_training: bool = True) -> jnp.ndarray:
    y_pred = forward_fn(params, rng, x, is_training)
    return jnp.sqrt(jnp.mean((jnp.square(y - y_pred))))


class GradientUpdater:
    def __init__(self, net_init, loss_fn, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    def init(self, master_rng, x):
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, x)
        opt_state = self._opt.init(params)
        return jnp.array(0), out_rng, params, opt_state

    def update(self, num_steps, rng, params, opt_state, x:jnp.ndarray, y: jnp.ndarray):
        rng, new_rng = jax.random.split(rng)

        loss, grads = jax.value_and_grad(self._loss_fn)(params, rng, x, y)

        #loss = jax.lax.pmean(loss, axis_name='j')

        grads = jax.lax.pmean(grads, axis_name='j')

        updates, opt_state = self._opt.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        metrics = {
            'step': num_steps,
            'loss': loss,
        }

        return num_steps + 1, new_rng, params, opt_state, metrics


def load_dataset(filename='./data/sales_train.csv', filename1='./data/test.csv'):
    sales_data = pd.read_csv(filename)
    test_data = pd.read_csv(filename1)

    sales_data['date'] = pd.to_datetime(sales_data['date'],format = '%d.%m.%Y')
    dataset = sales_data.pivot_table(index = ['shop_id','item_id'], values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')
    dataset.reset_index(inplace = True)
    dataset = pd.merge(test_data,dataset,on = ['item_id','shop_id'],how = 'left')
    dataset.fillna(0,inplace = True)
    dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)


    x_train = np.expand_dims(dataset.values[:,:-1], axis=2)
    y_train = dataset.values[:,-1:]

    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)

    y_train = (y_train - y_mean) / y_std
    
    x_mean = x_train.mean(axis=0)
    std_dev = x_train.std(axis=0)

    x_train = (x_train - x_mean) / std_dev

    x_test = (np.expand_dims(dataset.values[:,1:], axis=2) - x_mean) / std_dev

    max_y = jnp.max(y_train, axis=0)

    return jnp.array(x_train), jnp.array(y_train), jnp.array(x_test), test_data, y_mean, y_std, max_y


def get_generator_parallel(x, y, rng_key, batch_size, num_devices):
    def batch_generator():
        n = x.shape[0]
        key = rng_key
        kk = batch_size // num_devices
        while True:
            key = jax.random.split(key)
            perm = jax.random.choice(key, n, shape=(batch_size,))
            
            yield x[perm, :].reshape(num_devices, kk, *x.shape[1:]), y[perm].reshape(num_devices, kk, *y.shape[1:])
    return batch_generator()


def replicate_tree(t, num_devices):
    return jax.tree_map(lambda x: jnp.array([x] * num_devices), t)


def replicate(t, num_devices):
    return jax.tree_map(lambda x: jnp.stack([x] * num_devices), t)


def main():
    max_steps = 13301
    num_heads = 8
    head_size = 128
    num_layers = 8
    dropout_rate = 0.2
    grad_clip_value = 0.1
    learning_rate = 0.001
    time2vec_dim = 32
    batch_size = 128
    
    num_devices = jax.local_device_count()

    print("Num devices :::: ", num_devices)

    x, y, x_test, test_ds, y_mean, y_std, max_y = load_dataset()

    print("Examples :::: ", x.shape)
    print("Testing Examples :::: ", x_test.shape)
    print("Max y cap, y_mean, y_std :::::: ", max_y, y_mean, y_std)

    rng1, rng = jr.split(jax.random.PRNGKey(0))
    train_dataset = get_generator_parallel(x, y, rng1, batch_size, num_devices)

    forward_fn = build_forward_fn(num_layers, time2vec_dim, num_heads, head_size, dropout=dropout_rate)

    forward_fn = hk.transform(forward_fn)

    forward_apply = forward_fn.apply
    loss_fn = ft.partial(lm_loss_fn, forward_apply)

    optimizer = optax.chain(
        optax.adaptive_grad_clip(grad_clip_value),
        optax.sgd(learning_rate=learning_rate, momentum=0.95, nesterov=True),
        #optax.radam(learning_rate=learning_rate)
    )

    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

    logging.info('Initializing parameters...')
    rng1, rng = jr.split(rng)
    a = next(train_dataset)
    w, z = a
    num_steps, rng, params, opt_state = updater.init(rng1, w[0, :, :, :])

    params_multi_device = replicate_tree(params, num_devices)
    opt_state_multi_device = opt_state
    num_steps_replicated = replicate_tree(num_steps, num_devices)
    rng_replicated = rng

    fn_update = jax.pmap(updater.update, axis_name='j', in_axes=(0, None, 0, None, 0, 0), out_axes=(0, None, 0, None, 0))

    logging.info('Starting train loop ++++++++...')
    for i, (w, z) in zip(range(max_steps), train_dataset):
        if i % 100 == 0:
            logging.info(f'Step {i} computing forward-backward pass')
        num_steps_replicated, rng_replicated, params_multi_device, opt_state_multi_device, metrics = \
            fn_update(num_steps_replicated, rng_replicated, params_multi_device, opt_state_multi_device, w, z)

        if i % 100 == 0:
            logging.info(f'At step {i} the loss is {metrics}')
    
    # Test part of the model
    forward_apply = jax.jit(forward_apply, static_argnames=['is_training'])
    params_reduced = jax.device_get(jax.tree_map(lambda x: x[0], params_multi_device))# Reduce parameters for single device
    N = x_test.shape[0]
    result = np.zeros((N,))
    rng = rng_replicated
    ff = lambda eli, rng: forward_apply(params_reduced, rng, eli, is_training=False)
    count = N // 64
    for i in range(count):
        if i % 200 == 0:
            print('Computing ', i * 64)
        (rng,) = jr.split(rng, 1)
        a, b = i * 64, (i + 1) * 64
        eli = x_test[a:b]
        result[a:b] = np.array(ff(eli, rng))

    for i in range(count * 64, N):
        print("Last :::::: ", i)
        eli = jnp.stack([x_test[i] for i in range(64)], axis=0)
        result[i] = np.array(ff(eli, rng)[0])
        
    submission = pd.DataFrame({'ID':test_ds['ID'],'item_cnt_month':(y_std * result + y_mean).clip(0, max_y).ravel()})
    submission.to_csv('./data/result_submissions.csv', index=False)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
