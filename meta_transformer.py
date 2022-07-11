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
        x = hk.BatchNorm(False, False, decay_rate=0.9, eps=1e-6)(x, is_training)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = layer_norm(x)

        x = hk.Conv1D(output_channels=self.ff_dim, kernel_shape=1, padding="same")(x)
        x = hk.BatchNorm(False, False, decay_rate=0.9, eps=1e-6)(x, is_training)
        x = jnn.gelu(x)
        x = hk.Conv1D(output_channels=out_features, kernel_shape=1, padding="same")(x)
        x = hk.BatchNorm(False, False, decay_rate=0.9, eps=1e-6)(x, is_training)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.gelu(x)

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
        x = jnp.dot(x, hk.get_parameter('wfinal', shape=(x.shape[2], 256), init=w_init)) + hk.get_parameter('biasfinal', shape=(256,), init=hki.Constant(1e-6))
        x = jnn.gelu(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnp.dot(x, hk.get_parameter('wwfinal', shape=(256, 1), init=w_init)) + hk.get_parameter('bbiasfinal', shape=(1,), init=hki.Constant(1e-8))
        return x

def build_forward_fn(num_layers, time2vec_dim, num_heads, head_size, ff_dim=None, dropout=0.5):
    def forward_fn(x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        tr = TransformerThunk(time2vec_dim, num_heads, head_size, ff_dim, num_layers, dropout)
        return tr(x, is_training)

    return forward_fn
        
     
@ft.partial(jax.jit, static_argnums=(0, 6))
def lm_loss_fn(forward_fn, params, state, rng, x, y, is_training: bool = True) -> jnp.ndarray:
    y_pred, state = forward_fn(params, state, rng, x, is_training)
    return jnp.sqrt(jnp.mean((jnp.square(y - y_pred)))), state


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


def load_dataset(filename='./data/sales_train.csv', filename1='./data/test.csv',
        filename2='./data/shops.csv', filename3='./data/items.csv', filename4='./data/item_categories.csv'):
    train = pd.read_csv(filename)
    test = pd.read_csv(filename1)

    shops = pd.read_csv(filename2)
    items = pd.read_csv(filename3)
    cats = pd.read_csv(filename4)

    matrix = []
    cols = ["date_block_num", "shop_id", "item_id"]
    for i in range(34):
        sales = train[train.date_block_num == i]
        matrix.append(np.array(list(product( [i], sales.shop_id.unique(), sales.item_id.unique()))))
    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    matrix['date_block_num']=matrix['date_block_num'].astype(int)
    matrix['shop_id']=matrix['shop_id'].astype(int)
    matrix['item_id']=matrix['item_id'].astype(int)
    matrix.sort_values(cols,inplace=True)

    train['revenue']=train['item_cnt_day']*train['item_price']

    group=train.groupby(cols).agg({'item_cnt_day':['sum']})
    group.columns=['item_cnt_month']
    group.reset_index(inplace=True)
    matrix=pd.merge(matrix,group,on=cols,how='left')
    matrix['item_cnt_month']=matrix['item_cnt_month'].fillna(0).astype(np.float32)

    test['date_block_num']=34
    test["date_block_num"] = test["date_block_num"].astype(int)
    test['shop_id']=test['shop_id'].astype(int)
    test['item_id']=test['item_id'].astype(int)
    test.drop('ID',inplace=True,axis=1)
    matrix=pd.concat([matrix,test],ignore_index=True,sort=False,keys=cols)
    matrix.fillna(0,inplace=True)    

    matrix=pd.merge(matrix,shops,on=['shop_id'],how='left')
    matrix=pd.merge(matrix,items,on='item_id',how='left')
    matrix=pd.merge(matrix,cats,on='item_category_id',how='left')

    matrix["city"] = matrix["city"].astype(np.int8)
    matrix["category"] = matrix["category"].astype(np.int8)
    matrix["item_category_id"] = matrix["item_category_id"].astype(np.int8)
    matrix["sub_type_code"] = matrix["sub_type_code"].astype(np.int8)
    matrix["name2"] = matrix["name2"].astype(np.int8)
    matrix["name3"] = matrix["name3"].astype(np.int16)
    matrix["type_code"] = matrix["type_code"].astype(np.int8)

    def lag_feature(df,lags,cols ):
        for col in cols:
            print('Adding lag feature in ',col)
            tmp=df[['date_block_num','shop_id','item_id',col]]
            for i in lags:
                shifted=tmp.copy()
                shifted.columns=['date_block_num','shop_id','item_id',col+'_shifted_'+str(i)]
                shifted.date_block_num=shifted.date_block_num+i
                df=pd.merge(df,shifted,on=['date_block_num','shop_id','item_id'],how='left')
        return df

    matrix=lag_feature(matrix,[1,2,3],["item_cnt_month"])

    group=matrix.groupby(['date_block_num','item_category_id']).agg({'item_cnt_month':['mean']})
    group.columns=['date_item_cat_avg']
    group.reset_index(inplace=True)
    matrix=pd.merge(matrix,group,on=['date_block_num','item_category_id'],how='left')
    matrix['date_item_cat_avg']=matrix['date_item_cat_avg'].astype(np.float32)
    matrix=lag_feature(matrix,[1,2],['date_item_cat_avg'])
    matrix.drop(['date_item_cat_avg'],axis=1,inplace=True)

    group=matrix.groupby(['date_block_num','category']).agg({'item_cnt_month':['mean']})
    group.columns=['date_cat_avg']
    group.reset_index(inplace=True)

    matrix=pd.merge(matrix,group,on=['date_block_num','category'],how='left')
    matrix['date_cat_avg']=matrix['date_cat_avg'].astype(np.float32)

    matrix=lag_feature(matrix,[1,2],['date_cat_avg'])
    matrix.drop(['date_cat_avg'],axis=1,inplace=True)

    group=matrix.groupby(['date_block_num']).agg({'item_cnt_month':['mean']})
    group.columns=['date_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix=pd.merge(matrix,group,on='date_block_num',how='left')
    matrix.date_avg_item_cnt = matrix["date_avg_item_cnt"].astype(np.float32)
    matrix=lag_feature(matrix,[1,2],["date_avg_item_cnt"])
    matrix.drop(['date_avg_item_cnt'],inplace=True,axis=1)

    group=matrix.groupby(['date_block_num','item_id']).agg({'item_cnt_month':['mean']})
    group.columns=['date_item_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix=pd.merge(matrix,group,on=['date_block_num','item_id'],how='left')
    matrix.date_item_avg_item_cnt=matrix['date_item_avg_item_cnt'].astype(np.float32)
    matrix=lag_feature(matrix,[1,2,3],['date_item_avg_item_cnt'])
    matrix.drop(['date_item_avg_item_cnt'],inplace=True,axis=1)

    group=train.groupby(['item_id']).agg({'item_price':['mean']})
    group.columns=['item_id_price_avg']
    group.reset_index(inplace=True)

    matrix=pd.merge(matrix,group,on=['item_id'],how='left')
    matrix['item_id_price_avg']=matrix['item_id_price_avg'].astype(np.float32)

    group=train.groupby(['date_block_num','item_id']).agg({'item_price':['mean']})
    group.columns=['date_item_id_price_avg']
    group.reset_index(inplace=True)

    matrix=pd.merge(matrix,group,on=['date_block_num','item_id'],how='left')
    matrix['date_item_id_price_avg']=matrix['date_item_id_price_avg'].astype(np.float32)

    matrix=lag_feature(matrix,[1,2,3],['date_item_id_price_avg'])

    for i in [1,2,3]:
        matrix['delta_price_shifted_'+str(i)]=(matrix['date_item_id_price_avg_shifted_'+str(i)]-matrix['item_id_price_avg'])/matrix['item_id_price_avg']

    features_to_drop = ["item_id_price_avg", "date_item_id_price_avg"]

    matrix.drop(features_to_drop, axis = 1, inplace = True)

    X_train = matrix[matrix.date_block_num <= 33].drop(['item_cnt_month'], axis=1).values
    Y_train = matrix[matrix.date_block_num <= 33]['item_cnt_month']
    X_test = matrix[matrix.date_block_num == 34].drop(['item_cnt_month'], axis=1).values
    Y_train = Y_train.clip(0, 20)

    return jnp.array(X_train), jnp.array(Y_train), jnp.array(X_test), test


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
    max_steps = 2300
    num_heads = 8
    head_size = 128
    num_layers = 2
    dropout_rate = 0.4
    grad_clip_value = 1.0
    learning_rate = 0.0001
    time2vec_dim = 1
    batch_size = 128
    
    num_devices = jax.local_device_count()

    print("Num devices :::: ", num_devices)

    x, y, x_test, test_ds = load_dataset()

    print("Examples :::: ", x.shape)
    print("Examples :::: ", y.shape)
    print("Testing Examples :::: ", x_test.shape)

    rng1, rng = jr.split(jax.random.PRNGKey(111))
    train_dataset = get_generator_parallel(x, y, rng1, batch_size, num_devices)

    forward_fn = build_forward_fn(num_layers, time2vec_dim, num_heads, head_size, dropout=dropout_rate)

    forward_fn = hk.transform_with_state(forward_fn)

    forward_apply = forward_fn.apply
    loss_fn = ft.partial(lm_loss_fn, forward_apply)

    optimizer = optax.chain(
        optax.adaptive_grad_clip(grad_clip_value),
        #optax.sgd(learning_rate=learning_rate, momentum=0.95, nesterov=True),
        optax.radam(learning_rate=learning_rate)
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
        eli = x_test[a:b, :, None]
        fa, _ = forward_apply(params_reduced, state_reduced, rng,  eli, is_training=False)
        result = result.at[a:b].set(fa[:, 0, 0])

    result = np.array(result)

    output = pd.DataFrame({'ID': test_ds.index, 'item_cnt_month': result})
    output.to_csv('submission1.csv', index=False)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
