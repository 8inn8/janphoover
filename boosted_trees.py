import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def load_dataset(f1='./data/sales_train.csv', f2='./data/test.csv'):
    train_ds = pd.read_csv(f1)
    test_ds = pd.read_csv(f2)
    monthly_data = train_ds.pivot_table(index = ['shop_id','item_id'], values = ['item_cnt_day'], columns = ['date_block_num'], fill_value = 0, aggfunc='sum')
    monthly_data.reset_index(inplace=True)
    train_data = monthly_data.drop(columns=['shop_id','item_id'], level=0)
    train_data.fillna(0, inplace=True)

    y_train = train_data.values[:, -1:].clip(0, 20)

    sc = StandardScaler()
    x_train = sc.fit_transform(train_data.values[:,:-1])

    test_rows = monthly_data.merge(test_ds, on = ['item_id','shop_id'], how = 'right')
    x_test = test_rows.drop(test_rows.columns[:5], axis=1).drop('ID', axis=1)
    x_test.fillna(0, inplace=True)

    x_test = sc.transform(x_test)
    x_test = x_test

    return x_train, y_train, x_test, test_ds


print("Reading data.....................")
x, y, test, test_ds = load_dataset()

print("Fitting data...................")
model = XGBRegressor(tree_method='gpu_hist', gpu_id=0, objective='reg:squarederror', n_estimators=8192, max_depth=12, eta=0.01, subsample=0.8, colsample_bytree=0.8, seed=123)
model.fit(x, y)

print("Predicting data...............")
result = model.predict(test)

print("Writing data..................")
output = pd.DataFrame({'ID': test_ds['ID'], 'item_cnt_month': result.clip(0, 20).ravel()})
output.to_csv('./data/submission1.csv', index=False)
