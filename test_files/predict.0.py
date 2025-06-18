from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import date
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime, timedelta
import yfinance as yf
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import random
import os
import tensorflow as tf

today = pd.to_datetime("today").normalize()
data = yf.download("CNYGBP=X", start="2015-01-01", end=today)
all_test_df = data
all_test_df['ER'] = (all_test_df['High'] + all_test_df['Low']) / 2
all_test_df['inverse'] = 1/all_test_df['ER']
er_df = all_test_df[['inverse']].rename(columns={"Close": "CNY/GBP"})

er_df = er_df.reset_index().rename(columns={'index': 'Date','inverse': 'ER'})
er_df['Date'] = pd.to_datetime(er_df['Date'])

df_weekday_aligned = er_df.copy()

df_weekday_aligned['trade_war'] = 0
df_weekday_aligned.loc[(df_weekday_aligned['Date'] >= '2018-03-22') & (df_weekday_aligned['Date'] <= '2018-12-01'), 'trade_war'] = 1
df_weekday_aligned.loc[(df_weekday_aligned['Date'] >= '2025-02-01') , 'trade_war'] = 1

df_all = df_weekday_aligned

df_all['prev_change'] = df_all['ER'].pct_change().fillna(0)

# Create trend direction label (1 if next day up, 0 if down)
df_all['trend_up'] = (df_all['ER'].diff().shift(-1) > 0).astype(int)
df_all.dropna(inplace=True)  # drop last row if trend_up for next day is NaN
lstm_df = df_all.drop(['prev_change'], axis=1)

lstm_df['Date'] = pd.to_datetime(lstm_df['Date'])  # 保证为 datetime 类型
lstm_df = lstm_df.set_index('Date')  # 设置为 index

np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# 固定 TensorFlow 随机种子
tf.random.set_seed(42)

# 若用GPU，确保 TensorFlow 使用确定性算法
# 注意：这会牺牲一定速度
from tensorflow.keras import backend as K
os.environ['TF_DETERMINISTIC_OPS'] = '1'

lstm_df['Date'] = lstm_df.index
today = pd.to_datetime("today")
cutoff = pd.to_datetime("2025-05-26")

lstm_df['Date'] = pd.to_datetime(lstm_df['Date'])
if today > cutoff + timedelta(days=30):
    print("超过30天，重新训练模型...")
    lstm_df_model = lstm_df[lstm_df['Date'] < today]

    lstm_df_model = lstm_df_model.drop(columns=['Date'], axis=1)
    cutoff = today
else:
    lstm_df_model = lstm_df[lstm_df['Date'] < cutoff]
    lstm_df_model = lstm_df_model.drop(['Date'], axis=1)
lstm_df = lstm_df.drop(['Date'], axis=1)

# Split data into training and testing sets using 80/20 ratio
split_idx = int(len(lstm_df_model) * 0.91)
train_df = lstm_df_model.iloc[:split_idx]
test_df = lstm_df_model.iloc[split_idx:]

# Separate features (X) and target (y)
X_train = train_df
y_train = train_df['ER']
X_test = test_df
y_test = test_df['ER']

# Normalize features and target using Min-Max scaling (fit on training data only)
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

# Prepare data for LSTM: create sequences of length `lookback` for each training example
lookback = 12  # use past 12 months to predict the next month
X_train_seq, y_train_seq = [], []
for i in range(lookback, len(X_train_scaled)):
    # sequence of `lookback` feature vectors
    X_train_seq.append(X_train_scaled[i-lookback:i])
    # target is the exchange rate at this time (one month ahead of last input month)
    y_train_seq.append(y_train_scaled[i])
X_train_seq = np.array(X_train_seq)
y_train_seq = np.array(y_train_seq)

# Prepare sequences for the test set (using preceding data for initial sequence)
X_test_seq, y_test_seq = [], []
# Concatenate train and test for sequence generation to include trailing train data in initial test sequences
total_X = np.concatenate((X_train_scaled, X_test_scaled), axis=0)
total_y = np.concatenate((y_train_scaled, y_test_scaled), axis=0)
train_len = len(X_train_scaled)
for i in range(train_len, len(total_X)):
    if i < lookback:
        continue  # skip until we have enough history
    X_test_seq.append(total_X[i-lookback:i])
    y_test_seq.append(total_y[i])
X_test_seq = np.array(X_test_seq)
y_test_seq = np.array(y_test_seq)

# Build the LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
# 改进后的双层 LSTM 架构
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(lookback, X_train_seq.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1))

# 别忘了这一步！！
model.compile(optimizer='adam', loss='mean_squared_error')

# 然后再训练
history = model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=16, validation_split=0.1, verbose=1)


test_loss_all = model_all.evaluate(X_train_seq_all, y_train_seq_all, verbose=0)
print(f"Test MSE Loss: {test_loss_all:.6f}")

# 1. 准备输入
X_latest = X_train_scaled_all[-lookback:]
X_latest = X_latest.reshape(1, lookback, X_latest.shape[1])

# 2. 预测
y_pred_scaled = model_all.predict(X_latest)

# 3. 反归一化
y_pred = target_scaler.inverse_transform(y_pred_scaled)
print(f"Predicted next exchange rate: {y_pred[0][0]:.4f}")

# 初始化：最近 lookback 天的输入序列
X_input = X_train_scaled_all[-lookback:].copy()
future_preds_scaled = []

for _ in range(5):
    # 1. 重塑成 LSTM 输入格式
    X_latest = X_input.reshape(1, lookback, X_input.shape[1])

    # 2. 预测下一日（归一化后的值）
    y_next_scaled = model_all.predict(X_latest, verbose=0)
    future_preds_scaled.append(y_next_scaled[0][0])

    # 3. 构造一个 dummy 行作为下一天的“新特征”
    # 只更新 ER 一列，其他列你可以保留最后一天的值或自己设定
    next_row = X_input[-1].copy()
    next_row[0] = y_next_scaled[0][0]  # 假设 ER 是第 0 列
    X_input = np.vstack([X_input[1:], next_row])

# 4. 反归一化所有预测值
future_preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)
future_preds = target_scaler.inverse_transform(future_preds_scaled)

# 5. 打印预测结果
for i, val in enumerate(future_preds):
    print(f"Day {i+1} prediction: {val[0]:.4f}")

er_df.index = pd.to_datetime(er_df['Date'])
df_pred = er_df['ER'].copy()
df_pred = df_pred.reset_index()
last_date = df_pred['Date'].max()

# 生成未来工作日（不包括周六、周日）
#future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_preds))

# 创建预测DataFrame
df_future = pd.DataFrame({
    'Date': future_dates,
    'ER': future_preds.flatten()
})

# 合并
df_combined = pd.concat([df_pred, df_future], ignore_index=True)

df_to_save = df_combined.tail(30)
df_to_save['delta'] = df_to_save['ER'].diff()
df_to_save['trend'] = (df_to_save['delta'] > 0).astype(int)
from scipy.special import expit
df_to_save['down_prob'] = 1 - expit(df_to_save['delta'].fillna(0) * 100)

df_to_save.to_csv(f'output/prediction_{today}.csv', index=False)