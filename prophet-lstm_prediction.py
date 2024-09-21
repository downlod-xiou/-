import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

# 读取情感数据
data = pd.read_csv(r'D:\python\project\sentiment\Time series analysis\Results\daily_average_sentiment_filled.csv')
data['ds'] = pd.to_datetime(data['ds'])
data.set_index('ds', inplace=True)

# Prophet 模型部分
prophet_data = data.reset_index()[['ds', 'y']].rename(columns={'ds': 'ds', 'y': 'y'})

# 节假日信息
holidays_df = pd.read_csv(r'D:\python\project\sentiment\Time series analysis\predict\holidays.csv', encoding='GBK')
holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])

# 提取节假日名称和日期作为节假日信息
holidays = pd.DataFrame({
    'holiday': holidays_df['holiday'],
    'ds': holidays_df['ds'],
    'lower_window': 0,
    'upper_window': 1
})

# 设计节假日的lower_window和upper_window
holiday_windows = {
    '双十一购物节': {'lower': -7, 'upper': 7},
    '春节': {'lower': -5, 'upper': 10},
    '618购物节': {'lower': -3, 'upper': 3},
    '元旦': {'lower': -3, 'upper': 3},
    '中秋节': {'lower': -3, 'upper': 3},
    '劳动节': {'lower': -3, 'upper': 3},
    '国庆节': {'lower': -3, 'upper': 3}
}

# 更新节假日的持续时间参数
for holiday, window in holiday_windows.items():
    holidays.loc[holidays['holiday'] == holiday, 'lower_window'] = window['lower']
    holidays.loc[holidays['holiday'] == holiday, 'upper_window'] = window['upper']

# 创建 Prophet 模型实例
prophet_model = Prophet(
    growth='linear',  # 趋势为线性增长
    changepoint_prior_scale=0.1,  # 提高变化点先验比例
    changepoints=None,  # 默认值为None，表示自动检测变化点
    yearly_seasonality=True,  # 默认值为True
    weekly_seasonality=True,  # 重新启用周季节性
    seasonality_mode='multiplicative',
    seasonality_prior_scale=15.0,  # 提高季节性先验比例
    holidays_prior_scale=20.0,  # 提高节假日先验比例
    holidays=holidays,  # 加载的节假日信息
    interval_width=0.95  # 提高置信区间宽度
)

# 拟合模型
prophet_model.fit(prophet_data)

# 定义测试集大小
test_size = 365

# 创建未来日期的DataFrame
future = prophet_model.make_future_dataframe(periods=test_size)

# 进行预测
forecast = prophet_model.predict(future)

# 将 Prophet 的预测结果添加到原始数据集
data['Prophet_Pred'] = forecast.set_index('ds')['yhat'][:len(data)]

# 使用最后一年的数据作为测试集
train_data = data[:-test_size]
test_data = data[-test_size:]

# 修改特征选择
features = ['Prophet_Pred']
target = 'y'

X_train = train_data[features].values
y_train = train_data[target].values
X_test = test_data[features].values
y_test = test_data[target].values

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

# 数据转换为LSTM的三维输入
look_back = 5
def create_dataset(X, y, look_back=look_back):
    dataX, dataY = [], []
    for i in range(len(X) - look_back):
        dataX.append(X[i:(i + look_back), 0])  # 只选择第一个特征列
        dataY.append(y[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

train_X, train_y = create_dataset(X_train_scaled, y_train_scaled)
test_X, test_y = create_dataset(X_test_scaled, y_test_scaled)


# 创建LSTM模型
model = Sequential()
model.add(LSTM(200, activation='tanh', input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
history = model.fit(train_X, train_y, epochs=100, batch_size=16, validation_split=0.1, verbose=0)


# 模型预测
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

# 反归一化预测值
train_predict = y_scaler.inverse_transform(train_predict)
test_predict = y_scaler.inverse_transform(test_predict)

# 计算评价指标
train_mae = mean_absolute_error(y_train[look_back:], train_predict)
train_rmse = np.sqrt(mean_squared_error(y_train[look_back:], train_predict))
train_mape = np.mean(np.abs((y_train[look_back:] - train_predict.flatten()) / y_train[look_back:])) * 100

test_mae = mean_absolute_error(y_test[look_back:], test_predict)
test_rmse = np.sqrt(mean_squared_error(y_test[look_back:], test_predict))
test_mape = np.mean(np.abs((y_test[look_back:] - test_predict.flatten()) / y_test[look_back:])) * 100

print(f'Train MAE: {train_mae:.6f}')
print(f'Train RMSE: {train_rmse:.6f}')
print(f'Train MAPE: {train_mape:.6f}%')

print(f'Test MAE: {test_mae:.6f}')
print(f'Test RMSE: {test_rmse:.6f}')
print(f'Test MAPE: {test_mape:.6f}%')

# 绘制预测结果图
date_train = train_data.index[look_back:]
date_test = test_data.index[look_back:]

plt.figure(figsize=(16, 6))
plt.plot(date_train, y_train[look_back:], label='Actual')
plt.plot(date_train, train_predict, label='Train Predictions')
plt.plot(date_test, y_test[look_back:], label='Actual')
plt.plot(date_test, test_predict, label='Test Predictions')
plt.xlabel('Date')
plt.ylabel('Sentiment Value')
plt.title('Sentiment Value Predictions')
plt.legend()
plt.show()

# 保存真实值和预测值到本地csv文件
results = pd.DataFrame({
    'Date': np.concatenate((date_train, date_test)),
    'Actual': np.concatenate((y_train[look_back:], y_test[look_back:])),
    'Train Predictions': np.concatenate((train_predict.flatten(), np.full(len(y_test[look_back:]), np.nan))),
    'Test Predictions': np.concatenate((np.full(len(y_train[look_back:]), np.nan), test_predict.flatten()))
})
results.to_csv('predictions.csv', index=False)

# 保存评价指标到本地csv文件
evaluation = pd.DataFrame({
    'Metric': ['Train MAE', 'Train RMSE', 'Train MAPE', 'Test MAE', 'Test RMSE', 'Test MAPE'],
    'Value': [train_mae, train_rmse, train_mape, test_mae, test_rmse, test_mape]
})
evaluation.to_csv('evaluation.csv', index=False)
