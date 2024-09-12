import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset

# 1.数据处理
df = pd.read_csv('perrin-freres-monthly-champagne-.csv')

# print(df.head())
# print(df.tail())

df.columns = ["Month", "Sales"]

# 删除最后两行
df.drop(106, axis=0, inplace=True)
df.drop(105, axis=0, inplace=True)

df["Month"] = pd.to_datetime(df["Month"])  # 转成日期时间
df.set_index('Month', inplace=True)  # 用日期时间 替换 序号
# print(df.describe())  # 描述文件(数量、平均值、标准差(std)、最值……)

# 2. 数据可视化
ax = df.plot(y='Sales')  # 默认折线图
plt.title("Monthly Sales")
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

# 3.检验
test_result = adfuller(df['Sales'])


def adfuller_test(sales):
    # 执行 ADF 检验
    result = adfuller(sales)

    # 输出 ADF 检验的结果
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    # ADF Test Statistic：ADF 统计量。如果该值小于临界值（如 1%、5% 和 10% 的临界值），则认为时间序列是平稳的。
    # p-value：如果 p 值小于显著性水平（通常是 0.05），则认为时间序列是平稳的。
    # #Lags Used：使用的滞后期数。
    # Number of Observations Used：用于检验的有效观测数。
    # Critical Values：在不同的显著性水平下（如 1%、5% 和 10%），ADF 统计量的临界值。
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        # 强有力的证据反对零假设(Ho)，拒绝零假设。数据没有单位根，是平稳的
        print(
            "strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        # 弱证据反对零假设，时间序列有一个单位根，表明它是非平稳的
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


adfuller_test(df['Sales'])

# 4.处理，再检验
df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)  # 一次差分
df['Seasonal First Difference'] = df['Sales'] - df['Sales'].shift(12)  # 季节性差分

# 再一次 检验
adfuller_test(df['Seasonal First Difference'].dropna())
# 绘制图像
df["Seasonal First Difference"].plot(y="Seasonal First Difference")
plt.title("Seasonal First Difference")
plt.ylabel("Seasonal First Difference")
plt.show()

# 模型

autocorrelation_plot(df['Sales'])
plt.show()

# 绘制时间序列数据的自相关函数（Autocorrelation Function, ACF）和偏自相关函数（Partial Autocorrelation Function, PACF）图
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].dropna(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].dropna(), lags=40, ax=ax2)
plt.show()

# 6. 使用 ARIMA 模型
model = ARIMA(df['Sales'], order=(1, 1, 1), freq='MS')
model_fit = model.fit()

# 输出模型的摘要信息
# print(model_fit.summary())

df['forecast'] = model_fit.predict(start=90, end=103, dynamic=True)
df[['Sales', 'forecast']].plot(figsize=(12, 8))
plt.show()

model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), freq='MS')
results = model.fit(disp=False)
df['forecast'] = results.predict(start=90, end=103, dynamic=True)
df[['Sales', 'forecast']].plot(figsize=(12, 8))
plt.show()

# 生成未来日期的列表，假设我们想要预测接下来24个月的数据
future_dates = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
# 创建一个新的DataFrame，其索引为上面生成的未来日期，从第二个开始（第一个是当前数据集的最后一个索引）
future_datest_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
# 查看新创建的DataFrame的最后几条记录
future_datest_df.tail()
# 将现有的DataFrame（包含历史数据）与新创建的、用于预测的空DataFrame连接起来
future_df = pd.concat([df, future_datest_df])
# 使用之前训练好的模型（results）来预测未来的时间点（从历史数据的第105个月开始，预测到第120个月）
future_df['forecast'] = results.predict(start=104, end=120, dynamic=True)
# 绘制实际销售额与预测销售额
future_df[['Sales', 'forecast']].plot(figsize=(12, 8))
plt.show()
