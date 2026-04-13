import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 读取数据集 2分
data = pd.read_csv('sensor_data.csv')

# 1. 传感器数据统计
# 对传感器类型进行分组，并计算每个组的数据数量和平均值 3分
sensor_stats = data.groupby(data["SensorType"])["Value"].agg(['count', 'mean'])
# 输出结果
print("传感器数据数量和平均值:")
print(sensor_stats)



# 2. 按位置统计温度和湿度数据
# 筛选出温度和湿度数据，然后按位置和传感器类型分组，计算每个组的平均值 2分 => 题目不明确，应该说筛选出 SensorType 为Temperature 和 Humidity 的数据
location_stats = data[data['SensorType'].isin
(['Temperature', 'Humidity'])].groupby(['Location', 'SensorType'])['Value'].mean().unstack()
# 输出结果
print("每个位置的温度和湿度数据平均值:")
print(location_stats)


# 3. 数据清洗和异常值处理
# 标记异常值 3分
data["is_abnormal"] = np.where(
    ((data["SensorType"] == 'Temperature') & ((data["Value"] < -10) | (data["Value"] > 50)))
    | ((data["SensorType"] == 'Humidity') & ((data["Value"] < 0) | (data["Value"] > 100))),
    True,
    False,
)
# 输出异常值数量 2分 => 统计boolean列中true的数量，最简单的方法是对该列进行求和， 在python中true为1， false为 0
print("异常值数量:", data["is_abnormal"].sum())
# 填补缺失值
# 使用前向填充(Forward fill)和后向填充(backward fill)的方法填补缺失值 4分
data["Value"].fillna(method='ffill', inplace=True)
data["Value"].fillna(method='bfill', inplace=True)
# 保存清洗后的数据
# 删除用于标记异常值的列，并将清洗后的数据保存到新的CSV文件中 4分
cleaned_data = data.drop(columns=["is_abnormal"])
data.to_csv("cleaned_sensor_data.csv", index=False) # 不保存索引
# print("数据清洗完成，已保存为 'cleaned_sensor_data.csv'")




# 备注， 也是考察 pd、np的相关操作

# agg(['count', 'mean']), pandas中非常强大的聚合方法， 允许一次性应用多个统计函数，（如技计数、平均值、总和、标准差等）。会返回一个dataFrame， 列名即为count、mean

# isin([])过滤的方法

# data['value'].fillna(method, inplace=True)  => 数据填充