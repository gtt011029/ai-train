import pandas
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据采集
# 从本地文件中读取数据  2分
data = pandas.read_csv("user_behavior_data.csv")
print("数据采集完成，已加载到DataFrame中")

# 打印数据的前5条记录  2分
print(data.head())

# 2. 数据清洗与预处理
# 处理缺失值（删除）  2分
data = data.dropna(axis=0)

# 数据类型转换
data["Age"] = data["Age"].astype(int)  # Age数据类型转换为int 2分
data["PurchaseAmount"] = data["PurchaseAmount"].astype(float)  # PurchaseAmount数据类型转换为float  2分
data["ReviewScore"] = data["ReviewScore"].astype(int)  # 数据类型转换为int 2分

# 处理异常值  2分
data = data[
    (data["Age"].between(18, 70))
    & (data["PurchaseAmount"] > 0)
    & (data["ReviewScore"].between(1, 5))
]

# 数据标准化： z-score 另均值标准差， 目的是将数据缩放到均值为0、标准差为1的分布， 公式：z = (x - mu）/sigma
data["PurchaseAmount"] = (
    data["PurchaseAmount"] - data["PurchaseAmount"].mean()
) / data["PurchaseAmount"].std() # PurchaseAmount数据标准化 2分
data["ReviewScore"] = (
    data["ReviewScore"] - data["ReviewScore"].mean()
) / data["ReviewScore"].std()  # ReviewScore数据标准化 2分

# 保存清洗后的数据  1分
data.to_csv("cleaned_user_behavior_data.csv", index=False)
print("数据清洗完成，已保存为 'cleaned_user_behavior_data.csv'")