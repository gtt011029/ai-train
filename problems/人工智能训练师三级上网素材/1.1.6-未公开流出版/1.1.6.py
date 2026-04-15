import pandas as pd
import numpy as np

# 1. 数据采集
data = pd.read_csv("ecommerce_user_data.csv")
print("数据采集完成，已加载到DataFrame中")

# 显示前五行数据
print(data.head())

# 2. 数据清洗与预处理
# 处理缺失值
data = data.dropna()

# 数据类型转换
data["Age"] = data["Age"].astype(int)  # Age数据类型转换为int
data["PurchaseAmount"] = data["PurchaseAmount"].astype(
    float
)  # PurchaseAmount数据类型转换为float
data["Rating"] = data["Rating"].astype(int)  # Rating数据类型转换为int

# 处理异常值
data = data[
    data["Age"].between(18, 70)
    & (data["PurchaseAmount"] > 0)
    & data["Rating"].between(1, 5)
]

# 新增 AgeGroup 字段
bins = [18, 25, 35, 45, 55, 70]
labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
data["AgeGroup"] = pd.cut(data["Age"], bins=bins, labels=labels, right=True)

# 保存清洗后数据
data.to_csv("cleaned_ecommerce_data.csv", index=False)
print("数据清洗完成，已保存为 'cleaned_ecommerce_data.csv'")

# 3. 数据统计分析
# 每个商品类别的购买人数
category_count = data["ProductCategory"].value_counts()
print("\n每个商品类别的购买人数:\n", category_count)

# 不同性别的平均购买金额
avg_purchase_by_gender = data.groupby(["Gender"])["PurchaseAmount"].mean()
print("\n不同性别的平均购买金额:\n", avg_purchase_by_gender)

# 各年龄段用户数量
age_group_count = data["AgeGroup"].value_counts.sort_index()
print("\n各年龄段的用户数量:\n", age_group_count)
