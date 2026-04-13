import pandas as pd

data = pd.read_csv("finance数据集.csv")


# 显示前五行的数据
print(data.head())

import matplotlib.pyplot as plt
import seaborn as sns

# 设置图像尺寸
plt.figure(figsize=(12, 8))

# 识别数值列用于箱线图
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns

# 创建箱线图
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(x=data[col])
    plt.title(col)

plt.tight_layout()
plt.show()  # 图片显示

# 使用IQR处理异常值
# IQR（四分位距）科普：
#   将数据从小到大排列后，分成四等份：
#   Q1（第25百分位）= 下四分位数
#   Q3（第75百分位）= 上四分位数
#   IQR = Q3 - Q1，表示中间50%数据的分布范围
#   异常值判定标准：
#     低于 Q1 - 1.5*IQR 或高于 Q3 + 1.5*IQR 的点视为异常值
#   优点：不受极端值影响，比标准差更稳健
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1


# 移除异常值
data_cleaned = data[
    ~(
        (data[numeric_cols] < (Q1 - 1.5 * IQR))
        | (data[numeric_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)
]

# 检查处理重复值
duplicates = data_cleaned.duplicated()
num_duplicates = duplicates.sum()
data_cleaned = data_cleaned[~duplicates]

print(f"删除的重复行数: {num_duplicates}")


# 对数据进行归一化处理
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_cleaned[numeric_cols] = scaler.fit_transform(data_cleaned[numeric_cols])

# 设定目标变量
target_variable = data_cleaned["SeriousDlqin2yrs"]

from sklearn.model_selection import train_test_split

# 定义特征和目标
X = data_cleaned.drop(columns=["SeriousDlqin2yrs"])  # 1分
y = target_variable


# 划分数据（训练集占80%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 显示划分后的数据形状
print(f"训练数据形状: {X_train.shape}")
print(f"测试数据形状: {X_test.shape}")

# 保存清洗后的数据到CSV
cleaned_file_path = "2.1.3_cleaned_data.csv"
data_cleaned.to_csv("cleaned_file_path", index=False)