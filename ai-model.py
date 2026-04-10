import pandas as pd
import numpy as np

# (1)正确加载数据集，并显示前五行的数据及数据类型。
data = pd.read_csv('auto-mpg.csv')
print(data.head())
# print(data.info())
print(data.dtypes) # 显示每一列的数据类型


# (2)检查数据集中的缺失值并删除缺失值所在的行
# inplace ： 是否原地修改
# dropna ： 删除缺失值所在的行
print('\n 检查缺失值：')
print(data.isnull().sum()) # 检查每一列的缺失值
data.dropna() # 删除缺失值所在的行

# (3)将“horsepower”列转换为数值类型，并处理转换中的异常值。
# errors="coerce" ： 将异常值转换为NaN
data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")
data.dropna()

# (4)对数值型数据进行标准化处理，确保数据在同一量纲下进行分析。
# dtypes： data types
print('\n 数据类型为：')
print(data.horsepower.dtypes)

# 标准化处理
from sklearn.preprocessing import StandardScaler
numerical_features = ['displacement', 'horsepower', 'weight', 'acceleration'] # 数值型特征
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])


# (5)根据业务需求和数据特性，选择对燃油效率预测最有用的特征：选择以下特征：'cylinders'、'displacement'、'horsepower'、'weight'、'acceleration'、'model year'、'origin'
# (6)将“mpg”设为目标变量并标注；

from sklearn.model_selection import train_test_split
selected_features = ['cylinders','displacement','horsepower','weight','acceleration','model year','origin']
X = data[selected_features]
y = data['mpg'] # 目标变量


# (7)对数据进行标注和划分；
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# (8)保存处理后的数据，并命名为：2.1.1_cleaned_data.csv，保存到考生文件夹；
# 将特征和目标变量合并到一个数据框中
cleaned_data = X.copy()
cleaned_data["mpg"] = y

cleaned_data.to_csv("2.1.1_cleaned_data.csv", index=False)
print('\n 数据已保存到：2.1.1_cleaned_data.csv')


# (9)制定数据清洗和标注规范，将答案写到答题卷文件中，答题卷文件命名为“2.1.1.docx”，保存到考生文件夹；
# 数据清洗规范：
# 1.数据加载：使用pandas库加载数据集，检查数据的基本结构和类型
# 2.检查缺失值：统计每列的缺失值，并删除缺失值所在的行，保证数据完整
# 3.转换与异常处理：将数值列（如“horsepower”）转换为数值类型，处理转换中的异常值
# 4.数据标准化：对数值型数据进行标准化，以消除量纲影响，使用标准化方法
# 5.保存清洗后的数据：将经过清洗和处理后的数据保存为新的 CSV 文件，以便后续使用。

# 数据标注规范：
# 1.数据来源：标注数据的来源，包括数据集的名称、获取日期和数据提供者
# 2.数据描述：提供详细的数据描述，包括每列数据的含义、单位和可能的取值范围
# 3.特征选择：确定对目标变量预测最有用的特征
# 4.目标变量设定：将数据集中用于预测的目标变量定义为“mpg”（燃油效率）
# 5.数据划分：将数据集划分为训练集和测试集，用于模型训练和评估
# 6.保存处理后的数据
# 7.数据清洗和标注规范文档

