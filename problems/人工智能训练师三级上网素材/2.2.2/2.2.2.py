import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from sklearn.ensemble import RandomForestRegressor

# 总结该题目，主要是多了一个森林回归模型 RandomForestRegressor ， 需要增加一个决策树数量 n_estimators
# 还有一个pipeline ，主要是把标准化和线性回归组合在一起

# 加载数据集
df = pd.read_csv("auto-mpg.csv")

# 显示前五行数据
print(df.head())

# 处理缺失值
# 将 'horsepower' 列中的所有值转换为数值类型
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
# 删除包含缺失值的行
df = df.dropna(subset=["horsepower"])

# 选择相关特征进行建模（定义自变量（返回一个DataFrame）和因变量）
X = df[
    [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model year",
        "origin",
    ]
]
y = df["mpg"]

# 将数据集划分为训练集和测试集（测试集占比20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建包含标准化和线性回归的管道
# 是sklearn提供的流水线工具， 把多个处理步骤串联成一个整体
# 原始数据 =》 StandardScaler 标准化 =》  LinearRegression 线性回归
# 防止数据泄露：fit 时只用训练集的均值/方差来标准化，predict 时自动用同样的参数处理测试集，避免手动操作出错

# 代码简洁：不用分开写 scaler.fit_transform(X_train) + scaler.transform(X_test)，直接：


# 步骤统一管理：标准化和模型作为一个整体，保存模型时连同标准化参数一起保存
pipeline = Pipeline([("scaler", StandardScaler()), ("linreg", LinearRegression())])

# 训练模型
pipeline.fit(X_train, y_train)

# 保存训练好的模型
with open("2.2.2_model.pkl", "wb") as model_file:
    pickle.dump(pipeline, model_file)

# 预测并保存结果
y_pred = pipeline.predict(X_test)
results_df = pd.DataFrame(y_pred, columns=["预测结果"])
results_df.to_csv("2.2.2_results.txt", index=False)

# 测试模型
with open("2.2.2_report.txt", "w") as results_file:
    results_file.write(f"训练集得分: {pipeline.score(X_train, y_train)}\n")
    results_file.write(f"测试集得分: {pipeline.score(X_test, y_test)}\n")

# 创建随机森林回归模型实例（创建的决策树的数量为100）
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# 训练随机森林回归模型
rf_model.fit(X_train, y_train)

# 使用随机森林模型进行预测
y_pred_rf = rf_model.predict(X_test)

# 保存新的结果
results_rf_df = pd.DataFrame(y_pred_rf, columns=["预测结果"])
results_rf_df.to_csv("2.2.2_results_rf.txt", index=False)

# 测试模型并保存得分
with open("2.2.2_report_rf.txt", "w") as results_rf_file:
    results_rf_file.write(f"训练集得分: {rf_model.score(X_train, y_train)}\n")
    results_rf_file.write(f"测试集得分: {rf_model.score(X_test, y_test)}\n")