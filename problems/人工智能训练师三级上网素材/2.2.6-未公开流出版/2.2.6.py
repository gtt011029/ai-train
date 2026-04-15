import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据加载与探索 (2分)
data = pd.read_csv("ecommerce_customer_behavior.csv")
print("数据集形状：", data.shape)
print("\n数据基本信息：")
print(data.info())
print("\n目标变量分布：")
print(data["HighValuePurchase"].value_counts())

# 2. 数据预处理 (3分)
# 处理分类变量编码
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])
# 选择特征和目标变量
features = [
    "Age",
    "Gender",
    "CityTier",
    "Tenure",
    "AvgSessionDuration",
    "PageViewsPerSession",
    "BounceRate",
    "TotalTransactions",
    "AvgOrderValue",
    "DaysSinceLastPurchase",
]
X = data[features]
y = data["HighValuePurchase"]
# 3. 数据集划分 (1分)  stratify 按照y进行分层
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练样本数: {X_train.shape[0]}, 测试样本数: {X_test.shape[0]}")

# 4. 处理数据不平衡（2分）
print("处理前类别分布:")
print(y_train.value_counts())
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print("处理后类别分布:")
print(pd.Series(y_resampled).value_counts())

# 5. 构建逻辑回归模型（3分）
# 创建包含标准化和逻辑回归的管道
lr_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)
# 训练模型
lr_pipeline.fit(X_resampled, y_resampled)

# 预测
y_pred_lr = lr_pipeline.predict(X_test)

# 6. 构建随机森林模型（2分）
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)
y_pred_rf = rf_model.predict(X_test)


# 7. 模型评估与比较（4分）
def evaluate_model(model_name, y_true, y_pred):
    print(f"\n{model_name} 模型评估:")
    print("分类报告:")
    print(classification_report(y_true, y_pred))
    # 计算准确率
    accuracy = roc_auc_score(y_true, y_pred)
    print(f"准确率: {accuracy:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("混淆矩阵:")
    print(cm)

    return accuracy


# 评估逻辑回归模型
accuracy_lr = evaluate_model("逻辑回归", y_test, y_pred_lr)

# 评估随机森林模型
accuracy_rf = evaluate_model("随机森林", y_test, y_pred_rf)

# 8. 模型保存与结果输出（3分）
# 保存最佳模型
if accuracy_lr > accuracy_rf:
    best_model = lr_pipeline
    best_model_name = "逻辑回归"
else:
    best_model = rf_model
    best_model_name = "随机森林"

print(f"\n最佳模型: {best_model_name}")

# 保存最佳模型
with open("best_ecommerce_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

# 保存预测结果（不需要概率列）
results_df = pd.DataFrame(
    {"Actual": y_test, "Predicted_LR": y_pred_lr, "Predicted_RF": y_pred_rf}
)

results_df.to_csv("model_comparison_results.csv", index=False)

print("建模完成！所有结果已保存。")
print("\n=== 模型比较总结 ===")
print(f"逻辑回归准确率: {accuracy_lr:.4f}")
print(f"随机森林准确率: {accuracy_rf:.4f}")
print(f"最佳模型: {best_model_name}")
