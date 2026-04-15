import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# 知识点总结：
# shape 表示数据的维度大小 =》 返回一个元组(行数、列数)，就是返回表里就多少行记录，多少列字段



# 1. 数据加载与探索（2分）
data = pd.read_csv("diabetes_health_data.csv")
print("数据集形状：", data.shape)
print("\n数据基本信息：")
print(data.info())
print("\n缺失值统计：")
print(data.isnull().sum())

# 2. 数据清洗 - 处理不一致的分类变量（3分）
# 统一Gender列的编码
data["Gender"] = data["Gender"].replace({"M": "Male", "F": "Female"})
data["Gender"] = data["Gender"].fillna("Male")  # 填充剩余不一致值

# 统一Smoking列的编码
smoking_mapping = {"Y": "Yes", "N": "No", "yes": "Yes", "no": "No"}
data["Smoking"] = data["Smoking"].replace(smoking_mapping)

# 3. 缺失值处理 - 采用不同策略（3分）
# 对数值型特征，使用中位数填充
numerical_cols = ["BMI", "BloodPressure", "Glucose", "Insulin", "SkinThickness"]
for col in numerical_cols:
    data[col] = data[col].fillna(data[col].median())

# 对分类特征，使用众数填充
categorical_cols = ["Smoking"]
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])


# 4. 异常值检测与处理（3分）
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 将异常值截断到边界值
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df


# 处理关键特征的异常值
outlier_columns = ["BMI", "BloodPressure", "Glucose"]
for col in outlier_columns:
    data = handle_outliers(data, col)


# 5. 特征工程 - 创建新特征（2分）
# 创建BMI分类特征
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"


data["BMI_category"] = data["BMI"].apply(categorize_bmi)

# 6. 数据编码与标准化（3分）
# 对分类变量进行标签编码
categorical_to_encode = ["Gender", "Smoking", "BMI_category"]
label_encoders = {}
for col in categorical_to_encode:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 对数值型特征进行标准化
features_to_scale = [
    "Age",
    "BMI",
    "BloodPressure",
    "Glucose",
    "Insulin",
    "SkinThickness",
    "DiabetesPedigree",
    "PhysicalActivity",
]
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# 7. 数据集划分与保存（2分）
# 选择特征和目标变量
selected_features = [
    "Age",
    "Gender",
    "BMI",
    "BloodPressure",
    "Glucose",
    "Insulin",
    "DiabetesPedigree",
    "PhysicalActivity",
    "Smoking",
    "BMI_category",
]

X = data[selected_features]
y = data["Outcome"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
# 保存处理后的数据
final_data = X.copy()
final_data["Outcome"] = y
final_data.to_csv("diabetes_processed_data.csv", index=False)

print("数据预处理完成！")
print("\n处理后的数据统计信息：")
print(final_data.describe())


# 8. 数据质量报告生成（2分）
def generate_data_quality_report(df):
    report = {
        "total_records": len(df),
        "total_features": df.shape[1], # 列数用来当做特征数量
        "missing_values": df.isnull().sum(),
        "duplicate_rows": df.duplicated().sum(),
    }
    return report


quality_report = generate_data_quality_report(final_data)
print("数据质量报告：")
for key, value in quality_report.items():
    print(f"{key}: {value}")
