import pandas as pd
import numpy as np


data = pd.read_csv('patient_data.csv')
# print(data.head()) # 输出前5行数据， 快速了解数据概况
# print(data.info()) # 输出数据类型和缺失值信息
# print(data.describe()) # 输出数据统计信息
# print(data.isnull().sum()) # 输出数据缺失值统计
# print(data.corr()) # 输出数据相关性矩阵
# print(data.skew()) # 输出数据偏度
# print(data.kurt()) # 输出数据峰度
# print(data.mode()) # 输出数据众数
# print(data.median()) # 输出数据中位数
# print(data.mean()) # 输出数据平均值
# print(data.std()) # 输出数据标准差
# 创建RiskLevel列，根据住院天数判断风险等级
data["RiskLevel"] = np.where(data["DaysInHospital"] > 7, "高风险患者", "低风险患者")
#(1)统计不同风险等级的患者
risk_count = data["RiskLevel"].value_counts()
print(risk_count)
high_risk_ratio = risk_count["高风险患者"] / len(data)
low_risk_ratio = risk_count["低风险患者"] / len(data)
print(f"高风险患者比例: {high_risk_ratio:.2%}")
print(f"低风险患者比例: {low_risk_ratio:.2%}")

# (2)统计不同BMI区间中高风险患者的比例和统计不同BMI区间中的患者数
bmi_bins = [0, 18.5, 24, 28, np.inf]  # 定义BMI区间 Binning（分箱）
bmi_labels = ['偏瘦', '正常', '超重', '肥胖'] # 定义BMI标签
#根据BMI值划分指定区域
data['BMIRange'] = pd.cut(data["BMI"], bins=bmi_bins, labels=bmi_labels, right=False)
# 计算每个区间中，高风险患者的比例
bmi_risk_rate = data.groupby("BMIRange")["RiskLevel"].apply(lambda x:(x == '高风险患者')).mean()
print(bmi_risk_rate)
bmi_patient_count = data["BMIRange"].value_counts()
print(bmi_patient_count)


# (3)统计不同年龄区间中高风险患者的比例和统计不同年龄区间中的患者数
age_bins = [0, 26, 36, 46, 56, 66, np.inf]
age_labels = ['25岁及以下', '26-35岁', '36-45岁', '46-55岁', '56-65岁', '65岁以上']
data['age_range'] = pd.cut(data['Age'], bins = age_bins, labels = age_labels, right = False)
age_risk_rate = data.groupby("age_range")["RiskLevel"].apply(lambda x:(x == '高风险患者')).mean()
age_risk_count = data["age_range"].value_counts()
print(age_risk_rate)
print(age_risk_count)
