import pandas as pd
import numpy as np


# 读取数据集 1分
data = pd.read_csv("patient_data.csv")
# 1. 统计住院天数超过7天的患者数量及其占比
# 创建新列'RiskLevel'，根据住院天数判断风险等级 3分
data["RiskLevel"] = np.where(data["DaysInHospital"] > 7, "高风险患者", "低风险患者")
# 统计不同风险等级的患者数量 2分
risk_counts = data["RiskLevel"].value_counts()
# 计算高风险患者占比 1分
high_risk_ratio = risk_counts["高风险患者"] / len(data["RiskLevel"])
# 计算低风险患者占比 1分
low_risk_ratio = risk_counts["低风险患者"] / len(data["RiskLevel"])

# 输出结果
print("高风险患者数量:", risk_counts["高风险患者"])
print("低风险患者数量:", risk_counts["低风险患者"])
print("高风险患者占比:", high_risk_ratio)
print("低风险患者占比:", low_risk_ratio)


# 2. 统计不同BMI区间中高风险患者的比例和统计不同BMI区间中的患者数
# 定义BMI区间和标签
bmi_bins = [0, 18.5, 24, 28, np.inf]
bmi_labels = ["偏瘦", "正常", "超重", "肥胖"]
# 根据BMI值划分(cut)指定区间 4分
data["BMIRange"] = pd.cut(
    x=data["BMI"], bins=bmi_bins, labels=bmi_labels, right=False
)  # 使用左闭右开区间
# 计算每个BMI区间中高风险患者的比例 2分 => 首先要根据BMIRange 进行分组， 然后在过滤， 然后再得到比例
bmi_risk_rate = data.groupby(data["BMIRange"])["RiskLevel"].apply(
    lambda x: (x == "高风险患者").mean()
)
# 统计每个BMI区间的患者数量 1分
bmi_patient_count = data["BMIRange"].value_counts()

# 输出结果
print("BMI区间中高风险患者的比例和患者数:")
print(bmi_risk_rate)
print(bmi_patient_count)


# 3. 统计不同年龄区间中高风险患者的比例和统计不同年龄区间中的患者数
# 定义年龄区间和标签
age_bins = [0, 26, 36, 46, 56, 66, np.inf]
age_labels = ["≤25岁", "26-35岁", "36-45岁", "46-55岁", "56-65岁", "＞65岁"]
# 根据年龄值划分指定区间 4分
data["AgeRange"] = pd.cut(
    x=data["Age"], bins=age_bins, labels=age_labels, right=False
)  # 使用左闭右开区间
# 计算每个年龄区间中高风险患者的比例 2分
age_risk_rate = data.groupby(data["AgeRange"])["RiskLevel"].apply(
    lambda x: (x == "高风险患者").mean()
)
# 统计每个年龄区间的患者数量 1分
age_patient_count = data["AgeRange"].value_counts()

# 输出结果
print("年龄区间中高风险患者的比例和患者数:")
print(age_risk_rate)
print(age_patient_count)




# 备注： 主要考察 np和pd的用法， np是读取csv文件， 进行数据清洗、等， np主要是单纯处理数据的， 进行搜索、过滤、数据操作等

# value_counts
# 非常高频且使用的函数， 用于统计数列中各个“唯一值”出现的频率
# 返回的数据类型为 pandas.Series
# 索引（index）、值（values）,排序

# pd.cut(x=values, bins= , values=, right=)
# 按照什么分类和范围进行划分