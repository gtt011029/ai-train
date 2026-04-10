# 数据集说明：
# Unnamed: 0 - 索引号。
# SeriousDlqin2yrs - 个人在过去两年内是否出现过严重的拖欠（1 表示有严重拖欠，0 表示没有）。
# RevolvingUtilizationOfUnsecuredLines - 这是指个人未偿还的信用额度与总信用额度的比例。
# age - 客户的年龄。
# NumberOfTime30-59DaysPastDueNotWorse - 在过去一段时间内，贷款逾期30至59天的次数。
# DebtRatio - 债务比率。
# MonthlyIncome - 客户的月收入。
# NumberOfOpenCreditLinesAndLoans - 正在使用的信贷账户或贷款的数量。
# NumberOfTimes90DaysLate - 贷款逾期超过90天的次数。
# NumberRealEstateLoansOrLines - 持有的房地产相关贷款或信贷的数量。
# NumberOfTime60-89DaysPastDueNotWorse - 贷款逾期60至89天的次数。
# NumberOfDependents - 家庭中依赖该个人的人数。


# 科普 sklearn =》 sclikit-learn 机器学习库 =》 将几乎所有景点的机器学习算法都封装成了统一的、易于使用的api

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle # 序列化和反序列化， 将模型保存到文件中， 以便以后使用
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# （1）正确加载数据集，显示前五行的数据。
data = pd.read_csv('finance.csv')
print(data.head())

# （2）使用Logistic模型进行模型训练，要求设定自变量和因变量，并根据自变量特征进行模型训练，最终将训练好的模型以文件名2.2.1_model.pkl保存到考生文件夹，结果文件以2.2.1_results.txt保存到考生文件夹。

# 从原始数据总剔除不需要的列
# axis=1 : 按列删除；axis=0: 按行删除
data.dropna(inplace=True)  # 删除缺失值所在的行， 清洗数据， 保证都是可用数据
X = data.drop(["SeriousDlqin2yrs", "Unnamed: 0"], axis=1)
y = data["SeriousDlqin2yrs"]
#分割训练集和数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 训练Logistic回归模型
# LogisticRegression ： 实际上是一个分类算法， 预测某件事发生的概率， 通常结果是二选一
# max_iter 规定模型最多跑多少步
model = LogisticRegression(max_iter=1000) # 实例化
model.fit(X_train, y_train) # 训练（喂数据， 让模型学习）得到模型


#保存模型（保存模型是为了以后使用， 不需要每次都训练模型）
# wb : write binary 写入二进制文件
with open('2.2.1_model.pkl', 'wb') as file:
    pickle.dump(model, file)



# （3）使用测试工具对模型进行测试，并记录测试结果，命名2.2.1_report.txt，保存到考生文件夹
y_pred = model.predict(X_test)
# DataFrame 将一个裸的数据（通常是numpy数组或列表）包装成一个带话题、有表结构的表格对象，以便能够调用pandas丰富的导出功能
# index=False 告诉pandas 不要把索引也写进去
pd.DataFrame(y_pred, columns=["预测结果"]).to_csv("2.2.1_report.txt", index=False)

# （4）对测试结果进行详细分析，并编写测试报告，包括模型性能评估、错误分析及改进建议，将答案写到答题卷文件中，答题卷文件命名为"2.2.1.docx"，保存到考生文件夹。
# zero_division 除以0， 防御性设置，当某个类别没有被预测到， 导致计算分母为0，将结果设为1
report = classification_report(y_test, y_pred, zero_division=1)
with open('2.2.1.docx', 'w') as file:
    file.write(report)


# （5）运用工具分析算法中错误案例产生的原因并进行纠正，重新得到模型训练结果，以文件名2.2.1_results_xg.txt保存到考生文件夹。
# 分析测试结果
accuracy = (y_test == y_pred).mean() # 计算准确率
print(f"准确率: {accuracy:.2%}")

# 处理数据不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 重新模型训练
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
pd.DataFrame(y_pred, columns=["预测结果"]).to_csv("2.2.1_results_xg.txt", index=False)

# （6）将以上代码以及运行结果，以html格式保存并命名为2.2.1.html，保存到考生文件夹，考生文件夹命名为"准考证号+身份证后6位"。