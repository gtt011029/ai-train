import pandas as pd

data = pd.read_excel("大学生低碳生活行为的影响因素数据集.xlsx")
# 打印出数据集的前5行
print(data.head())

# 处理数据集中的缺失值
initial_row_count = len(data)  # 处理前的数据行数
data = data.dropna()  # 删除缺失值所在行
final_row_count = len(data)  # 处理后的数据行数
print(
    f"处理后数据行数: {final_row_count}, 删除的行数: {initial_row_count - final_row_count}"
)

# 删除重复行
data = data.drop_duplicates()

from sklearn.preprocessing import StandardScaler

numerical_features = [
    "4.您的月生活费○≦1,000元   ○1,001-2,000元   ○2,001-3,000元   ○≧3,001元"
]
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# 选择特征
selected_features = [
    "1.您的性别○男性   ○女性",
    "2.您的年级○大一   ○大二   ○大三   ○大四",
    "3.您的生源地○农村   ○城镇（乡镇）   ○地县级城市  ○省会城市及直辖市",
    "4.您的月生活费○≦1,000元   ○1,001-2,000元   ○2,001-3,000元   ○≧3,001元",
    "5.您进行过绿色低碳的相关生活方式吗?",
    "6.您觉得“低碳”，与你的生活关系密切吗？",
    "7.低碳生活是否会成为未来的主流生活方式？",
    "8.您是否认为低碳生活会提高您的生活质量？",
    "9.您从以下哪种途径获取有关于“低碳”的信息？(电视广播传媒)",
    "9 (专业环保机构)",
    "9 (报刊杂志)",
    "9(社交互动平台)",
    "9 (家人朋友同学)",
    "9 (学校宣传教育)",
    "10.您认为以下哪些方法能提高绿色低碳意识? (加强低碳宣传力度)",
    "10 (加强低碳普及教育)",
    "10 (完善绿色低碳设施)",
    "10(建立健全相关的政策法规)",
    "11.......会影响我对低碳生活的看法—家人亲友的实践",
    "身边的人的响应",
    "周围的人身体力行",
    "舆论媒体积极倡导",
    "12.我认为低碳生活…—是明智的行为 ",
    "对大家都有好处",
    "能够减缓气候变暖",
    "对解决环境问题是有益的",
    "13.—我有能力选择低碳生活",
    "我有条件实施低碳生活",
    "我有机会执行低碳生活",
    "我有足够知识进行低碳生活",
    "14.我打算以后…—减少使用一次性产品",
    "合理处理生活中的废弃物",
    "在日常生活中会进行垃圾分类",
    "尽可能劝说周围的人进行低碳生活",
    "15.日常生活中，我会......—对垃圾进行分类",
    "重复利用废旧物品",
    "避免使用一次性产品",
    "劝说周围的人进行低碳生活",
]
X = data[selected_features]

# 创建目标变量
y = data["低碳行为积极性"]

from sklearn.model_selection import train_test_split

# 数据划分（测试集取20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 合并处理后得数据，并将其保存（保存中不用额外创建索引）
cleaned_data = pd.concat([X, y], axis=1)  # 拼接的方式
cleaned_data.to_csv("2.1.2_cleaned_data.csv", index=False)
