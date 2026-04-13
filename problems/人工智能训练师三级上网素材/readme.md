# 数据分析核心库速查手册

## 一、NumPy：高性能数值计算基础

NumPy 的核心是 `ndarray` 对象，专注于数组操作和数学运算。

| 功能分类 | 常用指令 | 说明 |
|--------|--------|------|
| 创建数组 | `np.array([1,2,3])` | 从列表创建数组 |
| | `np.zeros((3,3))` | 创建 3x3 的全 0 矩阵 |
| | `np.arange(0, 10, 2)` | 创建步长为 2 的数组 |
| 属性查看 | `arr.shape` | 查看维度（行列数） |
| | `arr.dtype` | 查看元素数据类型 |
| 形状变换 | `arr.reshape(m, n)` | 改变数组形状 |
| | `arr.flatten()` | 将数组展平为一维 |
| 数学运算 | `np.sum(arr)` / `arr.mean()` | 求和、平均值 |
| | `np.sqrt(arr)` / `np.exp(arr)` | 逐元素开方、指数运算 |
| 索引切片 | `arr[0, 1]` | 获取第 0 行第 1 列元素 |
| | `arr[:, 0]` | 获取所有行的第 0 列 |

---

## 二、Pandas：数据分析核心工具

Pandas 提供了 `Series`（一维）和 `DataFrame`（二维表格）结构。

### 1. 数据查看与加载

- 读取数据：`pd.read_csv('file.csv')`，`pd.read_excel('file.xlsx')`
- 查看概览：
  - `df.head()` — 前 5 行
  - `df.info()` — 数据类型与缺失值
  - `df.describe()` — 数值列统计摘要

### 2. 数据选择与筛选

- 选择列：`df['column_name']`
- 按标签选择：`df.loc[row_label, col_label]`
- 按位置选择：`df.iloc[0:3, 1:2]`（选择前 3 行，第 2 列）
- 条件筛选：`df[df['age'] > 25]`

### 3. 数据清洗

- 查看缺失值：`df.isnull().sum()`
- 填充缺失值：`df.fillna(0)`
- 删除缺失值：`df.dropna(axis=0)`
- 重命名列：`df.rename(columns={'old_name': 'new_name'})`

### 4. 数据统计与聚合

- 聚合操作：`df.groupby('category')['value'].mean()`（按类别计算平均值）
- 排序：`df.sort_values(by='column', ascending=False)`
- 唯一值：`df['column'].unique()`

---

## 三、学习建议与概念理解

为了更好地理解这些库在数据分析流程中的位置，可以将其看作一个处理流水线：

- **NumPy** 就像是"地基"，负责底层的计算速度，所有数值计算都依赖它。
- **Pandas** 就像是"建筑"，它基于 NumPy 构建，增加了标签、索引和对复杂数据类型的处理能力，是处理现实世界表格数据的核心。
