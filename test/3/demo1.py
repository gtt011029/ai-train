import onnxruntime as ort # 高性能的ONNX推理引擎
import numpy as np
import scipy.special  # 特殊函数库， 包含很多数学函数
from PIL import Image


# 预处理图像
# corp_size: 裁剪大小
# mean: 均值
# std: 标准差
def preprocess_image(
    image,
    resize_size=256,
    corp_size=224,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    # 调整图像大小
    image = image.resize((resize_size, resize_size), Image.BILINEAR)
    w, h = image.size
    left = (w - corp_size) / 2
    top = (h - corp_size) / 2
    # 裁剪图像
    image = image.crop((left, top, left + corp_size, top + corp_size))
    # 转换为numpy数组
    image = np.array(image).astype(np.float32)

    # 归一化
    image = image / 255.0 # 将像素值从【0，255】归一化待【0，1】之间
    # 标准化， 将数据转换为均值为0，标准差为1的分布，让数据以0为中心， 便于模型训练
    # 为什么要花那么大力气处理呢：
    # 激活函数的需要： 神经网络里的激活函数（如 ReLU、Sigmoid、Tanh）在0附近最敏感。通过标准化数据把数据拉回0附近，可以加快模型的收敛速度（即让模型学到更快、更稳）
    # 提取核心特征： 减去均值本质上是去掉了图像的“背景亮度”，让模型更关注物体的形状和纹理，而不是整张图是亮还是暗
    #匹配训练环境： 如果你使用的模型在训练时用了特定的mean和std，那么你在推理时必须用一模一样的参数，否则会认不出图片
    image = (image - mean) / std

    # 转换为CHW格式（模型专用）【颜色通道， 高度， 宽度】 =》 channel, height, width =》 通道数， 高度， 宽度
    image = image.transpose(2, 0, 1)
    # 转换为四维数组
    image = image.reshape((1,) + image.shape)
    return image


# 模型加载
session = ort.InferenceSession("model.onnx")


# 加载类型标签
labels_path = "label.txt"
with open(labels_path, "r") as file:
    labels = [line.strip() for line in file.readlines()]

# 获取模型输入和输出的名称
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name



# 加载图片
image = Image.open('img_test.jpg').convert('RGB')

# 预处理图片
reprocessed_image = preprocess_image(image)

# 确保输入类型的浮点类型
reprocessed_image = reprocessed_image.astype('float32')

# 进行图片识别(使用模型开始处理)
output = session.run([output_name], {input_name: reprocessed_image})[0]

# 使用softmax函数获取概率
# softmax 将一组数值转换为概率分布（所有数之和为1）
# axis 决定函数沿哪个维度进行计算 -1 表示最后一个维度
# 数据格式为 np.array([[0.05, 0.01, 0.70, 0.02, 0.03, 0.15, 0.01, 0.03]])
probabilities = scipy.special.softmax(output, axis=-1)

#获取最高的5个概率和对应的类别索引
top5_idx = np.argsort(probabilities[0])[-5:][::-1] # 拿到排序后的index
top5_prod = probabilities[0][top5_idx] # 根据index拿值


# 打印结果
print("top 5 predicted classes:")
for i in range(5):
    print(f"{i + 1}: {labels[top5_idx[i]]} - Probility: {top5_prod[i]}")