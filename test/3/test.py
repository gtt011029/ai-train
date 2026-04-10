from PIL import Image
import cv2
import numpy as np
import scipy.special # 特殊函数库， 包含很多数学函数

# 1、加载模型
# 2、读取图片


# 3、图片预处理：尺寸、格式、数值范围
# 图像归一化
image = Image.open("img_test.jpg").convert("RGB")  # 转化成RGB模式
orig_image = cv2.imread("img_test.jpg")  # 使用opencv读取图片， 默认为bgr格式

# 调整图像大小
image = image.resize(28, 28)
cv2.resize(image, (320, 240))


# 转化为numpy数组
image_array = np.array(image, dtype=np.float32)

# 4、模型推理
#


# 5、后处理
# 分类任务
predicted_class = np.argmax(output)  # output 是上面模型推理的输出， 取最大值对应的索引， 即预测的类别
print(f"predicted_class: {predicted_class}")

proabilities = scipy.special.softmax(output, axis=-1)  # softmax函数： 将输出转为概率
top5_idx = np.argsort(proabilities[0])[-5:][::-1] # 取概率最大的5个值的索引， 并倒序
top5_prod = proabilities[0][top5_idx]
for i in range(5):
    print(f"{i + 1}:类别： {top5_idx[i]} 概率： {top5_prod[1]}")


with open('label.txt') as file:
    labels = [line.strip() for line in file.readlines()]
predicted_label = labels[predicted_class]
print(f"预测类别名称: {predicted_label}")


# 字典映射
emotion_tabel = {
    'neutral': 0,
    'happiness': 1,
    'sadness': 2,
    'anger': 3,
    'fear': 4,
    'surprise': 5,
    'disgust': 6,
    'contempt': 7,
    'unknown': 8,
    'NF': 9,
}

# 获取到对应的情感名称
predicted_emotion = list[emotion_tabel.keys()][predicted_label]
print(f"预测情感名称: {predicted_emotion}")