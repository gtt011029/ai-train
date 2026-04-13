import onnxruntime
import numpy as np
from PIL import Image

# 加载ONNX模型  2分
ort_session = onnxruntime.inferencesession('mnist.onnx')

# 加载图像 2分
image = Image.open('L')  # 转为灰度图


image = image.resize((28, 28))  # 调整大小为MNIST模型的输入尺寸2分
image_array = np.array(image, dtype=np.float32)  # 转为numpy数组2分
image_array = np.expand_dims(image_array, axis=0)  # 添加batch维度2分
image_array = np.expand_dims(image_array, axis=0)  # 添加通道维度2分
