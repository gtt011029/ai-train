import onnxruntime
import numpy as np
from PIL import Image
import scipy.special

# 1. 加载ONNX模型（2分）
session = onnxruntime.InferenceSession("animal_classifier.onnx")

# 2. 获取模型输入输出信息（2分）
input_name = session.get_input[0].name
output_name = session.get_output[0].name

# 3. 加载类别标签并输出结果（2分）
with open("animal_labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]


# 4. 图像预处理函数（4分）
def preprocess_image(image_path, target_size=(224, 224)):
    # 加载图像并转换为RGB格式
    image = _____________(image_path)._____________("RGB")

    # 调整图像大小
    image = image._____________(target_size)

    # 转换为numpy数组并归一化
    image_array = _____________(image)._____________(np.float32)
    image_array = image_array / 255.0

    # 标准化（ImageNet标准）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # 确保mean是float32
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # 确保std是float32
    image_array = (image_array - _____________) / _____________

    # 调整维度顺序为CHW
    image_array = _____________(image_array, (2, 0, 1))

    # 添加batch维度
    image_array = _____________(image_array, axis=0)
    image_array = image_array.astype(np.float32)

    return image_array


# 5. 加载并预处理测试图像（2分）
processed_image = _____________("test_animal.jpg")

# 6. 执行模型推理（2分）
outputs = _____________([output_name], {_____________: processed_image})

# 7. 后处理预测结果（4分）
# 应用softmax获取概率
probabilities = _____________._____________(outputs[0], axis=1)

# 获取最高概率的类别索引
predicted_class_idx = _____________(probabilities[0])

# 获取最高概率值
confidence = _____________[0][predicted_class_idx]

# 8. 输出结果（4分）
predicted_label = _____________[predicted_class_idx]

print(f"预测结果: {predicted_label}")
print(f"置信度: {confidence:.4f}")

# 9. 输出Top-3预测结果（2分）
top3_indices = _____________(probabilities[0])[-3:][_____________]
top3_confidences = _____________[0][top3_indices]

print("\nTop-3 预测结果:")
for i, (idx, conf) in enumerate(zip(top3_indices, top3_confidences)):
    print(f"{i + 1}. {labels[idx]} - {conf:.4f}")
