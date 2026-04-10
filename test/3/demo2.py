import onnxruntime as ort  # 高性能的ONNX推理引擎
import numpy as np
from PIL import Image

# 加载onnx模型
session = ort.InferenceSession('mnist.onnx')