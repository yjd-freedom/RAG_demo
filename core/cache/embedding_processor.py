# core/embedding_processor.py
import os
import torch
from PIL import Image
from typing import Union, List
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
import logging

logger = logging.getLogger(__name__)


class ChineseClipImageEmbedder:
    """中文 CLIP 图像嵌入生成器，仅支持图像向量化"""

    def __init__(self, model_path: str):
        """
        初始化中文 CLIP 模型（仅用于图像编码）

        Args:
            model_path (str): 本地模型路径，例如 "../../models/chinese-clip"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"正在从 {model_path} 加载中文 CLIP 模型（仅图像分支）...")

        # 加载完整模型（必须），但只使用视觉部分
        self.model = ChineseCLIPModel.from_pretrained(model_path)
        self.processor = ChineseCLIPProcessor.from_pretrained(model_path, use_fast=False)

        self.model.eval()
        self.model.to(self.device)
        logger.info(f"中文 CLIP 模型加载完成，运行设备: {self.device}")


    def encode_image(self, image: Union[str, Image.Image]) -> List[float]:
        """
        对单张图片进行向量化（支持文件路径或 PIL Image）
        Args:
            image (str or PIL.Image.Image): 图片路径 或 RGB PIL 图像

        Returns:
            List[float]: 归一化的 512 维嵌入向量，例如 [x1, x2, ..., x512]
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"图像文件不存在: {image}")
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"无法打开或解析图像 {image}: {e}")
        elif isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            raise TypeError("输入必须是图片路径（str）或 PIL.Image.Image 对象")

        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            embedding = image_features.cpu().numpy().tolist()[0]  # 关键：取 [0]

        return embedding  # [x1, x2, ..., x512]