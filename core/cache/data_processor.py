# core/data_processor.py
"""
该文件主要是将图片向量化存入数据库
"""#
import os
import sys
import json
import logging
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

# logging.basicConfig(level=logging.INFO)
# ========== 绝对路径导入方案 ==========
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
sys.path.insert(0, project_root)

try:
    from config.config import DATA_CONFIG, MODEL_CONFIG, MILVUS_CONFIG
except ImportError as e:
    print(f"导入配置失败: {e}")
    raise

try:
    from embedding_processor import ChineseClipImageEmbedder
    from milvus_manager import MilvusImageManager  # 使用简化版
except ImportError as e:
    print(f"导入本地模块失败: {e}")
    raise

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        # 1. 获取模型路径和模型名
        model_path_str = MODEL_CONFIG.local_model_path
        # print(model_path_str)# D:\RAG_img\models_manage\models\chinese-clip
        model_path = Path(model_path_str).resolve()
        # print(model_path)# D:\RAG_img\models_manage\models\chinese-clip

        # 2. 检查是否需要下载
        config_file = model_path / "config.json"
        safetensors_file = model_path / "model.safetensors"
        pytorch_file = model_path / "pytorch_model.bin"

        has_config = config_file.exists()
        has_weight = safetensors_file.exists() or pytorch_file.exists()

        if not (has_config and has_weight):

            model_name = "OFA-Sys/chinese-clip-vit-base-patch16"

            # 自动下载模型和处理器
            print("本地模型不完整或不存在，正在下载........")
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            # 保存到本地
            model.save_pretrained(model_path)
            processor.save_pretrained(model_path)

            print(f"模型下载完成，将保存到: {model_path}")

            # 确保父目录存在
            model_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedding_processor = ChineseClipImageEmbedder(model_path=str(model_path))
        self.milvus_manager = MilvusImageManager()  # 只存图片 + 描述
    def _read_text_file(self, file_path: str) -> str:
        """读取文本文件，支持 utf-8 / gbk / gb2312"""
        encodings = ['utf-8', 'gbk', 'gb2312']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read().strip()
            except Exception as e:
                logger.warning(f"尝试编码 {encoding} 失败: {e}")
        logger.error(f"无法读取文件: {file_path}")
        return ""

    def _load_images(self, folder_path: str) -> List[Tuple[str, Image.Image]]:
        """递归加载指定文件夹中的所有图片（包括子目录）"""
        images = []
        for root, _, files in os.walk(folder_path):  # 递归遍历所有子目录
            for file in files:
                if file.lower().endswith(tuple(DATA_CONFIG.supported_image_formats)):
                    img_path = os.path.join(root, file)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        # 使用相对路径作为 filename，避免重复名冲突
                        rel_path = os.path.relpath(img_path, folder_path)
                        images.append((rel_path, img))
                        logger.info(f"成功加载图片: {rel_path}")
                    except Exception as e:
                        logger.warning(f"加载图片失败: {img_path}, 错误: {e}")
        return images

    def process_single_attraction(self, attraction_folder: str, attraction_id: str) -> Dict:
        results = {
            "attraction_id": attraction_id,
            "image_count": 0,
            "text_content": "",
            "errors": []
        }

        # 查找 .txt 文件
        txt_files = [f for f in os.listdir(attraction_folder) if f.endswith('.txt')]
        if not txt_files:
            results["errors"].append("未找到 .txt 文件")
            return results

        txt_file = os.path.join(attraction_folder, txt_files[0])
        text_content = self._read_text_file(txt_file)
        if not text_content:
            results["errors"].append("文本为空")
            return results

        results["text_content"] = text_content

        # 加载图片
        images = self._load_images(attraction_folder)
        results["image_count"] = len(images)

        # 插入每张图片的向量 + 共享描述
        for i, (filename, img) in enumerate(images):
            image_id = f"{attraction_id}_{i}"  # 唯一 ID
            try:
                vector = self.embedding_processor.encode_image(img)
                self.milvus_manager.insert_image(
                    image_id=image_id,
                    vector=vector,
                    description=text_content  # 所有图片共用同一个描述
                )
                logger.info(f"插入图片: {image_id}, 描述长度: {len(text_content)} 字")
            except Exception as e:
                logger.error(f"插入图片失败: {image_id}, 错误: {e}")

        return results

    def process_all_attractions(self) -> Dict:
        """遍历所有手办文件夹并处理"""
        stats = {
            "total_folders": 0,
            "processed_folders": 0,
            "total_images": 0,
            "failed_folders": []
        }

        # 遍历 data 根目录下的所有文件夹
        for folder_name in os.listdir(DATA_CONFIG.data_root):
            folder_path = os.path.join(DATA_CONFIG.data_root, folder_name)
            if not os.path.isdir(folder_path):
                continue

            try:
                attraction_id = folder_name
                logger.info(f"正在处理手办信息: {attraction_id}")

                result = self.process_single_attraction(folder_path, attraction_id)

                stats["total_folders"] += 1
                if not result["errors"]:
                    stats["processed_folders"] += 1
                    stats["total_images"] += result["image_count"]
                else:
                    stats["failed_folders"].append(attraction_id)

            except Exception as e:
                logger.error(f"处理手办信息失败: {folder_name}, 错误: {e}")
                stats["failed_folders"].append(folder_name)

        logger.info("处理完成!")
        logger.info(f"总文件夹数: {stats['total_folders']}")
        logger.info(f"成功处理: {stats['processed_folders']}")
        logger.info(f"图片总数: {stats['total_images']}")
        logger.info(f"失败文件夹: {stats['failed_folders']}")

        return stats

def main():
    processor = DataProcessor()
    stats = processor.process_all_attractions()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()