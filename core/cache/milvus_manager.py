# core/milvus_manager.py
"""
字段数过少，后续得加上新字段
"""#

import os
import sys
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType,
    utility
)
import logging
from typing import List, Optional

# ========== 添加路径设置 ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入配置
try:
    from config.config import MILVUS_CONFIG, MODEL_CONFIG
except ImportError as e:
    print(f"导入配置失败: {e}")
    raise

logger = logging.getLogger(__name__)

class MilvusImageManager:
    """专用于图片向量 + 文本描述的 Milvus 管理器"""

    def __init__(self):
        self.host = MILVUS_CONFIG.host
        self.port = MILVUS_CONFIG.port
        self.collection_name = MILVUS_CONFIG.collection_name  # 现在仅用于图片+描述
        self.index_type = MILVUS_CONFIG.index_type
        self.metric_type = MILVUS_CONFIG.metric_type
        self.nlist = MILVUS_CONFIG.nlist
        self.nprobe = MILVUS_CONFIG.nprobe

        logger.info(f"初始化 MilvusImageManager:")
        logger.info(f"  主机: {self.host}:{self.port}")
        logger.info(f"  集合: {self.collection_name}")

        self._connect()

    def _connect(self):
        """连接 Milvus"""
        try:
            connections.connect(host=self.host, port=self.port)
            logger.info(f"已连接到 Milvus: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            raise

    def create_collection(self):
        """创建仅用于图片检索的集合"""
        if utility.has_collection(self.collection_name):
            logger.info(f"集合 {self.collection_name} 已存在，跳过创建")
            return True

        try:
            fields = [
                FieldSchema(
                    name="image_id",
                    dtype=DataType.VARCHAR,
                    max_length=128,
                    is_primary=True
                ),
                FieldSchema(
                    name="vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=MODEL_CONFIG.embedding_dim
                ),
                FieldSchema(
                    name="description",
                    dtype=DataType.VARCHAR,
                    max_length=65535  # 支持较长描述
                )
            ]

            schema = CollectionSchema(fields=fields, description="图片向量 + 文本描述")
            collection = Collection(name=self.collection_name, schema=schema)

            # 创建向量索引
            collection.create_index(
                field_name="vector",
                index_params={
                    "index_type": self.index_type,
                    "metric_type": self.metric_type,
                    "params": {"nlist": self.nlist}
                }
            )
            logger.info(f"集合 {self.collection_name} 创建成功，维度: {MODEL_CONFIG.embedding_dim}")
            return True

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False

    def insert_image(self, image_id: str, vector: List[float], description: str):
        """
        插入一张图片的向量和描述

        Args:
            image_id (str): 图片唯一标识，如 "img_001"
            vector (List[float]): 512 维归一化向量
            description (str): 对应的中文文本描述
        """
        try:
            if not utility.has_collection(self.collection_name):
                self.create_collection()

            collection = Collection(self.collection_name)

            data = [
                [image_id],
                [vector],
                [description]
            ]

            mr = collection.upsert(data)
            collection.flush()
            logger.debug(f"插入图片: {image_id}, 描述: {description[:30]}...")
            return mr

        except Exception as e:
            logger.error(f"插入图片失败 (image_id={image_id}): {e}")
            return None

    def close(self):
        """关闭连接"""
        try:
            connections.disconnect(alias="default")
            logger.info("已断开 Milvus 连接")
        except Exception as e:
            logger.warning(f"断开连接时出错: {e}")