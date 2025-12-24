# config/config.py
import os
from dataclasses import dataclass
from typing import Optional
import torch

# 获取当前文件路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 获取获取 上一级目录（父目录）
parent_dir = os.path.dirname(current_file_dir)
# print("当前文件路径",current_file_dir)
# print("父目录",parent_dir)
@dataclass           # 装饰器
class ModelConfig:
    """模型配置"""
    clip_model_name: str = "Chinese-clip-vit-base-patch16"          # 多模态Embedding model 可以把图片和文本转为向量
    embedding_dim: int = 512                                        # 向量维度（不能自定义，固定锁死）
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # local_model_path: str = "D:/RAG_project/models_manage/models/chinese-clip"
    local_model_path: str = os.path.join(parent_dir, "models_manage", "models", "chinese-clip")
    # print(local_model_path)

@dataclass
class MilvusConfig:
    """Milvus 配置"""
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "yolo_RAG_data"

    # 索引配置
    index_type: str = "IVF_FLAT"    # 索引类型
    metric_type: str = "IP"         # 内积，适合CLIP归一化向量
    nlist: int = 1024               # nlist：聚类中心数量（Number of Clusters）
    nprobe: int = 10                # nprobe：查询时扫描的簇数量

    # 分片配置
    # shards_num: int = 2 # 可以在MIlvus集群时用，这里是（单机）模式，不是集群（Cluster）模式
    """
    分片
    分片（Sharding） 是把同一个集合（collection）中的数据（比如你提到的“数据A”——即所有景点向量）水平切分成多份。
    这些分片可以分布到不同的物理节点（在 Milvus 集群模式下）。
    查询时，Milvus 会并行地在所有分片上执行搜索，然后合并结果。
    所以，分片的主要目的之一就是提升检索（和写入）的吞吐与速度，尤其是在大规模数据或高并发场景下
    """

@dataclass
class DataConfig:
    """数据配置"""
    # data_root: str = "D:/RAG_img/data"
    data_root: str = os.path.join(parent_dir,"data")
    # print(data_root)

    # 图片处理
    supported_image_formats: list = None

    def __post_init__(self):
        if self.supported_image_formats is None:
            self.supported_image_formats = ['.png', '.jpg', '.jpeg', '.webp']

# RAGConfig 后续如果需要用到LLM来生成回答可以用上，目前用不到
@dataclass
class RAGConfig:
    """RAG 配置"""
    # 回答生成配置
    use_llm: bool = True
    llm_model: Optional[str] = "qwen3-vl:4b-instruct"
    llm_api_key: Optional[str] = None

# 全局配置实例
MODEL_CONFIG = ModelConfig()
MILVUS_CONFIG = MilvusConfig()
DATA_CONFIG = DataConfig()
RAG_CONFIG = RAGConfig()
