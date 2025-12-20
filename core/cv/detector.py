# detector.py
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import os
import torch
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
import logging
from typing import List, Union
from pymilvus import connections, Collection, DataType
import time
from collections import OrderedDict
import threading
import queue
import asyncio
import tempfile
import pygame  # ← 使用 pygame.mixer 播放音频
import edge_tts

# =============== 配置 ===============
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "RAG_demo_data_copy"
TOP_K = 1
SAVE_CROPPED_IMAGES = False
RETRIEVAL_THRESHOLD = 0.7

DETECTION_CACHE = OrderedDict()
CACHE_TTL_SECONDS = 3.0
IOU_THRESHOLD = 0.8

DISPLAY_CACHE = OrderedDict()
DISPLAY_TTL_SECONDS = 20.0

# === 全局 1 分钟防重复播报机制 ===
GLOBAL_SPOKEN_RECORD = {}      # {description: last_spoken_timestamp}
SPOKEN_TTL_SECONDS = 60.0      # 1 分钟

# === TTS 控制 ===
TTS_QUEUE = queue.Queue()
TTS_THREAD_RUNNING = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== 初始化 pygame.mixer ==========
try:
    pygame.mixer.pre_init(frequency=24000, size=-16, channels=2, buffer=1024)
    pygame.mixer.init()
    logger.info("pygame.mixer 初始化成功")
except Exception as e:
    logger.error(f"pygame.mixer 初始化失败: {e}")
    raise


# ========== TTS 工作线程（edge-tts + pygame）==========
def tts_worker():
    """
    使用 edge-tts 生成高质量中文语音，并通过 pygame.mixer 播放。
    增强临时文件处理，避免 Windows 文件锁定导致的播放中断。
    """
    while TTS_THREAD_RUNNING:
        try:
            text = TTS_QUEUE.get(timeout=1)
            if not text or not text.strip():
                TTS_QUEUE.task_done()
                continue

            tmp_path = None
            try:
                # 创建临时 MP3 文件
                tmp_path = tempfile.mktemp(suffix=".mp3")
                logger.info(f"开始合成语音: '{text}' → {tmp_path}")

                # 使用 edge-tts 异步生成
                communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
                asyncio.run(communicate.save(tmp_path))

                # 验证文件是否生成成功且非空
                if not os.path.exists(tmp_path):
                    logger.error(f"音频文件未生成: {tmp_path}")
                    raise FileNotFoundError("Audio file not created")

                file_size = os.path.getsize(tmp_path)
                if file_size == 0:
                    logger.error(f"音频文件为空: {tmp_path}")
                    raise ValueError("Empty audio file")

                logger.info(f"音频文件已生成: {tmp_path} (大小: {file_size} bytes)")

                #  播放音频
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()

                # 等待播放完成
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)

                logger.info(f"播报完成: {text}")

            except Exception as e:
                logger.error(f"TTS 播放失败 (文本: '{text}') → 错误: {e}")
                # 可选：尝试使用备用语音
                try:
                    # 尝试其他语音，如 zh-CN-YunjianNeural
                    communicate = edge_tts.Communicate(text, "zh-CN-YunjianNeural")
                    asyncio.run(communicate.save(tmp_path))
                    pygame.mixer.music.load(tmp_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
                    logger.info(f"切换语音后成功播报: {text}")
                except:
                    logger.error("备用语音也失败")

            finally:
                # 必须 stop + unload，否则 Windows 无法删除文件
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()

                # 尝试删除临时文件
                if tmp_path and os.path.exists(tmp_path):
                    for _ in range(3):
                        try:
                            os.unlink(tmp_path)
                            break
                        except OSError:
                            time.sleep(0.1)
                    else:
                        logger.warning(f"无法删除临时音频文件: {tmp_path}")

                TTS_QUEUE.task_done()
        except queue.Empty:
            continue

# 启动 TTS 线程
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# =============== 工具函数 ===============
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def is_duplicate_detection(current_box, current_desc):
    current_time = time.time()
    expired_keys = [k for k, t in DETECTION_CACHE.items() if current_time - t > CACHE_TTL_SECONDS]
    for k in expired_keys:
        del DETECTION_CACHE[k]
    for (x1, y1, x2, y2, desc), _ in DETECTION_CACHE.items():
        if desc == current_desc:
            iou = calculate_iou(current_box, (x1, y1, x2, y2))
            if iou >= IOU_THRESHOLD:
                return True
    return False


def add_to_cache(box, desc):
    key = (*box, desc)
    DETECTION_CACHE[key] = time.time()
    if len(DETECTION_CACHE) > 20:
        DETECTION_CACHE.popitem(last=False)


# =============== 初始化 ===============
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
milvus_collection = Collection(COLLECTION_NAME)
VECTOR_FIELD_NAME = None
for field in milvus_collection.schema.fields:
    if field.dtype == DataType.FLOAT_VECTOR:
        VECTOR_FIELD_NAME = field.name
        break
if not hasattr(milvus_collection, 'is_loaded') or not milvus_collection.is_loaded():
    milvus_collection.load()

# model = YOLO(r'D:\RAG_img\data\new_dataset\runs\detect\new_yolov8n\weights\best.pt')
model = YOLO(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"data","new_dataset","runs","detect","new_yolov8n","weights","best.pt"))

def load_chinese_font(size=20):
    font_candidates = [
        "simhei.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "msyh.ttc",
    ]
    for font_path in font_candidates:
        try:
            font = ImageFont.truetype(font_path, size)
            logger.info(f"成功加载中文字体: {font_path}")
            return font
        except OSError:
            continue
    logger.warning("未找到可用中文字体，使用默认字体")
    return ImageFont.load_default()


font = load_chinese_font(20)


class ChineseClipImageEmbedder:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ChineseCLIPModel.from_pretrained(model_path)
        self.processor = ChineseCLIPProcessor.from_pretrained(model_path, use_fast=False)
        self.model.eval()
        self.model.to(self.device)

    def encode_image(self, image: Union[str, Image.Image]) -> List[float]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("Unsupported image type")
        if image.mode != "RGB":
            image = image.convert("RGB")
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            return feats.cpu().numpy()[0].tolist()


# clip_embedder = ChineseClipImageEmbedder(r"D:\RAG_img\models_manage\models\chinese-clip")
clip_embedder = ChineseClipImageEmbedder(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"models_manage","models","chinese-clip"))


if SAVE_CROPPED_IMAGES:
    os.makedirs("cropped_objects", exist_ok=True)


# =============== 核心检测函数 ===============
def detect_and_retrieve(frame):
    """
    检测并检索描述，实现：
      - 播放完整的 description（如“迪迦奥特曼是光之巨人”）
      - 1 分钟内同一描述不重复播报
      - 使用 edge-tts + pygame 高质量语音
    """#
    current_time = time.time()

    # 清理 DISPLAY_CACHE
    expired_keys = [k for k, v in DISPLAY_CACHE.items() if current_time - v['last_seen'] > DISPLAY_TTL_SECONDS]
    for k in expired_keys:
        del DISPLAY_CACHE[k]

    results = model(frame, conf=0.7, iou=0.5)
    original_frame = frame
    new_detections = []

    for result in results:
        for box in result.boxes:
            yolo_conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            object_img = frame[y1:y2, x1:x2]
            if object_img.size == 0:
                continue

            try:
                embedding = clip_embedder.encode_image(object_img)
                search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
                hits = milvus_collection.search(
                    data=[embedding],
                    anns_field=VECTOR_FIELD_NAME,
                    param=search_params,
                    limit=TOP_K,
                    output_fields=["description"]
                )

                desc = None
                similarity = 0.0

                if hits and len(hits[0]) > 0:
                    hit = hits[0][0]
                    cosine_similarity = hit.distance
                    if cosine_similarity >= RETRIEVAL_THRESHOLD:
                        desc = hit.entity.get("description", "未知角色")
                        similarity = round(float(cosine_similarity), 2)
                        logger.info(
                            f"检测详情 | YOLO置信度: {yolo_conf:.3f} | RAG相似度: {cosine_similarity:.3f} | 描述: {desc[:30]}..."
                        )
                    else:
                        continue
                else:
                    continue

                # === 1 分钟全局去重 ===
                last_spoken = GLOBAL_SPOKEN_RECORD.get(desc, 0)
                if current_time - last_spoken < SPOKEN_TTL_SECONDS:
                    logger.debug(f"'{desc}' 在过去 1 分钟内已播报，跳过")
                    continue

                current_box = (x1, y1, x2, y2)
                if is_duplicate_detection(current_box, desc):
                    continue

                # 记录播报时间
                GLOBAL_SPOKEN_RECORD[desc] = current_time
                add_to_cache(current_box, desc)

                # 更新显示缓存
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                cache_key = (desc, x_center, y_center)
                DISPLAY_CACHE[cache_key] = {
                    'bbox': [x1, y1, x2, y2],
                    'desc': desc,
                    'last_seen': current_time
                }

                new_detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "description": desc,
                    "similarity": similarity
                })

                # === 关键修改：播放完整 description ===
                tts_text = f"发现{desc}"  # 使用完整文本
                TTS_QUEUE.put(tts_text)
                logger.info(f" 触发语音: {tts_text}")

            except Exception as e:
                logger.error(f"处理失败: {e}")
                continue

    logger.info(
        f"本次检测返回 {len(new_detections)} 个结果: {[d['description'][:20] + '...' for d in new_detections]}")
    return original_frame, new_detections