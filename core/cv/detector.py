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
import requests  # ← 新增：用于调用 Ollama API
import json

# =============== 配置 ===============
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "RAG_demo_data"
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

# === Ollama LLM 配置（新增）===
USE_OLLAMA = True
OLLAMA_MODEL = "qwen3-vl:4b-instruct"  # 可替换为 llama3:8b、phi3、mistral 等支持中文的模型
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 5.0  # 秒

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


# ========== Ollama 调用函数（新增）==========
def generate_llm_reply(description: str) -> str:
    """
    调用 Ollama 生成一句自然、友好的中文回复。
    输入：RAG 检索到的 description（如“迪迦奥特曼是光之巨人”）
    输出：如“看！这是迪迦奥特曼，他是守护地球的光之巨人！”
    若失败，返回 None。
    """
    if not USE_OLLAMA:
        return None

    prompt = (
        f"你是一个专业的手办售卖员。请用几句简短、自然、口语化的中文描述下面的角色，"
        f"要突出这个手办的特点与卖点，让人有购买欲望。角色信息：{description}\n"
        f"⚠️ 重要说明：\n"
        f"1. 必须写满200个字以上，最多不超过500字。\n"
        f"2. 不要加引号、不要加标点符号结尾，直接输出一段连续文字。\n"
        f"3. 如果不够200字，请自动补充细节，比如角色背景、战斗能力、稀有度、收藏价值等。\n"
        f"4. 使用生动的语言，激发买家兴趣，像在直播间介绍商品一样。\n"
        f"5. 严禁使用‘等等’、‘总之’这类模糊词，必须具体描述。\n"
        f"6. 输出前请检查字数是否达标，若不足则补足。\n"
        f"7. 请确保内容连贯、流畅，避免重复。\n"
        f"8. 开头可以称呼观众：‘嘿兄弟姐妹们’之类的。\n"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 400
        }
    }

    try:
        logger.debug(f"正在调用 Ollama 生成回复，模型: {OLLAMA_MODEL}")
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )
        if response.status_code == 200:
            result = response.json()
            reply = result.get("response", "").strip()
            if reply:
                # 清理可能的多余空格或换行
                reply = " ".join(reply.split())
                logger.info(f"Ollama 生成回复: '{reply}'")
                return reply
            else:
                logger.warning("Ollama 返回空回复")
        else:
            logger.error(f"Ollama 请求失败，状态码: {response.status_code}, 响应: {response.text}")
    except Exception as e:
        logger.error(f"调用 Ollama 出错: {e}")

    return None


# ========== TTS 工作线程（保持不变）==========
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
                tmp_path = tempfile.mktemp(suffix=".mp3")
                logger.info(f"开始合成语音: '{text}' → {tmp_path}")

                communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
                asyncio.run(communicate.save(tmp_path))

                if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                    raise FileNotFoundError("Audio file invalid")

                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)

                logger.info(f"播报完成: {text}")

            except Exception as e:
                logger.error(f"TTS 播放失败 (文本: '{text}') → 错误: {e}")
                try:
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
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
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

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()


# =============== 工具函数（保持不变）==============
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


# =============== 初始化（保持不变）==============
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
milvus_collection = Collection(COLLECTION_NAME)
VECTOR_FIELD_NAME = None
for field in milvus_collection.schema.fields:
    if field.dtype == DataType.FLOAT_VECTOR:
        VECTOR_FIELD_NAME = field.name
        break
if not hasattr(milvus_collection, 'is_loaded') or not milvus_collection.is_loaded():
    milvus_collection.load()

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

clip_embedder = ChineseClipImageEmbedder(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"models_manage","models","chinese-clip"))

if SAVE_CROPPED_IMAGES:
    os.makedirs("cropped_objects", exist_ok=True)


# =============== 核心检测函数（关键修改：添加原始帧尺寸字段）==============
def detect_and_retrieve(frame):
    """
    检测并检索描述，实现：
      - 若启用 Ollama，则用 LLM 生成自然语言回复
      - 否则使用默认 "发现{desc}"
      - 1 分钟内同一描述不重复播报
      - 使用 edge-tts + pygame 高质量语音
    """
    current_time = time.time()

    expired_keys = [k for k, v in DISPLAY_CACHE.items() if current_time - v['last_seen'] > DISPLAY_TTL_SECONDS]
    for k in expired_keys:
        del DISPLAY_CACHE[k]

    results = model(frame, conf=0.7, iou=0.5)
    original_frame = frame
    new_detections = []

    # ✅ 获取原始帧的宽高（用于前端正确裁剪）
    original_height, original_width = frame.shape[:2]

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

                last_spoken = GLOBAL_SPOKEN_RECORD.get(desc, 0)
                if current_time - last_spoken < SPOKEN_TTL_SECONDS:
                    logger.debug(f"'{desc}' 在过去 1 分钟内已播报，跳过")
                    continue

                current_box = (x1, y1, x2, y2)
                if is_duplicate_detection(current_box, desc):
                    continue

                GLOBAL_SPOKEN_RECORD[desc] = current_time
                add_to_cache(current_box, desc)

                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                cache_key = (desc, x_center, y_center)
                DISPLAY_CACHE[cache_key] = {
                    'bbox': [x1, y1, x2, y2],
                    'desc': desc,
                    'last_seen': current_time
                }

                detection_item = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "description": desc,
                    "similarity": similarity,
                    # 新增字段：供前端裁剪使用（必须与 YOLO 坐标一致）
                    "original_width": original_width,
                    "original_height": original_height
                }

                # === 关键修改：生成 Ollama 回复，并添加到检测结果中 ===
                llm_reply = None
                if USE_OLLAMA:
                    llm_reply = generate_llm_reply(desc)
                    if not llm_reply:
                        llm_reply = f"发现{desc}"  # fallback
                else:
                    llm_reply = f"发现{desc}"

                detection_item["llm_reply"] = llm_reply  # ← 添加 LLM 回复

                new_detections.append(detection_item)

                # 发送 TTS 请求
                TTS_QUEUE.put(llm_reply)
                logger.info(f" 触发语音: {llm_reply}")

            except Exception as e:
                logger.error(f"处理失败: {e}")
                continue

    logger.info(
        f"本次检测返回 {len(new_detections)} 个结果: {[d['description'] + '...' for d in new_detections]}")
    return original_frame, new_detections