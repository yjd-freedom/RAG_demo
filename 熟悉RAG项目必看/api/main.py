"""
FastAPI 接口入口：提供 HTTP API 触发手办向量数据库构建。
支持两种模式：
  1. 不传 data_root → 使用 config/config.py 中的默认路径
  2. 传入 data_root → 使用指定的本地绝对路径（必须存在）

新增功能（2025年12月）：
  - 上传多种压缩包（ZIP / 7z / TAR 系列）并解压到 E:/YOLO_RAG_demo/data
  - 查看 data 目录下所有手办文件夹
  - 删除指定手办文件夹（用于清理）
"""

import os
import sys
import logging
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, ConfigDict
from typing import Optional, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ========== 添加项目根目录到 Python 路径 ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ========== 导入核心模块 ==========
try:
    from core.data_processor import DataProcessor
except ImportError as e:
    print(f"❌ 无法导入 DataProcessor: {e}")
    raise

# ========== 尝试导入可选压缩库（不影响主流程）==========
HAS_PY7ZR = False

try:
    import py7zr

    HAS_PY7ZR = True
except ImportError:
    logging.warning("未安装 py7zr，7z 格式将不可用。请运行: pip install py7zr")

# ========== 固定数据目录 ==========
UPLOAD_DATA_ROOT = Path("E:/YOLO_RAG_demo/data")
UPLOAD_DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ========== FastAPI 应用初始化 ==========
app = FastAPI(
    title="手办向量数据库构建服务",
    description="""
    提供 RESTful API，用于构建手办图片与文本的向量数据库。

    新增管理功能：
    - 上传 ZIP/7z/TAR 等压缩包（自动解压到 E:/YOLO_RAG_demo/data）
    - 查看/删除已上传的手办文件夹
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ========== 【新增】静态文件服务与前端页面支持 ==========
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def redirect_to_manager():
    """
    访问 http://localhost:8000 时自动跳转到前端管理页面。
    """
    return RedirectResponse(url="/static/manager.html")


# ========== 请求模型 ==========
class BuildVectorDBRequest(BaseModel):
    data_root: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data_root": "E:/YOLO_RAG_demo/data/小新"
            }
        }
    )


# ========== 健康检查 ==========
@app.get("/health", summary="健康检查")
def health_check():
    return {
        "status": "ok",
        "service": "Handmade Figure Vector Builder API",
        "message": "Service is up and running!"
    }


# ========== 构建向量库 ==========
@app.post("/build_vector_db", summary="构建向量数据库")
def build_vector_db(request: BuildVectorDBRequest):
    data_root = request.data_root

    if data_root is not None:
        if not os.path.isabs(data_root):
            raise HTTPException(
                status_code=400,
                detail="❌ data_root 必须是绝对路径（例如：C:\\data）"
            )
        if not os.path.exists(data_root):
            raise HTTPException(status_code=400, detail=f"❌ 路径不存在: {data_root}")
        if not os.path.isdir(data_root):
            raise HTTPException(status_code=400, detail=f"❌ 路径不是有效目录: {data_root}")

    try:
        processor = DataProcessor(data_root=data_root)
        stats = processor.process_all_attractions()
        return {
            "status": "success",
            "message": "✅ 向量数据库构建完成",
            "stats": stats
        }
    except ValueError as ve:
        logging.error(f"路径错误: {ve}")
        raise HTTPException(status_code=400, detail=f"路径错误: {str(ve)}")
    except Exception as e:
        logging.exception("构建失败堆栈跟踪:")
        raise HTTPException(status_code=500, detail=f"❌ 构建失败: {str(e)}")


# ========== 多格式压缩包上传 ==========
@app.post("/upload_folder", summary="上传手办数据压缩包（支持 ZIP/7z/TAR 等）")
async def upload_folder(file: UploadFile = File(...)):
    # ✅ 修改：移除 .rar 支持，防止因 unrar 缺失导致失败
    supported_exts = ['.tar.gz', '.tar.bz2', '.tar.xz', '.tgz', '.tbz2', '.zip', '.7z', '.tar']

    filename_lower = file.filename.lower()
    matched_ext = None
    for ext in supported_exts:
        if filename_lower.endswith(ext):
            matched_ext = ext
            break

    if matched_ext is None:
        ext_list = ', '.join(supported_exts)
        raise HTTPException(status_code=400, detail=f"仅支持以下格式: {ext_list}")

    # ✅ 提示用户不推荐使用 RAR
    if 'rar' in file.filename.lower():
        raise HTTPException(status_code=400, detail="⚠️ 不支持 RAR 格式，请使用 ZIP、TAR 或 7z 格式上传。")

    # 提取文件夹名（移除扩展）
    folder_name = file.filename[:-len(matched_ext)]
    if folder_name.endswith('.tar'):  # 处理 .tar.gz -> 移除 .tar
        folder_name = folder_name[:-4]

    target_path = UPLOAD_DATA_ROOT / folder_name

    # 安全校验：防止路径穿越
    try:
        target_path.resolve().relative_to(UPLOAD_DATA_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="非法文件夹名称")

    if target_path.exists():
        shutil.rmtree(target_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_path = tmp_path / file.filename

        with open(archive_path, "wb") as f:
            f.write(await file.read())

        extracted_root = tmp_path / "extracted"
        extracted_root.mkdir()

        # 解压逻辑
        try:
            if matched_ext == '.zip':
                import zipfile
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(extracted_root)

            elif matched_ext == '.7z':
                if not HAS_PY7ZR:
                    raise HTTPException(status_code=400, detail="服务器未安装 7z 支持，请使用 ZIP 或 TAR 格式")
                with py7zr.SevenZipFile(archive_path, mode='r') as zf:
                    zf.extractall(path=extracted_root)

            elif matched_ext in ('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz'):
                import tarfile
                with tarfile.open(archive_path, 'r:*') as tf:
                    tf.extractall(path=extracted_root)

            else:
                raise ValueError("未知压缩格式")

        except Exception as e:
            logging.exception("解压过程出错")
            raise HTTPException(status_code=400, detail=f"解压失败: {str(e)}")

        # 处理解压结构
        items = list(extracted_root.iterdir())
        if len(items) == 1 and items[0].is_dir():
            shutil.move(str(items[0]), str(target_path))
        else:
            target_path.mkdir()
            for item in extracted_root.iterdir():
                shutil.move(str(item), str(target_path / item.name))

    return {
        "status": "success",
        "message": f"✅ 已解压到 {target_path}",
        "folder_name": folder_name,
        "format": matched_ext
    }


# ========== 列出所有文件夹 ==========
@app.get("/list_folders", summary="列出所有手办文件夹")
def list_folders():
    folders = []
    for item in UPLOAD_DATA_ROOT.iterdir():
        if item.is_dir():
            # ✅ 统计图片数量（常见图像格式）
            image_count = sum(
                1 for f in item.rglob("*")
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            )

            # ✅ 新增：统计文本文件数量（如 .txt, .md）
            text_count = sum(
                1 for f in item.rglob("*")
                if f.is_file() and f.suffix.lower() in ['.txt', '.md', '.csv']
            )

            folders.append({
                "name": item.name,
                "path": str(item),
                "image_count": image_count,
                "text_count": text_count,  # ✅ 新增字段
                "created_time": item.stat().st_ctime
            })
    folders.sort(key=lambda x: x["name"])
    return {"folders": folders}


# ========== 删除文件夹 ==========
@app.delete("/delete_folder/{folder_name}", summary="删除指定手办文件夹")
def delete_folder(folder_name: str):
    if any(c in folder_name for c in ["..", "/", "\\", ":"]):
        raise HTTPException(status_code=400, detail="文件夹名包含非法字符")

    folder_path = UPLOAD_DATA_ROOT / folder_name

    try:
        folder_path.resolve().relative_to(UPLOAD_DATA_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="非法路径")

    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="文件夹不存在")
    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail="路径不是目录")

    try:
        shutil.rmtree(folder_path)
        return {"status": "success", "message": f"✅ 已删除: {folder_name}"}
    except Exception as e:
        logging.error(f"删除失败 {folder_path}: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")