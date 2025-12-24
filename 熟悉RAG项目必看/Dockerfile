# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（OpenCV 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements 并安装（利用缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 默认不复制代码（开发时通过 volume 挂载，生产时才 COPY）
 COPY . .   # ← 这行先注释掉或删除！

EXPOSE 8000