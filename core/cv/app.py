# realtime_app.py （修改版）

import cv2
import logging
from flask import Flask, Response, jsonify, render_template
from detector import detect_and_retrieve
import threading
import time
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

# 全局变量
latest_detections = []
frame_lock = threading.Lock()
current_frame = None

# === 新增：独立检测线程 ===
def detection_worker():
    global latest_detections, current_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ 无法打开摄像头！检测线程将退出。")
        return

    logger.info("✅ 检测线程已启动，开始读取摄像头...")
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("无法读取帧，退出检测线程")
            break

        # 保存当前帧供 video_feed 使用（可选）
        with frame_lock:
            current_frame = frame.copy()

        # 执行检测
        annotated_frame, detections = detect_and_retrieve(frame)

        # 更新全局结果
        latest_detections = detections

        # 控制频率（避免过快）
        time.sleep(0.3)  # ~3 FPS

    cap.release()

# === 视频流：从 current_frame 读取（不负责检测）===
def generate_frames():
    global current_frame
    while True:
        with frame_lock:
            frame = current_frame.copy() if current_frame is not None else None

        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # 返回黑屏或占位图
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', black)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)  # ~30 FPS


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/detections')
def api_detections():
    # 直接返回最新检测结果（无需锁，因为 list 赋值是原子的）
    return jsonify(latest_detections)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    # 启动独立检测线程
    det_thread = threading.Thread(target=detection_worker, daemon=True)
    det_thread.start()

    logger.info("启动实时卡通角色识别服务...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)