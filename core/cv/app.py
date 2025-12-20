# realtime_app.py
import cv2
import logging
from flask import Flask, Response, jsonify, render_template
from detector import detect_and_retrieve  # ← 确保 detector.py 在同目录

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

# 全局变量存储最新检测结果（线程不安全，仅用于 demo）
latest_detections = []
video_capture = None

def generate_frames():
    """生成 MJPEG 视频流"""
    global latest_detections, video_capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        logger.error("无法打开摄像头！视频流将不可用，但 API 仍可访问。")
        # 即使摄像头打不开，也不 raise，让 Flask 继续运行
        return b''  # 返回空字节流

    logger.info("摄像头已启动")
    try:
        while True:
            success, frame = video_capture.read()
            if not success:
                logger.warning(" 无法读取摄像头帧")
                break

            # 调用检测函数
            annotated_frame, detections = detect_and_retrieve(frame)
            latest_detections = detections  # 更新全局结果

            # 编码为 JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        logger.error(f"视频流生成异常: {e}")
    finally:
        if video_capture and video_capture.isOpened():
            video_capture.release()


@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/detections')
def api_detections():
    """返回最新检测结果（直接返回列表，适配前端）"""
    return jsonify(latest_detections)  #  关键修改：不再包裹 {"detections": ...}

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

if __name__ == '__main__':
    logger.info("启动实时卡通角色识别服务...")
    logger.info("请确保 detector.py 和模型文件路径正确")
    logger.info("访问 http://localhost:5000 查看实时画面")
    # 关键：确保服务启动，即使摄像头失败也不退出
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)