from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import threading
import time
from Scripts.test_camera import process_frame, is_initializing  # 导入边坡检测函数和初始化状态

app = Flask(__name__)

# 全局变量
streams = {}  # 存储所有视频流的信息
current_frame = None

class VideoStream:
    def __init__(self, url):
        self.url = url
        self.is_streaming = False
        self.is_initializing = False
        self.lane_mask = None
        self.prev_gray = None
        self.thread = None
        self.current_frame = None

    def generate_frames(self):
        while self.is_streaming:
            if self.current_frame is not None:
                try:
                    # 将OpenCV图像编码为JPEG
                    ret, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ret:
                        continue
                    # 将图像转换为字节流
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"生成帧时出错: {e}")
                    continue
            time.sleep(0.033)  # 约30fps

    def process_video_stream(self):
        print("正在连接网络视频流...")
        cap = cv2.VideoCapture(self.url)

        if not cap.isOpened():
            print("无法连接到视频流，请检查网络连接和URL是否正确")
            return

        # 设置视频参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 获取第一帧用于光流计算
        ret, frame = cap.read()
        if not ret:
            print("无法读取第一帧")
            return
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        while self.is_streaming:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取视频帧，尝试重新连接...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(self.url)
                    if not cap.isOpened():
                        print("重新连接失败，程序退出")
                        break
                    continue

                # 处理帧
                processed_frame, has_landslide = process_frame(frame)
                
                # 更新当前帧
                self.current_frame = processed_frame.copy()
                
                # 更新光流计算的灰度图
                if not self.is_initializing:
                    self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            except Exception as e:
                print(f"处理帧时出错: {e}")
                continue

        cap.release()

@app.route('/')
def index():
    return render_template('MonitoringFrame.html')

@app.route('/video_feed/<int:stream_id>')
def video_feed(stream_id):
    if stream_id in streams:
        return Response(streams[stream_id].generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Stream not found", 404

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global current_frame, streams
    try:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({'error': '未提供URL'}), 400

        # 创建新的视频流
        stream = VideoStream(url)
        stream.is_streaming = True
        stream.is_initializing = False
        stream.lane_mask = None
        stream.prev_gray = None
        stream.thread = threading.Thread(target=stream.process_video_stream)
        stream.thread.start()
        
        # 保存到全局字典
        stream_id = len(streams)
        streams[stream_id] = stream
        
        return jsonify({'message': '视频流已启动', 'stream_id': stream_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_stream/<int:stream_id>', methods=['POST'])
def stop_stream(stream_id):
    global streams
    try:
        if stream_id in streams:
            stream = streams[stream_id]
            stream.is_streaming = False
            stream.is_initializing = False
            if stream.thread:
                stream.thread.join()
            del streams[stream_id]
            return jsonify({'message': '视频流已停止'})
        return jsonify({'error': '未找到指定的视频流'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/toggle_initialization/<int:stream_id>', methods=['POST'])
def toggle_initialization(stream_id):
    global streams
    try:
        if stream_id in streams:
            stream = streams[stream_id]
            stream.is_initializing = not stream.is_initializing
            if not stream.is_initializing:
                # 保存车道掩膜
                stream.lane_mask = lane_mask
            return jsonify({'is_initializing': stream.is_initializing})
        return jsonify({'error': '未找到指定的视频流'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/toggle_detection/<int:stream_id>', methods=['POST'])
def toggle_detection(stream_id):
    global streams
    try:
        if stream_id in streams:
            stream = streams[stream_id]
            stream.is_detecting = not stream.is_detecting
            return jsonify({'is_detecting': stream.is_detecting})
        return jsonify({'error': '未找到指定的视频流'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 