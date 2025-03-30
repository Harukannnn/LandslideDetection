from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import threading
import time
from Scripts.vehicle_detection import VehicleDetection
from Scripts.test_camera import process_frame

app = Flask(__name__)

# 全局变量
streams = {}  # 存储所有视频流的信息
current_frame = None
is_initializing = False
lane_mask = None

class VideoStream:
    def __init__(self, url):
        self.url = url
        self.is_streaming = False
        self.is_initializing = False
        self.is_detecting = False
        self.accumulated_mask = None
        self.prev_gray = None
        self.thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30.0
        self.vehicle_detector = None  # 车辆检测器实例

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
        cap.set(cv2.CAP_PROP_FPS, 30)

        while self.is_streaming:
            try:
                current_time = time.time()
                if current_time - self.last_frame_time < self.frame_interval:
                    time.sleep(0.001)
                    continue

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

                # 保存原始帧用于检测
                original_frame = frame.copy()
                # 调整显示尺寸
                display_frame = cv2.resize(frame, (640, 480))
                processed_frame = display_frame.copy()

                # 处理帧
                if self.is_initializing and self.vehicle_detector:
                    # 使用原始帧进行检测和掩膜生成
                    processed_frame, mask = self.vehicle_detector.detect_vehicles_and_generate_mask(original_frame)
                    # 调整处理后的帧大小用于显示
                    processed_frame = cv2.resize(processed_frame, (640, 480))
                    if mask is not None:
                        # 调整掩膜大小以匹配显示尺寸
                        mask = cv2.resize(mask, (640, 480))
                        # 显示掩膜
                        mask_overlay = processed_frame.copy()
                        mask_overlay[mask > 0] = [0, 255, 0]
                        processed_frame = cv2.addWeighted(processed_frame, 0.7, mask_overlay, 0.3, 0)
                elif self.is_detecting:
                    if self.accumulated_mask is not None:
                        # 确保掩膜尺寸与帧匹配
                        mask = cv2.resize(self.accumulated_mask, (640, 480))
                        processed_frame = process_frame(display_frame, mask)
                    else:
                        processed_frame = process_frame(display_frame)
                else:
                    # 非初始化模式下显示累计掩膜
                    processed_frame = display_frame.copy()
                    if self.accumulated_mask is not None:
                        # 确保掩膜尺寸与帧匹配
                        mask = cv2.resize(self.accumulated_mask, (640, 480))
                        mask_overlay = processed_frame.copy()
                        mask_overlay[mask > 0] = [0, 255, 0]
                        processed_frame = cv2.addWeighted(processed_frame, 0.7, mask_overlay, 0.3, 0)
                
                # 更新当前帧
                with self.frame_lock:
                    self.current_frame = processed_frame.copy()
                self.last_frame_time = current_time

            except Exception as e:
                print(f"处理帧时出错: {e}")
                continue

        cap.release()

    def get_frame(self):
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def start_initialization(self):
        """开始初始化"""
        self.is_initializing = True
        self.is_detecting = False
        self.vehicle_detector = VehicleDetection()
        self.vehicle_detector.start_initialization()

    def stop_initialization(self):
        """停止初始化"""
        if self.vehicle_detector:
            # 获取累计掩膜并调整大小
            mask = self.vehicle_detector.stop_initialization()
            if mask is not None:
                self.accumulated_mask = cv2.resize(mask, (640, 480))
            self.vehicle_detector.cleanup()
            self.vehicle_detector = None
        self.is_initializing = False

@app.route('/')
def index():
    return render_template('MonitoringFrame.html')

@app.route('/video_feed/<int:stream_id>')
def video_feed(stream_id):
    def generate_frames():
        last_frame_time = 0
        frame_interval = 1.0 / 30.0  # 限制帧率为30fps
        
        while True:
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)  # 短暂休眠以减少CPU使用
                continue
                
            if stream_id in streams:
                stream = streams[stream_id]
                frame = stream.get_frame()
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    last_frame_time = current_time
            time.sleep(0.001)  # 短暂休眠以减少CPU使用

    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global streams
    try:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({'error': '未提供URL'}), 400

        # 创建新的视频流
        stream = VideoStream(url)
        stream.is_streaming = True
        stream.is_initializing = False
        stream.is_detecting = False
        stream.accumulated_mask = None
        stream.prev_gray = None
        stream.thread = threading.Thread(target=stream.process_video_stream)
        stream.thread.start()
        
        # 保存到全局字典
        stream_id = len(streams)
        streams[stream_id] = stream
        
        return jsonify({'success': True, 'message': '视频流已启动', 'stream_id': stream_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    try:
        data = request.get_json()
        stream_id = data.get('streamId')
        
        if stream_id in streams:
            # 停止视频流
            stream = streams[stream_id]
            stream.is_streaming = False
            stream.is_initializing = False
            if stream.thread:
                stream.thread.join()
            del streams[stream_id]
            
            # 清理全局变量
            global lane_mask, is_initializing
            lane_mask = None
            is_initializing = False
            
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': '视频流不存在'})
            
    except Exception as e:
        print(f"停止视频流时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/toggle_initialization/<int:stream_id>', methods=['POST'])
def toggle_initialization(stream_id):
    global streams
    try:
        if stream_id in streams:
            stream = streams[stream_id]
            if not stream.is_initializing:
                stream.start_initialization()
            else:
                stream.stop_initialization()
            return jsonify({
                'success': True, 
                'initializing': stream.is_initializing,
                'has_mask': stream.accumulated_mask is not None
            })
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
            if stream.is_detecting:
                stream.is_initializing = False  # 检测时停止初始化
            return jsonify({'success': True, 'detecting': stream.is_detecting})
        return jsonify({'error': '未找到指定的视频流'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 