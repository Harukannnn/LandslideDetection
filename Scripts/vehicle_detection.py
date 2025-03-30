# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import os
import time

class VehicleDetection:
    def __init__(self):
        self.model = None
        self.lane_mask = None
        self.is_initializing = False
        self.accumulated_mask = None
        self.load_model()

    def load_model(self):
        """加载YOLO模型"""
        try:
            # 获取当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建yolov5文件夹的路径
            yolov5_path = os.path.join(current_dir, '..', 'yolov5')
            
            # 加载模型
            self.model = torch.hub.load(yolov5_path, 'yolov5s', source='local')
            self.model.conf = 0.3  # 降低置信度阈值以增加检测数量
            self.model.classes = [2,3,5,7]  # 车辆类别
            print(f"成功加载模型: yolov5s")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print(f"请确保yolov5文件夹存在于: {yolov5_path}")
            self.model = None

    def detect_vehicles_and_generate_mask(self, frame):
        """
        检测车辆并生成掩膜
        :param frame: 输入图像帧
        :return: 处理后的图像帧和掩膜
        """
        if self.model is None:
            print("模型未加载，无法进行检测")
            return frame, None
        
        try:
            # 创建掩膜
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            
            # 使用YOLO进行检测
            results = self.model(frame)
            
            # 在图像上绘制检测结果并生成掩膜
            for det in results.xyxy[0]:  # 遍历检测结果
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                
                # 处理所有车辆类别（2:car, 3:motorcycle, 5:bus, 7:truck）
                if int(cls) in [2, 3, 5, 7] and conf > 0.3:
                    # 扩大检测框的范围
                    width = x2 - x1
                    height = y2 - y1
                    x1 = max(0, int(x1 - width * 0.3))
                    x2 = min(frame.shape[1], int(x2 + width * 0.3))
                    y1 = max(0, int(y1 - height * 0.2))
                    y2 = min(frame.shape[0], int(y2 + height * 0.2))
                    
                    # 在掩膜上标记车辆区域
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # 添加标签
                    class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
                    label = f'{class_names[int(cls)]} {conf:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 对掩膜进行膨胀操作，使掩膜更连续
            kernel = np.ones((30, 30), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # 更新累计掩膜
            if self.accumulated_mask is None:
                self.accumulated_mask = mask
            else:
                self.accumulated_mask = cv2.bitwise_or(self.accumulated_mask, mask)
            
            return frame, self.accumulated_mask
        except Exception as e:
            print(f"检测车辆和生成掩膜时出错: {e}")
            return frame, None

    def initialize_lane_mask(self, frame):
        """生成车道掩膜"""
        try:
            # 检测车辆并生成掩膜
            frame, mask = self.detect_vehicles_and_generate_mask(frame)
            return mask
        except Exception as e:
            print(f"生成车道掩膜时出错: {e}")
            return None

    def start_initialization(self):
        """开始初始化"""
        self.is_initializing = True
        self.accumulated_mask = None

    def stop_initialization(self):
        """停止初始化并返回累计掩膜"""
        self.is_initializing = False
        return self.accumulated_mask

    def cleanup(self):
        """清理资源"""
        self.model = None
        self.lane_mask = None
        self.is_initializing = False
        self.accumulated_mask = None

def process_video_stream(url, test_mode=False):
    """
    处理视频流
    :param url: 视频流URL
    :param test_mode: 是否为测试模式（生成车道掩膜）
    """
    # 创建 VehicleDetection 实例
    detector = VehicleDetection()
    print("正在连接网络视频流...")
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("无法连接到视频流，请检查网络连接和URL是否正确")
        return

    # 设置视频参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 创建窗口
    window_name = 'Lane Mask Generation' if test_mode else 'Vehicle Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 960)

    # 重置初始化状态
    detector.is_initializing = False
    detector.accumulated_mask = None

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧，尝试重新连接...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    print("重新连接失败，程序退出")
                    break
                continue

            # 处理帧
            if test_mode:
                if detector.is_initializing:
                    # 先进行车辆检测
                    processed_frame, mask = detector.detect_vehicles_and_generate_mask(frame)
                    if mask is not None:
                        # 显示掩膜
                        mask_overlay = processed_frame.copy()
                        mask_overlay[mask > 0] = [0, 255, 0]
                        processed_frame = cv2.addWeighted(processed_frame, 0.7, mask_overlay, 0.3, 0)
                else:
                    processed_frame, mask = detector.detect_vehicles_and_generate_mask(frame)
                    if detector.accumulated_mask is not None:
                        # 显示累计掩膜
                        mask_overlay = processed_frame.copy()
                        mask_overlay[detector.accumulated_mask > 0] = [0, 255, 0]
                        processed_frame = cv2.addWeighted(processed_frame, 0.7, mask_overlay, 0.3, 0)
            else:
                processed_frame, mask = detector.detect_vehicles_and_generate_mask(frame)
            
            # 显示结果
            cv2.imshow(window_name, processed_frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 退出
                break
            elif key == ord('i'):  # 切换初始化模式
                detector.is_initializing = not detector.is_initializing
                print(f"初始化模式: {'开启' if detector.is_initializing else '关闭'}")
            elif key == ord('c'):  # 清除掩膜
                detector.accumulated_mask = None
                detector.is_initializing = False
                print("已清除掩膜并停止掩膜生成")

        except Exception as e:
            print(f"处理帧时出错: {e}")
            continue

    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()  # 清理资源

if __name__ == "__main__":
    # 如果直接运行此文件，使用默认URL
    url = "http://192.168.102.23:8080/stream.mjpeg"
    # 设置为True以测试车道掩膜生成
    process_video_stream(url, test_mode=True) 
    process_video_stream(url, test_mode=True) 
