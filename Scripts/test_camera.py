# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import torch
import os

# 获取当前文件所在目录的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建yolov5文件夹的路径
yolov5_path = os.path.join(current_dir, '..', 'yolov5')

# 加载YOLO模型
try:
    model = torch.hub.load(yolov5_path, 'yolov5s', source='local')
    model.conf = 0.5
    model.classes = [2,3,5,7]  # 车辆类别
    print(f"成功加载模型: yolov5s")
except Exception as e:
    print(f"加载模型失败: {e}")
    print(f"请确保yolov5文件夹存在于: {yolov5_path}")
    model = None

# 全局变量
current_frame = None
is_streaming = False
is_initializing = False  # 是否正在初始化
lane_mask = None  # 车道掩膜
prev_gray = None  # 用于光流计算的上一帧灰度图

def initialize_lane_mask(frame):
    """
    使用YOLO检测车辆并生成车道掩膜
    """
    global lane_mask
    if model is None:
        print("模型未加载，无法进行初始化")
        return frame
        
    # 调整图像大小
    frame = cv2.resize(frame, (640, 480))
    
    # 使用YOLO进行检测
    results = model(frame)
    
    # 创建掩膜
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # 在图像上绘制检测结果
    for det in results.xyxy[0]:  # 遍历检测结果
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 添加标签
        label = f'Vehicle {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 对掩膜进行膨胀操作
    kernel = np.ones((30, 30), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 更新全局掩膜
    if lane_mask is None:
        lane_mask = mask
    else:
        # 使用OR操作合并掩膜
        lane_mask = cv2.bitwise_or(lane_mask, mask)
    
    # 在图像上显示掩膜区域
    mask_overlay = frame.copy()
    mask_overlay[mask > 0] = [0, 255, 0]  # 用绿色显示掩膜区域
    frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)
    
    return frame

def process_frame(frame):
    """
    处理单帧图像，进行边坡塌方检测
    :param frame: 输入图像帧
    :return: 处理后的图像帧和是否检测到塌方
    """
    global prev_gray, lane_mask
    try:
        # 调整图像大小
        frame = cv2.resize(frame, (640, 480))
        
        # 如果正在初始化，使用YOLO检测
        if is_initializing:
            frame = initialize_lane_mask(frame)
            return frame, False
            
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 确保有前一帧的灰度图
        if prev_gray is None:
            prev_gray = gray
            return frame, False
            
        # 如果掩膜已生成，进行光流检测
        if lane_mask is not None:
            # 生成非车道区域掩膜
            non_lane_mask = cv2.bitwise_not(lane_mask)
            masked_frame = cv2.bitwise_and(frame, frame, mask=non_lane_mask)
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            
            # 在图像上显示掩膜区域
            mask_overlay = frame.copy()
            mask_overlay[lane_mask > 0] = [0, 255, 0]  # 用绿色显示掩膜区域
            frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)
        
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # 更新前一帧的灰度图
        prev_gray = gray
        
        # 计算光流的幅度和方向
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 运动检测参数
        motion_threshold = 2.0
        area_threshold = 5000
        angle_threshold = 75
        
        # 创建运动掩膜
        motion_mask = cv2.inRange(magnitude, motion_threshold, 10.0)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # 轮廓检测
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 检测大面积运动区域
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > area_threshold:
                x, y, w, h = cv2.boundingRect(cnt)
                roi_angle = np.mean(angle[y:y+h, x:x+w]) * 180 / np.pi
                
                # 检查是否为垂直运动
                if abs(roi_angle - 90) < angle_threshold:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Landslide!", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    return frame, True
        
        # 显示光流结果
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 将光流结果叠加到原始帧上
        frame = cv2.addWeighted(frame, 0.7, flow_rgb, 0.3, 0)
        
        return frame, False
        
    except Exception as e:
        print(f"处理帧时出错: {e}")
        return frame, False

def process_video_stream(url):
    global current_frame, is_streaming, prev_gray
    
    print("正在连接网络视频流...")
    cap = cv2.VideoCapture(url)

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
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while is_streaming:
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
            processed_frame, has_landslide = process_frame(frame)
            
            # 更新全局帧
            current_frame = processed_frame.copy()
            
            # 更新光流计算的灰度图
            if not is_initializing:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        except Exception as e:
            print(f"处理帧时出错: {e}")
            continue

    cap.release()

if __name__ == "__main__":
    # 如果直接运行此文件，使用默认URL
    url = "http://192.168.26.23:8080/stream.mjpeg"
    is_streaming = True
    process_video_stream(url)

cv2.destroyAllWindows()