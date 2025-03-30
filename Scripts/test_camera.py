# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time

# 全局变量
prev_gray = None  # 用于存储前一帧的灰度图
prev_contours = []  # 用于存储前一帧的轮廓
detection_history = []  # 用于存储检测历史
HISTORY_LENGTH = 2  # 减少历史记录长度，使检测更快响应

# 处理尺寸
PROCESS_WIDTH = 320  # 减小处理尺寸
PROCESS_HEIGHT = 240

def check_contour_overlap(contour1, contour2):
    """
    检查两个轮廓是否重叠
    :param contour1: 第一个轮廓
    :param contour2: 第二个轮廓
    :return: 是否重叠
    """
    # 创建掩膜
    mask1 = np.zeros((PROCESS_HEIGHT, PROCESS_WIDTH), dtype=np.uint8)
    mask2 = np.zeros((PROCESS_HEIGHT, PROCESS_WIDTH), dtype=np.uint8)
    
    # 绘制轮廓
    cv2.drawContours(mask1, [contour1], -1, 255, -1)
    cv2.drawContours(mask2, [contour2], -1, 255, -1)
    
    # 计算重叠区域
    overlap = cv2.bitwise_and(mask1, mask2)
    
    # 如果重叠区域大于0，则存在重叠
    return np.sum(overlap) > 0

def process_frame(frame):
    """
    处理视频帧，进行边坡检测
    :param frame: 输入图像帧
    :return: 处理后的图像帧
    """
    global prev_gray, prev_contours, detection_history
    
    if frame is None:
        return None
        
    # 调整图像大小
    frame = cv2.resize(frame, (640, 480))
    process_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    
    # 转换为灰度图
    gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
    
    # 计算光流
    if prev_gray is None:
        prev_gray = gray.copy()
        return frame
        
    # 优化光流计算参数
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # 计算运动幅度和方向
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 设置运动检测阈值
    motion_threshold = 2.5  # 降低运动幅度阈值
    angle_threshold = 30  # 增加角度阈值范围
    
    # 创建运动掩膜
    motion_mask = np.zeros_like(gray)
    motion_mask[magnitude > motion_threshold] = 255
    
    # 创建方向掩膜（垂直运动）
    angle_mask = np.zeros_like(gray)
    angle_mask[angle > np.radians(angle_threshold)] = 255
    
    # 合并掩膜
    combined_mask = cv2.bitwise_and(motion_mask, angle_mask)
    
    # 优化形态学操作
    kernel = np.ones((3,3), np.uint8)  # 减小核大小
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 设置面积阈值（根据处理尺寸调整）
    area_threshold = 500  # 降低面积阈值
    
    # 计算整个图像的面积
    total_area = PROCESS_WIDTH * PROCESS_HEIGHT
    
    # 当前帧的检测结果
    current_detections = []
    
    # 在图像上绘制检测结果
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            # 计算运动区域占整个图像的比例
            area_ratio = area / total_area
            
            # 如果运动区域过大，可能是摄像头抖动
            if area_ratio > 0.8:  # 增加允许的运动区域比例
                continue
                
            # 计算轮廓的标准差（用于判断运动的稳定性）
            x, y, w, h = cv2.boundingRect(contour)
            roi = magnitude[y:y+h, x:x+w]
            std_dev = np.std(roi)
            
            # 计算运动方向的一致性
            roi_angle = angle[y:y+h, x:x+w]
            angle_std = np.std(roi_angle) * 180 / np.pi
            
            # 如果标准差小于阈值且方向一致，说明运动比较稳定
            if std_dev < 3.0 and angle_std < 45:  # 放宽标准差和角度一致性要求
                # 检查与前一帧轮廓的重叠
                overlap = False
                for prev_contour in prev_contours:
                    if check_contour_overlap(contour, prev_contour):
                        overlap = True
                        break
                
                if not overlap:
                    # 将坐标转换回原始尺寸
                    x = int(x * 640 / PROCESS_WIDTH)
                    y = int(y * 480 / PROCESS_HEIGHT)
                    w = int(w * 640 / PROCESS_WIDTH)
                    h = int(h * 480 / PROCESS_HEIGHT)
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    # 添加标签
                    cv2.putText(frame, f'Landslide {area:.0f}', (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    current_detections.append((x, y, w, h))
    
    # 更新检测历史
    detection_history.append(len(current_detections))
    if len(detection_history) > HISTORY_LENGTH:
        detection_history.pop(0)
    
    # 只有当连续多帧都检测到目标时才显示
    if len(detection_history) == HISTORY_LENGTH and all(d > 0 for d in detection_history):
        # 绘制所有检测到的区域
        for x, y, w, h in current_detections:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f'Landslide', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 更新前一帧的灰度图和轮廓
    prev_gray = gray.copy()
    prev_contours = contours
    
    return frame

def process_video_stream(url):
    """
    处理视频流
    :param url: 视频流URL
    """
    global prev_gray, prev_contours, detection_history
    prev_gray = None  # 重置prev_gray
    prev_contours = []  # 重置prev_contours
    detection_history = []  # 重置检测历史
    
    print("正在连接网络视频流...")
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("无法连接到视频流，请检查网络连接和URL是否正确")
        return

    # 设置视频参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率

    # 创建窗口
    cv2.namedWindow('Landslide Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Landslide Detection', 1280, 960)

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
            processed_frame = process_frame(frame)
            
            # 显示结果
            cv2.imshow('Landslide Detection', processed_frame)
            
            # 按键处理
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"处理帧时出错: {e}")
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 如果直接运行此文件，使用默认URL
    url = "http://192.168.102.23:8080/stream.mjpeg"
    process_video_stream(url)