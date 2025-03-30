# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
import time


model = torch.hub.load('E:/Pycharm/TestingFYP/yolov5','yolov5s',source='local')

# 视频流地址
url = "C:/Users/Heren/Documents/WeChat Files/wxid_6u4zst4m16e122/FileStorage/Video/2025-02/landsliding.mp4"  # 使用本地视频文件

print("正在打开视频文件...")
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("无法打开视频文件，请检查文件路径")
    exit(1)

# 获取视频信息
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"视频总帧数: {total_frames}")
print(f"视频帧率: {fps}")

# 设置视频参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

model.conf = 0.5
model.classes = [2,3,5,7]

# 背景减除器 (MOG2)
bg_subtractor = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=50, detectShadows=True)

# 用于光流法的初始帧
ret, frame1 = cap.read()
if not ret:
    print("无法读取第一帧，程序退出")
    cap.release()
    exit(1)

frame1 = cv2.resize(frame1, (640, 480))
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# 创建 HSV 图像用于显示光流
hsv_mask = np.zeros_like(frame1)
hsv_mask[..., 1] = 255

# 初始化车道掩膜
lane_mask = np.zeros(frame1.shape[:2], dtype=np.uint8)


# 图像亮度和对比度增强
def adjust_brightness_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    accumulator = np.cumsum(hist)
    maximum = accumulator[-1]

    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255.0 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# 帧率控制参数
frame_count = 0
frame_skip = 2  # 减少处理间隔，提高帧率
yolo_skip = 5   # 减少YOLO检测间隔
last_frame_time = time.time()
target_fps = 30
frame_interval = 1.0 / target_fps

# 车道掩膜初始化控制
initializing = False
lane_mask_history = []
history_size = 30
mask_dilation_kernel = np.ones((30, 30), np.uint8)
persistence_threshold = 0.7

def update_lane_mask(temp_mask):
    global lane_mask
    # 对掩膜进行膨胀操作
    dilated_mask = cv2.dilate(temp_mask, mask_dilation_kernel, iterations=2)
    # 对掩膜进行平滑处理
    dilated_mask = cv2.GaussianBlur(dilated_mask, (21, 21), 0)
    # 二值化处理
    _, dilated_mask = cv2.threshold(dilated_mask, 127, 255, cv2.THRESH_BINARY)
    
    # 更新车道掩膜（使用逻辑或操作，保持累积性）
    lane_mask = cv2.bitwise_or(lane_mask, dilated_mask)

def start_initialization():
    global initializing, lane_mask_history, lane_mask
    initializing = True
    lane_mask_history = []
    lane_mask = np.zeros(frame1.shape[:2], dtype=np.uint8)
    print("开始初始化车道掩膜...")

def stop_initialization():
    global initializing
    initializing = False
    print("停止初始化车道掩膜")

# 主循环
while True:
    # 帧率控制
    current_time = time.time()
    elapsed = current_time - last_frame_time
    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)
    last_frame_time = time.time()

    ret, frame2 = cap.read()
    if not ret:
        print("视频播放结束")
        break

    frame_count += 1
    frame2 = cv2.resize(frame2, (640, 480))

    # 显示当前进度
    progress = (frame_count / total_frames) * 100
    cv2.putText(frame2, f"Progress: {progress:.1f}%", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 每5帧进行一次图像增强
    if frame_count % 5 == 0:
        frame2 = adjust_brightness_contrast(frame2)

    # 每10帧进行一次YOLO检测
    if frame_count % yolo_skip == 0:
        results = model(frame2)
        detections = results.pandas().xyxy[0]

    # 车道掩膜初始化控制
    if initializing and frame_count % 3 == 0:
        vehicle_contours = []
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            # 扩大检测框的范围
            width = x2 - x1
            height = y2 - y1
            x1 = max(0, x1 - int(width * 0.3))
            x2 = min(frame2.shape[1], x2 + int(width * 0.3))
            y1 = max(0, y1 - int(height * 0.2))
            y2 = min(frame2.shape[0], y2 + int(height * 0.2))
            
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 255), 2)
            vehicle_contours.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))

        if len(vehicle_contours) > 0:
            all_points = np.concatenate(vehicle_contours)
            hull = cv2.convexHull(all_points)
            temp_mask = np.zeros(frame2.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(temp_mask, hull, 255)
            update_lane_mask(temp_mask)

        cv2.putText(frame2, "Initializing Lane Mask...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame2, "Monitoring...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 生成非车道区域掩膜
    non_lane_mask = cv2.bitwise_not(lane_mask)
    # 对非车道区域掩膜进行额外的膨胀操作
    non_lane_mask = cv2.dilate(non_lane_mask, np.ones((15, 15), np.uint8), iterations=1)
    masked_frame = cv2.bitwise_and(frame2, frame2, mask=non_lane_mask)

    # 初始化combined_mask
    combined_mask = np.zeros(frame2.shape[:2], dtype=np.uint8)

    # 每2帧进行一次背景减除和光流计算
    if frame_count % frame_skip == 0:
        # 背景减除
        fg_mask = bg_subtractor.apply(masked_frame)

        # 去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # 计算光流
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 计算光流的幅度和方向
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 设置光流颜色
        hsv_mask[..., 0] = angle * 180 / np.pi / 2
        hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # 转换为BGR颜色空间
        flow_rgb = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

        # 运动检测阈值
        motion_mask = cv2.inRange(magnitude, 2.0, 10.0)
        combined_mask = cv2.bitwise_and(fg_mask, motion_mask)

        # 轮廓检测
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame2, "Possible Landslide", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print("Landslide!")

        prev_gray = gray

    # 显示结果
    cv2.imshow('Original Frame', frame2)
    cv2.imshow('Landslide Detection', combined_mask)
    cv2.imshow('Lane Mask', lane_mask)

    # 按键控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 退出
        break
    elif key == ord('i'):  # 开始初始化
        start_initialization()
    elif key == ord('s'):  # 停止初始化
        stop_initialization()
    elif key == ord('r'):  # 重新播放
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        print("重新播放视频")

# 显示控制说明
print("\n控制说明：")
print("'i' - 开始初始化车道掩膜")
print("'s' - 停止初始化车道掩膜")
print("'r' - 重新播放视频")
print("'q' - 退出程序")

cap.release()
cv2.destroyAllWindows()