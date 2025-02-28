import cv2
import numpy as np
import torch


model = torch.hub.load('../yolov5','yolov5s',source='local')

# VEHICLE_CLASSES = ['car','truck','bus','motorbike']
# 视频流地址
url = "C:/Users/Heren/Documents/WeChat Files/wxid_6u4zst4m16e122/FileStorage/Video/2025-02/landsliding.mp4"
cap = cv2.VideoCapture(url)


# # 处理镜头晃动
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# # 用于特征点检测的光流法参数
# feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
# lk_params = dict(winSize=(15, 15), maxLevel=2,
#                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#
# # 读取第一帧
# ret, prev_frame = cap.read()
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#
# # 找到第一帧的特征点
# prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
#
# # 创建一个平移矢量列表
# transforms = np.zeros((n_frames-1, 3), np.float32)
#
# # 遍历视频帧
# for i in range(n_frames-1):
#     ret, curr_frame = cap.read()
#     if not ret:
#         break
#
#     curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
#
#     # 计算光流
#     curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
#
#     # 选择好的点
#     good_old = prev_pts[status == 1]
#     good_new = curr_pts[status == 1]
#
#     # 计算变换矩阵 (仅平移 + 旋转, 排除缩放)
#     matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new)
#
#     # 提取平移和旋转信息
#     dx = matrix[0, 2]
#     dy = matrix[1, 2]
#     da = np.arctan2(matrix[1, 0], matrix[0, 0])
#
#     transforms[i] = [dx, dy, da]
#
#     # 更新前一帧
#     prev_gray = curr_gray.copy()
#     prev_pts = good_new.reshape(-1, 1, 2)
#
# # 平滑运动 (移动平均)
# trajectory = np.cumsum(transforms, axis=0)
# smoothed_trajectory = cv2.blur(trajectory, (15, 1))
#
# # 计算平滑后的差值
# difference = smoothed_trajectory - trajectory
# transforms_smooth = transforms + difference
#
# # 应用平滑变换到视频
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
# for i in range(n_frames-1):
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     dx, dy, da = transforms_smooth[i]
#     matrix = np.array([[np.cos(da), -np.sin(da), dx],
#                        [np.sin(da),  np.cos(da), dy]])
#
#     # 应用仿射变换
#     stabilized_frame = cv2.warpAffine(frame, matrix, (width, height))
#
#     cv2.imshow("Stabilized Video", stabilized_frame)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#
# cap.release()
#
# cv2.destroyAllWindows()



# # 初始化背景减除器
# fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
#
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("视频读取结束或出错")
#         break
#
#     results = model(frame)
#     detections = results.xyxy[0].cpu().numpy()
#
#     mask = 255 * np.ones(frame.shape[:2], dtype=np.uint8)
#
#     # 3. 绘制车辆检测框并过滤车辆运动
#     for *xyxy, conf, cls in detections:
#         if int(cls) == 2 or int(cls) == 5 or int(cls) == 7:  # 2=car, 5=bus, 7=truck (COCO类别ID)
#             x1, y1, x2, y2 = map(int, xyxy)
#             cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), -1)  # 将车辆区域涂黑 (设置为背景)
#
#     # 应用背景减除 (使用掩膜处理后的帧)
#     fg_mask = fgbg.apply(frame)
#
#     # 将车辆区域在前景掩膜中移除 (保持背景颜色，车辆变为黑色)
#     fg_mask = cv2.bitwise_and(fg_mask, mask)
#
#     # 显示结果
#     cv2.imshow('Foreground Mask', fg_mask)
#     cv2.imshow('Original Frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


model.conf = 0.5
model.classes = [2,3,5,7]

# 背景减除器 (MOG2)
bg_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

# 用于光流法的初始帧
ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, (640, 480))  # 调整大小，提升处理速度
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# 创建 HSV 图像用于显示光流 (光流的可视化)
hsv_mask = np.zeros_like(frame1)
hsv_mask[..., 1] = 255  # 饱和度设置为最大值

# 初始化车道掩膜
lane_mask = np.zeros(frame1.shape[:2], dtype=np.uint8)


# 图像亮度和对比度增强
def adjust_brightness_contrast(image, alpha=1.5, beta=50):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# 车道掩膜更新次数
frame_count = 0
update_lane_mask_until = 30  # 在前500帧持续更新车道掩膜
monitoring_started = False

while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break

    frame2 = cv2.resize(frame2, (640, 480))
    frame2 = adjust_brightness_contrast(frame2)  # 图像增强

    # 使用 YOLOv5 进行车辆检测
    results = model(frame2)
    detections = results.pandas().xyxy[0]

    # 仅在初始阶段生成车道掩膜 (通过多边形拟合)
    if frame_count < update_lane_mask_until:
        vehicle_contours = []
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 车辆检测框

            # 将车辆框转换为轮廓点
            vehicle_contours.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))

        # 计算车道的多边形拟合 (凸包)
        if len(vehicle_contours) > 0:
            all_points = np.concatenate(vehicle_contours)
            hull = cv2.convexHull(all_points)

            # 绘制多边形车道掩膜
            temp_mask = np.zeros(frame2.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(lane_mask, hull, 255)

            # 将新的掩膜与旧的掩膜进行合并 (累积车道区域)
            lane_mask = cv2.bitwise_or(lane_mask, temp_mask)

        # 显示初始化状态
        cv2.putText(frame2, "Initializing Lane Mask...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        # 掩膜已稳定，可以开始滑坡监测
        monitoring_started = True
        cv2.putText(frame2, "Monitoring...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    frame_count += 1

    # 生成非车道区域掩膜
    non_lane_mask = cv2.bitwise_not(lane_mask)

    # 在应用背景减除和光流法前，先过滤车道区域
    masked_frame = cv2.bitwise_and(frame2, frame2, mask=non_lane_mask)

    # 背景减除
    fg_mask = bg_subtractor.apply(masked_frame)

    # 去除噪声 (形态学操作)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # 计算光流
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 计算光流的幅度和方向
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 设置光流颜色 (Hue 色调表示方向, Value 表示强度)
    hsv_mask[..., 0] = angle * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # 转换为 BGR 颜色空间进行显示
    flow_rgb = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    # 只保留运动强度大的区域 (滑坡通常是大面积运动)
    motion_mask = cv2.inRange(magnitude, 2.0, 10.0)  # 设置运动幅度阈值 (可调整)

    # 结合背景减除的结果
    combined_mask = cv2.bitwise_and(fg_mask, motion_mask)

    if monitoring_started:
        # 轮廓检测，筛选出较大的运动区域
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:  # 仅处理大于一定面积的轮廓 (可调整)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame2, "Possible Landslide", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print("Landslide!")
    # 显示结果
    cv2.imshow('Original Frame', frame2)
    cv2.imshow('Background Subtraction', fg_mask)
    cv2.imshow('Optical Flow', flow_rgb)
    cv2.imshow('Landslide Detection', combined_mask)
    cv2.imshow('Lane Mask', lane_mask)

    prev_gray = gray  # 更新前一帧

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()