"""
从原始脚本 Trail_Indicator_XAI_FusionMap_results_Web_0326.py 提取的函数和变量
供其他接口调用，避免执行顶层代码
"""

import os
import numpy as np
import cv2
import json

# POSITIVE_TO_NEGATIVE_PAIR_MAP 映射表
POSITIVE_TO_NEGATIVE_PAIR_MAP = {
    10: 0, 11: 1, 12: 2, 13: 3, 14: 4,  
    15: 5, 16: 6, 8: 1, 9: 4,          
    17: 7                                
}

ROI_AREA_THRESHOLD = 150
ROI_PADDING = 20
MODEL_INPUT_SIZE = (224, 224)


def find_static_content_mask(frame1, frame2, diff_threshold=5):
    """辅助函数：比较两张连续的帧，找到静态区域"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, static_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY_INV)
    return static_mask


def find_doppler_roi_from_video(video_frames, area_threshold, padding, frame_stride=4):
    """ROI提取 - 与原脚本一致，使用frame_stride=4"""
    max_area = 0
    best_frame_idx = -1
    
    for i, frame in enumerate(video_frames):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 70, 50]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50]); upper_red2 = np.array([180, 255, 255])
        lower_blue = np.array([100, 70, 50]); upper_blue = np.array([130, 255, 255])
        lower_turb = np.array([11, 70, 50]); upper_turb = np.array([90, 255, 255])
        
        mask_r1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        mask_b = cv2.inRange(hsv_img, lower_blue, upper_blue)
        mask_t = cv2.inRange(hsv_img, lower_turb, upper_turb)
        
        current_area = np.sum((mask_r1 + mask_r2 + mask_b + mask_t) > 0)
        
        if current_area > max_area:
            max_area = current_area
            best_frame_idx = i

    if best_frame_idx == -1: 
        return None
    
    total_frames = len(video_frames)
    best_frame = video_frames[best_frame_idx]
    compare_frame = None

    if best_frame_idx + frame_stride < total_frames:
        compare_frame = video_frames[best_frame_idx + frame_stride]
    elif best_frame_idx - frame_stride >= 0:
        compare_frame = video_frames[best_frame_idx - frame_stride]
    else:
        if best_frame_idx < total_frames - 1:
            compare_frame = video_frames[total_frames - 1] 
        elif best_frame_idx > 0:
            compare_frame = video_frames[0] 
        else:
            compare_frame = best_frame

    static_mask = find_static_content_mask(best_frame, compare_frame)
    
    hsv_best = cv2.cvtColor(best_frame, cv2.COLOR_BGR2HSV)
    mask_r1 = cv2.inRange(hsv_best, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv_best, lower_red2, upper_red2)
    mask_b = cv2.inRange(hsv_best, lower_blue, upper_blue)
    mask_t = cv2.inRange(hsv_best, lower_turb, upper_turb)
    color_mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), cv2.bitwise_or(mask_b, mask_t))
    
    cleaned_color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(static_mask))
    
    contours, _ = cv2.findContours(cleaned_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]

    if not final_contours:
        contours_raw, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = [cnt for cnt in contours_raw if cv2.contourArea(cnt) > area_threshold]

    if not final_contours: 
        return None

    all_points = np.concatenate(final_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    center_x, center_y = x + w // 2, y + h // 2
    max_side = max(w, h) + padding * 2
    
    x1 = max(0, center_x - max_side // 2)
    y1 = max(0, center_y - max_side // 2)
    x2 = min(best_frame.shape[1], center_x + max_side // 2)
    y2 = min(best_frame.shape[0], center_y + max_side // 2)
    
    return (x1, y1, x2, y2)


def create_string_token_mask(segment_np, spatial_resolution=4, vote_threshold=1, area_threshold_ratio=0.5): 
    """串Token掩码生成 - 与原脚本一致"""
    num_timesteps = segment_np.shape[0]
    num_spatial_locations = spatial_resolution * spatial_resolution
    model_input_size = (224, 224)
    patch_size = model_input_size[0] // spatial_resolution
    
    area_threshold = (patch_size * patch_size) * area_threshold_ratio

    lower_red1 = np.array([0, 70, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50]); upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 70, 50]); upper_blue = np.array([130, 255, 255])
    lower_turb = np.array([11, 70, 50]); upper_turb = np.array([90, 255, 255])

    vote_matrix = np.zeros((num_timesteps, num_spatial_locations), dtype=bool)
    
    for t, frame_bgr in enumerate(segment_np):
        frame_resized = cv2.resize(frame_bgr, model_input_size)
        hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        
        mask_r1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask_b = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        mask_t = cv2.inRange(hsv_frame, lower_turb, upper_turb)
        color_mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), cv2.bitwise_or(mask_b, mask_t))
        
        for s in range(num_spatial_locations):
            row = s // spatial_resolution
            col = s % spatial_resolution
            
            patch_mask = color_mask[
                row*patch_size : (row+1)*patch_size,
                col*patch_size : (col+1)*patch_size
            ]
            
            blood_flow_area = np.sum(patch_mask > 0)
            if blood_flow_area > area_threshold:
                vote_matrix[t, s] = True

    string_mask = np.zeros(num_spatial_locations, dtype=bool)
    for s in range(num_spatial_locations):
        num_votes = np.sum(vote_matrix[:, s])
        if num_votes <= vote_threshold:
            string_mask[s] = True
            
    return string_mask


def video_loader(video_path):
    """视频加载 - 使用OpenCV（与原脚本一致）"""
    import cv2
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: OpenCV cannot open video: {video_path}")
        cap.release()
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()

    if not frames:
        print(f"Warning: Video is empty or corrupted: {video_path}")
        return None
        
    return np.array(frames)


def transform_box_from_model_to_original_space(box_model, roi_bbox):
    """原脚本中的坐标变换函数"""
    if box_model is None or roi_bbox is None: return None
    mx1, my1, mx2, my2 = box_model
    rx1, ry1, rx2, ry2 = roi_bbox
    roi_w, roi_h = rx2 - rx1, ry2 - ry1
    if roi_w <= 0 or roi_h <= 0: return None
    model_w, model_h = (224, 224)
    scale_x_inv, scale_y_inv = roi_w / model_w, roi_h / model_h
    cx1, cy1 = mx1 * scale_x_inv, my1 * scale_y_inv
    cx2, cy2 = mx2 * scale_x_inv, my2 * scale_y_inv
    final_x1, final_y1, final_x2, final_y2 = cx1 + rx1, cy1 + ry1, cx2 + rx1, cy2 + ry1
    return [int(final_x1), int(final_y1), int(final_x2), int(final_y2)]


def get_center_and_draw_box(center_point, box_size, image_size=(224, 224)):
    """原脚本中的边界框生成函数"""
    cx, cy = center_point
    half_size = box_size // 2
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(image_size[0], cx + half_size)
    y2 = min(image_size[1], cy + half_size)
    return [int(x1), int(y1), int(x2), int(y2)]


class NpEncoder(json.JSONEncoder):
    """原脚本中的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)
