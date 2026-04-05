import numpy as np
import cv2
from PIL import Image
import PIL
import random
import torch
import json
import os

def fix_randomness(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pad_collate_fn(batch):
    """
    一个自定义的collate_fn，用于处理多视角数据，其中每个病例的视角数不同。
    它会将批次内的所有样本填充到该批次中最大的视角数。
    """
    # 1. 过滤掉加载失败的None样本
    batch = [item for item in batch if item is not None]
    if not batch: return None

    # 2. 找到当前批次中最大的视角数
    max_views = max([item[0].shape[0] for item in batch]) # item[0] is videos_tensor

    # 3. 准备用于存储填充后数据的列表
    padded_videos, padded_masks, labels, num_views_list, padded_orig_frames, case_ids = [], [], [], [], [], []

    # 4. 遍历批次中的每个样本
    for videos, masks, label, num_views, orig_frames, case_id in batch:
        num_padding = max_views - num_views
        
        # a. 如果需要填充
        if num_padding > 0:
            # 使用最后一个有效视角的数据进行填充
            video_pad = videos[-1].unsqueeze(0).repeat(num_padding, 1, 1, 1, 1)
            videos = torch.cat([videos, video_pad], dim=0)

            mask_pad = masks[-1].unsqueeze(0).repeat(num_padding, 1, 1, 1)
            masks = torch.cat([masks, mask_pad], dim=0)

            orig_frame_pad = orig_frames[-1].unsqueeze(0).repeat(num_padding, 1, 1, 1, 1)
            orig_frames = torch.cat([orig_frames, orig_frame_pad], dim=0)

        padded_videos.append(videos)
        padded_masks.append(masks)
        labels.append(label)
        num_views_list.append(num_views) # 存储的是【填充前】的真实视角数
        padded_orig_frames.append(orig_frames)
        case_ids.append(case_id)

    # 5. 将填充好的数据列表，堆叠成最终的批次张量
    final_videos = torch.stack(padded_videos)
    final_masks = torch.stack(padded_masks)
    final_labels = torch.stack(labels)
    final_num_views = torch.tensor(num_views_list, dtype=torch.long)
    final_orig_frames = torch.stack(padded_orig_frames)
    
    return final_videos, final_masks, final_labels, final_num_views, final_orig_frames, case_ids

def remove_info(array): #remove info on the image
    video = array.copy()
    l = video.shape[0] //2
    v1 = video[:l]
    v2 = video[-l:]
    mask = np.sum(v1-v2, axis=0)
    video[:,mask==0] = 0
    return video

def convertRGB(array):
    shape = array.shape
    assert len(shape) in [2,3]
    if len(shape) == 2: # image
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    else:
        array = np.array([cv2.cvtColor(a, cv2.COLOR_GRAY2RGB) for a in array])
    return array

def crop_resize_video(array, size=256, convert_RGB=True):
    imgs = [make_img_square(a) for a in array]
    video = np.array([resize_image(a, size, convert_RGB) for a in imgs])
    return video

def make_img_square(array):
    if array.shape[0] != array.shape[1]:
        diff = abs(array.shape[1] - array.shape[0]) // 2
        if len(array.shape) == 3: 
            if array.shape[0] < array.shape[1]:
                array = np.pad(array, ((diff,diff), (0,0), (0,0)), mode='constant')
            else:
                array = np.pad(array, ((0,0), (diff,diff), (0,0)), mode='constant')
        else:
            if array.shape[0] < array.shape[1]:
                array = np.pad(array, ((diff,diff), (0,0)), mode='constant')
            else:
                array = np.pad(array, ((0,0), (diff,diff)), mode='constant')
    return array

def resize_image(array, size, convert_RGB=True):
    img = Image.fromarray(array.astype('uint8'))
    if convert_RGB:
        img = img.convert('RGB')
    img = img.resize((size, size), resample=PIL.Image.BILINEAR)
    return np.array(img)


def clip_video(data_path, video, size = 64, random_clip=False): #clip video to size(64) padding itself
    # print('data_path',data_path)
    video0 = [v for v in video]
    video = [v for v in video]
    while (len(video) < size):
        video += video0
    try:
        start = random.choice(range(len(video) - size))
    except:
        start = 0
    if not random_clip:
        start = 0
    # print('data_path',data_path)
    # print('start',start)
    video = video[start:start + size]
    video = np.array(video)
    # print('video.shape',video.shape)
    return video

def create_attention_mask(batch_size, seq_len, indices_to_mask, device):
    """创建布尔类型的 attention mask (True的位置被忽略)。"""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i in indices_to_mask:
        if 0 <= i < seq_len:
            mask[:, i] = True
    return mask

def video_loader(path): 
    """
    从指定路径加载视频，并将其转换为 (T, H, W, C) 格式的RGB Numpy数组。
    """
    # 检查文件是否存在
    if not os.path.exists(path):
        print(f"错误：视频文件未找到: {path}")
        return None
        
    cap = cv2.VideoCapture(path)
    
    # 检查视频是否能被成功打开
    if not cap.isOpened():
        print(f"错误：OpenCV无法打开视频文件: {path}")
        cap.release()
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 将OpenCV默认的BGR格式转换为标准的RGB格式
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()

    if not frames:
        print(f"警告：视频文件为空或已损坏，无法读取任何帧: {path}")
        return None
        
    return np.array(frames)

def process_video_for_inference(video_frames, target_len, resize_dim):
    """
    返回处理后的视频帧和原始索引映射。
    """
    
    # 1. 移除静态信息
    # 注意: remove_info 可能会修改传入的数组，使用 .copy() 更安全
    processed_frames = remove_info(video_frames.copy())
    # processed_frames = video_frames.copy()

    # 2. 时间维度处理 (循环补帧 或 截断)
    original_indices = list(range(processed_frames.shape[0]))
    video_list = list(processed_frames) # 转为列表以方便追加

    # 如果帧数不足，循环补全
    if len(video_list) < target_len:
        print(f"帧数 ({len(video_list)}) < {target_len}，正在进行循环补帧...")
        original_video_copy = video_list.copy()
        original_indices_copy = original_indices.copy()
        while len(video_list) < target_len:
            video_list.extend(original_video_copy)
            original_indices.extend(original_indices_copy)

    # 截取到目标长度 (对于短视频和长视频都适用)
    final_frames_list = video_list[:target_len]
    final_index_map = original_indices[:target_len]
    
    final_frames_np = np.array(final_frames_list)
    
    # 3. 空间维度处理 (Pad-to-Square then Resize)
    #    直接调用训练时的函数
    print(f"正在进行图像处理 (Pad-to-Square & Resize to {resize_dim}x{resize_dim})...")
    final_frames_np = crop_resize_video(final_frames_np, size=resize_dim, convert_RGB=False)

    return final_frames_np, final_index_map


def find_static_content_mask(frame1, frame2, diff_threshold=5):
    """
    辅助函数：比较两张连续的帧，找到它们之间完全相同（静态）的区域。
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, static_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY_INV)
    return static_mask

def find_doppler_roi_from_video(video_frames, area_threshold, padding):
    """【V5版】ROI提取，输入的是固定长度的视频片段。"""
    max_area = 0
    best_frame_idx = -1
    for i, frame in enumerate(video_frames):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50]); upper_red2 = np.array([180, 255, 255])
        lower_blue = np.array([100, 70, 50]); upper_blue = np.array([130, 255, 255])
        lower_turb = np.array([11, 70, 50]); upper_turb = np.array([90, 255, 255])
        mask_r1 = cv2.inRange(hsv_img, lower_red1, upper_red1); mask_r2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        mask_b = cv2.inRange(hsv_img, lower_blue, upper_blue); mask_t = cv2.inRange(hsv_img, lower_turb, upper_turb)
        current_area = np.sum((mask_r1 + mask_r2 + mask_b + mask_t) > 0)
        if current_area > max_area:
            max_area = current_area
            best_frame_idx = i

    if best_frame_idx == -1 or best_frame_idx + 1 >= len(video_frames): return None
    
    best_frame = video_frames[best_frame_idx]
    next_frame = video_frames[best_frame_idx + 1]
    static_mask = find_static_content_mask(best_frame, next_frame)
    hsv_best = cv2.cvtColor(best_frame, cv2.COLOR_BGR2HSV)
    mask_r1 = cv2.inRange(hsv_best, lower_red1, upper_red1); mask_r2 = cv2.inRange(hsv_best, lower_red2, upper_red2)
    mask_b = cv2.inRange(hsv_best, lower_blue, upper_blue); mask_t = cv2.inRange(hsv_best, lower_turb, upper_turb)
    color_mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), cv2.bitwise_or(mask_b, mask_t))
    cleaned_color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(static_mask))
    contours, _ = cv2.findContours(cleaned_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]

    if not final_contours: return None

    all_points = np.concatenate(final_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    center_x, center_y = x + w // 2, y + h // 2
    max_side = max(w, h) + padding * 2
    x1 = max(0, center_x - max_side // 2); y1 = max(0, center_y - max_side // 2)
    x2 = min(best_frame.shape[1], center_x + max_side // 2)
    y2 = min(best_frame.shape[0], center_y + max_side // 2)
    return (x1, y1, x2, y2)


def calculate_iou(boxA, boxB):
    if boxA is None or boxB is None: return 0.0
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0: return 0.0
    return interArea / unionArea



def create_string_token_mask(segment_np, spatial_resolution=4, vote_threshold=1, area_threshold_ratio=0.5):  # 根据感受野改变生成的mask
    """
    【最终版 - 串Token掩码】
    根据整个视频片段的血流信息，生成一个 (16,) 的“串”Token掩码。

    Args:
        segment_np (np.array): 视频片段 (T, H, W, C)，例如 (16, H_orig, W_orig, 3)。
        spatial_resolution (int): 空间分辨率，例如 4 (对于4x4网格)。
        vote_threshold (int): 一个空间位置在16帧中至少需要多少帧有血流才被认为有效。
                              根据您的要求，设置为1。

    Returns:
        np.array: 一个 (16,) 的布尔掩码，True代表该“串”Token应被忽略。
    """
    num_timesteps = segment_np.shape[0]
    num_spatial_locations = spatial_resolution * spatial_resolution
    model_input_size = (224, 224)
    patch_size = model_input_size[0] // spatial_resolution
    
    # 阈值：一个56x56的patch里需要有多少血流像素才算有信号
    # 我们可以设置一个相对比例，例如 0.5%
    area_threshold_ratio = 0.5 
    area_threshold = (patch_size * patch_size) * area_threshold_ratio

    # 定义颜色阈值
    lower_red1 = np.array([0, 70, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50]); upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 70, 50]); upper_blue = np.array([130, 255, 255])
    lower_turb = np.array([11, 70, 50]); upper_turb = np.array([90, 255, 255])

    # 1. 构建“投票矩阵”
    # vote_matrix[t, s] = True 表示第t帧的第s个patch有足够血流
    vote_matrix = np.zeros((num_timesteps, num_spatial_locations), dtype=bool)
    
    for t, frame_bgr in enumerate(segment_np):
        # 预处理帧
        frame_resized = cv2.resize(frame_bgr, model_input_size)
        hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        
        # 一次性计算整帧的颜色掩码
        mask_r1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask_b = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        mask_t = cv2.inRange(hsv_frame, lower_turb, upper_turb)
        color_mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), cv2.bitwise_or(mask_b, mask_t))
        
        # 遍历4x4的patch网格
        for s in range(num_spatial_locations):
            row = s // spatial_resolution
            col = s % spatial_resolution
            
            # 提取当前patch对应的颜色掩码区域
            patch_mask = color_mask[
                row*patch_size : (row+1)*patch_size,
                col*patch_size : (col+1)*patch_size
            ]
            
            # 计算patch内的血流面积
            blood_flow_area = np.sum(patch_mask > 0)
            
            # 如果面积超过阈值，则为这个时空点“投有效票”
            if blood_flow_area > area_threshold:
                vote_matrix[t, s] = True

    # 2. 根据投票结果，为16个“串”Token做出最终决策
    # True代表“屏蔽”，False代表“不屏蔽”
    string_mask = np.zeros(num_spatial_locations, dtype=bool)
    
    for s in range(num_spatial_locations):
        # 统计每个空间位置在所有时间步上的得票数
        num_votes = np.sum(vote_matrix[:, s])
        
        # 如果有效帧数小于等于阈值，则屏蔽这个“串”
        if num_votes <= vote_threshold:
            string_mask[s] = True
            
    return string_mask

def parse_doctor_annotations(json_path, frame_limit=16):
    if not os.path.exists(json_path): return None
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    annotations = {}
    bounding_box_model = data.get("Models", {}).get("BoundingBoxLabelModel", [])
    if not bounding_box_model: return annotations
    for item in bounding_box_model:
        frame_idx = item.get("FrameCount")
        if frame_idx is not None and frame_idx < frame_limit:
            p1, p2 = item.get("p1"), item.get("p2")
            if p1 and p2:
                annotations[frame_idx] = [min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1])]
    return annotations