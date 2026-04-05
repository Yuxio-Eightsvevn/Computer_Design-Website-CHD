import argparse
import yaml
config_path = '/mnt/data1/zyh/CHD_1001/CHD_classify/test_configs/test_config_ResnetrTransformerDualToken_MutilViews_1118_18cls_layersdecouple55.yaml'    
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
for c in config:
    print(c, config[c])
cuda_device = config['cuda_device']

import os
import numpy as np
import torch
import torchvision
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nets_set.Multi_Views_dual_tokens_net import MultiViewDualTokensFusionSize
from functions_set.functions_RNT_dualclsentmax_mask_multiviews import *
from datasets.dataset_RNT_4cls_temporal_fusion_mutilviews import MultiViewDataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt  # 新增：绘图库
from sklearn.metrics import ConfusionMatrixDisplay  # 新增：混淆矩阵可视化
from utils import *
from timm.scheduler import CosineLRScheduler # 【新增】导入新的学习率调度器
from torch.utils.data import Dataset, DataLoader

# 尝试导入颜色库，方便终端查看错误病例
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class Fore: RED = ""; GREEN = ""; YELLOW = ""; RESET = ""
    class Style: BRIGHT = ""

num_classes = config['num_classes']
batch_size = config['batch_size']
learning_rate = float(config['learning_rate'])
total_epoch = config['total_epoch']
# steps = config['steps']
seed = config['seed']
num_workers = config['num_workers']
momentum = config['momentum']
weight_decay = config['weight_decay']

view_dropout = config['view_dropout']
drop_rate = config['drop_rate']
add_bias= config['add_bias']
# clip_size = config['clip_size']
num_segments = config['num_segments']
clip_size = config['clip_size']
train_tag = config['train_tag']

# 从config中读取ResNetTransformer的特定参数
d_model = config.get('d_model', 512)
spatial_size = config.get('spatial_size')
view_nhead = config.get('view_nhead')
view_encoder_layers = config.get('view_encoder_layers')
fusion_nhead = config.get('fusion_nhead')
fusion_layers = config.get('fusion_layers')
dropout = config.get('dropout', 0.1)


unfreeze_epoch = config['unfreeze_epoch']
# 余弦退火调度器参数 (保持不变)
use_scheduler = config['unfreeze_epoch']
warmup_epochs = config['warmup_epochs']
warmup_lr = config['warmup_lr']
lr_min = config['lr_min']

description = 'CHD_1views_{}'.format(train_tag)
print('description： {}'.format(description))
save_dir = '/mnt/data1/zyh/CHD_1001/CHD_classify/output/checkpoint/' + description
vis_dir = '/mnt/data1/zyh/CHD_1001/CHD_classify/output/visualize/' + description
train_class_weight = config['train_class_weight']
print('train class weight', train_class_weight)
valid_class_weight = config['valid_class_weight']
print('valid class weight', valid_class_weight)



#define
fix_randomness(seed)

best_valid_acc = 0
best_valid_acc_epoch = -1
best_loss_epoch = -1
best_loss = 100.0


# 预处理辅助函数
ROI_AREA_THRESHOLD = 150
# ROI_DISTANCE_THRESHOLD = 150
ROI_PADDING = 20

def find_static_content_mask(frame1, frame2, diff_threshold=5):
    """
    辅助函数：比较两张帧，找到它们之间完全相同（静态）的区域。
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    # 差值小于阈值的被认为是静态(255)，反之为动态(0)
    _, static_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY_INV)
    return static_mask

def find_doppler_roi_from_video(video_frames, area_threshold, padding, frame_stride=4):
    """
    【V6版 - 修复版】ROI提取
    改进点：
    1. 默认步长(frame_stride)设为4，提高动态检测灵敏度。
    2. 修复了当最佳帧位于视频末尾时无法计算静态掩码的问题（改为向前寻找对比帧）。
    """
    max_area = 0
    best_frame_idx = -1
    
    # 1. 遍历所有帧，寻找彩色信号面积最大的一帧 (Best Frame)
    for i, frame in enumerate(video_frames):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 定义红、蓝、湍流色的HSV范围
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

    # 如果整个视频都没有检测到颜色，返回None
    if best_frame_idx == -1: 
        return None
    
    # 2. 确定用于对比的帧 (处理边界情况)
    total_frames = len(video_frames)
    best_frame = video_frames[best_frame_idx]
    compare_frame = None

    # 优先向后找 stride 帧
    if best_frame_idx + frame_stride < total_frames:
        compare_frame = video_frames[best_frame_idx + frame_stride]
    # 如果后面不够，则向前找 stride 帧
    elif best_frame_idx - frame_stride >= 0:
        compare_frame = video_frames[best_frame_idx - frame_stride]
    else:
        # 如果视频非常短（总帧数 < stride），则退回到寻找最远的可用帧
        # 避免之前直接 return None 的情况
        if best_frame_idx < total_frames - 1:
            compare_frame = video_frames[total_frames - 1] # 找最后一帧
        elif best_frame_idx > 0:
            compare_frame = video_frames[0] # 找第一帧
        else:
            # 视频只有1帧的情况，无法做静态擦除，直接用原图
            compare_frame = best_frame

    # 3. 计算静态掩码
    static_mask = find_static_content_mask(best_frame, compare_frame)
    
    # 4. 重新计算最佳帧的颜色掩码
    hsv_best = cv2.cvtColor(best_frame, cv2.COLOR_BGR2HSV)
    mask_r1 = cv2.inRange(hsv_best, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv_best, lower_red2, upper_red2)
    mask_b = cv2.inRange(hsv_best, lower_blue, upper_blue)
    mask_t = cv2.inRange(hsv_best, lower_turb, upper_turb)
    color_mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), cv2.bitwise_or(mask_b, mask_t))
    
    # 5. 核心：颜色掩码 减去 静态掩码
    cleaned_color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(static_mask))
    
    # 6. 寻找轮廓并生成包围框
    contours, _ = cv2.findContours(cleaned_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]

    # 增加保底逻辑：如果静态擦除太狠导致没有轮廓，尝试退回使用原始颜色掩码
    if not final_contours:
        contours_raw, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = [cnt for cnt in contours_raw if cv2.contourArea(cnt) > area_threshold]

    if not final_contours: 
        return None

    all_points = np.concatenate(final_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # 添加 Padding
    center_x, center_y = x + w // 2, y + h // 2
    max_side = max(w, h) + padding * 2
    
    # 确保不越界
    x1 = max(0, center_x - max_side // 2)
    y1 = max(0, center_y - max_side // 2)
    x2 = min(best_frame.shape[1], center_x + max_side // 2)
    y2 = min(best_frame.shape[0], center_y + max_side // 2)
    
    return (x1, y1, x2, y2)


def create_string_token_mask_size(segment_np, 
                             spatial_resolution, # <-- 【核心修改1】现在是必需参数
                             model_input_size=(224, 224), 
                             vote_threshold=1, 
                             area_threshold_ratio=0.05): # <-- 【核心修改2】默认阈值更合理
    """
    【最终通用版 - 串Token掩码】
    根据整个视频片段的血流信息，为任意分辨率的网格生成“串”Token掩码。

    Args:
        segment_np (np.array): 视频片段 (T, H, W, C)。
        spatial_resolution (int): 空间网格的分辨率，例如 5 (对于5x5网格)。
        model_input_size (tuple): 模型期望的输入图像尺寸。
        vote_threshold (int): 一个空间位置在所有帧中至少需要多少帧有血流才被认为有效。
        area_threshold_ratio (float): 一个patch内血流像素面积占总面积的比例阈值。

    Returns:
        np.array: 一个 (spatial_resolution*spatial_resolution,) 的布尔掩码，
                  True代表该“串”Token应被忽略。
    """
    num_timesteps = segment_np.shape[0]
    num_spatial_locations = spatial_resolution * spatial_resolution
    
    # 【核心修改3】patch_size 现在根据传入的 spatial_resolution 动态计算
    patch_size_h = model_input_size[0] // spatial_resolution
    patch_size_w = model_input_size[1] // spatial_resolution
    
    area_threshold = (patch_size_h * patch_size_w) * area_threshold_ratio

    # 定义颜色阈值 (保持不变)
    lower_red1 = np.array([0, 70, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50]); upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 70, 50]); upper_blue = np.array([130, 255, 255])
    lower_turb = np.array([11, 70, 50]); upper_turb = np.array([90, 255, 255])

    # 1. 构建“投票矩阵”
    vote_matrix = np.zeros((num_timesteps, num_spatial_locations), dtype=bool)
    
    for t, frame_bgr in enumerate(segment_np):
        frame_resized = cv2.resize(frame_bgr, model_input_size)
        hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        
        mask_r1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        mask_b = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        mask_t = cv2.inRange(hsv_frame, lower_turb, upper_turb)
        color_mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), cv2.bitwise_or(mask_b, mask_t))
        
        # 遍历网格
        for s in range(num_spatial_locations):
            row = s // spatial_resolution
            col = s % spatial_resolution
            
            # 【核心修改4】使用动态计算的 patch_size
            patch_mask = color_mask[
                row * patch_size_h : (row + 1) * patch_size_h,
                col * patch_size_w : (col + 1) * patch_size_w
            ]
            
            blood_flow_area = np.sum(patch_mask > 0)
            
            if blood_flow_area > area_threshold:
                vote_matrix[t, s] = True

    # 2. 根据投票结果做出最终决策
    # num_votes 是一个 (num_spatial_locations,) 的数组，包含了每个“串”的得票数
    num_votes = np.sum(vote_matrix, axis=0) 
    
    # 如果得票数小于等于阈值，则屏蔽 (返回 True)
    string_mask = (num_votes < vote_threshold) 
            
    return string_mask


# --- 4. 【核心】为外部数据定义Dataset ---
class ExternalMultiViewDataset(Dataset):
    def __init__(self, json_path, transform, clip_size=16, max_views=12, random_sample_views=True):
        """
        专门用于处理外部测试数据的Dataset。
        - json_path: 指向我们之前生成的、带前缀的病例JSON文件。
        - random_sample_views: 是否在视角数超过max_views时进行随机采样。
        """
        with open(json_path, 'r') as f:
            self.case_data = json.load(f)
        
        self.case_ids = list(self.case_data.keys())
        self.transform = transform
        self.clip_size = clip_size
        self.max_views = max_views
        self.random_sample_views = random_sample_views

    def get_labels(self):
        return [info['label'] for info in self.case_data.values()]

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        case_info = self.case_data[case_id]
        
        all_view_paths = case_info['video_paths']
        
        # 存储通过血流验证的、【已处理好】的视频numpy数组
        valid_views_data_pairs = [] 

        # --- 步骤 1: 遍历所有原始路径，进行加载、验证和预处理 ---
        for video_path in all_view_paths:
            try:
                # a. 加载原始视频
                original_frames = video_loader(video_path)
                if original_frames is None or original_frames.shape[0] < 1:
                    continue

                # b. 剪辑/填充到固定长度 (e.g., 16帧)
                if original_frames.shape[0] >= self.clip_size:
                    processed_16_frames = original_frames[0 : self.clip_size]
                else:
                    padding_needed = self.clip_size - original_frames.shape[0]
                    padding_frames = np.repeat(original_frames[0:1], padding_needed, axis=0)
                    processed_16_frames = np.concatenate([original_frames, padding_frames], axis=0)
                
                # c. 检查是否存在有效的血流ROI
                roi_bbox = find_doppler_roi_from_video(processed_16_frames, ROI_AREA_THRESHOLD, ROI_PADDING)
                
                # d. 如果有血流，则进行裁剪和缩放，并存入列表
                if roi_bbox is not None:
                    x1, y1, x2, y2 = roi_bbox
                    frames_for_model = [cv2.resize(f[y1:y2, x1:x2], (224, 224)) for f in processed_16_frames if f[y1:y2, x1:x2].size > 0]
                    
                    if len(frames_for_model) != self.clip_size:
                        continue
                    
                    frames_for_model_np = np.array(frames_for_model)

                    # --- ▼▼▼【核心新增：为处理好的视频生成Mask】▼▼▼ ---
                    blood_flow_mask = create_string_token_mask_size(
                        frames_for_model_np, 
                        spatial_resolution=spatial_size
                    )
                    # --- ▲▲▲【核心新增结束】▲▲▲ ---
                    
                    # 将处理好的 (视频, mask) 对存起来
                    valid_views_data_pairs.append((frames_for_model_np, blood_flow_mask))

            except Exception as e:
                print(f"警告: 在处理视频 {os.path.basename(video_path)} 时发生异常: {e}")
                continue

         # --- 对过滤后的有效视角进行采样 ---
        views_to_process_pairs = valid_views_data_pairs
        if len(views_to_process_pairs) > self.max_views:
            if self.random_sample_views:
                indices_to_sample = random.sample(range(len(views_to_process_pairs)), self.max_views)
                views_to_process_pairs = [views_to_process_pairs[i] for i in indices_to_sample]
            else:
                views_to_process_pairs = views_to_process_pairs[:self.max_views]

        if not views_to_process_pairs:
            # print(f"警告: 病例 {case_id} 没有找到任何带血流的有效视角，将跳过该样本。")
            return None

        # --- 将最终选定的数据转换为Tensor ---
        view_tensors_list = []
        mask_tensors_list = []
        for frames_np, mask_np in views_to_process_pairs:
            # 处理视频
            video_tensor = torch.from_numpy(frames_np.copy()).permute(0, 3, 1, 2)
            if self.transform:
                video_tensor = self.transform(video_tensor)
            view_tensors_list.append(video_tensor)
            # print(video_path)
            # print(video_tensor.max())
            # print(video_tensor.min())

            # 处理Mask
            mask_tensors_list.append(torch.from_numpy(mask_np))

        # --- 进行视角数量的填充 (Padding) ---
        num_valid_views = len(view_tensors_list)
        padding_needed = self.max_views - num_valid_views
        
        if padding_needed > 0:
            # 获取填充所需的形状
            _, c, h, w = view_tensors_list[0].shape
            mask_shape = mask_tensors_list[0].shape
            
            # 创建用于填充的 placeholder
            zero_padding_view = torch.zeros((self.clip_size, c, h, w), dtype=torch.float32)
            # Mask的padding值是True (代表忽略)
            true_padding_mask = torch.ones(mask_shape, dtype=torch.bool)
            
            for _ in range(padding_needed):
                view_tensors_list.append(zero_padding_view)
                mask_tensors_list.append(true_padding_mask)
        
        # --- 堆叠并返回所有数据 ---
        multi_view_video = torch.stack(view_tensors_list, dim=0)
        multi_view_mask = torch.stack(mask_tensors_list, dim=0)
        label = torch.tensor(case_info['label'], dtype=torch.long)
        num_valid_views_tensor = torch.tensor(num_valid_views, dtype=torch.long)
        
        # 【重要】更新返回值，现在包含mask
        return multi_view_video, multi_view_mask, label, num_valid_views_tensor, case_id


    def __len__(self):
        return len(self.case_ids)

def collate_fn_skip_none(batch):
    """
    一个自定义的 collate_fn。
    它会过滤掉 batch 中所有值为 None 的样本。
    """
    # 过滤掉所有 None 元素
    batch = [item for item in batch if item is not None]
    
    # 如果过滤后 batch 为空，需要特殊处理
    if not batch:
        # 返回一个空的 batch 或者你可以根据需要抛出异常
        # 这里我们返回一个可以被解包的元组，但里面的tensor是空的
        # 这需要你的训练循环能处理空batch的情况
        return None 
    
    # 如果 batch 不为空，就使用 PyTorch 默认的 collate 函数来打包剩下的好样本
    return torch.utils.data.dataloader.default_collate(batch)

test_transform =  v2.Compose([
    # v2.Resize(size=(224, 224), antialias=True), # <--- 【核心修正】在这里添加Resize
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# 【重要】: 替换为您的外部数据JSON文件路径
# external_json_path = config['split_json_path']
external_json_path = "/mnt/data1/zyh/CHD_1001/CHD_classify/Clinical_trial_design_and_indicator_calculation_1127/XAI_Results_Final_0123/EXP_Final_Dataset_0123.json"
OUTPUT_DIR = '/mnt/data1/zyh/CHD_1001/CHD_classify/Clinical_trial_design_and_indicator_calculation_1127/XAI_Results_Final_0123/EXP_Video_Datas'
OUTPUT_JSON_NAME = '/mnt/data1/zyh/CHD_1001/CHD_classify/Clinical_trial_design_and_indicator_calculation_1127/XAI_Results_Final_0123/ai_diagnosis_summary.json'

test_dataset = ExternalMultiViewDataset(
    json_path=external_json_path,
    transform=test_transform,
    clip_size=clip_size,
    max_views=5,
    random_sample_views=False # 测试时设为False以保证结果可复现
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1, # 通常测试时 batch_size 设为 8, 16, 32 等
    shuffle=False,
    num_workers=num_workers, 
    pin_memory=True,
    collate_fn=collate_fn_skip_none # 使用自定义的collate_fn
)
print(f'外部测试数据大小: {len(test_dataset)} 个病例。')


# 选择训练模型 
torch.cuda.set_device(cuda_device) # 制定GPU

# 选择不同的网络模型 共用同样的训练逻辑 方便对比实验 add_bias控制某些层是否加入bias项，与训练稳定性有关
model_name = config['model_name']
model = MultiViewDualTokensFusionSize(view_num_classes=18, d_model=d_model, 
                                    view_nhead=view_nhead, view_encoder_layers=view_encoder_layers,
                                    case_num_classes=num_classes,
                                    fusion_layers=fusion_layers, fusion_nhead=fusion_nhead,
                                    max_views=5,dropout=dropout,
                                    spatial_size=spatial_size,
                                )

# 加载预训练权重（面对实验被迫终止情况 或者有预训练权重）
# model = torch.nn.DataParallel(model)
resume_path = config['resume_path']
if resume_path != '': 
    resume_model_dict = torch.load(resume_path,map_location='cpu') # map_location='cpu'保证当前模型在cGPU上
    resume_model_filter_dict = {k: v for k, v in resume_model_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    # 只有当模型的名字出现在
    model.load_state_dict(resume_model_filter_dict,strict=False)
    print('load model from {}'.format(resume_path))


model = model.cuda().eval()      

# Results Container
final_results_summary = {}
class_names = ['Normal', 'VSD', 'ASD', 'PDA']

# Counters
total_processed = 0
correct_predictions = 0

print(f"\n{Fore.CYAN}--- 开始推理 ---{Fore.RESET}")

with torch.no_grad():
    for data in tqdm(test_loader, desc="Inference"):
        if data is None: continue
        
        videos, masks, label, num_views, case_ids = data
        case_id = case_ids[0] # Batch size is 1
        gt_label_idx = label.item()
        gt_label_str = class_names[gt_label_idx]

        # Inference
        video_tensor = videos.cuda(non_blocking=True)
        mask_tensor = masks.cuda(non_blocking=True)
        num_views_tensor = num_views.cuda(non_blocking=True)
        
        with autocast(device_type='cuda', dtype=torch.float16):
                case_logits = model(video_tensor, num_views_tensor, mask_tensor)
        
        probs = torch.softmax(case_logits, dim=1).squeeze().cpu().tolist()
        pred_idx = np.argmax(probs)
        pred_str = class_names[pred_idx]
        confidence = probs[pred_idx]

        # 保存结果到大字典
        final_results_summary[case_id] = {
            "ground_truth": gt_label_str,
            "predicted_class": pred_str,
            "confidence": round(confidence, 4),
            "all_probs": {class_names[i]: round(probs[i], 4) for i in range(4)}
        }

        # 统计
        total_processed += 1
        if pred_idx == gt_label_idx:
            correct_predictions += 1
        else:
            # 【核心功能】: 打印预测错误的病例
            print(f"\n{Fore.RED}[Misclassified] Case: {case_id}{Fore.RESET}")
            print(f"  {Fore.YELLOW}Ground Truth: {gt_label_str}{Fore.RESET}")
            print(f"  {Fore.RED}Predicted:    {pred_str} (Conf: {confidence:.4f}){Fore.RESET}")
            print(f"  All Probs:    {final_results_summary[case_id]['all_probs']}")

# ================= 5. 保存结果 =================

os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON_NAME)

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(final_results_summary, f, indent=4)

accuracy = correct_predictions / total_processed if total_processed > 0 else 0
print(f"\n{Fore.GREEN}--- 推理完成 ---{Fore.RESET}")
print(f"总处理病例: {total_processed}")
print(f"准确率 (Accuracy): {accuracy:.4f} ({correct_predictions}/{total_processed})")
print(f"完整结果已保存至: {save_path}")