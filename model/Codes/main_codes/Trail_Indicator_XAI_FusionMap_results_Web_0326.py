import argparse
import yaml
config_path ='/mnt/data1/zyh/CHD_1001/CHD_classify/test_configs/test_config_ResnetrTransformerDualToken_1107_18cls_layersdecouple44.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
for c in config:
    print(c, config[c])
cuda_device = config['cuda_device']


import os
import sys
import re
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from PIL import Image, ImageDraw
import torch 
import torch.nn.functional as F
from nets_set.dual_tokens_net import ResnetTransformerDualTokensTemporalSpatialDecouplesize
# from functions_set.functions_RNT_dualclsentmax_mask_spatial_cls18 import *
# from dataset_pickle_2classs import OneViewDataset
from torchvision.transforms import v2
from collections import defaultdict
from tqdm import tqdm # 【已修正】导入tqdm
import matplotlib.pyplot as plt  # 新增：绘图库
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

# hyper-params
num_classes = config['num_classes']
batch_size = config['batch_size']
learning_rate = float(config['learning_rate'])
total_epoch = config['total_epoch']
# steps = config['steps']
seed = config['seed']
num_workers = config['num_workers']
weight_decay = config['weight_decay']
clip_size = config['clip_size']
# view_dropout = config['view_dropout']
drop_rate = config['drop_rate']
# add_bias= config['add_bias']
train_tag = config['train_tag']

d_model_transformer = config['d_model_transformer']
d_model_cnn = config['d_model_cnn']
compressed_dim = config['compressed_dim']
nhead = config['nhead']
num_layers = config['num_layers']
dropout = config['dropout']

def fix_randomness(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

fix_randomness(seed)

DATASET_ROOT_DIR = '/mnt/data1/zyh/CHD_1001/CHD_classify/Clinical_trial_design_and_indicator_calculation_1127/Trail_datasets' # <--- 【【【请修改为你的新数据集路径】】】
OUTPUT_DIR = '/mnt/data1/zyh/CHD_1001/CHD_classify/Clinical_trial_design_and_indicator_calculation_1127/XAI_Test_Results_Web_0325'
FOLDERS_TO_PROCESS = ['ASD_Abnormal', 'ASD_Normal', 'VSD_Abnormal', 'VSD_Normal', 'PDA_Abnormal', 'PDA_Normal']
# FOLDERS_TO_PROCESS = ['ASD_Normal', 'VSD_Normal', 'PDA_Normal']
# FOLDERS_TO_PROCESS = [ 'ASD_Abnormal_1221', 'VSD_Abnormal_1221']
# FOLDERS_TO_PROCESS = [ 'Normal_0123']


ROI_AREA_THRESHOLD = 150
ROI_PADDING = 20
MODEL_INPUT_SIZE = (224, 224)
SPATIAL_RESOLUTION = 4 # 我们的模型是 4x4 网格
PATCH_SIZE = MODEL_INPUT_SIZE[0] // SPATIAL_RESOLUTION


# POSITIVE_TO_NEGATIVE_MAP = {
#     "normal_view1": 0, "normal_view2": 1, "normal_view3": 2, "normal_view4": 3,
#     "normal_view5": 4, "normal_view6": 5, "normal_view7": 6, "normal_view8": 7,
#     "ASD_view2": 8, "ASD_view5": 9,
#     "VSD_view1": 10, "VSD_view2": 11, "VSD_view3": 12, "VSD_view4": 13, "VSD_view5": 14,
#     "ASD_view6": 15, "ASD_view7": 16,
#     "PDA_view8": 17,
# }

POSITIVE_TO_NEGATIVE_PAIR_MAP = {
    10: 0, 11: 1, 12: 2, 13: 3, 14: 4,  
    15: 5, 16: 6, 8: 1, 9: 4,          
    17: 7                                
}

def find_static_content_mask(frame1, frame2, diff_threshold=5):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, static_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY_INV)
    return static_mask

def find_doppler_roi_from_video(video_frames, area_threshold, padding, frame_stride=4):
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

def extract_case_id_robust(filename, video_path):
    path_lower = video_path.lower()
    if os.path.join('vsd_abnormal_1221', '') in path_lower:
        return filename.split('_', 1)[0]
    
    if '_ser' in filename:
        return filename.split('_ser', 1)[0]
    elif '_view' in filename:
        return filename.split('_view', 1)[0]
    else:
        return os.path.splitext(filename)[0]

def get_coarse_label_from_folder_name(folder_name):
    folder_lower = folder_name.lower()
    if 'vsd_abnormal' in folder_lower: return 1
    elif 'asd_abnormal' in folder_lower: return 2
    elif 'pda_abnormal' in folder_lower: return 3
    elif 'normal' in folder_lower: return 0
    else:
        print(f"警告：无法从文件夹名 '{folder_name}' 中确定粗粒度标签。")
        return -1
    
def transform_box_from_model_to_original_space(box_model, roi_bbox):
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
    cx, cy = center_point
    half_size = box_size // 2
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(image_size[0], cx + half_size)
    y2 = min(image_size[1], cy + half_size)
    return [int(x1), int(y1), int(x2), int(y2)]

# ================= 可视化相关 (仅保留注意力热图部分) =================
def create_ensemble_visualization(
    output_path, 
    background_image, 
    individual_heatmaps, 
    fused_heatmap,
    pred_box, 
    full_video_filename,
    pred_label,
    pred_key_frame
):
    model_names = sorted(individual_heatmaps.keys())
    num_models = len(model_names)
    
    fig, axes = plt.subplots(2, num_models, figsize=(num_models * 6, 12), dpi=120)
    
    title = (f"Attention Analysis for: {full_video_filename}\n"
             f"Predicted Class: {pred_label} | Predicted Key Frame: {pred_key_frame} ")
    fig.suptitle(title, fontsize=20)

    # 第一行：各模型的注意力热力图
    for i, name in enumerate(model_names):
        ax = axes[0, i]
        heatmap = individual_heatmaps[name]
        
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(background_image, 0.6, heatmap_colored, 0.4, 0)

        ax.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Attention CAM: {name}', fontsize=14)

    # 第二行：融合图与预测框
    ax_fused = axes[1, 0]
    im_fused = ax_fused.imshow(fused_heatmap, cmap='viridis')
    ax_fused.set_title('Fused Heatmap (Sum)', fontsize=14)
    ax_fused.axis('off')
    fig.colorbar(im_fused, ax=ax_fused, fraction=0.046, pad=0.04)

    ax_compare = axes[1, 1]
    ax_compare.imshow(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
    
    if pred_box:
        p_rect = patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1],
                                linewidth=3, edgecolor='lime', facecolor='none')
        ax_compare.add_patch(p_rect)
    
    ax_compare.set_title('BBox Comparison', fontsize=14)
    ax_compare.axis('off')

    for i in range(2, num_models):
        axes[1, i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)

class NpEncoder(json.JSONEncoder):
    """ 自定义JSON编码器，用于处理Numpy数据类型 """
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    test_transform =  v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    torch.cuda.set_device(cuda_device)

    MODEL_CONFIGS = {
        '4x4': {
            'resume_path': '/mnt/data1/zyh/CHD_1001/CHD_classify/output/checkpoint/CHD_1views_1101_ResnetTransformerDualToken_CNNfeatures_18cls_layersdecouple44/best_valid_acc/epoch_92.19575995160726.pth',
            'baseline_path': '/mnt/data1/zyh/CHD_1001/CHD_classify/output/checkpoint/CHD_1views_1101_ResnetTransformerDualToken_CNNfeatures/final_baseline_features_4x4_512dim_1101.pt',
            'spatial_resolution': 4
        },
        '5x5': {
            'resume_path': '/mnt/data1/zyh/CHD_1001/CHD_classify/output/checkpoint/CHD_1views_1101_ResnetTransformerDualToken_CNNfeatures_18cls_layersdecouple55/best_valid_acc/epoch_92.95020331726518.pth',
            'baseline_path': '/mnt/data1/zyh/CHD_1001/CHD_classify/output/checkpoint/CHD_1views_1101_ResnetTransformerDualToken_CNNfeatures/final_baseline_features_5x5_512dim_1101.pt', 
            'spatial_resolution': 5
        },
        '6x6': {
            'resume_path': '/mnt/data1/zyh/CHD_1001/CHD_classify/output/checkpoint/CHD_1views_1101_ResnetTransformerDualToken_CNNfeatures_18cls_layersdecouple66/best_valid_acc/epoch_91.37044700524723.pth',
            'baseline_path': '/mnt/data1/zyh/CHD_1001/CHD_classify/output/checkpoint/CHD_1views_1101_ResnetTransformerDualToken_CNNfeatures/final_baseline_features_6x6_512dim_1101.pt', 
            'spatial_resolution': 6
        },
        '7x7': {
            'resume_path': '/mnt/data1/zyh/CHD_1001/CHD_classify/output/checkpoint/CHD_1views_1101_ResnetTransformerDualToken_CNNfeatures_18cls_layersdecouple77/best_valid_acc/epoch_91.09501451063932.pth',
            'baseline_path': '/mnt/data1/zyh/CHD_1001/CHD_classify/output/checkpoint/CHD_1views_1101_ResnetTransformerDualToken_CNNfeatures/final_baseline_features_7x7_512dim_1101.pt', 
            'spatial_resolution': 7
        },
    }

    FINAL_BOX_SIZE = 56 

    print("--- 正在加载所有模型 ---")
    models = {}
    baselines = {} 

    for name, config in MODEL_CONFIGS.items():
        print(f"  -> 加载模型: {name}")
        model = ResnetTransformerDualTokensTemporalSpatialDecouplesize(num_classes=num_classes, 
                                                                        d_model_cnn=d_model_cnn,
                                                                        num_layers=num_layers,
                                                                        dropout=dropout,
                                                                        spatial_resolution=config['spatial_resolution'])
        resume_path = config['resume_path']
        if resume_path != '': 
            resume_model_dict = torch.load(resume_path,map_location='cpu')
            resume_model_filter_dict = {k: v for k, v in resume_model_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
            model.load_state_dict(resume_model_filter_dict,strict=False)
            print('load model from {}'.format(resume_path))
            model.to(cuda_device).eval()
            models[name] = model

        if os.path.exists(config['baseline_path']):
            baselines[name] = torch.load(config['baseline_path'], map_location='cpu').to(cuda_device)
            print(f"     -> 已加载基线: {baselines[name].shape}")
        else:
            raise FileNotFoundError(f"未找到模型 '{name}' 的基线文件: {config['baseline_path']}")

    print("--- 所有模型加载完毕 ---\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n--- 步骤 A: 预扫描所有数据，按病例ID组织 ---")
    
    cases_database = defaultdict(lambda: {"videos": [], "label": -1, "subtype_folder": ""})

    for folder_path_suffix in tqdm(FOLDERS_TO_PROCESS, desc="Pre-scanning Folders"):
        source_folder_path = os.path.join(DATASET_ROOT_DIR, folder_path_suffix)
        if not os.path.isdir(source_folder_path): 
            print(f"找不到文件夹: {source_folder_path}")
            continue

        toplevel_label = get_coarse_label_from_folder_name(folder_path_suffix)
        if toplevel_label == -1: 
            print(f"标签解析错误: {folder_path_suffix}")
            continue
        
        for item_name in os.listdir(source_folder_path):
            item_path = os.path.join(source_folder_path, item_name)
            video_path = None
            if os.path.isfile(item_path) and item_name.lower().endswith(('.mp4', '.avi')):
                video_path = item_path
            elif os.path.isdir(item_path):
                video_files = [f for f in os.listdir(item_path) if f.lower().endswith(('.mp4', '.avi'))]
                if video_files:
                    video_path = os.path.join(item_path, video_files[0])
            
            if video_path:
                video_filename = os.path.basename(video_path)
                case_id = extract_case_id_robust(video_filename, video_path)
                cases_database[case_id]["videos"].append(video_path)
                cases_database[case_id]["label"] = toplevel_label
                cases_database[case_id]["subtype_folder"] = folder_path_suffix
    
    print(f"预扫描完成！共找到 {len(cases_database)} 个独立病例。")
    print("\n--- 步骤 B: 开始按病例处理，生成XAI结果 ---")

    for case_id, case_data in tqdm(cases_database.items(), desc="Processing Cases"):
        is_normal_case = case_data['label'] == 0
        case_output_dir = os.path.join(OUTPUT_DIR, case_data['subtype_folder'], case_id)
        os.makedirs(case_output_dir, exist_ok=True)
        
        for video_path in case_data['videos']:
            try:
                shutil.copy2(video_path, case_output_dir)
            except Exception as e:
                tqdm.write(f"警告：复制原始视频失败 {video_path}: {e}")

        for video_path in case_data['videos']:
            video_filename = os.path.basename(video_path)
            try:
                original_frames = video_loader(video_path)
                if original_frames is None or len(original_frames) < 1: continue
                
                indices = np.arange(len(original_frames))
                if len(original_frames) < clip_size:
                    indices = np.tile(indices, (clip_size + len(original_frames) - 1) // len(original_frames))
                processed_16_frames = original_frames[indices[:clip_size]]
                
                roi_bbox = find_doppler_roi_from_video(processed_16_frames, ROI_AREA_THRESHOLD, ROI_PADDING)
                if roi_bbox is None: continue
                
                frames_for_model_np = np.array([
                    cv2.resize(frame[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]], MODEL_INPUT_SIZE) 
                    for frame in processed_16_frames
                ])
                
                video_tensor_uint8 = torch.from_numpy(frames_for_model_np.copy()).permute(0, 3, 1, 2)
                input_tensor = test_transform(video_tensor_uint8).unsqueeze(0).to(cuda_device)

                doppler_masks_by_size = {}
                for name, model_config in MODEL_CONFIGS.items():
                    res = model_config['spatial_resolution']
                    mask_np = create_string_token_mask(
                        frames_for_model_np, 
                        spatial_resolution=res,
                        vote_threshold=1 
                    )
                    doppler_masks_by_size[name] = torch.from_numpy(mask_np).unsqueeze(0).to(cuda_device)

                # =====================================================================
                # 集成推理与热力图叠加计算 (仅使用原生 Attention)
                # =====================================================================
                fused_heatmap = np.zeros(MODEL_INPUT_SIZE, dtype=np.float32)
                individual_heatmaps_for_vis = {}

                all_logits, raw_strings_dict, key_patch_indices = [], {}, {}
                key_frame_votes = []

                for name, model in models.items():
                    with torch.no_grad():
                        outputs = model(input_tensor, None, doppler_masks_by_size[name])
                        
                        all_logits.append(outputs["predicted_scores"])
                        raw_strings_dict[name] = outputs["raw_strings_for_loss"]

                # 投票决定最终的预测类别
                avg_logits = torch.stack(all_logits).mean(dim=0)
                final_pred_label_idx = torch.argmax(avg_logits, dim=1).item()

                if final_pred_label_idx <= 7:
                    normal_id = final_pred_label_idx
                else:
                    normal_id = POSITIVE_TO_NEGATIVE_PAIR_MAP.get(final_pred_label_idx)
                
                if normal_id is None:
                    print(f"警告: 预测标签 {final_pred_label_idx} 没有定义。跳过此病例。")
                    continue
                
                # 计算时序残差找关键帧及计算各模型热力图
                for name, model in models.items():
                    with torch.no_grad():
                        outputs = model(input_tensor, None, doppler_masks_by_size[name])
                        pred_label_idx_single = torch.argmax(outputs["predicted_scores"], dim=1).item()
                        
                        # 找关键帧
                        attention_vector = outputs["all_layer_attention_scores"][-1][0, pred_label_idx_single, :]
                        key_patch_idx = torch.argmax(attention_vector).item()
                        
                        key_patch_raw_string = raw_strings_dict[name][0, key_patch_idx, :, :]
                        key_patch_baseline = baselines[name][normal_id, key_patch_idx, :]
                        residuals = torch.linalg.norm(key_patch_raw_string - key_patch_baseline.unsqueeze(0), dim=-1)
                        key_frame_votes.append(torch.argmax(residuals).item())

                        # 提取当前模型的 Attention 热力图用于融合
                        res_spatial = model.spatial_size
                        heatmap_attn_small = attention_vector.cpu().numpy().reshape(res_spatial, res_spatial)
                        heatmap_attn_224 = cv2.resize(heatmap_attn_small, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
                        
                        # 归一化并叠加
                        if heatmap_attn_224.max() > heatmap_attn_224.min():
                            heatmap_attn_norm = (heatmap_attn_224 - heatmap_attn_224.min()) / (heatmap_attn_224.max() - heatmap_attn_224.min())
                            fused_heatmap += heatmap_attn_norm
                            individual_heatmaps_for_vis[name] = heatmap_attn_norm
                        else:
                            individual_heatmaps_for_vis[name] = np.zeros(MODEL_INPUT_SIZE, dtype=np.float32)

                # 投票决定最终的关键帧
                final_pred_key_frame = Counter(key_frame_votes).most_common(1)[0][0] if key_frame_votes else -1
                
                if np.sum(fused_heatmap) == 0:
                    print("Fusionmap is None")
                    continue
                    
                peak_y, peak_x = np.unravel_index(np.argmax(fused_heatmap), fused_heatmap.shape)
                pred_box_ensemble = get_center_and_draw_box((peak_x, peak_y), FINAL_BOX_SIZE)
                
                # 坐标系转换
                pred_box_in_original = transform_box_from_model_to_original_space(pred_box_ensemble, roi_bbox)

                # =====================================================================
                # 构建并保存JSON结果
                # =====================================================================
                prediction_info = {
                    "roi_crop_box_original_space": roi_bbox,
                    "predicted_key_frame_index": final_pred_key_frame,
                    "predicted_bounding_box_original_space": pred_box_in_original,
                    "fused_heatmap_224x224": fused_heatmap
                }

                json_filename = f"{os.path.splitext(video_filename)[0]}.json"
                json_save_path = os.path.join(case_output_dir, json_filename)
                
                with open(json_save_path, 'w', encoding='utf-8') as f:
                    json.dump(prediction_info, f, indent=4, cls=NpEncoder)

            except Exception as e:
                tqdm.write(f"\n处理视频 {video_filename} 时出错: {e}")
                import traceback; traceback.print_exc()
 
    print(f"\n--- 所有病例处理完成! 结果保存在: {OUTPUT_DIR} ---")