"""
心脏超声诊断接口

功能:
    - 单视角推理: 生成热力图(Heatmap)、边界框(BBox)、关键帧
    - 多视角推理: 生成诊断置信度 {Normal, VSD, ASD, PDA}

说明:
    完全复用Codes目录下的原始函数，不修改任何内部逻辑
    仅修改输出路径以适配预期目录结构
"""

import os
import sys
import json
import yaml
import torch
import cv2
import numpy as np
import shutil
import zipfile
from tqdm import tqdm
from collections import Counter

# 设置路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'Codes', 'Nets'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'Codes', 'main_codes'))

# 设置随机种子 - 与原脚本一致
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)
torch.cuda.manual_seed_all(42)

# 配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_INPUT_SIZE = (224, 224)
FINAL_BOX_SIZE = 56
ROI_AREA_THRESHOLD = 150
ROI_PADDING = 20
clip_size = 16
CLASS_NAMES = ['Normal', 'VSD', 'ASD', 'PDA']

# 模型配置 - 使用绝对路径
MODEL_CONFIGS = {
    '4x4': {
        'resume_path': os.path.join(CURRENT_DIR, 'singleview', 'spatial_size_4.pth'),
        'baseline_path': os.path.join(CURRENT_DIR, 'Baseline_4size', 'final_baseline_features_4x4_512dim_1101.pt'),
        'spatial_resolution': 4
    },
    '5x5': {
        'resume_path': os.path.join(CURRENT_DIR, 'singleview', 'spatial_size_5.pth'),
        'baseline_path': os.path.join(CURRENT_DIR, 'Baseline_4size', 'final_baseline_features_5x5_512dim_1101.pt'),
        'spatial_resolution': 5
    },
    '6x6': {
        'resume_path': os.path.join(CURRENT_DIR, 'singleview', 'spatial_size_6.pth'),
        'baseline_path': os.path.join(CURRENT_DIR, 'Baseline_4size', 'final_baseline_features_6x6_512dim_1101.pt'),
        'spatial_resolution': 6
    },
    '7x7': {
        'resume_path': os.path.join(CURRENT_DIR, 'singleview', 'spatial_size_7.pth'),
        'baseline_path': os.path.join(CURRENT_DIR, 'Baseline_4size', 'final_baseline_features_7x7_512dim_1101.pt'),
        'spatial_resolution': 7
    },
}

# 加载配置
with open(os.path.join(CURRENT_DIR, 'Codes', 'config', 'test_config_ResnetrTransformerDualToken_1107_18cls_layersdecouple44.yaml'), 'r', encoding='utf-8') as f:
    sv_config = yaml.load(f, Loader=yaml.FullLoader)

# 从原函数导入（完全复用，不修改内部逻辑）
from Codes.Nets.dual_tokens_net import ResnetTransformerDualTokensTemporalSpatialDecouplesize
from Codes.Nets.Multi_Views_dual_tokens_net import MultiViewDualTokensFusionSize
from Codes.main_codes.original_ref import (
    video_loader, 
    find_doppler_roi_from_video, 
    create_string_token_mask,
    POSITIVE_TO_NEGATIVE_PAIR_MAP,
    transform_box_from_model_to_original_space,
    get_center_and_draw_box,
    NpEncoder
)

# 导入torchvision transforms
from torchvision.transforms import v2


def load_single_models():
    """加载单视角模型 - 完全复用原脚本逻辑"""
    models = {}
    baselines = {}
    
    for name, config in MODEL_CONFIGS.items():
        model = ResnetTransformerDualTokensTemporalSpatialDecouplesize(
            num_classes=sv_config['num_classes'],
            d_model_cnn=sv_config['d_model_cnn'],
            num_layers=sv_config['num_layers'],
            dropout=sv_config['dropout'],
            spatial_resolution=config['spatial_resolution']
        )
        
        resume_path = config['resume_path']
        if resume_path and os.path.exists(resume_path):
            resume_model_dict = torch.load(resume_path, map_location='cpu')
            resume_model_filter_dict = {k: v for k, v in resume_model_dict.items() 
                                       if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
            model.load_state_dict(resume_model_filter_dict, strict=False)
            model.to(DEVICE).eval()
            models[name] = model
        
        if os.path.exists(config['baseline_path']):
            baselines[name] = torch.load(config['baseline_path'], map_location='cpu').to(DEVICE)
    
    return models, baselines


def load_multi_model():
    """加载多视角模型 - 完全复用原脚本逻辑"""
    model = MultiViewDualTokensFusionSize(
        spatial_size=5
    ).to(DEVICE).eval()
    
    mv_path = os.path.join(CURRENT_DIR, 'multiviews', 'epoch_97.98295452219057.pth')
    if os.path.exists(mv_path):
        model.load_state_dict(torch.load(mv_path, map_location=DEVICE), strict=False)
    
    return model


def process_single_video(video_path, models, baselines, transform):
    """
    处理单个视频 - 完全复用原脚本逻辑
    生成：热图、关键帧、边界框
    """
    video_filename = os.path.basename(video_path)
    
    # 1. 加载视频帧
    original_frames = video_loader(video_path)
    if original_frames is None or len(original_frames) < 1:
        return None
    
    # 2. 处理帧到16帧
    indices = np.arange(len(original_frames))
    if len(original_frames) < clip_size:
        indices = np.tile(indices, (clip_size + len(original_frames) - 1) // len(original_frames))
    processed_16_frames = original_frames[indices[:clip_size]]
    
    # 3. 找ROI
    roi_bbox = find_doppler_roi_from_video(processed_16_frames, ROI_AREA_THRESHOLD, ROI_PADDING)
    if roi_bbox is None:
        return None
    
    # 4. 裁剪并resize
    frames_for_model_np = np.array([
        cv2.resize(frame[roi_bbox[1]:roi_bbox[3], roi_bbox[0]:roi_bbox[2]], MODEL_INPUT_SIZE)
        for frame in processed_16_frames
    ])
    
    # 5. 转换为tensor
    video_tensor_uint8 = torch.from_numpy(frames_for_model_np.copy()).permute(0, 3, 1, 2)
    input_tensor = transform(video_tensor_uint8).unsqueeze(0).to(DEVICE)
    
    # 6. 创建doppler mask
    doppler_masks = {}
    for name, model in models.items():
        mask_np = create_string_token_mask(frames_for_model_np, model.spatial_size, vote_threshold=1)
        doppler_masks[name] = torch.from_numpy(mask_np).unsqueeze(0).to(DEVICE)
    
    # 7. 推理 - 完全复用原脚本逻辑
    fused_heatmap = np.zeros(MODEL_INPUT_SIZE, dtype=np.float32)
    key_frame_votes = []
    all_logits = []
    raw_strings_dict = {}
    
    # 第一次循环：获取logits和raw_strings
    for name, model in models.items():
        with torch.no_grad():
            outputs = model(input_tensor, None, doppler_masks[name])
            all_logits.append(outputs["predicted_scores"])
            raw_strings_dict[name] = outputs["raw_strings_for_loss"]
    
    # 投票决定预测类别
    avg_logits = torch.stack(all_logits).mean(dim=0)
    final_pred_label_idx = torch.argmax(avg_logits, dim=1).item()
    
    # 获取normal_id
    if final_pred_label_idx <= 7:
        normal_id = final_pred_label_idx
    else:
        normal_id = POSITIVE_TO_NEGATIVE_PAIR_MAP.get(final_pred_label_idx)
    
    if normal_id is None:
        return None
    
    # 第二次循环：关键帧和热力图 - 完全复用原脚本逻辑
    for name, model in models.items():
        with torch.no_grad():
            outputs = model(input_tensor, None, doppler_masks[name])
            
            # 使用每个模型自己的预测结果（原脚本逻辑）
            pred_label_idx_single = torch.argmax(outputs["predicted_scores"], dim=1).item()
            attn = outputs["all_layer_attention_scores"][-1][0, pred_label_idx_single, :]
            p_idx = torch.argmax(attn).item()
            
            # 关键帧计算 - 完全复用原脚本逻辑
            key_patch_raw_string = raw_strings_dict[name][0, p_idx, :, :]
            key_patch_baseline = baselines[name][normal_id, p_idx, :]
            residuals = torch.linalg.norm(key_patch_raw_string - key_patch_baseline.unsqueeze(0), dim=-1)
            key_frame_votes.append(torch.argmax(residuals).item())
            
            # 热力图计算 - 完全复用原脚本逻辑
            res_spatial = model.spatial_size
            heatmap_attn_small = attn.cpu().numpy().reshape(res_spatial, res_spatial)
            heatmap_attn_224 = cv2.resize(heatmap_attn_small, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            
            if heatmap_attn_224.max() > heatmap_attn_224.min():
                heatmap_attn_norm = (heatmap_attn_224 - heatmap_attn_224.min()) / (heatmap_attn_224.max() - heatmap_attn_224.min())
                fused_heatmap += heatmap_attn_norm
    
    # 最终关键帧
    final_pred_key_frame = Counter(key_frame_votes).most_common(1)[0][0]
    
    if np.sum(fused_heatmap) == 0:
        return None
    
    # 边界框 - 完全复用原脚本逻辑
    peak_y, peak_x = np.unravel_index(np.argmax(fused_heatmap), fused_heatmap.shape)
    pred_box_ensemble = get_center_and_draw_box((peak_x, peak_y), FINAL_BOX_SIZE)
    pred_box_in_original = transform_box_from_model_to_original_space(pred_box_ensemble, roi_bbox)
    
    # 判断是否为Normal
    is_normal = final_pred_label_idx <= 7
    
    return {
        "roi": roi_bbox,
        "kf": final_pred_key_frame,
        "heatmap": fused_heatmap,
        "bbox": pred_box_in_original,
        "frames": original_frames,
        "frames_for_model": frames_for_model_np,
        "pred": "Normal" if is_normal else "Abnormal"
    }


def multi_view_inference(case_preps, multi_model):
    """
    多视角推理 - 完全复用原脚本逻辑（Confidence_scores.py）
    生成置信度分数
    支持单视频或多视频输入
    """
    MAX_VIEWS = 5  # 与原模型训练时的max_views一致
    
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    view_tensors = []
    view_masks = []
    
    # 处理有效视角
    for prep in case_preps[:MAX_VIEWS]:
        raw_frames = prep['frames']
        indices = np.arange(len(raw_frames))
        if len(raw_frames) < clip_size:
            indices = np.tile(indices, (clip_size + len(raw_frames) - 1) // len(raw_frames))
        processed_frames = raw_frames[indices[:clip_size]]
        
        roi = prep['roi']
        frames_for_model = np.array([
            cv2.resize(f[roi[1]:roi[3], roi[0]:roi[2]], MODEL_INPUT_SIZE)
            for f in processed_frames
        ])
        
        tensor = transform(torch.from_numpy(frames_for_model.copy()).permute(0, 3, 1, 2)).unsqueeze(0).to(DEVICE)
        view_tensors.append(tensor.squeeze(0))
        
        mask_np = create_string_token_mask(frames_for_model, spatial_resolution=5)
        view_masks.append(torch.from_numpy(mask_np))
    
    # 记录有效视角数量（原脚本逻辑）
    num_valid_views = len(view_tensors)
    
    # Padding到MAX_VIEWS（原脚本逻辑：使用零tensor和True mask）
    while len(view_tensors) < MAX_VIEWS:
        view_tensors.append(torch.zeros_like(view_tensors[0]))
        view_masks.append(torch.ones_like(view_masks[0]))
    
    # 多视角推理
    multi_view_video = torch.stack(view_tensors[:MAX_VIEWS]).unsqueeze(0).to(DEVICE)
    multi_view_mask = torch.stack(view_masks[:MAX_VIEWS]).unsqueeze(0).to(DEVICE)
    # 使用有效视角数量（原脚本逻辑）
    num_views_tensor = torch.tensor([num_valid_views], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        logits = multi_model(multi_view_video, num_views_tensor, multi_view_mask)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    return {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(4)}


def save_visuals(video_path, result, output_dir, video_name):
    """保存可视化结果 - 仅修改路径"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存原视频到根目录
    shutil.copy2(video_path, os.path.join(output_dir, f"{video_name}_original.mp4"))
    
    # 2. 保存帧到_original/frames/目录
    original_dir = os.path.join(output_dir, f"{video_name}_original")
    frames_dir = os.path.join(original_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    from PIL import Image
    for idx, frame in enumerate(result['frames'][:16], 1):
        frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(frame_rgb).save(frame_path)
    
    # 保存热力图和边界框视频 - 增强版可视化
    h_orig, w_orig = result['frames'][0].shape[:2]
    rx1, ry1, rx2, ry2 = result['roi']
    rw, rh = rx2-rx1, ry2-ry1
    
    # 判断是否为Normal
    is_normal = result.get('pred') == 'Normal'
    
    # 热力图：仅在ROI区域应用，非ROI区域保持原图
    h_roi = cv2.resize(result['heatmap'], (rw, rh), interpolation=cv2.INTER_LINEAR)
    if h_roi.max() > h_roi.min():
        h_roi_norm = (h_roi - h_roi.min()) / (h_roi.max() - h_roi.min())
    else:
        h_roi_norm = np.zeros_like(h_roi)
    h_roi_bgr = cv2.applyColorMap(np.uint8(255 * h_roi_norm), cv2.COLORMAP_JET)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw_box = cv2.VideoWriter(os.path.join(output_dir, f"{video_name}_bbox.mp4"), fourcc, 30.0, (w_orig, h_orig))
    vw_heat = cv2.VideoWriter(os.path.join(output_dir, f"{video_name}_heatmap.mp4"), fourcc, 30.0, (w_orig, h_orig))
    
    bx = result['bbox']
    kf = result['kf']
    for i, frame in enumerate(result['frames']):
        # 边界框可视化
        f_box = frame.copy()
        # 绘制ROI白色细框（所有帧）
        cv2.rectangle(f_box, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)
        # 非Normal时叠加绿色边界框和AI Pred标注
        if not is_normal:
            cv2.rectangle(f_box, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
            cv2.putText(f_box, "AI Pred", (bx[0], bx[3] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if i == kf:
                cv2.putText(f_box, "KEY FRAME", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        vw_box.write(f_box)
        
        # 热力图可视化 - 仅在ROI区域叠加
        f_heat = frame.copy()
        roi_frame = f_heat[ry1:ry2, rx1:rx2]
        blended_roi = cv2.addWeighted(roi_frame, 0.6, h_roi_bgr, 0.4, 0)
        f_heat[ry1:ry2, rx1:rx2] = blended_roi
        vw_heat.write(f_heat)
    
    vw_box.release()
    vw_heat.release()


def zip_directory(directory_path, zip_path):
    """压缩目录"""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory_path)
                zipf.write(file_path, arcname)


def diagnose(target_dir: str, output_dir: str, save_videos: bool = True) -> dict:
    """
    处理诊断请求
    入口函数 - 完全复用原函数，仅修改路径
    """
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # 加载模型
    models, baselines = load_single_models()
    multi_model = load_multi_model()
    
    request_name = os.path.basename(os.path.abspath(target_dir))
    output_request_dir = os.path.join(output_dir, request_name)
    
    # 获取所有case
    case_dirs = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
    
    results = []
    
    for case_id in tqdm(case_dirs, desc="处理病例"):
        case_input_dir = os.path.join(target_dir, case_id)
        
        # 创建输出目录
        case_output_dir = os.path.join(output_request_dir, case_id, "output_data")
        os.makedirs(case_output_dir, exist_ok=True)
        
        # 获取视频文件
        video_files = sorted([f for f in os.listdir(case_input_dir) if f.lower().endswith(('.mp4', '.avi'))])
        
        case_preps = []
        video_results = []
        
        for video_file in video_files:
            video_path = os.path.join(case_input_dir, video_file)
            video_base = os.path.splitext(video_file)[0]
            
            try:
                # 单视角推理 - 调用原函数
                result = process_single_video(video_path, models, baselines, transform)
                
                if result is not None:
                    case_preps.append(result)
                    
                    # 保存JSON - 仅修改路径
                    json_data = {
                        "roi_crop_box_original_space": list(result['roi']),
                        "predicted_key_frame_index": int(result['kf']),
                        "predicted_bounding_box_original_space": result['bbox'],
                        "fused_heatmap_224x224": result['heatmap'].tolist()
                    }
                    
                    json_path = os.path.join(case_output_dir, f"{video_base}.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, cls=NpEncoder)
                    
                    # 复制原视频 - 仅修改路径
                    shutil.copy2(video_path, case_output_dir)
                    
                    # 保存可视化 - 仅修改路径
                    if save_videos:
                        save_visuals(video_path, result, os.path.join(output_request_dir, case_id, "output_videos"), video_base)
                    
                    video_results.append({
                        'video_name': video_file,
                        'status': 'success',
                        'json_path': json_path
                    })
                else:
                    video_results.append({
                        'video_name': video_file,
                        'status': 'error',
                        'message': '处理失败'
                    })
            except Exception as e:
                video_results.append({
                    'video_name': video_file,
                    'status': 'error',
                    'message': str(e)
                })
        
        # 多视角推理 - 调用原函数（支持单视频或多视频）
        multi_view_result = None
        if case_preps and len(case_preps) >= 1:
            multi_view_result = multi_view_inference(case_preps, multi_model)
            
            # 保存置信度 - 仅修改路径
            with open(os.path.join(case_output_dir, 'confidence_scores.json'), 'w', encoding='utf-8') as f:
                json.dump(multi_view_result, f, indent=2)
        
        # 压缩 - 仅修改路径
        if case_preps:
            zip_path = os.path.join(output_request_dir, case_id, 'output_data.zip')
            zip_directory(case_output_dir, zip_path)
        
        results.append({
            'case_id': case_id,
            'video_results': video_results,
            'multi_view_result': multi_view_result,
            'success_count': sum(1 for r in video_results if r['status'] == 'success'),
            'error_count': sum(1 for r in video_results if r['status'] == 'error'),
            'output_dir': os.path.join(output_request_dir, case_id)
        })
    
    return {
        "request_name": request_name,
        "target_dir": target_dir,
        "output_dir": output_request_dir,
        "total_cases": len(case_dirs),
        "success_cases": sum(1 for r in results if r['success_count'] > 0),
        "failed_cases": sum(1 for r in results if r['success_count'] == 0),
        "results": results
    }


def process_single_case(case_input_dir: str, output_dir: str, case_id: str = None, save_videos: bool = True) -> dict:
    """处理单个case"""
    return diagnose(case_input_dir, output_dir, save_videos)


def initialize():
    """初始化"""
    pass
