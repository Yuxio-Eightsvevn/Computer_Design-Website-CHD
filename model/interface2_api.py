"""
心脏超声诊断接口 - API模式

功能:
    - 提供单例模式的推理引擎
    - 支持单病例和多病例诊断
    - 适合集成到其他系统

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

# 配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_INPUT_SIZE = (224, 224)
FINAL_BOX_SIZE = 56
ROI_AREA_THRESHOLD = 150
ROI_PADDING = 20
clip_size = 16
CLASS_NAMES = ['Normal', 'VSD', 'ASD', 'PDA']

# 模型配置
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

# 从原函数导入
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
from torchvision.transforms import v2


# 单例引擎
class HeartInferenceEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        self.single_models = {}
        self.baselines = {}
        self.multi_model = None
        self._load_resources()
        self._initialized = True
        print(f"HeartInferenceEngine 初始化完成 (device: {DEVICE})")
    
    def _load_resources(self):
        for res in [4, 5, 6, 7]:
            m = ResnetTransformerDualTokensTemporalSpatialDecouplesize(
                num_classes=18, d_model_cnn=512, num_layers=6, spatial_resolution=res
            ).to(DEVICE).eval()
            
            p = os.path.join(CURRENT_DIR, 'singleview', f'spatial_size_{res}.pth')
            if os.path.exists(p):
                ckpt = torch.load(p, map_location=DEVICE)
                if 'pos_encoder_temporal.pe' in ckpt:
                    m_pe = m.state_dict()['pos_encoder_temporal.pe']
                    new_pe = m_pe.clone()
                    l = min(ckpt['pos_encoder_temporal.pe'].shape[0], m_pe.shape[0])
                    new_pe[:l, :, :] = ckpt['pos_encoder_temporal.pe'][:l, :, :]
                    ckpt['pos_encoder_temporal.pe'] = new_pe
                m.load_state_dict(ckpt, strict=False)
                self.single_models[f"{res}x{res}"] = m
            
            bp = os.path.join(CURRENT_DIR, 'Baseline_4size', f'final_baseline_features_{res}x{res}_512dim_1101.pt')
            if os.path.exists(bp):
                self.baselines[f"{res}x{res}"] = torch.load(bp, map_location=DEVICE)
        
        self.multi_model = MultiViewDualTokensFusionSize(spatial_size=5).to(DEVICE).eval()
        mp = os.path.join(CURRENT_DIR, 'multiviews', 'epoch_97.98295452219057.pth')
        if os.path.exists(mp):
            self.multi_model.load_state_dict(torch.load(mp, map_location=DEVICE), strict=False)
    
    @torch.no_grad()
    def infer_single(self, prep):
        all_outs = {}
        for name, model in self.single_models.items():
            mask = torch.from_numpy(create_string_token_mask(prep['model_frames'], model.spatial_size)).unsqueeze(0).to(DEVICE)
            all_outs[name] = model(prep['tensor'], None, mask)
        
        logits = torch.stack([v['predicted_scores'] for v in all_outs.values()]).mean(0)
        pred_label = torch.argmax(logits, dim=1).item()
        
        norm_id = POSITIVE_TO_NEGATIVE_PAIR_MAP.get(pred_label, pred_label if pred_label <= 7 else 0)
        
        fused_h = np.zeros(MODEL_INPUT_SIZE, dtype=np.float32)
        key_frame_votes = []
        raw_strings_dict = {}
        
        for name, model in self.single_models.items():
            raw_strings_dict[name] = all_outs[name]["raw_strings_for_loss"]
        
        for name, model in self.single_models.items():
            out = all_outs[name]
            
            # 使用每个模型自己的预测结果（原脚本逻辑）
            pred_label_single = torch.argmax(out["predicted_scores"], dim=1).item()
            attn = out["all_layer_attention_scores"][-1][0, pred_label_single, :]
            p_idx = torch.argmax(attn).item()
            
            key_patch_raw_string = raw_strings_dict[name][0, p_idx, :, :]
            key_patch_baseline = self.baselines[name][norm_id, p_idx, :]
            residuals = torch.linalg.norm(key_patch_raw_string - key_patch_baseline.unsqueeze(0), dim=-1)
            key_frame_votes.append(torch.argmax(residuals).item())
            
            res_spatial = model.spatial_size
            heatmap_attn_small = attn.cpu().numpy().reshape(res_spatial, res_spatial)
            heatmap_attn_224 = cv2.resize(heatmap_attn_small, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
            
            if heatmap_attn_224.max() > heatmap_attn_224.min():
                heatmap_attn_norm = (heatmap_attn_224 - heatmap_attn_224.min()) / (heatmap_attn_224.max() - heatmap_attn_224.min())
                fused_h += heatmap_attn_norm
        
        final_kf = Counter(key_frame_votes).most_common(1)[0][0]
        
        py, px = np.unravel_index(np.argmax(fused_h), fused_h.shape)
        pred_box_ensemble = get_center_and_draw_box((px, py), FINAL_BOX_SIZE)
        bbox_orig = transform_box_from_model_to_original_space(pred_box_ensemble, prep['roi'])
        
        is_normal = pred_label <= 7
        return {"pred": "Normal" if is_normal else "Abnormal", "kf": final_kf, "heatmap": fused_h, "bbox": bbox_orig}
    
    @torch.no_grad()
    def infer_multi_view(self, case_preps):
        """
        多视角推理 - 完全复用原脚本逻辑（Confidence_scores.py）
        支持单视频或多视频输入
        """
        if not case_preps:
            return {'Normal': 0.0, 'VSD': 0.0, 'ASD': 0.0, 'PDA': 0.0}
        
        MAX_VIEWS = 5  # 与原模型训练时的max_views一致
        
        # 重新处理帧（与原脚本逻辑一致）
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        view_tensors = []
        view_masks = []
        
        # 处理有效视角
        for prep in case_preps[:MAX_VIEWS]:
            raw_frames = prep['raw_frames']
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
        
        # Padding到MAX_VIEWS（原脚本逻辑）
        while len(view_tensors) < MAX_VIEWS:
            view_tensors.append(torch.zeros_like(view_tensors[0]))
            view_masks.append(torch.ones_like(view_masks[0]))
        
        multi_view_video = torch.stack(view_tensors[:MAX_VIEWS]).unsqueeze(0).to(DEVICE)
        multi_view_mask = torch.stack(view_masks[:MAX_VIEWS]).unsqueeze(0).to(DEVICE)
        # 使用有效视角数量（原脚本逻辑）
        num_views_tensor = torch.tensor([num_valid_views], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            logits = self.multi_model(multi_view_video, num_views_tensor, multi_view_mask)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        return {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(4)}


def process_single_case(case_input_dir, output_dir, case_id=None, save_videos=True):
    """处理单个case - 完全复用原函数"""
    engine = HeartInferenceEngine()
    
    if case_id is None:
        case_id = os.path.basename(os.path.abspath(case_input_dir))
    
    case_output_dir = os.path.join(output_dir, case_id, "output_data")
    os.makedirs(case_output_dir, exist_ok=True)
    
    video_files = sorted([f for f in os.listdir(case_input_dir) if f.lower().endswith(('.mp4', '.avi'))])
    
    case_preps = []
    video_results = []
    
    for video_file in video_files:
        video_path = os.path.join(case_input_dir, video_file)
        video_base = os.path.splitext(video_file)[0]
        
        try:
            raw_frames = video_loader(video_path)
            if raw_frames is None:
                video_results.append({'video_name': video_file, 'status': 'error', 'message': '视频读取失败'})
                continue
            
            indices = np.arange(len(raw_frames))
            if len(raw_frames) < clip_size:
                indices = np.tile(indices, (clip_size + len(raw_frames) - 1) // len(raw_frames))
            processed_frames = raw_frames[indices[:clip_size]]
            
            roi = find_doppler_roi_from_video(processed_frames, ROI_AREA_THRESHOLD, ROI_PADDING)
            if roi is None:
                video_results.append({'video_name': video_file, 'status': 'error', 'message': 'ROI检测失败'})
                continue
            
            frames_for_model = np.array([
                cv2.resize(f[roi[1]:roi[3], roi[0]:roi[2]], MODEL_INPUT_SIZE)
                for f in processed_frames
            ])
            
            tensor = engine.transform(torch.from_numpy(frames_for_model.copy()).permute(0, 3, 1, 2)).unsqueeze(0).to(DEVICE)
            
            prep = {'tensor': tensor, 'roi': roi, 'model_frames': frames_for_model, 'raw_frames': raw_frames}
            result = engine.infer_single(prep)
            
            # 合并prep和result，确保save_visuals有所需的所有字段
            full_result = {**prep, **result}
            case_preps.append(full_result)
            
            if save_videos:
                save_visuals(video_path, full_result, os.path.join(output_dir, case_id, "output_videos"), video_base)
            
            json_data = {
                "roi_crop_box_original_space": list(roi),
                "predicted_key_frame_index": int(result['kf']),
                "predicted_bounding_box_original_space": result['bbox'],
                "fused_heatmap_224x224": result['heatmap'].tolist()
            }
            
            json_path = os.path.join(case_output_dir, f"{video_base}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, cls=NpEncoder)
            
            shutil.copy2(video_path, case_output_dir)
            
            video_results.append({'video_name': video_file, 'status': 'success', 'json_path': json_path})
        
        except Exception as e:
            video_results.append({'video_name': video_file, 'status': 'error', 'message': str(e)})
    
    multi_view_result = None
    if case_preps:
        multi_view_result = engine.infer_multi_view(case_preps)
        
        with open(os.path.join(case_output_dir, 'confidence_scores.json'), 'w', encoding='utf-8') as f:
            json.dump(multi_view_result, f, indent=2)
        
        zip_path = os.path.join(output_dir, case_id, 'output_data.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(case_output_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), case_output_dir))
    
    return {
        'case_id': case_id,
        'video_results': video_results,
        'multi_view_result': multi_view_result,
        'success_count': sum(1 for r in video_results if r['status'] == 'success'),
        'error_count': sum(1 for r in video_results if r['status'] == 'error'),
        'output_dir': os.path.join(output_dir, case_id)
    }


def save_visuals(video_path, result, output_dir, video_name):
    """保存可视化结果 - 增强版可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存原视频到根目录
    shutil.copy2(video_path, os.path.join(output_dir, f"{video_name}_original.mp4"))
    
    # 2. 保存帧到_original/frames/目录
    original_dir = os.path.join(output_dir, f"{video_name}_original")
    frames_dir = os.path.join(original_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    from PIL import Image
    raw_frames = result.get('raw_frames', [])
    for idx, frame in enumerate(raw_frames[:16], 1):
        frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(frame_path)
    
    # 3. 保存热力图和边界框视频 - 增强版可视化
    if 'frames' not in result or 'heatmap' not in result or 'roi' not in result or 'bbox' not in result:
        return
    
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


def initialize():
    return HeartInferenceEngine()
