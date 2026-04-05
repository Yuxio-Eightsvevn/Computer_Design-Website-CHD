import os
import json
import cv2
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')  # 或者 'Agg' (Agg是不显示图形，只用于保存图片)
import matplotlib.pyplot as plt

# 1. 指向您已生成的XAI结果的根目录
XAI_RESULTS_ROOT = r'D:\实验室\workTable\26_3_17_web\worktable\models\input_path\request2'

# 2. 指定保存可视化验证图片的根目录
VISUALIZATION_OUTPUT_ROOT = r'D:\实验室\workTable\26_3_17_web\worktable\models\output_path'


def video_loader(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        return np.array(frames) if frames else None
    except: return None

def create_visualization_pair(
    output_dir,
    video_base_name,
    original_frame,      # 原始视频帧 (H, W, C)
    heatmap_224,         # (224, 224) float32
    roi_bbox,            # (x1, y1, x2, y2)
    pred_box,
    case_info_str,
    generate_bbox=True # <--- 【新增参数】控制是否生成BBox图
):
    """
    【全覆盖版】生成热图叠加图 & 预测框图
    特点：热图铺满整个ROI，无阈值截断。
    """
    h_orig, w_orig, _ = original_frame.shape
    heatmap_overlay_img = original_frame.copy()
    
    if heatmap_224 is not None and roi_bbox is not None:
        rx1, ry1, rx2, ry2 = roi_bbox
        roi_w, roi_h = rx2 - rx1, ry2 - ry1
        
        if roi_w > 0 and roi_h > 0 and rx1 >= 0 and ry1 >= 0:
            try:
                # 1. Resize 回 ROI 大小 (使用平滑插值)
                heatmap_roi = cv2.resize(heatmap_224, (roi_w, roi_h), interpolation=cv2.INTER_CUBIC)
                
                # 2. 高斯模糊 (消除锯齿，保持丝滑)
                blur_ksize = int(roi_w * 0.1) | 1 
                heatmap_roi = cv2.GaussianBlur(heatmap_roi, (blur_ksize, blur_ksize), 0)
                
                # 3. 归一化 (确保范围在 0-1)
                if heatmap_roi.max() > heatmap_roi.min():
                    heatmap_roi = (heatmap_roi - heatmap_roi.min()) / (heatmap_roi.max() - heatmap_roi.min())
                else:
                    heatmap_roi[:] = 0
                
                # 4. 生成彩色热图 (Blue=Low, Red=High)
                heatmap_uint8 = np.uint8(255 * heatmap_roi)
                heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                
                # 5. 【核心修改】全区域融合，不设阈值
                # 提取原图ROI
                roi_original = original_frame[ry1:ry2, rx1:rx2]
                
                # 直接将彩色热图与原图按比例混合
                # 参数建议：原图 0.6 + 热图 0.4。这样既能看清解剖结构，又能看清颜色
                blended_roi = cv2.addWeighted(roi_original, 0.6, heatmap_colored, 0.4, 0)
                
                # 6. 贴回原图
                heatmap_overlay_img[ry1:ry2, rx1:rx2] = blended_roi
                
                # (可选) 绘制 ROI 框，标示模型视野边界
                cv2.rectangle(heatmap_overlay_img, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

            except Exception as e:
                print(f"热图渲染出错: {e}")

    # 保存热图
    fig1, ax1 = plt.subplots(figsize=(10, 10 * h_orig / w_orig))
    ax1.imshow(cv2.cvtColor(heatmap_overlay_img, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Heatmap Overlay (Full ROI)\n{case_info_str}", fontsize=12)
    ax1.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{video_base_name}_heatmap.png"), dpi=150)
    plt.close(fig1)

    # --- 2. 生成预测框图 (BBox) - 【受控生成】 ---
    if generate_bbox:
        bbox_img = original_frame.copy()
        if pred_box is not None and len(pred_box) == 4:
            bx1, by1, bx2, by2 = [int(p) for p in pred_box]
            cv2.rectangle(bbox_img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
            cv2.putText(bbox_img, "AI Pred", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        fig2, ax2 = plt.subplots(figsize=(10, 10 * h_orig / w_orig))
        ax2.imshow(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"AI Detection Box\n{case_info_str}", fontsize=12)
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_base_name}_bbox.png"), dpi=150)
        plt.close(fig2)


# ==========================================================================
#                                 主程序
# ==========================================================================
if __name__ == '__main__':
    print("--- 开始可视化验证脚本 (按 病情 -> 病例ID -> 视频/JSON 结构) ---")
    
    if not os.path.exists(XAI_RESULTS_ROOT):
        print(f"错误: 找不到输入目录 {XAI_RESULTS_ROOT}")
        exit()

    # 1. 遍历第一层：病情文件夹 (e.g., ASD_Abnormal, VSD_Normal)
    for subtype_folder in tqdm(os.listdir(XAI_RESULTS_ROOT), desc="Processing Subtypes"):
        subtype_path = os.path.join(XAI_RESULTS_ROOT, subtype_folder)
        if not os.path.isdir(subtype_path): continue

        # 【核心新增】判断当前是否为 Normal 组
        is_abnormal_group = 'Abnormal' in subtype_folder

        # 2. 遍历第二层：病例ID文件夹 (e.g., 20210903_2)
        for case_id_folder in tqdm(os.listdir(subtype_path), desc=f"Cases in {subtype_folder}", leave=False):
            case_path = os.path.join(subtype_path, case_id_folder)
            if not os.path.isdir(case_path): continue

            # 3. 在病例文件夹内，查找所有视频文件
            video_files = [f for f in os.listdir(case_path) if f.lower().endswith(('.mp4', '.avi'))]

            for video_file in video_files:
                video_base_name = os.path.splitext(video_file)[0]
                video_full_path = os.path.join(case_path, video_file)
                
                # 寻找同名的JSON文件
                json_file = f"{video_base_name}.json"
                json_full_path = os.path.join(case_path, json_file)

                # 必须两者都存在才处理
                if not os.path.exists(json_full_path):
                    # tqdm.write(f"  [Skip] JSON not found for: {video_file}")
                    continue

                try:
                    # --- 读取数据 ---
                    original_frames = video_loader(video_full_path)
                    if original_frames is None: continue
                    
                    with open(json_full_path, 'r') as f:
                        pred_info = json.load(f)

                    # --- 提取信息 ---
                    # 1. 关键帧索引 (如果没有，默认为中间帧)
                    key_frame_idx = pred_info.get("predicted_key_frame_index")
                    if key_frame_idx is None:
                        key_frame_idx = len(original_frames) // 2
                    
                    # 确保索引不越界
                    if key_frame_idx >= len(original_frames):
                        key_frame_idx = len(original_frames) - 1

                    key_frame_img = original_frames[key_frame_idx]

                    # 2. ROI 坐标 (用于热图还原)
                    roi_bbox = pred_info.get("roi_crop_box_original_space")
                    # 如果JSON里没存ROI(旧数据)，或者它是阴性样本(全图)，默认全图
                    if not roi_bbox: 
                        h, w, _ = key_frame_img.shape
                        roi_bbox = [0, 0, w, h]

                    # 3. 预测框坐标
                    pred_box = pred_info.get("predicted_bounding_box_original_space")

                    # 4. 热图数据
                    heatmap_224 = None
                    if "fused_heatmap_224x224" in pred_info:
                        heatmap_224 = np.array(pred_info["fused_heatmap_224x224"])

                    # --- 构建输出路径 ---
                    # 保持完全相同的目录结构: Output_Root / Subtype / CaseID /
                    output_dir = os.path.join(VISUALIZATION_OUTPUT_ROOT, subtype_folder, case_id_folder)
                    os.makedirs(output_dir, exist_ok=True)

                    # --- 执行可视化 ---
                    case_info_str = f"Case: {case_id_folder}\nVideo: {video_file}\nFrame: {key_frame_idx}"
                    
                    generate_bbox_flag =  is_abnormal_group

                    create_visualization_pair(
                        output_dir=output_dir,
                        video_base_name=video_base_name,
                        original_frame=key_frame_img,
                        heatmap_224=heatmap_224,
                        roi_bbox=roi_bbox,
                        pred_box=pred_box,
                        case_info_str=case_info_str,
                        generate_bbox=generate_bbox_flag # <-- 传入控制参数
                    )

                except Exception as e:
                    tqdm.write(f"Error processing {video_file}: {e}")

    print(f"\n--- 处理完成！输出保存至: {VISUALIZATION_OUTPUT_ROOT} ---")