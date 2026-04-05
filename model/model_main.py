"""
心脏超声诊断系统 - 主程序

功能:
    - 单视角推理: 生成热力图(Heatmap)、边界框(BBox)、关键帧
    - 多视角推理: 生成诊断置信度 {Normal, VSD, ASD, PDA}

输入目录结构:
    {target_dir}/
    ├── case1/
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   └── ...
    ├── case2/
    │   └── video1.mp4
    └── ...

输出目录结构:
    {output_dir}/
    └── {request_name}/
        ├── case1/
        │   ├── output_videos/
        │   ├── output_data/
        │   └── output_data.zip
        └── case2/
            └── ...

单视角模型 (4个):
    - spatial_size_4.pth  (4x4 网格)
    - spatial_size_5.pth  (5x5 网格)
    - spatial_size_6.pth  (6x6 网格)
    - spatial_size_7.pth  (7x7 网格)
    输出: 热力图 + 边界框 + 关键帧

多视角模型:
    - epoch_97.98295452219057.pth
    输出: 诊断置信度 {Normal, VSD, ASD, PDA}

使用方法:
    python main.py

    或在代码中调用:
    from main import run_diagnosis
    result = run_diagnosis(target_dir, output_dir)
"""

# ==================== 配置区域 ====================
# 请在此处修改输入输出路径
# 路径说明: 支持绝对路径和相对路径
#   - 绝对路径: r"D:\实验室\workTable\26_3_17_web\worktable\models\input_path\request1"
#   - 相对路径: "./input_path/request1" 或 "input_path/request1" (相对于运行目录)
#   注意: 相对路径是相对于运行Python脚本时的工作目录(os.getcwd())，而非脚本所在目录
TARGET_DIR = r"D:\实验室\workTable\26_3_17_web\worktable\models\input_path\request_test"
OUTPUT_DIR = r"D:\实验室\workTable\26_3_17_web\worktable\models\output_path"
# =================================================

import os
import sys

# 确保可以导入heart_diagnosis模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from heart_diagnosis import diagnose


def run_diagnosis(target_dir: str, output_dir: str) -> dict:
    """
    运行诊断流程
    
    Args:
        target_dir: 输入目录路径
        output_dir: 输出目录路径
    
    Returns:
        dict: 诊断结果
    """
    print(f"\n{'='*70}")
    print(f"心脏超声诊断系统")
    print(f"{'='*70}")
    print(f"输入目录: {target_dir}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")
    
    result = diagnose(target_dir=target_dir, output_dir=output_dir)
    
    print(f"\n{'='*70}")
    print(f"诊断完成")
    print(f"{'='*70}")
    print(f"Request名称: {result['request_name']}")
    print(f"总病例数: {result['total_cases']}")
    print(f"成功病例: {result['success_cases']}")
    print(f"失败病例: {result['failed_cases']}")
    print(f"输出目录: {result['output_dir']}")
    print(f"{'='*70}\n")
    
    return result


def main():
    """主函数"""
    # 运行诊断
    result = run_diagnosis(TARGET_DIR, OUTPUT_DIR)
    
    # 打印每个case的诊断结果
    print("\n各病例诊断结果:")
    print("-" * 70)
    for case_result in result['results']:
        case_id = case_result['case_id']
        success = case_result['success_count']
        error = case_result['error_count']
        
        print(f"\n病例: {case_id}")
        print(f"  成功: {success}, 失败: {error}")
        
        if case_result['multi_view_result']:
            confidence = case_result['multi_view_result']
            print(f"  诊断置信度:")
            for cls, score in confidence.items():
                print(f"    - {cls}: {score}")
    
    print("\n" + "=" * 70)
    print(f"所有结果已保存到: {result['output_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
