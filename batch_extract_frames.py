#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量视频帧提取工具
将多个视频的每一帧提取为独立的PNG图片文件，保存到与视频同名的文件夹中
"""

import cv2
import os
import sys
import glob
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir=None, max_frames=100, target_width=512, output_format='png'):
    """
    从视频中提取帧并保存为图片

    参数:
        video_path: 视频文件路径
        output_dir: 输出目录（如果为None，则使用视频同名文件夹）
        max_frames: 最大提取帧数
        target_width: 目标图片宽度（高度会自动按比例计算）
        output_format: 输出格式 ('png' 或 'jpg')
    """

    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 错误：视频文件不存在: {video_path}")
        return False

    # 如果没有指定输出目录，为每个视频创建独立的frames文件夹
    if output_dir is None:
        video_dir = os.path.dirname(video_path)
        video_name = Path(video_path).stem  # 获取视频文件名（不含扩展名）
        # 创建以视频文件名命名的文件夹，里面放frames子文件夹
        # 例如：video1.mp4 -> video1/frames/
        output_dir = os.path.join(video_dir, video_name, 'frames')

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 打开视频
    print(f"📹 正在处理: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ 错误：无法打开视频文件")
        return False

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"   📊 分辨率: {width}x{height}")
    print(f"   📊 帧率: {fps:.2f} FPS")
    print(f"   📊 总帧数: {total_frames}")
    print(f"   📊 时长: {duration:.2f}秒")

    # 计算实际提取的帧数
    actual_frames = min(total_frames, max_frames)
    print(f"   🎯 将提取 {actual_frames} 帧")

    # 计算目标高度（保持宽高比）
    aspect_ratio = width / height if height > 0 else 1
    target_height = int(target_width / aspect_ratio)

    # 提取帧
    success_count = 0

    for i in range(actual_frames):
        # 计算当前帧在原视频中的索引
        frame_index = int(i * total_frames / actual_frames)

        # 定位到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # 读取帧
        ret, frame = cap.read()

        if ret:
            # 调整图片尺寸
            resized_frame = cv2.resize(frame, (target_width, target_height),
                                      interpolation=cv2.INTER_LANCZOS4)

            # 保存图片（使用4位数字编号）
            file_ext = 'png' if output_format == 'png' else 'jpg'
            output_path = os.path.join(output_dir, f'frame_{i:04d}.{file_ext}')

            if output_format == 'png':
                cv2.imwrite(output_path, resized_frame)
            else:
                cv2.imwrite(output_path, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

            success_count += 1
        else:
            print(f"   ⚠️  警告：第 {frame_index} 帧读取失败")

    # 释放资源
    cap.release()

    print(f"   ✅ 成功提取 {success_count} 帧 -> {output_dir}")
    print()

    return True


def batch_extract_videos(video_paths, max_frames=100, target_width=512, output_format='png'):
    """
    批量提取多个视频的帧

    参数:
        video_paths: 视频文件路径列表
        max_frames: 每个视频最大提取帧数
        target_width: 目标图片宽度
        output_format: 输出格式 ('png' 或 'jpg')
    """
    total = len(video_paths)
    success_count = 0

    print("=" * 60)
    print("🎬 批量视频帧提取工具")
    print("=" * 60)
    print(f"📁 待处理视频数: {total}")
    print(f"🎯 每个视频提取帧数: {max_frames}")
    print(f"📐 输出尺寸宽度: {target_width}")
    print(f"📄 输出格式: {output_format.upper()}")
    print("=" * 60)
    print()

    for i, video_path in enumerate(video_paths, 1):
        print(f"[{i}/{total}] ", end="")
        if extract_frames(video_path, max_frames=max_frames, target_width=target_width,
                        output_format=output_format):
            success_count += 1

    print("=" * 60)
    print(f"✅ 处理完成！成功: {success_count}/{total}")
    print("=" * 60)


def find_videos(directory, extensions=['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']):
    """
    在目录中递归查找所有视频文件（包括子目录）

    参数:
        directory: 目录路径
        extensions: 视频文件扩展名列表

    返回:
        视频文件路径列表
    """
    video_paths = []
    for ext in extensions:
        # 使用 **/*.ext 模式递归搜索所有子目录
        pattern = os.path.join(directory, '**', ext)
        video_paths.extend(glob.glob(pattern, recursive=True))
        # 也搜索大写扩展名
        pattern = os.path.join(directory, '**', ext.upper())
        video_paths.extend(glob.glob(pattern, recursive=True))

    return sorted(video_paths)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量提取视频帧为PNG图片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 处理单个视频
  python batch_extract_frames.py video.mp4

  # 处理多个视频
  python batch_extract_frames.py video1.mp4 video2.mp4 video3.mp4

  # 处理整个文件夹的视频
  python batch_extract_frames.py --dir ./videos

  # 自定义参数
  python batch_extract_frames.py video.mp4 --frames 50 --width 256 --format jpg
        '''
    )

    parser.add_argument('videos', nargs='*', help='视频文件路径（可多个）')
    parser.add_argument('--dir', '-d', help='包含视频的文件夹路径')
    parser.add_argument('--frames', '-f', type=int, default=100,
                       help='每个视频提取的最大帧数 (默认: 100)')
    parser.add_argument('--width', '-w', type=int, default=512,
                       help='输出图片宽度 (默认: 512)')
    parser.add_argument('--format', '-fmt', choices=['png', 'jpg'], default='png',
                       help='输出格式 (默认: png)')

    args = parser.parse_args()

    # 收集所有视频路径
    video_paths = []

    # 添加直接指定的视频
    if args.videos:
        video_paths.extend(args.videos)

    # 添加文件夹中的视频
    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"❌ 错误：目录不存在: {args.dir}")
            sys.exit(1)
        videos_from_dir = find_videos(args.dir)
        if not videos_from_dir:
            print(f"❌ 错误：目录中没有找到视频文件: {args.dir}")
            sys.exit(1)
        video_paths.extend(videos_from_dir)

    # 验证
    if not video_paths:
        parser.print_help()
        print("\n❌ 错误：请指定至少一个视频文件或包含视频的目录")
        sys.exit(1)

    # 检查所有文件是否存在
    valid_paths = []
    for path in video_paths:
        if os.path.isfile(path):
            valid_paths.append(path)
        else:
            print(f"⚠️  警告：跳过不存在的文件: {path}")

    if not valid_paths:
        print("❌ 错误：没有有效的视频文件")
        sys.exit(1)

    # 执行批量提取
    try:
        batch_extract_videos(valid_paths, max_frames=args.frames,
                           target_width=args.width, output_format=args.format)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
