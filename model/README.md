# 心脏超声多视角智能诊断系统

## 项目概述

本系统用于心脏超声多视角诊断，基于深度学习模型实现：
- **单视角推理**：生成热力图(Heatmap)和边界框(BBox)可视化
- **多视角推理**：生成诊断置信度 {Normal, VSD, ASD, PDA}

## 目录结构

```
{项目根目录}/
├── main.py                      # 主程序入口
├── heart_diagnosis.py           # 核心诊断模块
├── interface1_batch.py          # 批处理接口
├── interface2_api.py           # API接口
├── requirements.txt             # Python依赖
├── README.md                    # 项目文档
├── require_file.txt             # 依赖文件清单
├── Codes/
│   ├── config/
│   │   ├── test_config_ResnetrTransformerDualToken_1107_18cls_layersdecouple44.yaml
│   │   └── test_config_ResnetrTransformerDualToken_MutilViews_1118_18cls_layersdecouple55.yaml
│   ├── Nets/
│   │   ├── dual_tokens_net.py
│   │   └── Multi_Views_dual_tokens_net.py
│   └── main_codes/
│       └── original_ref.py      # 提取的原函数
├── singleview/                  # 单视角模型权重
│   ├── spatial_size_4.pth
│   ├── spatial_size_5.pth
│   ├── spatial_size_6.pth
│   └── spatial_size_7.pth
├── multiviews/                  # 多视角模型权重
│   └── epoch_97.98295452219057.pth
├── Baseline_4size/              # 基准特征文件
│   ├── final_baseline_features_4x4_512dim_1101.pt
│   ├── final_baseline_features_5x5_512dim_1101.pt
│   ├── final_baseline_features_6x6_512dim_1101.pt
│   └── final_baseline_features_7x7_512dim_1101.pt
└── input_path/                  # 输入视频目录
    └── request1/
        ├── case1/
        │   └── *.mp4
        └── ...
```

## 使用方法

### 方法1：运行主程序 (main.py)

```bash
python main.py
```

### 方法2：直接调用接口 (heart_diagnosis.py)

```python
from heart_diagnosis import diagnose

result = diagnose(
    target_dir=r"D:\path\to\input",
    output_dir=r"D:\path\to\output"
)
```

### 方法3：批处理模式 (interface1_batch.py)

```bash
python interface1_batch.py -t <输入目录> -o <输出目录>
```

### 方法4：API接口 (interface2_api.py)

```python
from interface2_api import HeartInferenceEngine

# 初始化引擎（单例模式）
engine = HeartInferenceEngine()

# 单病例诊断
result = engine.process_single_case(
    case_input_dir=r"D:\path\to\case1",
    output_dir=r"D:\path\to\output"
)

# 获取诊断结果
print(result['multi_view_result'])  # {'Normal': 0.001, 'VSD': 0.001, 'ASD': 0.998, 'PDA': 0.0}
```

## 输入输出格式

### 输入目录结构
```
{输入目录}/
├── case1/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── case2/
│   └── video1.mp4
└── ...
```

### 输出目录结构
```
{输出目录}/
└── {request_name}/
    ├── case1/
    │   ├── output_videos/
    │   │   ├── video1_original.mp4       # 原视频副本
    │   │   ├── video1_original/          # 序列帧目录
    │   │   │   └── frames/
    │   │   │       ├── frame_0001.png
    │   │   │       └── ...
    │   │   ├── video1_bbox.mp4           # 边界框可视化
    │   │   └── video1_heatmap.mp4        # 热力图可视化
    │   ├── output_data/
    │   │   ├── video1.json              # 热力图数据+预测框
    │   │   ├── video1.mp4               # 原始视频副本
    │   │   ├── confidence_scores.json   # 多视角诊断置信度
    │   │   └── ...
    │   └── output_data.zip               # 数据压缩包
    └── case2/
        └── ...
```

## 模型说明

### 单视角模型 (4个)
| 模型 | 网格大小 | 路径 |
|------|----------|------|
| spatial_size_4 | 4x4 | singleview/spatial_size_4.pth |
| spatial_size_5 | 5x5 | singleview/spatial_size_5.pth |
| spatial_size_6 | 6x6 | singleview/spatial_size_6.pth |
| spatial_size_7 | 7x7 | singleview/spatial_size_7.pth |

**输出**：热力图 + 边界框 + 关键帧

### 多视角模型
| 模型 | 路径 |
|------|------|
| 多视角融合 | multiviews/epoch_97.98295452219057.pth |

**输出**：诊断置信度 {Normal, VSD, ASD, PDA}

## 诊断类别

- **Normal**: 正常
- **VSD**: 室间隔缺损 (Ventricular Septal Defect)
- **ASD**: 房间隔缺损 (Atrial Septal Defect)
- **PDA**: 动脉导管未闭 (Patent Ductus Arteriosus)

## 注意事项

1. **仅调用原函数**：系统完全复用Codes目录下的原始函数，不修改任何内部逻辑
2. **仅路径适配**：仅修改输出路径以适配预期目录结构
3. **多视频支持**：每个case可包含多个视频(view1, view2, view3等)
4. **GPU支持**：自动检测并使用GPU（如可用）
5. **随机种子**：使用固定种子(42)确保结果可复现
