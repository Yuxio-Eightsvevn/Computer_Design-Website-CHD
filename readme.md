# 儿童先心病超声诊断平台 - README

---

## 一、协作协定

你是一个精通python、网页开发等编程技能的编程助手，善于接管大型项目，富有经验，代码风格简洁。在与用户的协作中，你必须先充分理解用户对大型项目的各种期望、以及大体的项目结构再进行行动。行动过程中必须记住用户的项目的代码结构，并了解其内大概是什么样的结构。在某些期望要求等方面不明确时，你会立刻停止任务并向用户询问要求；当对项目的任何地方不了解、或出现任何无法根据现有的信息解决的问题、或用户要求调试时，你也会停下，自行请求项目代码或用代码进行调试；在对用户的项目代码进行更改时，请先行指出要更改的项目结构，并对需要进行的改动进行调整，即使是在可读模式也不允许在未得到我的直接允许时直接对代码进行更改。必要时也必须明确指出要改动的地方以及其上下文。必要时采用md格式来进行说明，来使得说明美观。在不必要时，不允许对不明确绝对没用的代码逻辑进行更改。必要时采用md格式来进行说明，来使得说明美观。

必须单独强调的以下几点：
1.在你对代码计划进行更改时，请考察你的实现方式是否会影响其它功能的异常，并尽可能的不删除原有的任何逻辑，除非我明确指出且有必要。
2.在你对某些变量的作用不明确时，即使可以“望文生义”也不允许这样做，必须对代码进行搜索以明确其用途，并尽可能的不影响原有功能逻辑，除非的确有必要或我明确指出。



**⚠️ 重要约定：所有对项目的更新都必须同步更新本 README.md 文档，保持文档与代码的一致性。**

---

## 二、项目概述

### 2.1 项目意图

本项目是一个**儿童先天性心脏病（先心病）超声诊断辅助平台**，旨在：

1. **辅助医师诊断**：通过AI模型对心脏超声视频进行分析，生成热力图、边界框、关键帧标注及诊断置信度
2. **支持多病例批量处理**：支持批量上传病例视频，后台自动处理
3. **提供教育培训功能**：教育模式下，医师可进行诊断练习并获得自动评分

### 2.2 诊断类别

| 类别 | 说明 |
|------|------|
| Normal | 正常 |
| VSD | 室间隔缺损 |
| ASD | 房间隔缺损 |
| PDA | 动脉导管未闭 |

### 2.3 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| 数据库 | SQLite |
| AI模型 | PyTorch + C3D (单视角/多视角) |
| 前端 | 原生HTML/CSS/JavaScript |
| 视频处理 | ffmpeg |
| 运行时 | Python 3.x + Conda环境 |

---

## 三、系统架构

### 3.1 目录结构

```
Diagnostic_Platform_Stable_Version_C3D/
├── main.py                    # FastAPI主入口
├── routes_oridata.py          # 核心路由：上传、AI处理、教育模式
├── database.py                # 用户数据库操作
├── users.db                   # SQLite数据库文件
│
├── login.html                 # 登录页面
├── flow.html                  # 模式选择主页
├── admin.html                 # 用户管理页面（管理员专用）
├── dashboard.html             # 任务上传页面
├── task_status.html           # 任务进度页面
├── diagnosis.html             # 医师判读界面
├── edu_admin.html             # 教育管理后台（管理员专用）
├── edu_status.html            # 教育练习中心（用户端）
│
├── model/                     # AI模型目录
│   ├── model_main.py          # 模型入口函数
│   ├── heart_diagnosis.py     # 核心诊断逻辑
│   ├── singleview/            # 单视角模型权重
│   ├── multiviews/            # 多视角模型权重
│   ├── Baseline_4size/        # 基线特征
│   └── Codes/                 # 模型代码
│       ├── Nets/              # 网络结构
│       ├── main_codes/        # 核心算法
│       └── config/            # 配置文件
│
└── data_batch_storage/        # 数据存储目录
    ├── {username}/            # 用户私有空间
    │   ├── oridata/           # 原始上传数据
    │   ├── processed/         # AI处理结果
    │   └── data.json          # 任务索引
    └── SYSTEM/edu_data/       # 教育模式系统空间
        ├── {submission_id}/   # 教育批次原始数据
        ├── processed/         # 处理结果
        ├── Doctor_Diag_Result/# 用户成绩单
        └── data.json          # 教育任务索引
```

### 3.2 核心模块关系图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          前端页面层                                   │
├─────────────────────────────────────────────────────────────────────┤
│  login.html ──→ flow.html ──┬──→ dashboard.html ──→ task_status.html│
│                             │                   │                   │
│                             │                   └──→ diagnosis.html │
│                             │                                       │
│                             ├──→ edu_status.html ──→ diagnosis.html │
│                             │      (教育模式)                        │
│                             │                                       │
│                             ├──→ admin.html (管理员)                 │
│                             └──→ edu_admin.html (管理员)             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          后端API层                                   │
├─────────────────────────────────────────────────────────────────────┤
│  main.py (FastAPI主入口)                                             │
│      ├── 用户认证: /api/login, /api/logout                          │
│      ├── 用户管理: /api/users/*                                      │
│      ├── 任务获取: /api/tasks/*                                      │
│      └── 结果提交: /api/diagnosis/submit-json                       │
│                                                                      │
│  routes_oridata.py (核心路由)                                        │
│      ├── 判读模式: /api/users/{username}/upload-oridata             │
│      ├── 教育模式: /api/edu/*                                        │
│      └── 后台任务: run_model_inference_wrapper                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          AI模型层                                    │
├─────────────────────────────────────────────────────────────────────┤
│  model_main.py                                                       │
│      └── run_diagnosis() ──→ heart_diagnosis.py                     │
│                                  ├── 单视角推理 (4个模型)            │
│                                  └── 多视角推理 (置信度)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、用户管理模块

### 4.1 登录页面 (login.html)

**位置**: `login.html`

**功能**: 用户身份验证入口

**关键逻辑**:
- 表单提交至 `/api/login`
- 登录成功后设置cookie `username` 并跳转至 `/flow`

**后端接口**:
| 接口 | 位置 | 功能 |
|------|------|------|
| POST /api/login | main.py:89-122 | 验证用户并设置cookie |
| POST /api/logout | main.py:124-128 | 清除cookie登出 |

**数据库操作**:
| 函数 | 位置 | 功能 |
|------|------|------|
| verify_user() | database.py:84-122 | 验证用户名密码 |
| hash_password() | database.py:15-19 | SHA-256密码哈希 |

---

### 4.2 模式选择主页 (flow.html)

**位置**: `flow.html`

**功能**: 登录后的主入口，根据用户角色显示不同功能入口

**界面按钮**:
| 按钮 | 可见性 | 跳转目标 |
|------|--------|----------|
| 诊断模式 | 所有用户 | /dashboard |
| 教育模式 | 所有用户 | /edu_status |
| 任务管理 | 仅管理员 | /edu_admin |
| 用户管理 | 仅管理员 | /admin |

**权限判断逻辑** (flow.html:129):
```javascript
isAdmin = u.is_admin === true;
// 根据is_admin字段动态显示管理按钮
```

---

### 4.3 用户管理页面 (admin.html) - 管理员专用

**位置**: `admin.html`

**功能**: 管理员对用户进行增删改查

**安全机制**:
- 二次密码验证 (admin.html:79-92)
- 仅管理员可访问

**后端接口**:
| 接口 | 位置 | 功能 |
|------|------|------|
| GET /api/users | main.py:143-146 | 获取所有用户 |
| POST /api/users | main.py:215-231 | 创建用户 |
| PUT /api/users/{id} | main.py:234-250 | 更新用户 |
| DELETE /api/users/{id} | main.py:253-264 | 删除用户 |

**数据库操作**:
| 函数 | 位置 | 功能 |
|------|------|------|
| get_all_users() | database.py:124-143 | 获取用户列表 |
| create_user() | database.py:188-218 | 创建用户 |
| update_user() | database.py:220-245 | 更新用户信息 |
| delete_user() | database.py:247-267 | 删除用户 |
| check_username_exists() | database.py:269-283 | 检查用户名重复 |

---

## 五、判读模式

### 5.1 任务上传页面 (dashboard.html)

**位置**: `dashboard.html`

**功能**: 医生上传病例视频批次

**上传方式**:
1. 选择文件夹 (webkitdirectory)
2. 选择视频文件 (multiple)
3. 拖拽文件夹到放置区

**路径解析逻辑** (dashboard.html:183-231):
- CASE A: 无路径信息 → 自动生成病例名 `patient N`
- CASE B1: 路径层级≥3 → 第2层为病例名
- CASE B2: 路径层级=2 → 第1层为病例名

**上传限制**:
- 每个病例最多5个视频
- 支持格式: .mp4, .avi, .mov, .mkv

**后端接口**:
| 接口 | 位置 | 功能 |
|------|------|------|
| POST /api/users/{username}/upload-oridata | routes_oridata.py:288-371 | 接收上传并触发AI处理 |

---

### 5.2 任务进度页面 (task_status.html)

**位置**: `task_status.html`

**功能**: 查看已上传任务的处理状态

**任务状态**:
- 处理中: `is_cmp = False` (蓝色脉冲点)
- 已完成: `is_cmp = True` (绿色点)

**后端接口**:
| 接口 | 位置 | 功能 |
|------|------|------|
| GET /api/users/{username}/all-tasks | routes_oridata.py:381-387 | 获取任务列表 |
| DELETE /api/users/{username}/tasks/{id} | routes_oridata.py:389-403 | 删除任务 |

---

### 5.3 医师判读界面 (diagnosis.html)

**位置**: `diagnosis.html`

**功能**: 医师查看AI处理结果并进行诊断判读

**核心功能**:

| 功能模块 | 说明 |
|----------|------|
| 视频网格 | 最多显示5个视频，支持3种模态(original/heatmap/bbox) |
| 视频弹窗 | 点击视频放大查看，含AI置信度显示 |
| 3D可视化 | video_3d_modal.js 【待实现】 |
| 诊断录入 | 选择诊断结果(4类)和严重程度 |
| 计时统计 | 记录每个病例的查看时间 |
| 结果提交 | 批量提交所有病例的诊断结论 |

**后端接口**:
| 接口 | 位置 | 功能 |
|------|------|------|
| GET /api/tasks/{username}/{folder}/patients | main.py:324-419 | 获取病例列表 |
| GET /api/get-metadata | main.py:422-431 | 获取置信度元数据 |
| POST /api/diagnosis/submit-json | main.py:455-542 | 提交诊断结果 |

**关键函数**:
| 函数 | 位置 | 功能 |
|------|------|------|
| loadPatients() | diagnosis.html | 加载病例列表 |
| openVideoModal() | diagnosis.html | 打开视频弹窗 |
| submitDiagnosis() | diagnosis.html | 提交诊断结果 |

---

### 5.4 判读模式完整工作流

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 上传阶段                                                      │
├─────────────────────────────────────────────────────────────────┤
│ dashboard.html                                                   │
│     │                                                            │
│     ├── 用户选择/拖拽视频文件                                     │
│     ├── 系统自动解析病例分组                                      │
│     └── POST /api/users/{username}/upload-oridata               │
│              │                                                   │
│              ├── 保存到: data_batch_storage/{user}/oridata/{id}/ │
│              ├── 更新索引: {user}/data.json                      │
│              └── 触发后台任务: run_model_inference_wrapper       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. AI处理阶段                                                    │
├─────────────────────────────────────────────────────────────────┤
│ routes_oridata.py:run_model_inference_wrapper                   │
│     │                                                            │
│     └── run_diagnosis(target_dir, output_dir)                   │
│              │                                                   │
│              ├── 单视角推理: 生成热力图/边界框/关键帧             │
│              ├── 多视角推理: 生成置信度分数                       │
│              ├── 视频转码: ffmpeg H.264                          │
│              └── 输出到: {user}/processed/{id}/                  │
│                       ├── {case}/output_videos/                  │
│                       └── {case}/output_data/                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. 判读阶段                                                      │
├─────────────────────────────────────────────────────────────────┤
│ diagnosis.html                                                   │
│     │                                                            │
│     ├── GET /api/tasks/{user}/{folder}/patients                 │
│     ├── 显示病例列表和视频                                        │
│     ├── 医师查看视频 + AI置信度                                   │
│     ├── 医师录入诊断结论                                          │
│     └── POST /api/diagnosis/submit-json (mode="diag")           │
│              │                                                   │
│              └── 保存到: {user}/processed/{id}/                  │
│                          final_diagnosis_report_{timestamp}.json │
└─────────────────────────────────────────────────────────────────┘
```

---

## 六、教育模式

### 6.1 教育管理后台 (edu_admin.html) - 管理员专用

**位置**: `edu_admin.html`

**功能**: 管理员创建、发布、管理教育批次

**工作流程**:

```
上传ZIP → AI处理 → 未发布 → 选择用户 → 发布
```

**任务状态**:
| 状态 | 说明 | 卡片颜色 |
|------|------|----------|
| processing | 模型判读中 | 蓝色边框 |
| unreleased | 待发布 | 橙色边框 |
| published | 已发布 | 绿色边框 |

**ZIP包要求**:
```
education_batch.zip
├── epoch_data.json    # 必需：标准答案
└── videos/            # 必需：视频文件夹
    ├── case1_video1.mp4
    └── ...
```

**epoch_data.json格式**:
```json
{
  "case_id_1": {
    "label": 0,
    "videos": ["videos/case1_video1.mp4", ...]
  },
  "case_id_2": { ... }
}
```

**后端接口**:
| 接口 | 位置 | 功能 |
|------|------|------|
| POST /api/edu/parse-zip | routes_oridata.py:417-446 | 解析ZIP统计信息 |
| POST /api/edu/confirm-upload | routes_oridata.py:448-519 | 确认上传并触发AI处理 |
| GET /api/edu/admin/tasks | routes_oridata.py:522-535 | 获取任务列表(分状态) |
| POST /api/edu/publish-task | routes_oridata.py:537-544 | 发布任务给指定用户 |
| GET /api/edu/admin/task-status/{id} | routes_oridata.py:546-576 | 获取用户完成进度 |
| DELETE /api/edu/admin/tasks/{id} | routes_oridata.py:606-633 | 删除教育任务 |

---

### 6.2 教育练习中心 (edu_status.html) - 用户端

**位置**: `edu_status.html`

**功能**: 用户查看分配给自己的教育任务，进行判读练习

**任务卡片**:
- 已完成: 绿色边框 + 分数显示
- 待判读: 橙色边框 + "开始判读"按钮

**成绩报告**:
| 指标 | 说明 |
|------|------|
| 准确率 | 正确数/总数 |
| 敏感度 | TP/(TP+FN) |
| 特异性 | TN/(TN+FP) |
| 总耗时 | 判读时间 |

**后端接口**:
| 接口 | 位置 | 功能 |
|------|------|------|
| GET /api/edu/user/tasks/{username} | routes_oridata.py:579-594 | 获取用户的教育任务 |
| GET /api/edu/user/result/{username}/{id} | routes_oridata.py:596-602 | 获取成绩详情 |

---

### 6.3 教育模式完整工作流

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 管理员创建批次                                                │
├─────────────────────────────────────────────────────────────────┤
│ edu_admin.html                                                   │
│     │                                                            │
│     ├── 上传ZIP (含epoch_data.json标准答案)                       │
│     ├── POST /api/edu/parse-zip (解析统计)                       │
│     └── POST /api/edu/confirm-upload (确认上传)                  │
│              │                                                   │
│              ├── 保存到: SYSTEM/edu_data/{submission_id}/        │
│              ├── 重组目录结构适配AI模型                           │
│              ├── 触发AI处理                                      │
│              └── 状态: processing → unreleased                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. 管理员发布任务                                                │
├─────────────────────────────────────────────────────────────────┤
│ edu_admin.html                                                   │
│     │                                                            │
│     ├── 选择目标用户                                             │
│     └── POST /api/edu/publish-task                               │
│              │                                                   │
│              └── 状态: unreleased → published                    │
│                  记录target_users到data.json                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. 用户进行判读练习                                              │
├─────────────────────────────────────────────────────────────────┤
│ edu_status.html → diagnosis.html                                 │
│     │                                                            │
│     ├── GET /api/edu/user/tasks/{username} (获取任务列表)        │
│     ├── 点击"开始判读" → diagnosis.html?mode=edu&taskId=xxx     │
│     ├── 用户录入诊断结论                                         │
│     └── POST /api/diagnosis/submit-json (mode="edu")            │
│              │                                                   │
│              ├── 读取标准答案: SYSTEM/edu_data/{id}/epoch_data.json
│              ├── 计算准确率/敏感度/特异性                        │
│              └── 保存成绩单: SYSTEM/edu_data/Doctor_Diag_Result/{user}.json
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. 管理员查看进度                                                │
├─────────────────────────────────────────────────────────────────┤
│ edu_admin.html                                                   │
│     │                                                            │
│     └── GET /api/edu/admin/task-status/{submission_id}          │
│              │                                                   │
│              └── 返回各用户的完成状态和分数                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 七、AI诊断模块

### 7.1 模型入口 (model_main.py)

**位置**: `model/model_main.py`

**入口函数**:
```python
def run_diagnosis(target_dir: str, output_dir: str) -> dict
```

**参数说明**:
| 参数 | 说明 |
|------|------|
| target_dir | 输入目录，包含病例文件夹 |
| output_dir | 输出目录根路径 |

**输入目录结构**:
```
target_dir/
├── case1/
│   ├── video1.mp4
│   └── video2.mp4
└── case2/
    └── video1.mp4
```

**输出目录结构**:
```
output_dir/{request_name}/
├── case1/
│   ├── output_videos/
│   │   ├── video1_original.mp4
│   │   ├── video1_heatmap.mp4
│   │   ├── video1_bbox.mp4
│   │   └── ...
│   ├── output_data/
│   │   ├── video1.json
│   │   ├── confidence_scores.json
│   │   └── video1.mp4 (原视频)
│   └── output_data.zip
└── case2/
    └── ...
```

---

### 7.2 核心诊断逻辑 (heart_diagnosis.py)

**位置**: `model/heart_diagnosis.py`

**主函数**:
```python
def diagnose(target_dir: str, output_dir: str, save_videos: bool = True) -> dict
```

**核心流程**:

| 步骤 | 函数 | 位置 | 功能 |
|------|------|------|------|
| 1 | load_single_models() | 94-120 | 加载4个单视角模型 |
| 2 | load_multi_model() | 123-133 | 加载多视角模型 |
| 3 | process_single_video() | 136-244 | 单视频推理：热力图/边界框/关键帧 |
| 4 | multi_view_inference() | 247-301 | 多视角推理：置信度分数 |
| 5 | save_visuals() | 304-353 | 保存可视化结果 |

**单视角推理详细流程** (process_single_video):

```
加载视频 → 提取帧 → 找ROI → 裁剪Resize → 创建Doppler Mask
    → 4个模型分别推理 → 投票决定预测类别 → 计算关键帧
    → 融合热力图 → 计算边界框 → 返回结果
```

**关键参数**:
| 参数 | 值 | 位置 | 说明 |
|------|-----|------|------|
| DEVICE | cuda/cpu | 41 | 运行设备 |
| MODEL_INPUT_SIZE | (224, 224) | 42 | 模型输入尺寸 |
| FINAL_BOX_SIZE | 56 | 43 | 输出边界框大小 |
| clip_size | 16 | 46 | 视频帧数 |
| MAX_VIEWS | 12 | 253 | 多视角最大视角数 |

---

### 7.3 单视角模型与多视角模型

**单视角模型** (4个):

| 模型文件 | 网格大小 | 权重路径 |
|----------|----------|----------|
| spatial_size_4.pth | 4×4 | model/singleview/ |
| spatial_size_5.pth | 5×5 | model/singleview/ |
| spatial_size_6.pth | 6×6 | model/singleview/ |
| spatial_size_7.pth | 7×7 | model/singleview/ |

**网络结构**: `ResnetTransformerDualTokensTemporalSpatialDecouplesize`
**位置**: `model/Codes/Nets/dual_tokens_net.py`

**多视角模型**:

| 模型文件 | 说明 |
|----------|------|
| epoch_97.98295452219057.pth | 多视角融合模型 |

**网络结构**: `MultiViewDualTokensFusionSize`
**位置**: `model/Codes/Nets/Multi_Views_dual_tokens_net.py`

**模型输出**:

| 模型 | 输出 |
|------|------|
| 单视角 | 热力图、边界框、关键帧索引 |
| 多视角 | 置信度分数 {Normal, VSD, ASD, PDA} |

---

## 八、后端API与数据存储

### 8.1 主服务入口 (main.py)

**位置**: `main.py`

**生命周期管理**:
| 阶段 | 函数 | 位置 | 功能 |
|------|------|------|------|
| 启动 | init_system() | 28-53 | 初始化数据库、创建目录结构 |
| 运行 | lifespan() | 56-64 | 管理应用生命周期 |

**主要路由**:

| 路由 | 方法 | 位置 | 功能 |
|------|------|------|------|
| /api/login | POST | 89-122 | 用户登录 |
| /api/logout | POST | 124-128 | 用户登出 |
| /api/users | GET/POST | 143-231 | 用户管理 |
| /api/users/current | GET | 148-203 | 获取当前用户 |
| /api/tasks/{username} | GET | 267-320 | 获取用户任务列表 |
| /api/tasks/{username}/{folder}/patients | GET | 324-419 | 获取病例列表 |
| /api/get-metadata | GET | 422-431 | 获取元数据 |
| /api/diagnosis/submit-json | POST | 455-542 | 提交诊断结果 |

**静态文件挂载**:
```python
app.mount("/videos", StaticFiles(directory=DATA_BATCH_STORAGE))
app.mount("/data", StaticFiles(directory=DATA_BATCH_STORAGE))
```

---

### 8.2 路由模块 (routes_oridata.py)

**位置**: `routes_oridata.py`

**核心函数**:

| 函数 | 位置 | 功能 |
|------|------|------|
| generate_submission_id() | 68-71 | 生成唯一任务ID |
| update_user_task_index() | 138-198 | 维护用户任务索引(含文件锁) |
| update_edu_task_index() | 201-246 | 维护教育任务索引 |
| run_model_inference_wrapper() | 250-284 | AI模型后台任务包装器 |
| transcode_to_h264() | 95-134 | 视频编码转换 |
| generate_thumbnail_ffmpeg() | 84-92 | 生成视频缩略图 |

**主要路由**:

| 路由 | 方法 | 位置 | 功能 |
|------|------|------|------|
| /api/users/{username}/upload-oridata | POST | 288-371 | 判读模式上传 |
| /api/users/{username}/oridata-count | GET | 375-379 | 获取上传计数 |
| /api/users/{username}/all-tasks | GET | 381-387 | 获取所有任务 |
| /api/users/{username}/tasks/{id} | DELETE | 389-403 | 删除单个任务 |
| /api/users/{username}/clear-stuck-tasks | DELETE | 454-520 | 清空卡住的任务 |
| /api/edu/parse-zip | POST | 417-446 | 解析教育ZIP |
| /api/edu/confirm-upload | POST | 448-519 | 确认教育上传 |
| /api/edu/admin/tasks | GET | 522-535 | 获取教育任务列表 |
| /api/edu/publish-task | POST | 537-544 | 发布教育任务 |
| /api/edu/user/tasks/{username} | GET | 579-594 | 获取用户教育任务 |

---

### 8.3 数据库模块 (database.py)

**位置**: `database.py`

**表结构**:
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,           -- SHA-256哈希
    doctor TEXT NOT NULL,
    organization TEXT NOT NULL,
    is_admin BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**核心函数**:

| 函数 | 位置 | 功能 |
|------|------|------|
| init_database() | 32-82 | 初始化数据库，创建默认账户 |
| verify_user() | 84-122 | 验证用户登录 |
| get_all_users() | 124-143 | 获取所有用户 |
| get_user_by_id() | 145-164 | 按ID获取用户 |
| get_user_by_username() | 166-185 | 按用户名获取用户 |
| create_user() | 188-218 | 创建用户 |
| update_user() | 220-245 | 更新用户信息 |
| delete_user() | 247-267 | 删除用户 |
| check_username_exists() | 269-283 | 检查用户名是否存在 |

**默认账户**:
| 用户名 | 密码 | 角色 |
|--------|------|------|
| admin | 123456 | 管理员 |
| doctor1 | 123456 | 普通用户 |

---

### 8.4 SYSTEM保留名机制

**位置**: `database.py`

**保护点**:

```python
# create_user() 中 (database.py:191-193)
if username.upper() == "SYSTEM":
    print(f"🚫 拦截：禁止创建保留用户名 {username}")
    return False

# check_username_exists() 中 (database.py:271-272)
if username.upper() == "SYSTEM":
    return True  # 假装已存在，阻止注册
```

**设计目的**:

| 保护机制 | 说明 |
|----------|------|
| 禁止注册SYSTEM用户 | 确保无人能以SYSTEM身份登录 |
| 大小写不敏感 | 阻止 system/System/SYSTEM 等变体 |

**SYSTEM空间用途**:
```
data_batch_storage/SYSTEM/edu_data/
├── {submission_id}/          # 教育批次原始数据
├── processed/{submission_id}/ # AI处理结果
├── Doctor_Diag_Result/       # 用户成绩单
│   ├── doctor1.json          # 每个用户一个成绩文件
│   └── ...
└── data.json                 # 教育任务索引
```

---

## 九、开发指南

### 10.1 本地启动方式

```bash
# 1. 激活Conda环境
conda activate challengeserver

# 2. 进入项目目录
cd /path/to/Diagnostic_Platform_Stable_Version_C3D

# 3. 启动FastAPI服务（开发模式）
python main.py

# 或生产模式
nohup uvicorn main:app --host 0.0.0.0 --port 11000 --workers 4 > app.log 2>&1 < /dev/null &
```

**访问地址**:
- 登录页: http://127.0.0.1:11000/
- 管理后台: http://127.0.0.1:11000/admin

### 10.2 环境依赖

**Python依赖**:
```
fastapi
uvicorn
aiofiles
aiohttp  # [新增] 用于大模型API异步调用
pydantic
torch
torchvision
opencv-python
numpy
pillow
tqdm
pyyaml
einops  # [新增] 模型依赖
entmax  # [新增] 模型依赖
```

**系统依赖**:
- ffmpeg (视频处理)

### 10.3 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 模型导入失败 | 模型文件缺失 | 检查model/singleview/和multiviews/目录 |
| 视频无法播放 | 编码格式不支持 | 检查ffmpeg是否正确安装，视频会自动转码为H.264 |
| 404错误加载帧 | 帧文件路径错误 | 确认frames文件夹存在且命名正确 |
| 教育任务无法发布 | epoch_data.json格式错误 | 检查JSON格式和label值范围(0-3) |
| 用户无法登录 | 数据库未初始化 | 删除users.db重新启动，会自动创建默认账户 |
| 大模型调用失败 | API Key无效或网络错误 | 检查API Key是否正确，测试网络连接 |
| AI分析报告未生成 | 大模型未配置 | 在edu_admin.html配置大模型API |
| AI分析超时 | 网络延迟或模型响应慢 | 稍后刷新页面查看报告 |

### 10.4 文件锁机制说明

为防止并发写入冲突，`routes_oridata.py` 中使用文件锁：

```python
# update_user_task_index() 中 (routes_oridata.py:148-161)
lock_path = user_root / "data.json.lock"
for _ in range(50):  # 最长等待5秒
    try:
        with open(lock_path, "x") as _: pass  # 原子创建
        acquired = True
        break
    except FileExistsError:
        time.sleep(0.1)
```

### 10.5 任务状态追踪机制

为安全清理卡住的任务，系统使用文件标记追踪后台任务状态：

**状态文件**:
| 文件 | 位置 | 说明 |
|------|------|------|
| `.processing` | oridata/{submission_id}/ | 任务正在运行 |
| `.failed` | oridata/{submission_id}/ | 任务已失败 |

**状态转换**:
```
任务启动 → 创建 .processing
任务成功 → 删除 .processing
任务失败 → .processing 重命名为 .failed
```

**任务状态判断**:

| is_cmp | .processing | .failed | 状态 | 可清理 |
|--------|-------------|---------|------|--------|
| True | - | - | 已完成 | ❌ |
| False | ✓ | - | 正在运行 | ❌ |
| False | - | ✓ | 已失败 | ✅ |
| False | - | - | 未知/卡住 | ✅ |

### 10.6 清空卡住任务接口

**接口**: `DELETE /api/users/{username}/clear-stuck-tasks`

**位置**: `routes_oridata.py:454-520`

**功能**: 安全清空当前用户下所有未完成且未在运行的任务

**删除条件**:
1. `is_cmp = False`（未完成）
2. 不存在 `.processing` 文件（没有后台任务在运行）

**安全机制**:
- 已完成的任务（`is_cmp = True`）不受影响
- 正在运行的任务（存在 `.processing` 文件）不受影响
- 失败的任务（存在 `.failed` 文件）会被清理

**返回示例**:
```json
{
  "message": "已清理 3 个卡住的任务",
  "deleted_count": 3,
  "deleted_tasks": [
    {"submission_id": "20260401_123456", "request_name": "测试任务1"},
    {"submission_id": "20260401_234567", "request_name": "测试任务2"}
  ],
  "kept_count": 5
}
```

---

## 十一、教育模式增强功能

### 11.1 功能概述

教育模式新增了**大模型智能分析报告**功能，通过集成主流大模型API（如智谱AI、OpenAI、DeepSeek、Moonshot等），为用户的诊断练习提供深度分析和专业评价。

### 11.2 新增统计数据

#### A. 病种级别统计
- 各病种总数、正确数、错误数
- 按病种分类的时间分析

#### B. 时间分析
- 前3个时间最长的病例及其详情
- 病种级别的平均时间比较（正确vs错误）

#### C. AI依赖性分析
| 类型 | 说明 |
|------|------|
| 正确依赖 | AI正确且判读正确 |
| 依赖不足 | AI正确且判读错误 |
| 正确独立 | AI错误且判读正确 |
| 过度依赖 | AI错误且判读错误 |

#### D. 详细数据列表
- 真实标签列表
- AI判读标签列表
- 医师判读标签列表
- 判读使用时间列表

### 11.3 大模型分析内容

大模型会基于统计数据生成以下评价：

1. **整体表现评价** - 综合准确率、敏感度、特异性等指标
2. **诊断能力分析** - 分析在不同病种上的表现，识别强项和弱项
3. **AI工具使用情况** - 分析对AI辅助的依赖程度，是否过度依赖或利用不足
4. **改进建议** - 给出具体、可操作的改进建议（3-5条）
5. **优势点** - 识别用户的诊断优势（2-3条）
6. **不足点** - 识别需要改进的地方（2-3条）

### 11.4 配置与管理

#### 大模型配置API

| 接口 | 方法 | 位置 | 功能 |
|------|------|------|------|
| /api/admin/llm-config | POST | main.py:483-510 | 保存大模型配置 |
| /api/admin/llm-config | GET | main.py:513-553 | 获取配置（API Key脱敏） |
| /api/admin/test-llm | POST | main.py:556-566 | 测试大模型连接 |

#### 配置文件位置
```
config/llm_config.json
```

#### 配置参数说明
```json
{
  "base_url": "https://open.bigmodel.cn/api",  // API基础URL
  "api_key": "your-api-key-here",              // API密钥
  "model": "glm-4"                             // 模型名称
}
```

### 11.5 推荐大模型服务

| 服务商 | 免费额度 | 特点 | 注册地址 |
|--------|---------|------|---------|
| **智谱AI (推荐)** | 新用户送¥30 | 国内服务，响应快，GLM-4性能优秀 | https://open.bigmodel.cn/ |
| DeepSeek | 免费1元额度 | 性价比高 | https://platform.deepseek.com/ |
| Moonshot AI | 新用户送15元 | 长文本处理能力强 | https://platform.moonshot.cn/ |
| OpenAI | 无免费额度 | GPT-4性能最强 | https://platform.openai.com/ |

### 11.6 使用流程

1. **管理员配置大模型**
   - 访问教育管理后台（edu_admin.html）
   - 在底部"大模型配置"卡片填写配置
   - 点击"保存配置"
   - 点击"测试连接"验证配置

2. **用户进行判读练习**
   - 系统自动追踪每个病例的查看时间
   - 提交时自动记录所有统计数据
   - 后台异步调用大模型分析

3. **查看报告**
   - 初次查看显示"正在生成AI分析报告"
   - 系统自动轮询等待分析完成（最多30秒）
   - 分析完成后显示完整报告（包含大模型评价）

### 11.7 核心文件

| 文件 | 功能 | 关键功能 |
|------|------|---------|
| llm_analyzer.py | 大模型调用模块 | LLMAnalyzer类，支持多种大模型API |
| main.py | 后端API | 统计计算、大模型配置管理 |
| edu_admin.html | 管理界面 | 大模型配置界面 |
| edu_status.html | 用户界面 | 报告查看、轮询等待 |
| diagnosis.html | 判读界面 | 查看时间追踪 |

### 11.8 兼容性说明

**支持的大模型API**：
- ✅ 智谱AI (BigModel): `/v4/chat/completions`
- ✅ OpenAI: `/v1/chat/completions`
- ✅ DeepSeek: `/v1/chat/completions`
- ✅ Moonshot: `/v1/chat/completions`
- ✅ 其他兼容OpenAI格式的API

**自动识别机制**：
- 系统会自动检测URL中是否包含"bigmodel"或"zhipu"
- 智谱AI自动使用v4接口
- 其他服务使用v1接口

### 11.9 错误处理

| 错误情况 | 处理方式 |
|---------|---------|
| 大模型未配置 | 跳过AI分析，仅显示基础报告 |
| API调用失败 | 记录失败状态，显示基础报告 |
| 分析超时（30秒） | 显示基础报告，提示稍后再查看 |
| 响应格式错误 | 记录错误日志，显示基础报告 |

---

## 附录：关键文件索引

| 功能模块 | 文件 | 关键行号 |
|----------|------|----------|
| 用户登录 | main.py | 89-122 |
| 用户管理 | database.py | 188-283 |
| 任务上传 | routes_oridata.py | 288-371 |
| 任务状态追踪 | routes_oridata.py | 250-320 |
| 清空卡住任务 | routes_oridata.py | 454-520 |
| AI处理 | heart_diagnosis.py | 366-480 |
| 判读界面 | diagnosis.html | openVideoModal(), submitDiagnosis() |
| 教育上传 | routes_oridata.py | 448-519 |
| SYSTEM保护 | database.py | 191-193, 271-272 |
| 大模型分析 | llm_analyzer.py | 全文件 |
| 大模型配置API | main.py | 483-574 |
| 教育增强统计 | main.py | 578-825 |
| 查看时间追踪 | diagnosis.html | 1190-1206, 1472-1488, 2269-2299 |
| 报告查看优化 | edu_status.html | 114-300 |
| 大模型配置界面 | edu_admin.html | 105-239 |

---

## 十二、2026年4月更新记录

### 12.1 用户管理界面优化 (admin.html)

**更新内容**:
- 删除每行的"上传任务"和"下载结果"按钮
- 新增"角色"列，显示"管理员"或"普通用户"标签
- 编辑弹窗新增"设为管理员"复选框
- 支持创建/编辑用户时设置管理员角色

**后端变更**:
- `database.py`: `update_user()` 函数新增 `is_admin` 参数
- `main.py`: `UserUpdateRequest` 模型新增 `is_admin` 字段
- API接口 `/api/users/{user_id}` 支持更新用户角色

**相关文件**:
| 文件 | 修改内容 |
|------|----------|
| admin.html | 表格UI、编辑弹窗、提交逻辑 |
| database.py | update_user() 函数 |
| main.py | UserUpdateRequest、API接口 |

---

### 12.2 教育管理后台增强 (edu_admin.html)

**更新内容**:
- 已发布任务的用户列表中，已完成用户显示"查看报告"按钮
- 点击"查看报告"弹出完整评估报告（包含准确率、敏感度、特异性、AI依赖分析、AI专家评价）
- 任务详情弹窗底部新增"取消发布"按钮
- 点击"取消发布"清空所有用户成绩并将任务状态改为"未发布"

**新增API**:
| 接口 | 方法 | 位置 | 功能 |
|------|------|------|------|
| /api/edu/admin/unpublish-task | POST | routes_oridata.py | 取消发布并清空成绩 |

**相关文件**:
| 文件 | 修改内容 |
|------|----------|
| edu_admin.html | 用户报告弹窗、取消发布功能 |
| routes_oridata.py | 新增unpublish-task接口 |

---

### 12.3 诊断界面UI修复 (diagnosis.html)

**更新内容**:
- 修复教育模式完成后弹窗显示位置错误问题
- 弹窗改为独立居中显示，带遮罩背景
- 添加 `.modal-card` 样式和 `@keyframes slideUp` 动画
- 优化完成弹窗文字，提示用户去教育练习中心查看报告

**相关文件**:
| 文件 | 修改内容 |
|------|----------|
| diagnosis.html | 弹窗HTML结构、CSS样式、JS函数 |

---

### 12.4 教育状态页面优化 (edu_status.html)

**更新内容**:
- 修复 `loadUser` 函数处理API失败的情况
- 报告弹窗宽度从450px增加到700px
- AI依赖统计从垂直列表改为横向4列卡片布局
- 添加缺失的 `</style>` 闭合标签修复语法错误

**相关文件**:
| 文件 | 修改内容 |
|------|----------|
| edu_status.html | loadUser函数、报告弹窗UI、CSS |

---

### 12.5 大模型分析模块修复 (llm_analyzer.py)

**更新内容**:
- 修复 `prompt` 变量未初始化问题
- 修改 `_construct_prompt()` 函数使用 `prompt = f"""` 而非 `prompt += f"""`

**相关文件**:
| 文件 | 修改内容 |
|------|----------|
| llm_analyzer.py | _construct_prompt() 函数 |

---

### 12.6 置信度数据解析修复 (main.py + diagnosis.html)

**更新内容**:
- 后端API支持两种置信度数据格式：
  - 直接格式: `{ Normal: 0.85, VSD: 0.12, ... }`
  - 包装格式: `{ confidence_scores: { Normal: 0.85, ... } }`
- 前端 `loadModelConfidence()` 函数同步支持两种格式解析

**相关文件**:
| 文件 | 修改内容 |
|------|----------|
| main.py | /api/get-metadata 接口 |
| diagnosis.html | loadModelConfidence() 函数 |

---

### 12.7 数据库更新函数增强 (database.py)

**更新内容**:
- `update_user()` 函数新增 `is_admin` 可选参数
- 支持同时更新密码和角色、仅更新角色、仅更新密码等多种场景

**函数签名**:
```python
def update_user(user_id: int, doctor: str, organization: str, 
                password: Optional[str] = None, is_admin: Optional[bool] = None) -> bool
```

---

### 12.8 双阶段教育系统 (单独判读 + AI辅助判读)

**更新内容**:

教育模式新增**双阶段判读**功能，用户需要完成两个阶段：
1. **单独判读**: 仅显示原始视频，禁用heatmap/bbox模态按钮
2. **AI辅助判读**: 显示完整模态（original/heatmap/bbox），需先完成单独判读才可解锁
3. **对比报告**: 显示两个阶段的详细对比分析

**后端变更**:

| 接口/函数 | 修改位置 | 修改内容 |
|-----------|----------|----------|
| /api/edu/confirm-upload | routes_oridata.py | 新增 `is_dual_stage` 字段 |
| /api/edu/publish-task | routes_oridata.py | 新增 `publish_mode` 参数(single/dual) |
| DiagnosisSubmitJsonRequest | main.py | 新增 `eduSubMode` 字段 |
| /api/diagnosis/submit-json | main.py | 保存时添加 `_SINGLE` / `_AI-ASSIST` 后缀 |
| /api/edu/check-task-status | routes_oridata.py | 修复 `_SINGLE`/`_AI-ASSIST` 后缀处理 |

**前端变更**:

| 页面 | 修改内容 |
|------|----------|
| edu_admin.html | 发布模式下拉框(single/dual)，默认dual |
| edu_status.html | 双按钮垂直排列("单独判读"+"AI判读")，对比报告入口 |
| diagnosis.html | 单独判读模式禁用heatmap/bbox按钮，检查前置任务完成状态 |

**任务状态处理**:

```
任务ID: task123
├── task123_SINGLE      # 单独判读结果
└── task123_AI-ASSIST  # AI辅助判读结果
```

**对比报告内容**:
- 两个阶段的准确率、敏感度、特异性对比
- AI依赖性分析对比
- 各病种表现对比

**相关文件**:
| 文件 | 修改内容 |
|------|----------|
| routes_oridata.py | is_dual_stage、publish_mode、check-task-status修复 |
| main.py | DiagnosisSubmitJsonRequest、eduSubMode |
| edu_admin.html | 发布模式选择UI |
| edu_status.html | 双按钮、对比报告 |
| diagnosis.html | 模态按钮禁用、状态检查 |

---

### 12.11 AI分析等待界面与触发逻辑修复

**更新内容**:

| 问题 | 修复位置 | 详细说明 |
|------|----------|----------|
| 等待界面显示秒数 | edu_status.html | 简化等待界面，不显示秒数/次数 |
| AI分析key不匹配 | main.py | analyze_with_llm参数从original_task_folder改为save_key |
| 缺少手动触发API | routes_oridata.py | 新增 /api/edu/trigger-llm-analysis 接口 |
| 前端未触发分析 | edu_status.html | 检查failed/无状态时自动触发重新分析 |
| 双阶段failed处理 | edu_status.html | failed状态时触发重新分析 |

**技术细节**:

1. **等待界面简化**：
   - 移除 "正在生成AI分析报告" 文字
   - 移除轮询时的 `(attempts/maxAttempts)` 显示
   - 只显示旋转齿轮动画

2. **AI分析key修复**：
   ```python
   # 修改前（key不匹配）
   asyncio.create_task(analyze_with_llm(request.username, original_task_folder, stats))
   
   # 修改后（key匹配）
   asyncio.create_task(analyze_with_llm(request.username, save_key, stats))
   ```

3. **手动触发API**：
   - 支持 failed 状态重新触发
   - 自动更新状态为 pending

**相关文件**:
| 文件 | 修改内容 |
|------|----------|
| edu_status.html | 等待界面、failed处理、双阶段处理 |
| main.py | analyze_with_llm 参数 |
| routes_oridata.py | trigger-llm-analysis 接口 |

---

### 12.12 AI分析检测与双阶段报告修复

**更新内容**:

| 问题 | 修复位置 | 详细说明 |
|------|----------|----------|
| 已有报告仍超时 | edu_status.html | pollForLLMAnalysis开始时先检查是否completed |
| 双阶段已完成仍等待 | edu_status.html | 直接检查singleCompleted && assistCompleted |
| 超时后显示错误 | edu_status.html | 改为显示等待界面+重新触发按钮 |
| 按钮文字固定 | edu_status.html | 根据id后缀动态显示单独判别/AI辅助判别 |
| 管理员任务状态错误 | routes_oridata.py | 双阶段任务检查两个子任务是否都完成 |
| 管理员查看AI报告 | edu_admin.html | 双阶段报告显示AI评价内容 |

**相关文件**:
| 文件 | 修改内容 |
|------|----------|
| edu_status.html | 检测completed状态、超时处理、双阶段逻辑 |
| routes_oridata.py | task-status API支持双阶段检查 |
| edu_admin.html | 双阶段对比报告、显示AI评价 |

---

## 十三、协定的再次确认

作为编程助手，我承诺遵循以下协定：

1. **先理解再行动**：充分理解项目结构、用户期望和代码逻辑后再进行修改
2. **明确指出改动**：在修改代码前，先说明要改动的地方、上下文和原因
3. **不删除原有逻辑**：除非用户明确指出且的确有必要，否则不删除任何原有功能逻辑
4. **考察影响范围**：检查实现方式是否会影响其他功能的异常
5. **不明则问**：对变量的作用不明确时，必须搜索确认其用途
6. **保持文档同步**：所有更新必须同步更新本README.md文档
7. **遵循用户意图**：严格按照用户的具体要求实现功能，不自行添加额外逻辑

**本项目协定的核心原则**：
- 用户要求修改什么，就修改什么
- 不修改不必要的代码
- 修改前先计划，确认后再执行

---

*文档版本: 2026-04-06*
*最后更新: 2026-04-06 - 添加AI分析检测与双阶段报告修复*
