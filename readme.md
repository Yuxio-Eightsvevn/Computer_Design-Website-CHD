# 儿童先心病超声诊断平台 - README

---

## 一、协作协定

你是一个精通python、网页开发等编程技能的编程助手，善于接管大型项目，富有经验，代码风格简洁。在与用户的协作中，你必须先充分理解用户对大型项目的各种期望、以及大体的项目结构再进行行动。行动过程中必须记住用户的项目的代码结构，并了解其内大概是什么样的结构。在某些期望要求等方面不明确时，你会立刻停止任务并向用户询问要求；当对项目的任何地方不了解、或出现任何无法根据现有的信息解决的问题、或用户要求调试时，你也会停下，自行请求项目代码或用代码进行调试；在对用户的项目代码进行更改时，请先行指出要更改的项目结构，并对需要进行的改动进行调整，即使是在可读模式也不允许在未得到我的直接允许时直接对代码进行更改。必要时也必须明确指出要改动的地方以及其上下文。必要时采用md格式来进行说明，来使得说明美观。在不必要时，不允许对不明确绝对没用的代码逻辑进行更改。必要时采用md格式来进行说明，来使得说明美观。

必须单独强调的以下几点：
1.在你对代码计划进行更改时，请考察你的实现方式是否会影响其它功能的异常，并尽可能的不删除原有的任何逻辑，除非我明确指出且有必要。
2.在你对某些变量的作用不明确时，即使可以“望文生义”也不允许这样做，必须对代码进行搜索以明确其用途，并尽可能的不影响原有功能逻辑，除非的确有必要或我明确指出。
3.未经我的允许，不要对我的任何git状态管理以及相关文件进行修改，修改前请先再向我重复一遍才可执行。


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
1. **单独判读(Single)**: 仅显示原始视频，禁用heatmap/bbox模态按钮
2. **AI辅助判读(Assist)**: 显示完整模态（original/heatmap/bbox），需先完成单独判读才可解锁
3. **对比报告**: 显示两个阶段的详细对比分析

---

#### 12.8.1 系统架构

##### 工作流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     教育批次上传与发布流程                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 管理员上传ZIP包 (edu_admin.html)                              │
│     └── epoch_data.json + videos/                                │
│                      │                                           │
│                      ▼                                           │
│  2. 后端解析并重组目录 (routes_oridata.py)                         │
│     └── 标记 is_dual_stage = True                                │
│                      │                                           │
│                      ▼                                           │
│  3. AI模型处理                                                     │
│     └── 生成热力图、边界框、置信度                                │
│                      │                                           │
│                      ▼                                           │
│  4. 管理员发布任务 (routes_oridata.py)                            │
│     └── 设置 publish_mode = "dual"                               │
│                      │                                           │
│                      ▼                                           │
│  5. 用户开始判读 (edu_status.html → diagnosis.html)              │
│     ┌─────────────────────────────────────────────────────────┐  │
│     │ 阶段1: 单独判读 (_SINGLE)                                │  │
│     │   • 仅显示 original 视频                                 │  │
│     │   • 禁用 heatmap/bbox 按钮                              │  │
│     │   • 提交后保存为 taskId_SINGLE                          │  │
│     └─────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│     ┌─────────────────────────────────────────────────────────┐  │
│     │ 阶段2: AI辅助判读 (_AI-ASSIST)                           │  │
│     │   • 显示 original/heatmap/bbox                         │  │
│     │   • 需阶段1完成后才可进入                                 │  │
│     │   • 提交后保存为 taskId_AI-ASSIST                        │  │
│     └─────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

#### 12.8.2 数据结构

##### 任务索引 (SYSTEM/edu_data/data.json)

```json
{
  "tasks": [
    {
      "submission_id": "task123",
      "request_name": "测试批次",
      "is_dual_stage": true,
      "status": "published",
      "target_users": ["doctor1", "doctor2"],
      "request_pos": "processed/task123",
      "request_case_cnt": 5,
      "request_video_cnt": 15
    }
  ]
}
```

##### 用户成绩单 (SYSTEM/edu_data/Doctor_Diag_Result/{username}.json)

**当前实现** (使用后缀):

```json
{
  "task123_SINGLE": {
    "accuracy": 0.85,
    "sensitivity": 0.80,
    "specificity": 0.90,
    "edu_sub_mode": "single",
    "total_duration": 120.5,
    "category_stats": {...},
    "ai_dependency": {...}
  },
  "task123_AI-ASSIST": {
    "accuracy": 0.92,
    "sensitivity": 0.88,
    "specificity": 0.95,
    "edu_sub_mode": "assist",
    "total_duration": 95.3,
    "category_stats": {...},
    "ai_dependency": {...}
  }
}
```

**问题**:
- 任务ID被污染（包含 `_SINGLE` / `_AI-ASSIST` 后缀）
- 需要字符串操作提取父任务ID
- 代码中多处需要后缀处理逻辑
- 不易扩展第三阶段

---

#### 12.8.3 API接口详解

##### 发布任务

**接口**: `POST /api/edu/publish-task`

**参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| submission_id | string | 任务ID |
| target_users | string (JSON数组) | 发布对象用户列表 |
| publish_mode | string | "single" 或 "dual"，默认 "dual" |

**代码位置**: `routes_oridata.py:747`

```python
@router.post("/api/edu/publish-task")
async def publish_edu_task(
    submission_id: str = Form(...),
    target_users: str = Form(...),
    publish_mode: str = Form("dual")
):
    is_dual_stage = (publish_mode == "dual")
    update_edu_task_index({
        "submission_id": submission_id,
        "status": "published",
        "target_users": users,
        "is_dual_stage": is_dual_stage
    })
```

##### 提交诊断结果

**接口**: `POST /api/diagnosis/submit-json`

**请求体** (DiagnosisSubmitJsonRequest):
```typescript
{
  username: string,
  taskFolder: string,        // 任务文件夹
  mode: "diag" | "edu",     // 模式
  eduSubMode: "single" | "assist" | null,  // 教育子模式
  submittedAt: string,
  totalTime: object,
  patientCount: number,
  records: DiagnosisRecordSimple[]
}
```

**代码位置**: `main.py:842`

```python
@app.post("/api/diagnosis/submit-json")
async def submit_diagnosis_json(request: DiagnosisSubmitJsonRequest):
    if request.mode == "edu":
        # 1. 读取标准答案
        gt_path = Path(DATA_BATCH_STORAGE) / "SYSTEM/edu_data" / original_task_folder / "epoch_data.json"
        
        # 2. 计算统计数据
        stats = calculate_stats(request.records, ground_truth, ai_data)
        
        # 3. 保存成绩 (带后缀)
        save_key = request.taskFolder
        if request.eduSubMode == 'single':
            save_key = f"{request.taskFolder}_SINGLE"
        elif request.eduSubMode == 'assist':
            save_key = f"{request.taskFolder}_AI-ASSIST"
        
        user_data[save_key] = stats
        
        # 4. 触发AI分析
        asyncio.create_task(analyze_with_llm(request.username, save_key, stats))
```

##### 获取用户任务列表

**接口**: `GET /api/edu/user/tasks/{username}`

**返回结构**:
```json
{
  "tasks": [
    {
      "submission_id": "task123",
      "request_name": "测试批次",
      "is_dual_stage": true,
      "is_completed": false,
      "single_done": true,      // 双阶段: 单独判读是否完成
      "assist_done": false,      // 双阶段: AI辅助是否完成
      "edu_sub_mode": "dual",
      "last_score": 0.85
    }
  ]
}
```

**代码位置**: `routes_oridata.py:867`

##### 获取用户成绩

**接口**: `GET /api/edu/user/result/{username}/{taskId}`

**代码位置**: `routes_oridata.py:923`

```python
@router.get("/api/edu/user/result/{username}/{taskId}")
async def get_user_edu_result(username: str, taskId: str):
    # 处理双阶段任务：提取父任务ID
    parent_id = taskId
    if taskId.endswith('_SINGLE') or taskId.endswith('_AI-ASSIST'):
        parent_id = taskId.rsplit('_', 1)[0]
    
    # 读取用户成绩单
    result_file = EDU_RESULTS_DIR / f"{username}.json"
    return user_results.get(taskId)  # 按完整taskId查找
```

---

#### 12.8.4 前端实现

##### edu_status.html 任务卡片渲染

**代码位置**: `edu_status.html:193-216`

```javascript
if (isDualStage) {
    // 双阶段任务：两个垂直按钮
    div.innerHTML = `
        <button onclick="handleTask('${id}_SINGLE', ...)">
            ${singleDone ? '✅ 已完成' : '单独判读'}
        </button>
        <button ${!singleDone ? 'disabled' : ''} onclick="handleTask('${id}_AI-ASSIST', ...)">
            ${assistDone ? '✅ 已完成' : 'AI辅助判读'}
        </button>
    `;
} else {
    // 普通任务
    div.innerHTML = `<button onclick="handleTask('${id}', ...)">开始判读</button>`;
}
```

##### diagnosis.html 单独判读模式禁用

**代码位置**: `diagnosis.html:1636-1646`

```javascript
// 单独判读模式：禁用模态切换按钮
if (currentMode === 'edu' && eduSubMode === 'single') {
    const modalityGroup = document.querySelector('.diagnosis-group');
    const modalityButtons = modalityGroup.querySelectorAll('.option-btn');
    modalityButtons.forEach(btn => {
        btn.style.opacity = '0.5';
        btn.style.pointerEvents = 'none';
        btn.title = '单独判读模式不可切换';
    });
}
```

##### URL参数传递

```
# 单独判读
/diagnosis.html?mode=edu&taskId=xxx_SINGLE&eduSubMode=single

# AI辅助判读
/diagnosis.html?mode=edu&taskId=xxx_AI-ASSIST&eduSubMode=assist
```

---

#### 12.8.5 对比报告

双阶段任务完成后，显示**对比报告**，包含：

##### 指标对比表格

| 指标 | 单独判读 | AI辅助判读 | 差值 |
|------|----------|------------|------|
| 准确率 | 85.0% | 92.0% | +7.0% |
| 敏感度 | 80.0% | 88.0% | +8.0% |
| 特异性 | 90.0% | 95.0% | +5.0% |

##### AI依赖性分析

| 类型 | 说明 | 单独判读 | AI辅助 |
|------|------|----------|--------|
| 正确依赖 | AI正确+判读正确 | 12 | 18 |
| 依赖不足 | AI正确+判读错误 | 3 | 1 |
| 正确独立 | AI错误+判读正确 | 2 | 1 |
| 过度依赖 | AI错误+判读错误 | 3 | 0 |

##### 六维雷达图对比

使用统一六维指标进行叠加对比：
1. 准确率 (Accuracy)
2. 敏感度 (Sensitivity)
3. 特异性 (Specificity)
4. 病种均衡性 (Category Balance)
5. 诊断效率 (Diagnosis Efficiency)
6. 决策一致性 (Decision Consistency)

---

#### 12.8.6 关键代码索引

| 功能 | 文件 | 行号 |
|------|------|------|
| 发布任务，设置双阶段 | routes_oridata.py | 747-765 |
| 保存成绩（后缀逻辑） | main.py | 1032-1042 |
| 计算统计数据 | main.py | 866-1023 |
| 获取用户任务列表 | routes_oridata.py | 867-920 |
| 检查任务状态 | routes_oridata.py | 923-939 |
| 任务卡片渲染 | edu_status.html | 193-234 |
| 禁用模态按钮 | diagnosis.html | 1636-1646 |
| 提交诊断 | diagnosis.html | 2820-2850 |
| 对比报告渲染 | edu_status.html | 720-800 |

---

#### 12.8.7 相关文件

| 文件 | 职责 |
|------|------|
| main.py | 统计数据计算、成绩保存、AI分析触发 |
| routes_oridata.py | 任务发布、状态检查、成绩读取 |
| edu_admin.html | 发布任务界面 |
| edu_status.html | 任务列表、对比报告渲染 |
| diagnosis.html | 诊断界面、模态控制 |

---

### 12.16 三阶段教育系统重构计划 (Plan A - 嵌套结构)

**重构目标**: 将使用后缀的成绩存储改为嵌套结构，便于扩展第三阶段。

#### 12.16.1 新数据结构

**成绩单** (SYSTEM/edu_data/Doctor_Diag_Result/{username}.json)

```json
{
  "task123": {
    "stages": {
      "single": {
        "accuracy": 0.85,
        "sensitivity": 0.80,
        "specificity": 0.90,
        "edu_sub_mode": "single",
        "total_duration": 120.5,
        "category_stats": {...},
        "ai_dependency": {...},
        "completed_at": "2026-04-21T10:30:00"
      },
      "assist": {
        "accuracy": 0.92,
        "sensitivity": 0.88,
        "specificity": 0.95,
        "edu_sub_mode": "assist",
        "total_duration": 95.3,
        "category_stats": {...},
        "ai_dependency": {...},
        "completed_at": "2026-04-21T10:35:00"
      }
    },
    "llm_analysis": {
      "single": { "status": "completed", "report": "..." },
      "assist": { "status": "pending" }
    }
  }
}
```

#### 12.16.2 代码更改计划

##### Step 1: main.py - 保存逻辑重构

**文件**: `main.py`
**行号**: 1025-1045

**修改前**:
```python
save_key = request.taskFolder
if request.eduSubMode:
    if request.eduSubMode == 'single':
        save_key = f"{request.taskFolder}_SINGLE"
    elif request.eduSubMode == 'assist':
        save_key = f"{request.taskFolder}_AI-ASSIST"
user_data[save_key] = stats
```

**修改后**:
```python
parent_id = re.sub(r'_SINGLE|_AI-ASSIST$', '', request.taskFolder)
if parent_id not in user_data:
    user_data[parent_id] = {"stages": {}, "llm_analysis": {}}
if "stages" not in user_data[parent_id]:
    user_data[parent_id] = {"stages": {}, "llm_analysis": {}}
user_data[parent_id]["stages"][request.eduSubMode] = stats
```

##### Step 2: main.py - AI分析触发

**文件**: `main.py`
**行号**: 1049-1051

**修改后**:
```python
asyncio.create_task(analyze_with_llm(
    request.username,
    parent_id,  # 使用父任务ID
    request.eduSubMode,  # 传递阶段参数
    stats
))
```

##### Step 3: main.py - analyze_with_llm 函数

**文件**: `main.py`
**行号**: 1078

**函数签名修改**:
```python
async def analyze_with_llm(username: str, task_folder: str, stage: str, stats: Dict):
    # 修改保存逻辑
    user_data[parent_id]["llm_analysis"][stage] = {"status": "completed", "report": llm_report}
```

##### Step 4: routes_oridata.py - 读取逻辑

**文件**: `routes_oridata.py`
**行号**: 793-812

**修改后**:
```python
if is_dual_stage:
    parent_id = submission_id
    single_data = res_data.get(parent_id, {}).get("stages", {}).get("single")
    assist_data = res_data.get(parent_id, {}).get("stages", {}).get("assist")
    user_info["completed"] = single_data is not None and assist_data is not None
```

##### Step 5: routes_oridata.py - 获取用户任务

**文件**: `routes_oridata.py`
**行号**: 896-913

**修改后**:
```python
if is_dual:
    single_id = f"{sub_id}_SINGLE"  # 仍用于URL跳转
    assist_id = f"{sub_id}_AI-ASSIST"
    single_done = single_id in user_results
    assist_done = assist_id in user_results
    # 但保存时使用嵌套结构
```

##### Step 6: edu_status.html - 任务列表

**文件**: `edu_status.html`
**行号**: 193-216

**URL保持不变**（用于传递阶段信息）:
```javascript
handleTask('${t.submission_id}_SINGLE', '${t.request_name}_SINGLE', singleDone, 'single', '${t.submission_id}')
```

##### Step 7: edu_status.html - 成绩读取

**文件**: `edu_status.html`
**行号**: 294-302

**修改后**:
```javascript
const rSingle = await fetch(`/api/edu/user/result/${username}/${taskParentId}?stage=single`);
const rAssist = await fetch(`/api/edu/user/result/${username}/${taskParentId}?stage=assist`);
```

##### Step 8: diagnosis.html - 提交逻辑

**文件**: `diagnosis.html`
**行号**: 2830-2835

**保持不变**，因为 `eduSubMode` 已经在 URL 参数中

---

#### 12.16.3 API更改

##### 新增: 获取指定阶段成绩

**接口**: `GET /api/edu/user/result/{username}/{taskId}?stage={stage}`

**返回**:
```json
{
  "stage": "single",
  "accuracy": 0.85,
  "sensitivity": 0.80,
  ...
}
```

---

#### 12.16.4 影响范围

| 文件 | 改动量 | 说明 |
|------|--------|------|
| main.py | 中 | 保存逻辑、AI分析函数 |
| routes_oridata.py | 小 | 读取逻辑 |
| edu_status.html | 小 | 成绩读取 |
| diagnosis.html | 无 | 无需改动 |
| edu_admin.html | 小 | 管理员查看报告 |

---

#### 12.16.5 向后兼容

**已有数据迁移**: 可以写一个一次性脚本将旧格式转换为新格式

```python
# 迁移脚本逻辑
for username in users:
    old_data = read_json(f"{username}.json.old")
    new_data = {}
    for key, value in old_data.items():
        if key.endswith("_SINGLE"):
            parent = key.replace("_SINGLE", "")
            new_data[parent]["stages"]["single"] = value
        elif key.endswith("_AI-ASSIST"):
            parent = key.replace("_AI-ASSIST", "")
            new_data[parent]["stages"]["assist"] = value
        else:
            new_data[key] = value
    write_json(f"{username}.json", new_data)
```

---

#### 12.16.6 第三阶段扩展

新结构便于添加第三阶段，只需：

1. 在 `publish_mode` 添加 `"triple"` 选项
2. 在 `stages` 中添加 `"review"` 阶段
3. 前端添加第三个按钮

```json
{
  "task123": {
    "stages": {
      "single": {...},
      "assist": {...},
      "review": {...}  // 第三阶段
    }
  }
}
```

**相关文件**:
| 文件 | 修改内容 |
|------|----------|
| flow.html | 卡片、按钮、字体尺寸全面放大，按钮hover效果 |
| diagnosis.html | 置信度标签、临床判断按钮、option-btn样式 |
| edu_status.html | 导航栏、任务卡片、按钮、侧边栏、指标标签，按钮hover效果 |
| edu_admin.html | 按钮hover效果 |
| task_status.html | 按钮hover效果 |
| dashboard.html | 按钮hover效果 |
| admin.html | 按钮hover效果 |
| login.html | 登录按钮hover效果，介绍文字间距 |

**按钮hover效果已应用到的页面**:
- admin.html ✅
- dashboard.html ✅
- edu_admin.html ✅
- edu_status.html ✅
- flow.html ✅
- task_status.html ✅
- login.html (login-btn-title, login-btn) ✅

---

### 12.14 login.html 登录页面布局重构

**更新内容**:

应用户要求，对登录页面进行了全新布局设计：

#### 新布局结构

```
┌──────────────────────────────────────────────────────────────┐
│  Header (蓝色, 95%宽, 圆角30px, 固定高度60px)                  │
│  [header.title]                                             │
├──────────────────────────────────────────────────────────────┤
│  TitleField (90%宽, max-width:1440px, 白色背景, 圆角16px)    │
│  ┌────────────────────────┬───────────────────────────────┐ │
│  │    左栏(60%): 图片      │  右栏(40%):                    │ │
│  │   titleField.image     │  • welcomeText (淡色)         │ │
│  │   (max-height: 390px)  │  • mainTitle (50px粗体)       │ │
│  │                        │  • subtitle (蓝色)           │ │
│  │                        │  • description (介绍文本)      │ │
│  │                        │  • loginButtonText            │ │
│  └────────────────────────┴───────────────────────────────┘ │
│                                                                │
│  Keywords (横向排列, 圆角药片样式)                              │
├──────────────────────────────────────────────────────────────┤
│  MainContent (90%宽, max-width:1440px)                       │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  轮播图 (100%宽, 高度455px, 圆角16px)                    ││
│  └──────────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────────┐│
│  │  核心功能 (3列白色卡片)                                  ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

#### login_config.json 新结构

```json
{
  "header": {
    "title": "CHD-AIDE"
  },
  "titleField": {
    "image": "res/admin/images/pic_01.png",
    "welcomeText": "欢迎来到CHD-AIDE平台官方网站",
    "mainTitle": "CHD-AIDE",
    "subtitle": "推理辅助与教育平台",
    "description": "CHD-AIDE是一款专注于儿童先天性心脏病诊断...",
    "loginButtonText": "点击登录"
  },
  "keywords": [
    { "text": "精准诊断" },
    { "text": "智能辅助" }
  ],
  "background": "",
  "carousel": { "enabled": true },
  "features": [...]
}
```

#### 配置说明

| 字段 | 说明 | 可为空 |
|------|------|--------|
| header.title | Header区域显示的标题文字 | 否 |
| titleField.image | 左侧图片路径 | 可(使用默认图) |
| titleField.welcomeText | 欢迎文本(淡色) | 否 |
| titleField.mainTitle | 主标题(深色粗体) | 否 |
| titleField.subtitle | 副标题(蓝色) | 否 |
| titleField.description | 介绍文本 | 否 |
| titleField.loginButtonText | 登录按钮文字 | 否 |
| keywords | 关键词列表(横向排列) | 可(留空则不显示) |
| background | 背景图片路径 | 可(留空则不显示背景) |
| carousel.enabled | 是否显示轮播图 | 可(默认true) |
| features | 核心功能列表 | 否 |

#### 设计规范

| 属性 | 值 |
|------|-----|
| 最大内容宽度 | 1440px (增加20%) |
| Header高度 | 60px |
| Header圆角 | 30px |
| TitleField内边距 | 52px (增加30%) |
| TitleField圆角 | 16px |
| TitleField图片最大高度 | 390px (增加30%) |
| 轮播图高度 | 455px (增加30%) |
| 轮播图图片最小高度 | 85% |
| 欢迎文本字体 | 16px, letter-spacing: 1px |
| 主标题字体 | 50px, 800 weight, letter-spacing: 4px |
| 副标题字体 | 20px, letter-spacing: 2px |
| 介绍文本字体 | 15px, line-height: 2, letter-spacing: 2px, text-justify: inter-character |
| 按钮内边距 | 18px 50px |
| 按钮圆角 | 30px |
| 按钮字体 | 18px |
| 关键词样式 | 等间距排列，无卡片/分隔 |
| 轮播图图片 | object-fit: contain 完整显示 |

**相关文件**:
| 文件 | 修改内容 |
|------|----------|
| login.html | 全新布局结构、CSS样式、JS渲染逻辑，介绍文字间距，轮播图完整显示 |
| login_config.json | 重构JSON结构，支持独立配置 |

---

### 12.15 按钮hover效果与UI细节修复 (2026-04-21)

**更新内容**:

#### A. 按钮hover效果
为以下页面添加统一按钮hover效果：`filter: brightness(1.1)` + `transform: translateY(-1px)`

| 页面 | 状态 |
|------|------|
| admin.html | ✅ 已添加 |
| dashboard.html | ✅ 已添加 |
| edu_admin.html | ✅ 已添加 |
| edu_status.html | ✅ 已添加 |
| flow.html | ✅ 已添加 |
| task_status.html | ✅ 已添加 |
| login.html | ✅ 已添加 (login-btn-title, login-btn) |

#### B. login.html介绍文字间距
- 添加 `text-justify: inter-character` 支持中文文字间距
- 添加 `text-align: justify` 两端对齐

#### C. login.html轮播图完整显示
- 改用 `object-fit: contain` 确保图片完整显示，允许留白

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

### 12.17 三阶段教育系统重构实施 (2026-04-21)

**执行时间**: 2026-04-21

#### 实施内容

##### 1. main.py - 保存逻辑重构

**文件**: `main.py`
**行号**: ~1025-1050

**更改**:
- 使用嵌套结构保存成绩: `user_data[parent_id]["stages"][eduSubMode] = stats`
- 提取父任务ID时去掉 `_SINGLE` / `_AI-ASSIST` 后缀
- 增加 `completed_at` 时间戳

##### 2. main.py - analyze_with_llm 函数

**文件**: `main.py`
**行号**: ~1081

**更改**:
- 函数签名从 `analyze_with_llm(username, task_folder, stats)` 改为 `analyze_with_llm(username, parent_id, stage, stats)`
- 保存时使用嵌套结构: `user_data[parent_id]["llm_analysis"][stage]`

##### 3. routes_oridata.py - 管理员任务状态

**文件**: `routes_oridata.py`
**行号**: ~793

**更改**:
- 使用 `res_data.get(submission_id, {}).get("stages", {}).get("single")` 替代原来的 `res_data[single_key]`

##### 4. routes_oridata.py - 用户任务列表

**文件**: `routes_oridata.py`
**行号**: ~897

**更改**:
- 使用嵌套结构检查完成状态
- 获取分数使用 `stages["single"].get("accuracy")`

##### 5. routes_oridata.py - API支持新旧格式

**文件**: `routes_oridata.py`
**行号**: ~952, ~961

**更改**:
- `get_user_edu_result` API: 自动识别新旧格式
- `trigger_llm_analysis` API: 自动识别新旧格式

#### 兼容性

**向后兼容**: 前端代码无需修改，后端API自动兼容新旧两种格式：
- 新格式: `user_data[taskId]["stages"]["single"]`
- 旧格式: `user_data[taskId_SINGLE]`

---

### 12.18 删除任务时自动清理用户成绩 (2026-04-21)

**问题描述**：
- `delete_edu_task` 函数删除任务时，只删除了物理文件和索引，没有清理用户的成绩记录
- `unpublish-task` 函数只处理旧格式后缀结构，没有处理新格式嵌套结构

**修复内容**：

#### 1. delete_edu_task 函数 (routes_oridata.py:1114-1134)

**新增逻辑**：删除任务时同步清理所有用户的成绩记录

```python
# 3. 删除所有用户的成绩记录
target_users = task.get("target_users", [])
for username in target_users:
    result_file = EDU_RESULTS_DIR / f"{username}.json"
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            user_data = json.load(f)
        # 删除新格式嵌套结构（父键）
        if submission_id in user_data:
            del user_data[submission_id]
        # 删除旧格式后缀结构
        if f"{submission_id}_SINGLE" in user_data:
            del user_data[f"{submission_id}_SINGLE"]
        if f"{submission_id}_AI-ASSIST" in user_data:
            del user_data[f"{submission_id}_AI-ASSIST"]
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
```

#### 2. unpublish-task 函数 (routes_oridata.py:853-860)

**修复逻辑**：增加对新格式嵌套结构的删除支持

```python
# 新格式：直接删除父键（包含stages嵌套结构）
if submission_id in user_data:
    del user_data[submission_id]
# 旧格式后缀结构（兼容旧数据）
if f"{submission_id}_SINGLE" in user_data:
    del user_data[f"{submission_id}_SINGLE"]
if f"{submission_id}_AI-ASSIST" in user_data:
    del user_data[f"{submission_id}_AI-ASSIST"]
```

**相关文件**：
| 文件 | 修改内容 |
|------|----------|
| routes_oridata.py | delete_edu_task 增加删除用户成绩逻辑 |
| routes_oridata.py | unpublish-task 增加新格式删除支持 |

---

### 12.19 三阶段教育系统实现 (2026-04-21)

**更新内容**：

#### 系统架构

三阶段教育模式新增"复核判读"阶段：

| 阶段 | 后缀 | 说明 | UI限制 |
|------|------|------|--------|
| 第一阶段 | `_SINGLE` | 单独判读 | 仅显示原视频，禁用heatmap/bbox |
| 第二阶段 | `_AI-ASSIST` | AI辅助判读 | 显示所有模态 |
| 第三阶段 | `_REVIEW` | 复核判读 | 仅显示原视频，禁用heatmap/bbox |

**publish_mode 枚举**: `"single"` | `"dual"` | `"triple"`

#### 核心逻辑变更

##### 1. LLM分析触发时机 (main.py)

**修改前**：每个阶段完成后立即触发LLM分析

**修改后**：仅当三阶段全部完成后，触发一次综合分析（对比第一阶段和第三阶段）

```python
# 三阶段都完成时，触发综合分析
if is_triple and all_stages_done:
    asyncio.create_task(analyze_with_llm(
        username, parent_id, "triple",
        {"single": stages["single"], "assist": stages["assist"], "review": stages["review"]}
    ))
```

##### 2. LLM Prompt变更 (llm_analyzer.py)

三阶段分析时，prompt包含：
- 三个阶段的完整指标对比
- 第一阶段 vs 第三阶段详细对比
- 复核能力提升分析

##### 3. 后端修改 (routes_oridata.py)

| 函数 | 修改内容 |
|------|----------|
| publish-task | 支持 `publish_mode="triple"`，设置 `edu_sub_mode="triple"` |
| get_user_edu_tasks | 增加三阶段完成状态判断 |
| check_task_status | 增加 `_REVIEW` 后缀处理 |
| get_user_edu_result | 增加 `_REVIEW` 后缀处理 |
| trigger_llm_analysis | 增加 `_REVIEW` 后缀处理 |
| unpublish-task | 增加 `_REVIEW` 删除支持 |

##### 4. 前端修改 (edu_status.html)

| 修改点 | 内容 |
|--------|------|
| 任务列表渲染 | 三阶段任务显示三个垂直按钮 |
| 按钮解锁逻辑 | 阶段2需阶段1完成，阶段3需阶段2完成 |
| openResultModal | 增加 `_REVIEW` 后缀识别 |
| rerunTask | 增加 `_REVIEW` 后缀处理 |

##### 5. 前端修改 (edu_admin.html)

- 发布模式改为下拉选择：单阶段/双阶段/三阶段
- 动态显示发布成功消息

##### 6. 前端修改 (diagnosis.html)

- `_REVIEW` 后缀处理
- 复核判读模式与单独判读一致，禁用heatmap/bbox

#### 数据结构

```json
{
  "taskId": {
    "stages": {
      "single": { ... },
      "assist": { ... },
      "review": { ... }
    },
    "llm_analysis": {
      "triple": {
        "status": "completed",
        "report": { ... },
        "analyzed_at": "..."
      }
    }
  }
}
```

#### 相关文件

| 文件 | 修改内容 |
|------|----------|
| routes_oridata.py | publish-task三阶段支持、任务状态处理 |
| main.py | 正则表达式、LLM触发逻辑 |
| llm_analyzer.py | 三阶段prompt构造 |
| edu_admin.html | 发布模式下拉选择 |
| edu_status.html | 三按钮渲染、报告展示 |
| diagnosis.html | REVIEW后缀处理、UI限制 |

---

### 12.20 三阶段报告UI优化与置信度隐藏 (2026-04-21)

**更新内容**：

#### 1. 三阶段雷达图间距调整 (edu_status.html)

增大三个雷达图之间的间距，防止重叠：

| 属性 | 修改前 | 修改后 |
|------|--------|--------|
| gap | 25px | 50px |
| canvas尺寸 | 110x110 | 130x130 |

#### 2. 教育模式置信度隐藏 (diagnosis.html)

**修改原因**：在AI辅助判读阶段，医师不应依赖模型置信度进行诊断，应独立思考后再参考AI结果。

**修改内容**：
- 教育模式下所有阶段（single/assist/review）均隐藏模型置信度
- 仅在判读模式（非教育模式）下显示置信度

**代码位置**：`diagnosis.html:1856-1862`

```javascript
// 修改前：仅 single 和 review 隐藏置信度
if (currentMode === 'edu' && (eduSubMode === 'single' || eduSubMode === 'review'))

// 修改后：所有教育阶段都隐藏置信度
if (currentMode === 'edu')
```

---

### 12.21 三阶段报告重新练习按钮修复 (2026-04-21)

**问题**：在 edu_status 的三阶段报告界面中，没有重新进入三个阶段判读的按钮。

**修复内容**：

#### 1. 添加复核判读按钮 (edu_status.html:88)

新增 `btn-re-run-review` 按钮：

```html
<button class="btn btn-start" style="flex:1; display:none;" id="btn-re-run-review">重新进行复核判读</button>
```

#### 2. 修改 showTripleStageReport 函数 (edu_status.html:1248-1255)

**修改前**：隐藏所有重新练习按钮

**修改后**：显示三个阶段对应的重新练习按钮

```javascript
// 显示三阶段重新练习按钮
document.getElementById('btn-re-run').style.display = 'none';
document.getElementById('btn-re-run-single').style.display = '';
document.getElementById('btn-re-run-assist').style.display = '';
document.getElementById('btn-re-run-review').style.display = '';
document.getElementById('btn-re-run-single').onclick = () => rerunTask(parentId + '_SINGLE', name);
document.getElementById('btn-re-run-assist').onclick = () => rerunTask(parentId + '_AI-ASSIST', name);
document.getElementById('btn-re-run-review').onclick = () => rerunTask(parentId + '_REVIEW', name);
```

---

### 12.22 二阶段/三阶段日志区分修复 (2026-04-22)

**问题**：二阶段任务中，日志错误地显示"三阶段完成状态: False"，导致用户困惑。

**修复内容** (main.py:1071)：

```python
# 修改前
print(f"🔔 当前阶段: {request.eduSubMode}，三阶段完成状态: {all_stages_done}，跳过LLM分析")

# 修改后
print(f"🔔 当前阶段: {request.eduSubMode}，跳过LLM分析（仅在三阶段全部完成后触发综合分析）")
```

---

*文档版本: 2026-04-22*
*最后更新: 2026-04-22 - 清空任务时间戳检测修复*

### 12.23 清空任务时间戳检测修复 (2026-04-22)

**问题**：点击"原视频"或"病灶定位框"按钮时，视频只会闪烁并保持原模态。

**原因**：HTML按钮文本（原视频/病灶定位框）与 selectOption 函数中的判断条件（普通/模型识别框）不匹配。

**修复内容** (diagnosis.html:2478-2483):

`javascript
// 修改前
if (modalityText === '普通') { ... }
else if (modalityText === '模型识别框') { ... }

// 修改后
if (modalityText === '原视频') { ... }
else if (modalityText === '病灶定位框') { ... }
`

---
### 12.24 清空任务逻辑增强：基于时间戳判断系统中断残留 (2026-04-22)

**问题**：系统关闭时，AI 推理可能仍在执行并完成，但后续清理代码未执行，导致 `.processing` 文件残留。

**修复内容** (`routes_oridata.py`):

| 位置 | 修改 |
|------|------|
| `clear_stuck_tasks` (第549-565行) | 增加 `.processing` 文件时间戳检查 |
| `clear_stuck_edu_tasks` (第1239-1255行) | 增加 `.processing` 文件时间戳检查 |

**新逻辑**：

```python
if processing_flag.exists():
    age_seconds = time.time() - processing_flag.stat().st_mtime
    if age_seconds > 1800:  # 超过30分钟
        # 判定为系统中断残留，执行清理
    else:
        # 正在运行，保留
        continue
```

**效果**：

| 场景 | 修改前 | 修改后 |
|------|--------|--------|
| 正常在运行 | 保留 | 保留（文件较新） |
| 系统中断残留 | 保留 | 删除（文件超30分钟） |

---
### 12.25 login.html 核心功能栏 CSS Grid 自适应布局 (2026-04-22)

**问题**：核心功能栏使用 `flex` + 硬编码 `width: 33.33%`，只能固定3列，无法自适应配置项数量。

**修复内容** (`login.html:234-248`):

```css
/* 修改前 */
.features-row { display: flex; gap: 20px; }
.feature-item { flex: 1; width: 33.33%; }

/* 修改后 */
.features-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
}
@media (max-width: 1200px) { .features-row { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 900px)  { .features-row { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px)  { .features-row { grid-template-columns: 1fr; } }
```

**效果**：最多4列自适应，屏幕收窄时自动减少列数。

---
### 12.26 用户头像配置化 (2026-04-23)

**功能**：支持通过配置文件为指定用户设置自定义头像图标。

**新增文件**：`/UI/res/share/user_avatar_config.json`

```json
{
  "avatars": {
    "admin": "res/share/icon/admin.png",
    "TestAccount": "res/share/icon/TestAccount.png"
  }
}
```

**修改的文件**：

| 文件 | 修改内容 |
|------|---------|
| `flow.html` | 添加 `loadAvatarConfig()`、`setUserAvatar()` 函数 |
| `edu_status.html` | 添加 `loadAvatarConfig()`、`setUserAvatar()` 函数 |
| `dashboard.html` | 添加 `loadAvatarConfig()`、`setUserAvatar()` 函数 |
| `task_status.html` | 添加 `loadAvatarConfig()`、`setUserAvatar()` 函数 |

**逻辑**：
1. 页面加载时调用 `loadAvatarConfig()` 读取配置
2. 设置头像时调用 `setUserAvatar(elem, username)`
3. 若用户在配置文件中存在 → 显示图标图片
4. 若用户不在配置中 → 回退显示首字母

**向后兼容**：配置文件中未收录的用户仍显示首字母头像。

---
### 12.27 用户显示名从 username 改为 doctor (2026-04-23)

**功能**：界面显示的用户名称从登录用户名（username）改为医师姓名（doctor）。

**修改的文件**：

| 文件 | 行号 | 修改内容 |
|------|------|---------|
| `flow.html` | 232, 245-250 | 提取 `doctor` 字段，显示医师姓名 |
| `edu_status.html` | 140 | 显示 `doctor \|\| username` |
| `dashboard.html` | 207 | 显示 `doctor \|\| username` |
| `task_status.html` | 202 | 显示 `doctor \|\| username` |
| `diagnosis.html` | 1540 | 显示 `doctor \|\| username` |

**注意**：头像设置仍使用 `username`（作为用户标识）。

---
### 12.28 LLM分析触发逻辑修复与提示词约束增强 (2026-04-23)

**问题1**：三阶段任务在二阶段完成时错误触发二阶段对比分析

**原因**：`main.py:1050` 使用 `request.eduSubMode == "review"` 判断是否为三阶段，但完成第二阶段时 `eduSubMode` 是 `assist` 而非 `review`，导致 `is_triple=False`，错误进入二阶段分支。

**修复** (`main.py:1049-1058`)：从任务发布时的 `edu_sub_mode` 判断任务类型，而非当前提交的阶段。

```python
# 从任务发布时的edu_sub_mode判断
task_info_path = Path(DATA_BATCH_STORAGE) / "SYSTEM" / "edu_data" / "data.json"
is_task_triple = False
if task_info_path.exists():
    with open(task_info_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)
        for t in task_data.get("tasks", []):
            if t.get("submission_id") == request.taskFolder:
                is_task_triple = (t.get("edu_sub_mode") == "triple")
                break
```

---

**问题2**：AI评价与数据事实不符（如复核准确率下降却评价为"有所提升"）

**修复** (`llm_analyzer.py`)：在三阶段和二阶段提示词中增加【重要约束】和【数据核实】步骤：

```python
【重要约束 - 请务必遵守】
1. 你必须严格依照上述数据进行分析，禁止臆测或推理未给出的信息
2. 如果数据显示准确率下降，你必须如实描述下降，不能说"有所提升"
3. 如果数据显示准确率为0%或100%，你必须如实反映，不能暗示任何隐藏信息
4. 所有评价结论必须直接来源于给出的数据，禁止添加数据中不存在的信息
5. 对于数据中的每一个数值变化，必须明确指出是升高还是降低，并给出具体数值

请从以下几个方面给出评价：
0. **数据核实** - 在给出任何分析前，先核实数据：
   - 第一阶段准确率: XX%
   - 第二/第三阶段准确率: XX%
   - 相比首次的准确率变化: +/-XX%（必须如实反映）
1. 整体表现评价 - 必须如实反映数据中的准确率是升是降
...
```

---
---

## 📋 待修复问题 (Known Issues)

### [TODO] 后端接受返回绑定显示名而非内部名的 Bug

**问题描述**：系统存在前后端标签名不一致的问题：
- **前端显示**：`正常`、`VSD`、`ASD`、`PDA`
- **后端期望**：内部名 `Normal`、`VSD`、`ASD`、`PDA`
- **已修复**：当前 `LABEL_MAP` 已临时改为 `"正常": 0` 等，临时兼容前端

**根本修复方案（待实施）**：

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A. 前后端统一用内部名 | 前端按钮文字改为 `Normal` 等 | 后端无需改 | 用户界面显示英文 |
| B. 后端全面支持显示名 | 后端同时支持 `正常`/`Normal` | 用户体验好 | 需要改多处代码 |
| C. 建立映射层 | 前端传内部名，后端转显示名 | 最规范 | 改动较大 |

**建议**：采用方案 A（统一用内部名），对用户更专业，且避免乱码风险。

**涉及文件**：
- `UI/diagnosis.html` - 诊断按钮文字（第1172-1175行）
- `main.py` - `LABEL_MAP`/`LABEL_REVERSE_MAP`（第850-851行）

---
