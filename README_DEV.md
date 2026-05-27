# 儿童先心病超声诊断平台 - 开发文档

> 本文档为开发者提供项目的完整技术架构参考。

---

## 一、项目概述

### 1.1 项目定位

本项目是一个**儿童先天性心脏病（先心病）超声诊断辅助平台**，基于 FastAPI 构建，支持 AI 辅助诊断和医学教育培训。

**核心能力**：
- 对心脏超声视频进行 AI 分析，生成热力图、边界框、关键帧标注
- 支持单视角和多视角诊断，输出四分类置信度（Normal/VSD/ASD/PDA）
- 提供医学教育培训模式，支持双阶段判读（单独判读 + AI辅助判读）

### 1.2 诊断类别

| 类别 | 说明 |
|------|------|
| Normal | 正常 |
| VSD | 室间隔缺损（Ventricular Septal Defect） |
| ASD | 房间隔缺损（Atrial Septal Defect） |
| PDA | 动脉导管未闭（Patent Ductus Arteriosus） |

### 1.3 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + uvicorn |
| 数据库 | SQLite |
| AI模型 | PyTorch (C3D 双Token时空网络) |
| 前端 | 原生 HTML/CSS/JavaScript |
| 视频处理 | OpenCV + ffmpeg |
| 大模型 | 支持智谱AI/OpenAI/DeepSeek/Moonshot |

### 1.4 系统三大模式

| 模式 | 说明 | 主要页面 |
|------|------|----------|
| **诊断模式** | 用户上传视频 → AI处理 → 医师判读 | dashboard.html → task_status.html → diagnosis.html |
| **教育模式** | 管理员发布任务 → 多阶段学习 → 成绩统计 | edu_admin.html → edu_status.html → diagnosis.html |
| **管理后台** | 用户管理、任务管理 | admin.html, edu_admin.html |

---

## 二、目录结构

```
Diagnostic_Platform_Stable_Version_C3D/
│
├── main.py                        # FastAPI主入口 (1629行)
├── routes_oridata.py              # 核心路由模块 (1399行)
├── database.py                    # 数据库操作 (299行)
├── llm_analyzer.py                # 大模型分析器 (485行)
├── users.db                       # SQLite数据库文件
│
├── config/                        # 配置文件目录
│   ├── llm_config.json            # 旧版LLM配置
│   ├── llm_models.json            # 新版LLM配置(多模型)
│   └── LLM_MODELS_GUIDE.md       # LLM配置指南
│
├── model/                         # AI模型目录
│   ├── model_main.py              # 模型入口函数
│   ├── heart_diagnosis.py         # 核心诊断逻辑 (513行)
│   │
│   ├── singleview/               # 单视角模型权重
│   │   ├── spatial_size_4.pth     # 4×4网格模型
│   │   ├── spatial_size_5.pth     # 5×5网格模型
│   │   ├── spatial_size_6.pth     # 6×6网格模型
│   │   └── spatial_size_7.pth     # 7×7网格模型
│   │
│   ├── multiviews/               # 多视角模型权重
│   │   └── epoch_97_*.pth        # 多视角融合模型
│   │
│   ├── Baseline_4size/            # 基线特征
│   │
│   └── Codes/                     # 模型代码
│       ├── Nets/                  # 神经网络结构
│       │   ├── dual_tokens_net.py             # 单视角网络 (302行)
│       │   └── Multi_Views_dual_tokens_net.py # 多视角融合网络 (249行)
│       │
│       └── main_codes/            # 工具函数
│           ├── utils.py                        # 基础工具 (365行)
│           ├── Confidence_scores.py            # 置信度计算 (548行)
│           ├── original_ref.py                 # 原始参考函数 (219行)
│           ├── Trail_Indicator_valid_XAI.py    # XAI可视化 (214行)
│           ├── test_heatmap_copy.py            # 热图测试
│           └── Trail_Indicator_XAI_FusionMap_results_Web_0326.py
│
├── UI/                            # 前端页面目录
│   ├── login.html                 # 登录页面
│   ├── flow.html                  # 模式选择主页
│   ├── admin.html                 # 用户管理后台 (管理员)
│   ├── dashboard.html             # 任务上传页面
│   ├── task_status.html           # 任务进度页面
│   ├── diagnosis.html             # 医师判读界面
│   ├── edu_admin.html             # 教育管理后台 (管理员)
│   └── edu_status.html            # 教育练习中心 (用户端)
│
└── data_batch_storage/            # 数据存储根目录
    │
    ├── {username}/                # 用户私有空间
    │   ├── oridata/              # 原始上传数据
    │   │   └── {submission_id}/  # 任务ID目录
    │   │       ├── .processing   # 处理中标记
    │   │       ├── .failed       # 失败标记
    │   │       ├── case1/        # 病例文件夹
    │   │       │   ├── video1.mp4
    │   │       │   └── video2.mp4
    │   │       └── data.json     # 任务索引
    │   │
    │   └── processed/            # AI处理结果
    │       └── {submission_id}/  # 任务ID目录
    │           ├── case1/
    │           │   ├── output_videos/
    │           │   │   ├── video1_original.mp4
    │           │   │   ├── video1_heatmap.mp4
    │           │   │   ├── video1_bbox.mp4
    │           │   │   └── video1_keyframes.mp4
    │           │   ├── output_data/
    │           │   │   ├── confidence_scores.json
    │           │   │   ├── video1.json
    │           │   │   └── ...
    │           │   └── output_data.zip
    │           └── case2/
    │               └── ...
    │
    └── SYSTEM/                    # 系统保留空间
        └── edu_data/              # 教育模式数据
            ├── {submission_id}/   # 教育批次
            │   ├── epoch_data.json     # 标准答案
            │   ├── videos/             # 原始视频
            │   └── ...
            ├── processed/              # AI处理结果
            │   └── {submission_id}/
            │       └── ...
            ├── Doctor_Diag_Result/     # 用户成绩单
            │   ├── doctor1.json
            │   └── doctor2.json
            └── data.json               # 教育任务索引
```

---

## 三、核心模块关系图

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
│      ├── 判读模式: /api/users/{username}/upload-oridata            │
│      ├── 教育模式: /api/edu/*                                        │
│      └── 后台任务: run_model_inference_wrapper                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          AI模型层                                    │
├─────────────────────────────────────────────────────────────────────┤
│  model_main.py                                                       │
│      └── run_diagnosis() ──→ heart_diagnosis.py                     │
│                                  ├── 单视角推理 (4个模型)            │
│                                  │     → 热力图/边界框/关键帧        │
│                                  └── 多视角推理 (置信度)             │
│                                        → {Normal, VSD, ASD, PDA}   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、数据流向

### 4.1 诊断模式数据流

```
1. 上传阶段
   dashboard.html
       │
       ├── 用户选择/拖拽视频文件
       ├── 系统自动解析病例分组
       └── POST /api/users/{username}/upload-oridata
                │
                ├── 保存到: data_batch_storage/{user}/oridata/{id}/
                ├── 更新索引: {user}/data.json
                └── 触发后台任务: run_model_inference_wrapper

2. AI处理阶段
   routes_oridata.py:run_model_inference_wrapper
       │
       └── run_diagnosis(target_dir, output_dir)
                │
                ├── 单视角推理: 生成热力图/边界框/关键帧
                ├── 多视角推理: 生成置信度分数
                ├── 视频转码: ffmpeg H.264
                └── 输出到: {user}/processed/{id}/

3. 判读阶段
   diagnosis.html
       │
       ├── GET /api/tasks/{user}/{folder}/patients
       ├── 显示病例列表和视频
       ├── 医师查看视频 + AI置信度
       ├── 医师录入诊断结论
       └── POST /api/diagnosis/submit-json (mode="diag")
                │
                └── 保存到: {user}/processed/{id}/
                            final_diagnosis_report_{timestamp}.json
```

### 4.2 教育模式数据流

```
1. 管理员创建批次
   edu_admin.html
       │
       ├── 上传ZIP (含epoch_data.json标准答案)
       ├── POST /api/edu/parse-zip (解析统计)
       └── POST /api/edu/confirm-upload (确认上传)
                │
                ├── 保存到: SYSTEM/edu_data/{submission_id}/
                ├── 重组目录结构适配AI模型
                ├── 触发AI处理
                └── 状态: processing → unreleased

2. 管理员发布任务
   edu_admin.html
       │
       ├── 选择目标用户
       └── POST /api/edu/publish-task
                │
                └── 状态: unreleased → published

3. 用户进行判读练习
   edu_status.html → diagnosis.html
       │
       ├── GET /api/edu/user/tasks/{username}
       ├── 点击"开始判读" → diagnosis.html?mode=edu&taskId=xxx
       ├── 用户录入诊断结论
       └── POST /api/diagnosis/submit-json (mode="edu")
                │
                ├── 读取标准答案: SYSTEM/edu_data/{id}/epoch_data.json
                ├── 计算准确率/敏感度/特异性
                └── 保存成绩单: SYSTEM/edu_data/Doctor_Diag_Result/{user}.json

4. 管理员查看进度
   edu_admin.html
       │
       └── GET /api/edu/admin/task-status/{submission_id}
```

---

## 五、数据库设计

### 5.1 用户表 (users)

**位置**: `database.py`

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,           -- SHA-256哈希
    doctor TEXT NOT NULL,              -- 医生姓名
    organization TEXT NOT NULL,       -- 机构名称
    is_admin BOOLEAN DEFAULT 0,       -- 是否管理员
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### 5.2 SYSTEM保留名保护

**位置**: `database.py:191-193, 271-272`

```python
# create_user() 中
if username.upper() == "SYSTEM":
    print(f"🚫 拦截：禁止创建保留用户名 {username}")
    return False

# check_username_exists() 中
if username.upper() == "SYSTEM":
    return True  # 假装已存在，阻止注册
```

### 5.3 默认账户

| 用户名 | 密码 | 角色 |
|--------|------|------|
| admin | 123456 | 管理员 |
| doctor1 | 123456 | 普通用户 |

---

## 六、后端核心模块说明

### 6.1 main.py - FastAPI主入口

**位置**: `main.py` (1629行)

**核心功能**:
- 系统初始化 (`init_system`): 数据库初始化、目录创建、LLM配置迁移
- 用户认证 API: `/api/login`, `/api/logout`
- 用户管理 API: `/api/users`, `/api/users/{id}`, `/api/users/current`
- 任务管理 API: `/api/tasks/{username}`, `/api/tasks/{username}/{folder}/patients`
- 诊断提交 API: `/api/diagnosis/submit-json`
- LLM模型管理 API: `/api/admin/llm-models/*`
- 部署管理 API: `/api/admin/deploy-config`, `/api/admin/check-models`

**关键函数**:

| 函数 | 行号 | 功能 |
|------|------|------|
| `init_system()` | 102-147 | 系统初始化 |
| `migrate_llm_config()` | 49-100 | LLM配置迁移 |
| `login()` | 183-216 | 用户登录 |
| `get_current_user()` | 243-298 | 获取当前用户 |
| `get_task_patients()` | 420-562 | 获取病例列表（支持智慧寻址） |
| `submit_diagnosis_json()` | 1001-1280 | 提交诊断结果 |
| `analyze_with_llm()` | 1283-1350 | 异步调用大模型分析 |

**请求模型**:

```python
class LoginRequest(BaseModel):
    username: str
    password: str

class UserCreateRequest(BaseModel):
    username: str
    password: str
    doctor: str
    organization: str

class UserUpdateRequest(BaseModel):
    doctor: str
    organization: str
    password: Optional[str] = None
    is_admin: Optional[bool] = None

class DiagnosisRecordSimple(BaseModel):
    patientId: str
    diagnosis: str
    severity: Optional[int] = None
    viewTime: Optional[int] = 0

class DiagnosisSubmitJsonRequest(BaseModel):
    username: str
    taskFolder: str
    mode: Optional[str] = "diag"
    eduSubMode: Optional[str] = None
    submittedAt: str
    totalTime: Dict[str, Any]
    patientCount: int
    records: List[DiagnosisRecordSimple]
    skip_llm: Optional[bool] = False
```

---

### 6.2 routes_oridata.py - 核心路由模块

**位置**: `routes_oridata.py` (1399行)

**核心功能**:
- 原始数据上传: `/api/users/{username}/upload-oridata`
- 教育模式: `/api/edu/*`
- 后台任务管理: `run_model_inference_wrapper`

**关键函数**:

| 函数 | 行号 | 功能 |
|------|------|------|
| `generate_submission_id()` | 70-73 | 生成唯一任务ID |
| `safe_join()` | 75-80 | 安全路径拼接（防路径穿越） |
| `transcode_to_h264()` | 97-136 | 视频转码为H.264 |
| `generate_thumbnail_ffmpeg()` | 86-94 | 生成视频缩略图 |
| `update_user_task_index()` | 140-201 | 维护用户任务索引（含文件锁） |
| `update_edu_task_index()` | 204-250 | 维护教育任务索引 |
| `run_model_inference_wrapper()` | 254-351 | AI推理后台任务包装器 |

**文件锁机制** (routes_oridata.py:148-161):
```python
lock_path = user_root / "data.json.lock"
for _ in range(50):  # 最长等待5秒
    try:
        with open(lock_path, "x") as _: pass  # 原子创建
        acquired = True
        break
    except FileExistsError:
        time.sleep(0.1)
```

**任务状态追踪**:
| 标记文件 | 说明 |
|----------|------|
| `.processing` | 任务正在运行 |
| `.failed` | 任务已失败 |

**主要路由**:

| 路由 | 方法 | 行号 | 功能 |
|------|------|------|------|
| `/api/users/{username}/upload-oridata` | POST | 355-443 | 上传原始数据 |
| `/api/users/{username}/all-tasks` | GET | 453-488 | 获取所有任务 |
| `/api/users/{username}/tasks/{id}` | DELETE | 490-504 | 删除单个任务 |
| `/api/users/{username}/clear-stuck-tasks` | DELETE | 518-624 | 清空卡住的任务 |
| `/api/edu/parse-zip` | POST | 629-659 | 解析教育ZIP |
| `/api/edu/confirm-upload` | POST | 661-736 | 确认教育上传 |
| `/api/edu/admin/tasks` | GET | 739-752 | 获取教育任务列表 |
| `/api/edu/publish-task` | POST | 781-808 | 发布教育任务 |
| `/api/edu/admin/task-status/{id}` | GET | 810-893 | 获取用户完成进度 |

---

### 6.3 database.py - 数据库操作模块

**位置**: `database.py` (299行)

**核心功能**:
- SQLite数据库连接管理
- 用户CRUD操作
- SYSTEM保留名保护

**核心函数**:

| 函数 | 行号 | 功能 |
|------|------|------|
| `get_db_connection()` | 9-13 | 获取数据库连接 |
| `hash_password()` | 15-19 | SHA-256密码哈希 |
| `verify_password()` | 21-30 | 验证密码（兼容明文与哈希） |
| `init_database()` | 32-82 | 初始化数据库 |
| `verify_user()` | 84-122 | 验证用户登录 |
| `get_all_users()` | 124-143 | 获取所有用户 |
| `get_user_by_id()` | 145-164 | 按ID获取用户 |
| `get_user_by_username()` | 166-185 | 按用户名获取用户 |
| `create_user()` | 188-218 | 创建用户 |
| `update_user()` | 220-260 | 更新用户信息 |
| `delete_user()` | 262-282 | 删除用户 |
| `check_username_exists()` | 284-298 | 检查用户名是否存在 |

**密码迁移逻辑** (database.py:106-114):
```python
is_hashed = isinstance(stored_password, str) and len(stored_password) == 64
if not is_hashed:
    # 将明文密码迁移为哈希
    new_hashed = hash_password(password)
    cursor.execute('UPDATE users SET password = ? WHERE id = ?', (new_hashed, row['id']))
```

---

### 6.4 llm_analyzer.py - 大模型分析器

**位置**: `llm_analyzer.py` (485行)

**核心功能**:
- 支持多种大模型服务（智谱AI/OpenAI/DeepSeek/Moonshot）
- 三阶段/二阶段/单阶段分析
- 异步API调用

**核心类**:

```python
class LLMAnalyzer:
    def __init__(self, base_url: str, api_key: str, model: str = "glm-4")
    async def analyze_performance(self, stats: Dict[str, Any], stage: str = "single") -> Dict[str, Any]
```

**支持的模型**:

| 服务商 | 识别特征 | Endpoint |
|--------|----------|----------|
| 智谱AI | URL包含"bigmodel"或"zhipu" | `/api/paas/v4/chat/completions` |
| OpenAI | 其他 | `/v1/chat/completions` |
| DeepSeek | 其他 | `/v1/chat/completions` |
| Moonshot | 其他 | `/v1/chat/completions` |

**分析阶段**:

| stage参数 | 阶段类型 | 统计指标 |
|-----------|----------|----------|
| `"single"` | 单阶段分析 | accuracy, sensitivity, specificity, category_stats, ai_dependency |
| `"assist"` | 二阶段对比 | single vs assist对比, 四种依赖分析 |
| `"triple"` | 三阶段综合 | single/assist/review三阶段对比 |

---

## 七、AI诊断模块

### 7.1 模型架构概述

AI诊断模块位于 `model/` 目录下，采用**双Token时空网络**架构：

```
单视角模型 (4个)              多视角融合模型
┌─────────────────────┐       ┌─────────────────────────┐
│ spatial_size_4.pth │       │                         │
│ spatial_size_5.pth │       │ epoch_97.*.pth          │
│ spatial_size_6.pth │       │                         │
│ spatial_size_7.pth │       │ (多视角病例级分类)       │
└─────────┬───────────┘       └────────────┬────────────┘
          │                                 │
          ▼                                 ▼
  热力图 + 边界框 + 关键帧          诊断置信度
  {Normal, VSD, ASD, PDA}
```

### 7.2 模型入口 (model_main.py)

**位置**: `model/model_main.py` (128行)

**入口函数**:
```python
def run_diagnosis(target_dir: str, output_dir: str) -> dict
```

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
│   │   └── ...
│   └── output_data.zip
└── case2/
    └── ...
```

---

### 7.3 核心诊断逻辑 (heart_diagnosis.py)

**位置**: `model/heart_diagnosis.py` (513行)

**核心配置**:

| 参数 | 值 | 说明 |
|------|-----|------|
| DEVICE | cuda/cpu | 运行设备 |
| MODEL_INPUT_SIZE | (224, 224) | 模型输入尺寸 |
| FINAL_BOX_SIZE | 56 | 输出边界框大小 |
| clip_size | 16 | 视频帧数 |
| CLASS_NAMES | ['Normal', 'VSD', 'ASD', 'PDA'] | 诊断类别 |
| ROI_AREA_THRESHOLD | 150 | ROI面积阈值 |
| ROI_PADDING | 20 | ROI扩展padding |

**单视角模型配置** (MODEL_CONFIGS):

| 网格 | 模型文件 | 基线特征 |
|------|----------|----------|
| 4×4 | spatial_size_4.pth | final_baseline_features_4x4_512dim_1101.pt |
| 5×5 | spatial_size_5.pth | final_baseline_features_5x5_512dim_1101.pt |
| 6×6 | spatial_size_6.pth | final_baseline_features_6x6_512dim_1101.pt |
| 7×7 | spatial_size_7.pth | final_baseline_features_7x7_512dim_1101.pt |

**核心函数**:

| 函数 | 行号 | 功能 |
|------|------|------|
| `load_single_models()` | 95-121 | 加载4个单视角模型 |
| `load_multi_model()` | 124-134 | 加载多视角融合模型 |
| `process_single_video()` | 137-249 | 单视频推理：热力图/边界框/关键帧 |
| `multi_view_inference()` | 252-306 | 多视角推理：置信度分数 |
| `save_visuals()` | 309-380 | 保存可视化结果 |
| `diagnose()` | 383-503 | 主诊断流程 |

**单视角推理流程** (process_single_video):
```
加载视频 → 提取16帧 → 找ROI → 裁剪Resize → 创建Doppler Mask
    → 4个模型分别推理 → 投票决定预测类别 → 计算关键帧
    → 融合热力图 → 计算边界框 → 返回结果
```

**多视角推理流程** (multi_view_inference):
```
处理每个视频 → ROI裁剪 → 视角采样 → Padding到MAX_VIEWS=5
    → 多视角特征融合 → 输出4类置信度
```

---

### 7.4 神经网络结构 (Codes/Nets/)

#### 7.4.1 单视角网络 (dual_tokens_net.py)

**位置**: `model/Codes/Nets/dual_tokens_net.py` (302行)

**类**: `ResnetTransformerDualTokensTemporalSpatialDecouplesize`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| num_classes | int | 18 | 类别数 |
| d_model_cnn | int | 512 | CNN输出维度 |
| nhead | int | 8 | 注意力头数 |
| num_layers | int | 6 | Transformer层数 |
| dropout | float | 0.1 | Dropout概率 |
| spatial_resolution | int | 5 | 空间池化分辨率 |

**输入**: `(B, T, C, H_in, W_in)` - 视频帧序列
**输出**:
```python
{
    "predicted_scores": (B, 18),           # 预测分数
    "final_cls_tokens": (B, 18, 512),      # 最终CLS tokens
    "all_layer_cls_tokens": list,          # 每层的CLS tokens
    "raw_strings_for_loss": (B, 16, T, 512), # 用于损失的原始特征
    "all_layer_attention_scores": list,     # 每层的注意力分数
    "total_decouple_loss": scalar,         # 解耦损失
}
```

**网络架构**:
```
输入 (B,T,C,H,W)
    ↓
CNN特征提取 (ResNet50去掉最后两层)
    ↓
投影层 (Conv2d 2048->512)
    ↓
自适应池化 (spatial_resolution x spatial_resolution)
    ↓
拼接 Class Tokens + Patch Tokens
    ↓
添加空间位置编码 + 时序位置编码
    ↓
6层交错式时空Transformer块
    ↓
分类头 (LayerNorm -> Linear -> 1)
```

**关键组件**:

| 组件 | 行号 | 功能 |
|------|------|------|
| `PositionalEncoding` | 12-26 | 正弦/余弦位置编码 |
| `BasicTransformerBlock` | 36-80 | 标准Transformer编码器层 |
| `InterleavedSpatioTemporalBlock` | 83-125 | 交错式时空注意力块 |

#### 7.4.2 多视角融合网络 (Multi_Views_dual_tokens_net.py)

**位置**: `model/Codes/Nets/Multi_Views_dual_tokens_net.py` (249行)

**类**: `MultiViewDualTokensFusionSize`

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| view_num_classes | int | 18 | 每个视角的类别数 |
| d_model | int | 512 | 模型维度 |
| view_nhead | int | 8 | 视角内注意力头数 |
| view_encoder_layers | int | 6 | 视角内编码器层数 |
| case_num_classes | int | 4 | 病例级别类别数 |
| fusion_layers | int | 2 | 跨视角融合层数 |
| fusion_nhead | int | 8 | 融合阶段注意力头数 |
| max_views | int | 5 | 最大支持视角数 |
| spatial_size | int | 4 | 空间池化尺寸 |

**输入**:
```python
x_case: (B, V, T, C, H, W)        # B=批次, V=视角数, T=帧数
num_views_per_sample: (B,)        # 每个样本的实际视角数
masks: (B, V, NumPatches)         # 空间注意力掩码
```

**输出**: `logits: (B, case_num_classes)` - 病例级别分类logits

**网络架构**:
```
输入 (B, V, T, C, H, W)
    ↓
共享CNN特征提取 (B*V*T, C, H, W)
    ↓
投影 + 池化 (B*V, T, 16, D)
    ↓
视角内编码:
    - 拼接 Class Tokens + Patch Tokens
    - 添加空间/时序位置编码
    - 通过6层交错式时空Transformer
    ↓
单视角特征提炼:
    - 提取 view_cls_tokens
    - 时间维度平均 -> (B*V, 18, D)
    ↓
跨视角融合:
    - 展平拼接: (B, V*18, D)
    - 拼接 Case CLS Token: (B, 1+V*18, D)
    - 添加位置编码
    - 通过融合Transformer (2层)
    ↓
最终分类:
    Case CLS Token -> LayerNorm -> Linear -> (B, 4)
```

---

### 7.5 工具函数 (Codes/main_codes/)

#### 7.5.1 utils.py (365行)

**主要函数**:

| 函数 | 行号 | 功能 |
|------|------|------|
| `fix_randomness(seed)` | 10-17 | 设置随机种子 |
| `pad_collate_fn(batch)` | 19-64 | 多视角数据填充collate函数 |
| `video_loader(path)` | 139-171 | 加载视频为numpy数组 |
| `find_doppler_roi_from_video()` | 220-259 | 提取多普勒ROI区域 |
| `create_string_token_mask()` | 275-351 | 生成串Token掩码 |
| `calculate_iou(boxA, boxB)` | 262-271 | 计算边界框IoU |

#### 7.5.2 Confidence_scores.py (548行)

**主要类**:

| 类/函数 | 行号 | 功能 |
|---------|------|------|
| `ExternalMultiViewDataset` | 279-408 | 外部测试数据集类 |
| `create_string_token_mask_size()` | 204-275 | 通用串Token掩码生成 |
| `find_doppler_roi_from_video()` V6版 | 105-201 | ROI提取(支持frame_stride) |

**全局常量**:
```python
ROI_AREA_THRESHOLD = 150
ROI_PADDING = 20
class_names = ['Normal', 'VSD', 'ASD', 'PDA']
```

#### 7.5.3 original_ref.py (219行)

**主要函数**:

| 函数 | 行号 | 功能 |
|------|------|------|
| `video_loader()` | 158-184 | OpenCV视频加载 |
| `transform_box_from_model_to_original_space()` | 187-199 | 坐标变换(模型空间→原图空间) |
| `get_center_and_draw_box()` | 202-210 | 生成边界框 |

**HSV颜色检测阈值** (用于多普勒血流区域识别):
```python
# 红色: [0-10, 170-180] (饱和度高)
# 蓝色: [100-130] (饱和度中等)
# 湍流: [11-90] (饱和度变化区域)
```

---

## 八、前端UI模块详细说明

### 8.1 login.html - 用户登录页面

**位置**: `UI/login.html` (632行)

**核心功能**:
- 用户名密码认证
- 背景图片配置
- 登录按钮跳转

**关键API调用**:

| URL | 方法 | 功能 |
|------|------|------|
| `/api/login` | POST | 用户登录认证 |

**登录流程**:
```
用户输入用户名密码 → POST /api/login
    → 成功：设置cookie username，跳转 /flow
    → 失败：显示错误信息
```

**背景配置**: 从 `/res/share/background_config.json` 读取

---

### 8.2 flow.html - 模式选择主页

**位置**: `UI/flow.html` (320行)

**核心功能**:
- 显示用户信息（姓名、机构）
- 根据权限动态显示功能按钮
- 模式选择导航

**权限按钮**:

| 按钮 | 可见条件 | 跳转目标 |
|------|----------|----------|
| 诊断模式 | 所有用户 | /dashboard |
| 教育模式 | 所有用户 | /edu_status |
| 任务管理 | 仅管理员 | /edu_admin |
| 用户管理 | 仅管理员 | /admin |

**关键代码** (flow.html:237):
```javascript
isAdmin = u.is_admin === true;  // 严格判断
if (isAdmin) {
    tasksBtn.style.display = 'inline-block';
    usersBtn.style.display = 'inline-block';
}
```

---

### 8.3 dashboard.html - 任务上传页面

**位置**: `UI/dashboard.html`

**核心功能**:
- 文件夹/文件选择上传
- 拖拽上传
- 自动病例分组
- 视频缩略图生成
- 上传进度显示

**核心函数**:

| 函数 | 行号 | 功能 |
|------|------|------|
| `handleIncomingFiles()` | 239-291 | 路径解析核心逻辑 |
| `traverse()` | 219-236 | 递归遍历文件夹 |
| `setupDropzone()` | 293-308 | 拖拽区域设置 |
| `createVideoThumbnail()` | 310-354 | 视频缩略图生成 |
| `uploadBtn.onclick` | 432-478 | XHR上传核心 |

**路径解析逻辑** (dashboard.html:239-291):

| 情况 | 路径结构 | 病例名确定方式 |
|------|----------|----------------|
| CASE A | 无路径信息 | 自动生成 `patient N` |
| CASE B1 | ≥3层路径 | 第2层为病例名 |
| CASE B2 | 2层路径 | 第1层为病例名 |

**关键API调用**:

| URL | 方法 | 功能 |
|------|------|------|
| `/api/users/${username}/upload-oridata` | POST | 上传任务数据 |
| `/api/users/current` | GET | 获取当前用户 |

---

### 8.4 diagnosis.html - 医师判读界面

**位置**: `UI/diagnosis.html`

**核心功能**:
- 视频网格展示（最多5个视频）
- 三种模态切换 (original/heatmap/bbox)
- 视频弹窗播放
- 诊断结论录入
- 计时统计
- 草稿自动保存

**核心函数**:

| 函数 | 行号 | 功能 |
|------|------|------|
| `loadPatients()` | 1621-1726 | 加载病例列表 |
| `loadPatientVideos()` | 1728-1902 | 加载视频 |
| `loadModelConfidence()` | 1930-2021 | 加载AI置信度 |
| `openVideoModal()` | 2023-2421 | 视频弹窗 |
| `saveDraft()` | 1345-1364 | 保存草稿 |
| `submitDiagnosis()` | - | 提交诊断 |

**草稿缓存机制**:
```javascript
// sessionStorage key格式
`diagnosisDraft:${username}:${taskFolder}`

// 保存内容
{
    timerState: {...},
    currentPatientIndex: number,
    diagnosisRecords: {...},
    currentSelections: {...}
}
```

**URL参数**:
```
/diagnosis?username=xxx&task=xxx&taskName=xxx&mode=edu&eduSubMode=xxx
```

**多阶段任务后缀**:
- `_SINGLE`: 单独判读阶段
- `_AI-ASSIST`: AI辅助判读阶段
- `_REVIEW`: 复核判读阶段

**关键API调用**:

| URL | 方法 | 功能 |
|------|------|------|
| `/api/tasks/${username}/${taskFolder}/patients` | GET | 获取病例列表 |
| `/api/tasks/${username}/${taskId}/diagnosis` | POST | 提交诊断结果 |

---

### 8.5 edu_admin.html - 教育管理后台

**位置**: `UI/edu_admin.html`

**核心功能**:
- ZIP包上传与解析
- 任务批次管理（处理中/待发布/已发布）
- 用户分配与发布
- 成绩查看与报告
- 大模型配置
- 模型检测与部署

**看板状态**:

| 状态 | 说明 | 卡片颜色 |
|------|------|----------|
| processing | 模型判读中 | 蓝色边框 |
| unreleased | 待发布 | 橙色边框 |
| published | 已发布 | 绿色边框 |

**核心函数**:

| 函数 | 行号 | 功能 |
|------|------|------|
| `fetchTasks()` | 776-784 | 任务列表轮询(5秒) |
| `handleFile()` | 823-862 | ZIP处理 |
| `openPublishModal()` | 898-918 | 发布弹窗 |
| `viewUserReport()` | 1015-1069 | 查看报告 |
| `showTripleStageReport()` | 1172-1288 | 三阶段报告 |

**关键API调用**:

| URL | 方法 | 功能 |
|------|------|------|
| `/api/edu/admin/tasks` | GET | 获取任务列表 |
| `/api/edu/parse-zip` | POST | 解析ZIP |
| `/api/edu/confirm-upload` | POST | 确认上传 |
| `/api/edu/publish-task` | POST | 发布任务 |
| `/api/admin/llm-models` | GET/POST | LLM模型管理 |

---

### 8.6 edu_status.html - 教育练习中心

**位置**: `UI/edu_status.html`

**核心功能**:
- 任务列表展示
- 阶段状态显示
- 成绩报告展示
- LLM分析轮询

**核心函数**:

| 函数 | 行号 | 功能 |
|------|------|------|
| `fetchEduTasks()` | 181-325 | 获取教育任务 |
| `handleTask()` | 333-347 | 任务点击处理 |
| `openResultModal()` | 349-489 | 成绩报告弹窗 |
| `pollForLLMAnalysis()` | 491-571 | 轮询LLM分析(30次) |
| `showDualStageReport()` | 838-1074 | 双阶段对比报告 |
| `showTripleStageReport()` | 1077-1297 | 三阶段对比报告 |

**报告阶段标识**:
- 普通任务: 单一阶段
- 双阶段任务: `is_dual_stage = true` → Single + Assist
- 三阶段任务: `edu_sub_mode === 'triple'` → Single + Assist + Review

**关键API调用**:

| URL | 方法 | 功能 |
|------|------|------|
| `/api/edu/user/tasks/${username}` | GET | 获取用户任务 |
| `/api/edu/user/result/${username}/${taskId}` | GET | 获取成绩报告 |

---

### 8.7 admin.html - 用户管理后台

**位置**: `UI/admin.html` (688行)

**核心功能**:
- 用户列表展示
- 创建/编辑用户
- 角色管理（管理员/普通用户）
- 二次密码验证

**核心函数**:

| 函数 | 行号 | 功能 |
|------|------|------|
| `loadUsers()` | - | 加载用户列表 |
| `showCreateModal()` | - | 显示创建弹窗 |
| `showEditModal()` | - | 显示编辑弹窗 |
| `handleSubmit()` | - | 表单提交 |
| `deleteUser()` | - | 删除用户 |

**安全机制**:
- 二次密码验证 (admin.html:79-92)
- 仅管理员可访问

**关键API调用**:

| URL | 方法 | 功能 |
|------|------|------|
| `/api/users` | GET | 获取用户列表 |
| `/api/users` | POST | 创建用户 |
| `/api/users/{id}` | PUT | 更新用户 |
| `/api/users/{id}` | DELETE | 删除用户 |

---

## 九、API路由完整清单

### 9.1 main.py - 主服务API

#### 用户认证

| 路由 | 方法 | 位置 | 功能 |
|------|------|------|------|
| `/api/login` | POST | main.py:183-216 | 用户登录 |
| `/api/logout` | POST | main.py:218-222 | 用户登出 |

#### 用户管理

| 路由 | 方法 | 位置 | 功能 |
|------|------|------|------|
| `/api/users` | GET | main.py:238-241 | 获取所有用户 |
| `/api/users` | POST | main.py:310-326 | 创建用户 |
| `/api/users/current` | GET | main.py:243-298 | 获取当前用户 |
| `/api/users/{user_id}` | GET | main.py:302-307 | 获取单个用户 |
| `/api/users/{user_id}` | PUT | main.py:329-346 | 更新用户 |
| `/api/users/{user_id}` | DELETE | main.py:349-360 | 删除用户 |

#### 任务管理

| 路由 | 方法 | 位置 | 功能 |
|------|------|------|------|
| `/api/tasks/{username}` | GET | main.py:363-416 | 获取用户任务列表 |
| `/api/tasks/{username}/{task_folder}/patients` | GET | main.py:420-562 | 获取病例列表 |
| `/api/get-metadata` | GET | main.py:565-574 | 获取元数据 |

#### 诊断提交

| 路由 | 方法 | 位置 | 功能 |
|------|------|------|------|
| `/api/diagnosis/submit-json` | POST | main.py:1001-1280 | 提交诊断结果 |

#### LLM模型管理

| 路由 | 方法 | 位置 | 功能 |
|------|------|------|------|
| `/api/admin/llm-models` | GET | main.py:617-657 | 获取模型列表 |
| `/api/admin/llm-models` | POST | main.py:660-709 | 添加模型 |
| `/api/admin/llm-models/{model_id}` | PUT | main.py:712-757 | 更新模型 |
| `/api/admin/llm-models/{model_id}` | DELETE | main.py:760-797 | 删除模型 |
| `/api/admin/llm-models/select` | POST | main.py:800-834 | 选择模型 |
| `/api/admin/llm-models/{model_id}/test` | POST | main.py:837-878 | 测试模型 |

#### 部署管理

| 路由 | 方法 | 位置 | 功能 |
|------|------|------|------|
| `/api/admin/deploy-config` | GET | main.py:885-902 | 获取部署配置 |
| `/api/admin/deploy-config` | POST | main.py:905-921 | 保存部署配置 |
| `/api/admin/check-models` | POST | main.py:924-963 | 检测模型文件 |
| `/api/admin/reset-data` | POST | main.py:966-997 | 重置数据 |

---

### 9.2 routes_oridata.py - 核心路由API

#### 原始数据上传

| 路由 | 方法 | 位置 | 功能 |
|------|------|------|------|
| `/api/users/{username}/upload-oridata` | POST | routes_oridata.py:355-443 | 上传原始数据 |
| `/api/users/{username}/oridata-count` | GET | routes_oridata.py:447-451 | 获取上传计数 |
| `/api/users/{username}/all-tasks` | GET | routes_oridata.py:453-488 | 获取所有任务 |
| `/api/users/{username}/tasks/{id}` | DELETE | routes_oridata.py:490-504 | 删除任务 |
| `/api/users/{username}/re-sync-tasks` | GET | routes_oridata.py:506-513 | 重新同步任务 |
| `/api/users/{username}/clear-stuck-tasks` | DELETE | routes_oridata.py:518-624 | 清空卡住任务 |

#### 教育模式

| 路由 | 方法 | 位置 | 功能 |
|------|------|------|------|
| `/api/edu/parse-zip` | POST | routes_oridata.py:629-659 | 解析教育ZIP |
| `/api/edu/confirm-upload` | POST | routes_oridata.py:661-736 | 确认上传 |
| `/api/edu/admin/tasks` | GET | routes_oridata.py:739-752 | 获取教育任务列表 |
| `/api/edu/publish-task` | POST | routes_oridata.py:781-808 | 发布任务 |
| `/api/edu/admin/task-status/{id}` | GET | routes_oridata.py:810-893 | 获取任务状态 |
| `/api/edu/admin/unpublish-task` | POST | routes_oridata.py:896-930 | 取消发布 |
| `/api/edu/admin/tasks/{id}` | DELETE | routes_oridata.py:932-970 | 删除任务 |
| `/api/admin/inference-stats` | GET | routes_oridata.py:755-778 | 获取推理统计 |

---

## 十、数据流向详解

### 10.1 诊断模式完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 上传阶段 (dashboard.html)                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  用户选择文件                                                     │
│      │                                                          │
│      ├── traverse() 递归遍历文件夹                                │
│      ├── handleIncomingFiles() 路径解析分组                       │
│      │      CASE A: 无路径 → patient N                           │
│      │      CASE B1: ≥3层路径 → 第2层为病例名                     │
│      │      CASE B2: 2层路径 → 第1层为病例名                     │
│      │                                                          │
│      └── uploadBtn.onclick() XHR上传                            │
│              │                                                  │
│              POST /api/users/{username}/upload-oridata          │
│                  │                                              │
│                  ├── 保存到: oridata/{submission_id}/caseN/     │
│                  ├── 生成metadata.json                           │
│                  ├── update_user_task_index()                    │
│                  └── background_tasks.add_task()                  │
│                          run_model_inference_wrapper()           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. AI处理阶段 (routes_oridata.py后台任务)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  run_model_inference_wrapper()                                   │
│      │                                                          │
│      ├── 创建 .processing 标记文件                               │
│      ├── asyncio.to_thread(run_diagnosis)                        │
│      │      │                                                  │
│      │      ├── load_single_models() 加载4个单视角模型           │
│      │      ├── load_multi_model() 加载多视角模型                 │
│      │      │                                                  │
│      │      ├── 遍历病例目录                                     │
│      │      │   process_single_video()                          │
│      │      │     → 热力图/边界框/关键帧                        │
│      │      │                                                  │
│      │      │   multi_view_inference()                          │
│      │      │     → 置信度 {Normal, VSD, ASD, PDA}              │
│      │      │                                                  │
│      │      └── save_visuals() 保存视频                          │
│      │                                                          │
│      ├── transcode_to_h264() 视频转码                           │
│      ├── update_user_task_index(is_cmp=True)                     │
│      └── 删除 .processing 标记                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. 判读阶段 (diagnosis.html)                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  loadPatients()                                                  │
│      │                                                          │
│      GET /api/tasks/{username}/{taskFolder}/patients            │
│          │                                                      │
│          └── 返回病例列表 + 视频路径 + metadata URL              │
│                                                                  │
│  用户查看视频                                                    │
│      │                                                          │
│      ├── original 模态: 原始视频                                 │
│      ├── heatmap 模态: 热力图叠加视频                           │
│      └── bbox 模态: 带边界框视频                                │
│                                                                  │
│  加载AI置信度                                                    │
│      │                                                          │
│      GET /api/get-metadata?path=.../confidence_scores.json      │
│          │                                                      │
│          └── 返回 {Normal: 0.85, VSD: 0.12, ...}               │
│                                                                  │
│  医师诊断                                                       │
│      │                                                          │
│      ├── 选择诊断: 正常/VSD/ASD/PDA                             │
│      ├── 选择严重程度                                           │
│      ├── 计时统计 (每病例查看时间)                               │
│      └── submitDiagnosis()                                      │
│              │                                                  │
│              POST /api/diagnosis/submit-json                    │
│                  │                                              │
│                  └── 保存 final_diagnosis_report_{timestamp}.json│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 教育模式完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 管理员创建批次 (edu_admin.html)                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ZIP包上传                                                       │
│      │                                                          │
│      ├── handleFile() 处理文件                                  │
│      └── POST /api/edu/parse-zip                               │
│              │                                                  │
│              └── 返回: case_count, video_count, submission_id   │
│                                                                  │
│  确认上传                                                        │
│      │                                                          │
│      POST /api/edu/confirm-upload                               │
│          │                                                      │
│          ├── 解压到 SYSTEM/edu_data/{submission_id}/             │
│          ├── 重组目录结构 (按epoch_data.json)                     │
│          ├── update_edu_task_index(status=processing)           │
│          └── background_tasks.add_task()                        │
│                  run_model_inference_wrapper(is_system=True)     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. AI处理 (后台任务)                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  run_model_inference_wrapper(is_system=True)                     │
│      │                                                          │
│      ├── 创建 .processing 标记                                   │
│      ├── run_diagnosis()                                        │
│      │      │                                                  │
│      │      └── 输出到: SYSTEM/edu_data/processed/{id}/         │
│      │                                                          │
│      ├── transcode_to_h264()                                   │
│      └── update_edu_task_index(status=unreleased, is_cmp=True) │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. 管理员发布 (edu_admin.html)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  打开发布弹窗                                                    │
│      │                                                          │
│      ├── 选择目标用户                                            │
│      └── POST /api/edu/publish-task                             │
│              │                                                  │
│              └── update_edu_task_index(                         │
│                      status=published,                          │
│                      target_users=[...],                        │
│                      is_dual_stage=True/False                  │
│                  )                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. 用户判读 (edu_status.html → diagnosis.html)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  edu_status.html                                                 │
│      │                                                          │
│      GET /api/edu/user/tasks/{username}                         │
│          │                                                      │
│          └── 返回分配的任务列表                                  │
│                                                                  │
│  点击"开始判读"                                                  │
│      │                                                          │
│      → diagnosis.html?mode=edu&taskId=xxx&eduSubMode=single     │
│                                                                  │
│  单独判读阶段 (diagnosis.html)                                    │
│      │                                                          │
│      ├── 仅显示 original 视频                                    │
│      ├── 禁用 heatmap/bbox 按钮                                 │
│      ├── 医师录入诊断                                            │
│      └── POST /api/diagnosis/submit-json(mode=edu, eduSubMode=single)
│              │                                                  │
│              ├── 读取 epoch_data.json 标准答案                   │
│              ├── 计算 accuracy/sensitivity/specificity          │
│              ├── 计算 AI依赖性分析                              │
│              └── 保存到 Doctor_Diag_Result/{username}.json      │
│                                                                  │
│  AI辅助判读阶段                                                  │
│      │                                                          │
│      ├── 显示 original/heatmap/bbox                            │
│      ├── 医师录入诊断                                            │
│      └── POST /api/diagnosis/submit-json(mode=edu, eduSubMode=assist)
│              │                                                  │
│              └── 同样计算统计数据                                │
│                                                                  │
│  两阶段都完成后                                                   │
│      │                                                          │
│      asyncio.create_task(analyze_with_llm())                    │
│          │                                                      │
│          └── 调用大模型生成对比分析报告                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. 查看报告 (edu_status.html)                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  点击"查看报告"                                                  │
│      │                                                          │
│      GET /api/edu/user/result/{username}/{taskId}               │
│          │                                                      │
│          └── 返回: accuracy, sensitivity, specificity,           │
│                   category_stats, ai_dependency,                │
│                   llm_analysis_status                           │
│                                                                  │
│  llm_analysis_status = pending                                   │
│      │                                                          │
│      pollForLLMAnalysis() 轮询等待                               │
│          │                                                      │
│          └── 30次 × 1秒轮询                                     │
│                                                                  │
│  llm_analysis_status = completed                                 │
│      │                                                          │
│      showCompleteReport() / showDualStageReport()              │
│          │                                                      │
│          └── 显示完整评估报告                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 十一、配置文件说明

### 11.1 LLM配置文件

#### llm_models.json (新版)

**位置**: `config/llm_models.json`

**结构**:
```json
{
  "models": [
    {
      "id": "xxx",
      "display_name": "模型显示名",
      "base_url": "https://api.xxx.com/v1",
      "api_key": "sk-xxx",
      "model": "glm-4",
      "description": "模型描述",
      "created_at": "2024-01-01T00:00:00"
    }
  ],
  "selected_model_id": "xxx"
}
```

#### llm_config.json (旧版)

**位置**: `config/llm_config.json` (已弃用，自动迁移)

**结构**:
```json
{
  "base_url": "https://open.bigmodel.cn/api",
  "api_key": "xxx",
  "model": "glm-4"
}
```

### 11.2 背景配置

**位置**: `UI/res/share/background_config.json`

**结构**:
```json
{
  "enabled": true,
  "showBackground": true,
  "image": "bg.jpg",
  "showTitle": true,
  "title": "标题文字",
  "subtitle": "副标题文字"
}
```

### 11.3 用户头像配置

**位置**: `UI/res/share/user_avatar_config.json`

**结构**:
```json
{
  "avatars": {
    "admin": "avatars/admin.png",
    "doctor1": "avatars/doctor1.png"
  }
}
```

---

## 九、前端页面清单

| 文件 | 功能 | 访问权限 |
|------|------|----------|
| login.html | 用户登录 | 公开 |
| flow.html | 模式选择主页 | 已登录用户 |
| dashboard.html | 任务上传 | 已登录用户 |
| task_status.html | 任务进度查看 | 已登录用户 |
| diagnosis.html | 医师判读界面 | 已登录用户 |
| admin.html | 用户管理后台 | 仅管理员 |
| edu_admin.html | 教育管理后台 | 仅管理员 |
| edu_status.html | 教育练习中心 | 已登录用户 |

---

## 十、关键函数/类索引

### 10.1 后端核心函数

#### main.py

| 函数/类 | 行号 | 用途 |
|---------|------|------|
| `init_system()` | 102-147 | 系统初始化 |
| `migrate_llm_config()` | 49-100 | LLM配置迁移 |
| `login()` | 183-216 | 用户登录 |
| `get_current_user()` | 243-298 | 获取当前用户 |
| `get_task_patients()` | 420-562 | 获取病例列表 |
| `submit_diagnosis_json()` | 1001-1280 | 提交诊断结果 |
| `analyze_with_llm()` | 1283-1350 | 异步LLM分析 |
| `LoginRequest` | 174-176 | 登录请求模型 |
| `UserCreateRequest` | 225-229 | 创建用户请求 |
| `DiagnosisSubmitJsonRequest` | 604-613 | 诊断提交请求 |

#### routes_oridata.py

| 函数/类 | 行号 | 用途 |
|---------|------|------|
| `generate_submission_id()` | 70-73 | 生成任务ID |
| `safe_join()` | 75-80 | 安全路径拼接 |
| `transcode_to_h264()` | 97-136 | 视频转码 |
| `generate_thumbnail_ffmpeg()` | 86-94 | 生成缩略图 |
| `update_user_task_index()` | 140-201 | 维护用户索引 |
| `update_edu_task_index()` | 204-250 | 维护教育索引 |
| `run_model_inference_wrapper()` | 254-351 | AI推理包装器 |
| `upload_oridata()` | 355-443 | 上传原始数据 |
| `get_all_tasks()` | 453-488 | 获取所有任务 |
| `clear_stuck_tasks()` | 518-624 | 清空卡住任务 |
| `parse_edu_zip()` | 629-659 | 解析教育ZIP |
| `confirm_edu_upload()` | 661-736 | 确认教育上传 |
| `publish_edu_task()` | 781-808 | 发布教育任务 |
| `get_edu_task_detail_status()` | 810-893 | 获取任务状态 |
| `unpublish_edu_task()` | 896-930 | 取消发布任务 |

#### database.py

| 函数/类 | 行号 | 用途 |
|---------|------|------|
| `get_db_connection()` | 9-13 | 获取数据库连接 |
| `hash_password()` | 15-19 | SHA-256密码哈希 |
| `verify_password()` | 21-30 | 验证密码 |
| `init_database()` | 32-82 | 初始化数据库 |
| `verify_user()` | 84-122 | 验证用户登录 |
| `get_all_users()` | 124-143 | 获取所有用户 |
| `get_user_by_id()` | 145-164 | 按ID获取用户 |
| `get_user_by_username()` | 166-185 | 按用户名获取用户 |
| `create_user()` | 188-218 | 创建用户 |
| `update_user()` | 220-260 | 更新用户 |
| `delete_user()` | 262-282 | 删除用户 |
| `check_username_exists()` | 284-298 | 检查用户名存在 |

#### llm_analyzer.py

| 函数/类 | 行号 | 用途 |
|---------|------|------|
| `LLMAnalyzer` | 14-485 | 大模型分析器类 |
| `LLMAnalyzer.__init__()` | 17-31 | 初始化 |
| `analyze_performance()` | 33-65 | 执行分析 |
| `_get_endpoint()` | 67-87 | 获取API端点 |
| `_build_payload()` | 89-119 | 构造请求体 |
| `_make_request()` | 121-147 | 发送HTTP请求 |
| `_construct_prompt()` | 149-379 | 构造提示词 |
| `_parse_response()` | 381-430 | 解析响应 |

---

### 10.2 AI模型函数

#### heart_diagnosis.py

| 函数/类 | 行号 | 用途 |
|---------|------|------|
| `load_single_models()` | 95-121 | 加载单视角模型 |
| `load_multi_model()` | 124-134 | 加载多视角模型 |
| `process_single_video()` | 137-249 | 单视频推理 |
| `multi_view_inference()` | 252-306 | 多视角推理 |
| `save_visuals()` | 309-380 | 保存可视化 |
| `diagnose()` | 383-503 | 主诊断流程 |

#### dual_tokens_net.py

| 函数/类 | 行号 | 用途 |
|---------|------|------|
| `PositionalEncoding` | 12-26 | 位置编码 |
| `BasicTransformerBlock` | 36-80 | Transformer块 |
| `InterleavedSpatioTemporalBlock` | 83-125 | 交错时空块 |
| `ResnetTransformerDualTokensTemporalSpatialDecouplesize` | 127-302 | 主模型类 |

#### Multi_Views_dual_tokens_net.py

| 函数/类 | 行号 | 用途 |
|---------|------|------|
| `PositionalEncoding` | 12-26 | 位置编码 |
| `BasicTransformerBlock` | 36-56 | Transformer块 |
| `InterleavedSpatioTemporalBlock` | 60-102 | 交错时空块 |
| `MultiViewDualTokensFusionSize` | 104-248 | 多视角融合模型 |

---

### 10.3 前端JavaScript函数

#### dashboard.html

| 函数 | 行号 | 用途 |
|------|------|------|
| `generateDefaultRequestName()` | 157-161 | 生成任务名 |
| `fetchUploadCount()` | 166-174 | 获取上传计数 |
| `loadCurrentUser()` | 199-216 | 加载当前用户 |
| `traverse()` | 219-236 | 递归遍历文件夹 |
| `handleIncomingFiles()` | 239-291 | 处理文件入口 |
| `setupDropzone()` | 293-308 | 设置拖拽区 |
| `createVideoThumbnail()` | 310-354 | 创建缩略图 |
| `renderRows()` | 356-429 | 渲染病例列表 |
| `uploadBtn.onclick` | 432-478 | 上传按钮事件 |

#### diagnosis.html

| 函数 | 行号 | 用途 |
|------|------|------|
| `shuffleArray()` | 1317-1324 | 洗牌算法 |
| `getDraftKey()` | 1331-1333 | 获取草稿key |
| `saveDraft()` | 1345-1364 | 保存草稿 |
| `loadDraft()` | 1366-1378 | 加载草稿 |
| `applyDraft()` | 1398-1457 | 应用草稿 |
| `loadPatients()` | 1621-1726 | 加载病例 |
| `loadPatientVideos()` | 1728-1902 | 加载视频 |
| `loadModelConfidence()` | 1930-2021 | 加载AI置信度 |
| `openVideoModal()` | 2023-2421 | 打开视频弹窗 |
| `startTimer()` | 2506-2517 | 启动计时器 |
| `submitDiagnosis()` | - | 提交诊断 |

#### edu_admin.html

| 函数 | 行号 | 用途 |
|------|------|------|
| `checkDeployConfig()` | 417-429 | 检查部署配置 |
| `checkModels()` | 431-459 | 检测模型 |
| `resetData()` | 461-479 | 重置数据 |
| `loadModels()` | 518-535 | 加载模型列表 |
| `renderKanban()` | 787-809 | 渲染看板 |
| `handleFile()` | 823-862 | 处理文件 |
| `openPublishModal()` | 898-918 | 打开发布弹窗 |
| `openStatusDetail()` | 944-991 | 打开状态详情 |
| `viewUserReport()` | 1015-1069 | 查看报告 |
| `showTripleStageReport()` | 1172-1288 | 三阶段报告 |
| `unpublishTask()` | 1363-1382 | 取消发布 |

#### edu_status.html

| 函数 | 行号 | 用途 |
|------|------|------|
| `loadAvatarConfig()` | 99-108 | 加载头像配置 |
| `loadUser()` | 119-162 | 加载用户信息 |
| `fetchEduTasks()` | 181-325 | 获取教育任务 |
| `handleTask()` | 333-347 | 处理任务点击 |
| `openResultModal()` | 349-489 | 打开结果弹窗 |
| `pollForLLMAnalysis()` | 491-571 | 轮询LLM分析 |
| `showBasicReport()` | 641-683 | 显示基础报告 |
| `showCompleteReport()` | 685-814 | 显示完整报告 |
| `showDualStageReport()` | 838-1074 | 双阶段报告 |
| `showTripleStageReport()` | 1077-1297 | 三阶段报告 |

---

## 十一、启动与部署

### 11.1 本地启动

```bash
# 1. 激活Conda环境
conda activate challengeserver

# 2. 进入项目目录
cd /path/to/Diagnostic_Platform_Stable_Version_C3D

# 3. 启动FastAPI服务（开发模式）
python main.py

# 4. 访问地址
# 登录页: http://127.0.0.1:11000/
# 管理后台: http://127.0.0.1:11000/admin
```

### 11.2 生产环境部署

```bash
# 使用 nohup 后台运行
nohup uvicorn main:app --host 0.0.0.0 --port 11000 --workers 4 > app.log 2>&1 &

# 或使用 gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:11000
```

### 11.3 环境依赖

**Python依赖**:
```
fastapi
uvicorn
pydantic
torch
torchvision
opencv-python
numpy
pillow
tqdm
pyyaml
aiofiles
aiohttp
einops
entmax
```

**系统依赖**:
- ffmpeg (视频处理)

---

## 十二、文档结束

本文档已完整覆盖项目的技术架构、模块关系、API路由、数据流向和关键函数索引。

如需进一步信息，请参考：
- `README.md` - 项目原始说明文档
- 源代码注释 - 各模块内的详细注释
