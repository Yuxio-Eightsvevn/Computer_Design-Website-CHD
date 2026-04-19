# 儿童先心病超声诊断平台 - 技术设计文档

## 一句话概括

本平台是一个面向儿童先天性心脏病（先心病）超声诊断的AI辅助系统，采用FastAPI后端 + 原生HTML/JavaScript前端的架构，核心功能包括：视频上传与AI模型推理（生成热力图、边界框、关键帧）、医师交互式诊断、教育培训与评估。**系统面临六大技术挑战**：①**视频编码兼容性**：不同设备录制的视频使用不同编码格式，通过ffmpeg转码为H.264+AAC格式的MP4确保浏览器播放；②**目录结构自适应**：上传路径可能有多种层级（TaskName/patient/video.mp4 或 patient/video.mp4），后端自动解析并重组为AI模型期望的case/video结构；③**教育双阶段模式**：通过任务ID后缀机制（`_SINGLE`/`_AI-ASSIST`）隔离独立判读与AI辅助判读的成绩，前端根据`eduSubMode`参数禁用/启用heatmap/bbox按钮，雷达图采用统一六维指标（准确率、敏感度、特异性、病种均衡性、诊断效率、决策一致性）进行叠加对比；④**文件锁并发控制**：使用排他锁文件（`.lock`）和原子性`open(file, 'x')`操作防止多任务同时写入`data.json`索引文件；⑤**大模型异步分析**：诊断提交后通过`asyncio.create_task`触发后台分析，前端轮询`llm_analysis_status`状态，支持智谱AI/OpenAI/DeepSeek/Moonshot等多API格式自动适配；⑥**数据隔离与权限**：SYSTEM保留用户名保护、Cookie-based认证、用户私有目录隔离、`target_users`列表控制教育任务分发。

## 模块索引

| 序号 | 模块 | 状态 |
|------|------|------|
| 1 | 后端API与数据存储架构 | ✅ 已完成 |
| 2 | 双阶段教育模式 | ✅ 已完成 |
| 3 | 大模型智能分析集成 | ✅ 已完成 |
| 4 | 视频处理与帧提取 | ✅ 已完成 |
| 5 | 诊断数据流与状态管理 | ✅ 已完成 |
| 6 | 用户权限与数据隔离 | ✅ 已完成 |

---

## 1. 后端API与数据存储架构

### 设计难点与解决方案

#### 1.1 文件锁机制解决并发写入冲突

**问题**：多用户同时上传任务时，多个后台AI推理任务可能同时更新同一个用户的`data.json`索引文件，导致数据覆盖或文件损坏。

**解决方案**：采用排他锁文件（`.lock`）机制，使用原子性文件创建操作实现无死锁的乐观锁。

```python
# 核心实现 (routes_oridata.py:139-199)
lock_path = user_root / "data.json.lock"
acquired = False
for _ in range(50):  # 最多等待5秒
    try:
        with open(lock_path, "x") as _: pass  # 原子创建，文件存在则抛异常
        acquired = True
        break
    except FileExistsError:
        time.sleep(0.1)

if acquired:
    try:
        # 读取、更新、写入 data.json
        ...
    finally:
        if lock_path.exists(): lock_path.unlink()  # 必须释放锁
```

**关键点**：
- 使用`open(file, 'x')`模式利用文件系统原子性保证锁的互斥
- 锁获取失败最多等待5秒，超时放弃避免死锁
- `finally`块确保锁一定被释放

#### 1.2 任务状态追踪机制

**问题**：后台AI推理任务运行时间较长，前端需要知道任务当前处于什么状态（处理中/已完成/失败/卡住）。

**解决方案**：使用`.processing`和`.failed`状态标记文件，在任务启动、成功、失败时创建/删除/重命名标记文件。

```
任务目录结构：
data_batch_storage/{username}/oridata/{submission_id}/
├── .processing     ← 任务正在运行（后台任务创建）
├── .failed         ← 任务已失败
├── case1/
│   └── video.mp4
└── ...
```

**状态判断逻辑**：
| is_cmp | .processing | .failed | 最终状态 |
|--------|-------------|---------|----------|
| true | - | - | completed |
| false | ✓ | - | processing |
| false | - | ✓ | failed |
| false | - | - | stuck（卡住）|

#### 1.3 智慧寻址机制

**问题**：用户任务的原始数据和处理结果可能位于不同路径，前端只知道`submission_id`，需要动态查找实际数据位置。

**解决方案**：按优先级尝试多个可能的路径，找到第一个存在的。

```python
# 核心实现 (routes_oridata.py)
proc_path = Path(DATA_BATCH_STORAGE) / username / "processed" / submission_id
root_path = Path(DATA_BATCH_STORAGE) / username / submission_id
task_path = proc_path if proc_path.exists() else root_path
```

### 核心流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI 主入口 (main.py)                     │
├─────────────────────────────────────────────────────────────────────┤
│  /api/users/{username}/upload-oridata  →  接收上传 → 触发后台任务    │
│  /api/edu/*                                    →  教育模式专用路由    │
│  /api/diagnosis/submit-json                    →  诊断结果提交        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    路由模块 (routes_oridata.py)                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ update_user  │  │ update_edu   │  │ run_model    │                │
│  │ _task_index  │  │ _task_index  │  │ _inference   │                │
│  │              │  │              │  │ _wrapper     │                │
│  │ 文件锁保护   │  │ 文件锁保护   │  │ 状态文件标记 │                │
│  └──────────────┘  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         数据存储层                                    │
├─────────────────────────────────────────────────────────────────────┤
│  data_batch_storage/                                                │
│  ├── {username}/                   ← 用户私有空间                    │
│  │   ├── oridata/                                               │
│  │   │   └── {submission_id}/     ← 原始上传（带状态标记文件）      │
│  │   ├── processed/                                             │
│  │   │   └── {submission_id}/     ← AI处理结果                    │
│  │   └── data.json                 ← 用户任务索引（文件锁保护）      │
│  └── SYSTEM/                      ← 系统保留空间                    │
│      └── edu_data/                                              │
│          ├── {submission_id}/      ← 教育批次原始数据               │
│          ├── processed/            ← 教育批次AI结果                 │
│          ├── Doctor_Diag_Result/   ← 用户成绩单                     │
│          └── data.json             ← 教育任务索引（文件锁保护）      │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键数据结构

#### 用户任务索引 (data.json)
```json
{
  "tasks": [
    {
      "submission_id": "20260419_120000_123456",
      "request_name": "测试任务",
      "request_pos": "processed/20260419_120000_123456",
      "request_case_cnt": 5,
      "request_video_cnt": 12,
      "is_cmp": true,
      "upload_time": "2026-04-19 12:00:00",
      "cmp_time": "2026-04-19 12:05:30"
    }
  ]
}
```

#### 教育任务索引 (SYSTEM/edu_data/data.json)
```json
{
  "tasks": [
    {
      "submission_id": "edu_20260419_120000_123456",
      "request_name": "教育批次A",
      "status": "published",
      "is_dual_stage": true,
      "target_users": ["doctor1", "doctor2"],
      "request_case_cnt": 10,
      "request_video_cnt": 30
    }
  ]
}
```

### 模块交互关系

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  前端页面   │ ←→  │  FastAPI    │ ←→  │  路由模块   │
│ dashboard   │     │  main.py    │     │routes_oridata│
│ edu_admin   │     │             │     │  .py        │
│ edu_status  │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
           ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
           │ AI模型推理  │           │ 文件系统    │           │ 数据库      │
           │ run_diagnosis│          │ 索引+锁机制 │           │ database.py │
           └─────────────┘           └─────────────┘           └─────────────┘
```

---

## 2. 双阶段教育模式

### 设计难点与解决方案

#### 2.1 双阶段任务的数据关联

**问题**：教育任务分为"单独判读"和"AI辅助判读"两个阶段，需要：
- 单独阶段禁用AI辅助工具（heatmap/bbox）
- AI辅助阶段必须先完成单独阶段才能解锁
- 两个阶段的结果需要关联对比

**解决方案**：

1. **任务ID后缀机制**：在数据库和文件系统中使用后缀区分两个阶段
   ```
   原始任务ID: task123
   单独判读:   task123_SINGLE
   AI辅助判读: task123_AI-ASSIST
   ```

2. **前端状态检查**：AI辅助按钮在单独判读完成前被禁用
   ```javascript
   // edu_status.html 按钮禁用逻辑
   <button ${!singleDone ? 'disabled' : ''} ...>
     ${singleDone ? 'AI辅助判读' : '⚠️ AI辅助判读'}
   </button>
   ```

3. **后端提交时添加后缀**：根据`eduSubMode`参数在保存成绩时添加后缀
   ```python
   # main.py 诊断提交逻辑
   if request.eduSubMode == 'single':
       save_key = f"{request.taskFolder}_SINGLE"
   elif request.eduSubMode == 'assist':
       save_key = f"{request.taskFolder}_AI-ASSIST"
   ```

#### 2.2 单独阶段禁用AI辅助工具

**问题**：单独判读阶段不应显示heatmap/bbox等AI辅助信息，需要在前端强制禁用。

**解决方案**：通过URL参数`eduSubMode`传递阶段标识，在诊断页面根据参数控制界面。

```javascript
// diagnosis.html 模态按钮禁用逻辑
const urlParams = new URLSearchParams(window.location.search);
const eduSubMode = urlParams.get('eduSubMode');

if (eduSubMode === 'single') {
    // 禁用heatmap和bbox按钮
    document.getElementById('btn-heatmap').disabled = true;
    document.getElementById('btn-bbox').disabled = true;
}
```

#### 2.3 双阶段对比报告

**问题**：需要将两个阶段的诊断结果进行对比，直观展示AI辅助带来的提升。

**解决方案**：设计六维判读能力模型，使用雷达图叠加展示。

**六维指标**（已更新）：
| 维度 | 计算方式 | 含义 |
|------|---------|------|
| 准确率 | 正确数/总数 | 整体诊断准确性 |
| 敏感度 | TP/(TP+FN) | 疾病检出能力 |
| 特异性 | TN/(TN+FP) | 排除非病例能力 |
| 病种均衡性 | 1 - 各病种准确率的标准差归一化 | 对各类病种诊断的稳定性 |
| 诊断效率 | 病例数/总分钟数，归一化到0-1 | 单位时间诊断量 |
| 决策一致性 | 与AI预测一致的比率 | 与AI观点的一致程度 |

#### 2.4 ZIP包结构重组

**问题**：教育批次ZIP包内的目录结构可能与AI模型期望的结构不一致。

**解决方案**：上传后根据`epoch_data.json`自动重组目录结构。

```python
# routes_oridata.py:627-679
# 原始ZIP结构（可能是任意嵌套）
education_batch.zip
├── some_folder/          ← 可能多一层
│   ├── epoch_data.json
│   └── videos/
│       └── case1.mp4

# 重组后结构（适配AI模型）
{submission_id}/
├── epoch_data.json
├── case1/
│   └── case1.mp4         ← 视频被移到对应病例目录下
└── case2/
    └── case2.mp4
```

### 核心流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     教育批次上传流程                                   │
├─────────────────────────────────────────────────────────────────────┤
│  edu_admin.html                                                     │
│      │                                                             │
│      ├── 1. 上传ZIP包                                              │
│      ├── 2. POST /api/edu/parse-zip (解析统计)                      │
│      └── 3. POST /api/edu/confirm-upload (确认上传)                 │
│              │                                                      │
│              ├── 解压ZIP                                             │
│              ├── 根据epoch_data.json重组目录结构                      │
│              ├── 更新教育任务索引 (status: processing)               │
│              └── 触发后台AI推理                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     管理员发布任务                                     │
├─────────────────────────────────────────────────────────────────────┤
│  edu_admin.html                                                     │
│      │                                                             │
│      └── POST /api/edu/publish-task                                 │
│              │                                                      │
│              └── 更新任务状态 (unreleased → published)               │
│                  记录 target_users 列表                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     用户双阶段判读流程                                │
├─────────────────────────────────────────────────────────────────────┤
│  edu_status.html                                                    │
│      │                                                             │
│      ├── 查看任务列表                                                │
│      │                                                              │
│      ├── 阶段1: 单独判读 ──────────────────────────────────────────┐│
│      │   │                                                          ││
│      │   ├── 点击"单独判读" → diagnosis.html?eduSubMode=single      ││
│      │   ├── 禁用heatmap/bbox按钮                                    ││
│      │   └── 提交 → POST /api/diagnosis/submit-json (mode=edu)      ││
│      │                    保存为 taskId_SINGLE                       ││
│      │                                                              ││
│      └── 阶段2: AI辅助判读 ─────────────────────────────────────────┐│
│          │                                                          ││
│          ├── 点击"AI辅助判读" (需单独阶段完成)                        ││
│          ├── diagnosis.html?eduSubMode=assist                       ││
│          ├── 启用所有模态按钮                                         ││
│          └── 提交 → POST /api/diagnosis/submit-json (mode=edu)      ││
│                      保存为 taskId_AI-ASSIST                         ││
│                                                                      ││
│  edu_status.html                                                    ││
│      │                                                              ││
│      └── 查看报告 → showDualStageReport()                           ││
│              │                                                      ││
│              ├── 获取 taskId_SINGLE 和 taskId_AI-ASSIST 数据         ││
│              ├── 计算六维指标                                         ││
│              └── 渲染雷达图对比                                       ││
└──────────────────────────────────────────────────────────────────────┘
```

### 关键数据结构

#### 诊断提交请求
```python
class DiagnosisSubmitJsonRequest(BaseModel):
    username: str
    taskFolder: str
    mode: str = "diag"  # diag=判读模式, edu=教育模式
    eduSubMode: Optional[str] = None  # single / assist
    submittedAt: str
    totalTime: Dict[str, Any]  # {seconds: int, formatted: "MM:SS"}
    patientCount: int
    records: List[DiagnosisRecordSimple]
```

#### 成绩单存储结构 (Doctor_Diag_Result/{username}.json)
```json
{
  "task123_SINGLE": {
    "accuracy": 0.75,
    "sensitivity": 0.80,
    "specificity": 0.70,
    "totalTime": {"seconds": 300, "formatted": "5:00"},
    "patientCount": 10,
    "category_stats": {...},
    "ai_dependency": {
      "correct_reliance": 5,
      "insufficient_reliance": 1,
      "correct_independence": 2,
      "over_reliance": 2
    },
    "edu_sub_mode": "single",
    "llm_analysis_status": "pending"
  },
  "task123_AI-ASSIST": {
    "accuracy": 0.90,
    "sensitivity": 0.90,
    "specificity": 0.90,
    "totalTime": {"seconds": 360, "formatted": "6:00"},
    "patientCount": 10,
    "edu_sub_mode": "assist",
    "llm_analysis_status": "completed"
  }
}
```

### 模块交互关系

```
┌─────────────┐     ┌─────────────────────────────┐     ┌─────────────┐
│ edu_admin   │ ←→  │ /api/edu/*                  │ ←→  │ routes     │
│ .html       │     │ 路由                        │     │ _oridata   │
│             │     │                             │     │ .py        │
└─────────────┘     │ - parse-zip                 │     └─────────────┘
                    │ - confirm-upload            │             │
                    │ - publish-task              │             │
                    │ - trigger-llm-analysis      │             ▼
                    └─────────────────────────────┘     ┌─────────────┐
                                                       │ SYSTEM/     │
┌─────────────┐     ┌─────────────────────────────┐     │ edu_data/   │
│ edu_status  │ ←→  │ /api/edu/user/*            │     │             │
│ .html       │     │                             │     │ - 原始数据  │
│             │     │ - tasks/{username}          │     │ - processed│
│             │     │ - result/{username}/{id}    │     │ - 成绩单    │
└─────────────┘     └─────────────────────────────┘     └─────────────┘
                            ↑
                            │
                    ┌───────┴───────┐
                    │ main.py       │
                    │ /api/diagnosis│
                    │ /submit-json  │
                    └───────────────┘
```

---

## 3. 大模型智能分析集成

### 设计难点与解决方案

#### 3.1 多API格式兼容

**问题**：不同大模型服务（智谱AI、OpenAI、DeepSeek、Moonshot）的API格式和endpoint路径不完全一致，需要兼容处理。

**解决方案**：自动检测API类型并选择正确的endpoint路径。

```python
# llm_analyzer.py:66-86
def _get_endpoint(self) -> str:
    if self.is_zhipu:  # 智谱AI
        if base.endswith('/api/paas/v4'):
            return f"{base}/chat/completions"
        elif base.endswith('/api'):
            return f"{base}/paas/v4/chat/completions"
        else:
            return f"{base}/api/paas/v4/chat/completions"
    else:  # OpenAI兼容格式
        if base.endswith('/v1'):
            return f"{base}/chat/completions"
        else:
            return f"{base}/v1/chat/completions"
```

**支持的API格式**：

| 服务商 | 识别特征 | Endpoint格式 |
|--------|---------|-------------|
| 智谱AI | URL包含"bigmodel"或"zhipu" | `/api/paas/v4/chat/completions` |
| OpenAI | 标准v1格式 | `/v1/chat/completions` |
| DeepSeek | 兼容OpenAI | `/v1/chat/completions` |
| Moonshot | 兼容OpenAI | `/v1/chat/completions` |

#### 3.2 异步非阻塞调用

**问题**：大模型API调用耗时较长（可能数十秒），如果在前端请求中同步等待会导致请求超时。

**解决方案**：使用Python的`asyncio.create_task`在后台异步执行，不阻塞前端请求。

```python
# main.py:1049-1052
# 诊断提交后立即返回，前端开始轮询
asyncio.create_task(analyze_with_llm(request.username, save_key, stats))
return {"message": "评估完成，AI分析报告生成中", "stats": stats}
```

```python
# main.py:1078-1154
async def analyze_with_llm(username: str, task_folder: str, stats: Dict[str, Any]):
    """异步调用大模型分析，完成后更新成绩单"""
    # 1. 读取配置
    # 2. 调用API
    # 3. 更新成绩单中的 llm_analysis_status 和 llm_analysis 字段
```

#### 3.3 前端轮询等待机制

**问题**：异步任务完成后，前端需要知道何时可以获取分析结果。

**解决方案**：前端轮询机制 + 状态标记。

```javascript
// edu_status.html
async function pollForLLMAnalysis(id, name, maxAttempts = 30) {
    // 先检查是否已完成
    if (data.llm_analysis_status === 'completed') {
        showDualStageReport(...);
        return;
    }
    
    // 轮询等待
    if (attempts < maxAttempts) {
        setTimeout(() => pollForLLMAnalysis(...), 1000);
    } else {
        // 超时，显示基础报告
    }
}
```

#### 3.4 提示词工程与响应解析

**问题**：大模型需要准确的上下文和数据单位说明才能给出正确分析，且响应格式可能包含在markdown代码块中。

**解决方案**：结构化的提示词模板 + 鲁棒的JSON解析。

```python
# 提示词包含：
# 1. 背景说明（系统用途、AI辅助概念）
# 2. 四种依赖的定义（正确依赖、依赖不足、正确独立、过度依赖）
# 3. 数据单位说明（秒、百分比等）
# 4. 具体统计数据

# 响应解析：
# 1. 尝试匹配 ```json ... ``` 代码块
# 2. 尝试直接解析JSON
# 3. 验证必要字段存在
```

#### 3.5 失败重试与状态记录

**问题**：API调用可能因网络问题或服务不稳定而失败，需要记录状态供前端显示。

**解决方案**：失败时更新成绩单中的状态字段。

```python
# main.py:1160-1177
except Exception as e:
    print(f"❌ 大模型分析失败: {e}")
    # 记录失败状态到成绩单
    if task_folder in user_data:
        user_data[task_folder]["llm_analysis_status"] = "failed"
        user_data[task_folder]["llm_analysis"] = {"error": str(e)}
```

### 核心流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      异步分析触发流程                                  │
├─────────────────────────────────────────────────────────────────────┤
│  用户提交诊断 → /api/diagnosis/submit-json                          │
│       │                                                             │
│       ├── 保存成绩到 Doctor_Diag_Result/{username}.json             │
│       │                                                              │
│       └── asyncio.create_task(analyze_with_llm)                      │
│                │                                                      │
│                └── 函数立即返回，前端开始轮询                           │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    analyze_with_llm 异步执行                          │
├─────────────────────────────────────────────────────────────────────┤
│  1. 读取 config/llm_models.json                                    │
│  2. 查找 selected_model_id 对应的模型配置                            │
│  3. 创建 LLMAnalyzer 实例                                            │
│  4. 调用 analyzer.analyze_performance(stats)                         │
│       │                                                             │
│       ├── _construct_prompt() → 构造完整提示词                       │
│       ├── _get_endpoint() → 选择正确API端点                         │
│       ├── _make_request() → 发送HTTP请求                           │
│       └── _parse_response() → 解析JSON响应                          │
│  5. 更新成绩单: llm_analysis_status + llm_analysis                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      前端轮询获取结果                                  │
├─────────────────────────────────────────────────────────────────────┤
│  edu_status.html: openResultModal()                                 │
│       │                                                             │
│       ├── GET /api/edu/user/result/{username}/{id}                 │
│       ├── 检查 llm_analysis_status                                  │
│       ├── 'pending' → 等待1秒后再次请求                              │
│       ├── 'completed' → 显示完整报告（含AI评价）                      │
│       └── 'failed' → 显示基础报告 + 重新触发按钮                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键数据结构

#### LLM配置文件 (config/llm_models.json)
```json
{
  "models": [
    {
      "id": "glm-4-pro",
      "display_name": "智谱 GLM-4",
      "base_url": "https://open.bigmodel.cn/api/paas/v4",
      "api_key": "xxx",
      "model": "glm-4",
      "description": "智谱AI大模型，适合医学分析"
    }
  ],
  "selected_model_id": "glm-4-pro"
}
```

#### 分析结果结构
```json
{
  "llm_analysis_status": "completed",
  "llm_analysis": {
    "status": "completed",
    "analysis": {
      "overall_evaluation": "整体表现良好...",
      "diagnosis_ability": "在VSD诊断上表现优秀...",
      "ai_tool_usage": "正确依赖率较高，AI辅助效果明显...",
      "improvement_suggestions": ["建议加强ASD诊断能力", "注意判读时间分配"],
      "strength_points": ["VSD检出率高", "判读效率高"],
      "weakness_points": ["ASD敏感度待提高", "部分病例判读过快"]
    }
  },
  "llm_model_used": "智谱 GLM-4"
}
```

### 模块交互关系

```
┌─────────────┐     ┌─────────────────────────────┐     ┌─────────────┐
│ edu_status  │ ←→  │ main.py                     │ ←→  │ llm_analyzer│
│ .html       │     │                             │     │ .py         │
│             │     │ /api/diagnosis/submit-json   │     │             │
│ 前端轮询    │     │   → asyncio.create_task()    │     │ LLMAnalyzer │
│             │     │                             │     │             │
└─────────────┘     └─────────────────────────────┘     └─────────────┘
                            │                                  │
                            │                                  │
                            ▼                                  ▼
                    ┌─────────────┐                    ┌─────────────┐
                    │ Doctor_     │                    │ 大模型API   │
                    │ Diag_Result │                    │ (智谱/OpenAI│
                    │ /{user}.json                    │  /DeepSeek) │
                    └─────────────┘                    └─────────────┘
```

---

## 4. 视频处理与帧提取

### 设计难点与解决方案

#### 4.1 视频编码兼容性转码

**问题**：不同设备录制的视频可能使用不同的编码格式（H.264、H.265、VP9等），浏览器原生支持的格式有限，导致视频无法在网页中播放。

**解决方案**：上传后使用ffmpeg将所有视频转码为H.264+AAC格式的MP4文件。

```python
# routes_oridata.py:96-135
async def transcode_to_h264(video_path: Path):
    """将视频转换为 H.264 编码"""
    temp_output = video_path.with_name(f"temp_{video_path.name}")
    
    cmd = (
        f'ffmpeg -y -i "{video_path.absolute()}" '
        f'-vcodec libx264 -pix_fmt yuv420p '
        f'-preset fast -crf 23 '
        f'-acodec aac "{temp_output.absolute()}"'
    )
    
    proc = await asyncio.create_subprocess_shell(cmd, ...)
    if proc.returncode == 0:
        os.replace(temp_output, video_path)  # 原子替换
```

**技术细节**：
- 使用`libx264`视频编码器
- `yuv420p`像素格式确保最大兼容性
- `crf 23`平衡质量和文件大小
- `create_subprocess_shell`确保能读取系统PATH中的ffmpeg

#### 4.2 缩略图异步生成

**问题**：视频上传时需要生成缩略图供前端显示，但视频处理耗时较长，不能阻塞上传流程。

**解决方案**：使用`asyncio.create_subprocess_shell`异步调用ffmpeg生成缩略图。

```python
# routes_oridata.py:85-93
async def generate_thumbnail_ffmpeg(video_path: Path, thumb_path: Path, seek_time: float = 0.05):
    cmd = f'ffmpeg -y -ss {seek_time} -i "{video_path.absolute()}" -frames:v 1 -q:v 2 "{thumb_path.absolute()}"'
    proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    await proc.communicate()
    return proc.returncode == 0
```

**关键点**：
- `-ss 0.05`定位到视频开头5%位置（避免黑屏）
- `-frames:v 1`只提取一帧
- `-q:v 2`高质量输出

#### 4.3 帧提取与动态检测

**问题**：不同视频帧数不同（16帧、20帧、31帧等），不能硬编码帧数，否则会导致404错误或加载不存在的帧。

**解决方案**：前端动态检测实际存在的帧数。

```javascript
// diagnosis.html:2313-2335
async function detectFrameCount(framesDir) {
    let frameCount = 0;
    const maxFrames = 100;

    for (let i = 0; i < maxFrames; i++) {
        const framePath = `${framesDir}/frame_${String(i).padStart(4, '0')}.png`;
        const response = await fetch(framePath, { method: 'HEAD' });
        if (response.ok) {
            frameCount = i + 1;
        } else {
            break;  // 找不到就结束
        }
    }
    return frameCount;
}
```

#### 4.4 批量帧提取工具

**问题**：需要将视频帧提取为图片用于3D可视化展示，且每个视频需要独立的frames文件夹避免覆盖。

**解决方案**：批量提取工具为每个视频创建独立的`{video_name}/frames/`目录结构。

```bash
# 使用示例
python batch_extract_frames.py video1.mp4 video2.mp4

# 输出结构
video1/
├── video1.mp4
└── frames/
    ├── frame_0000.png
    ├── frame_0001.png
    └── ...
video2/
├── video2.mp4
└── frames/
    ├── frame_0000.png
    ├── frame_0001.png
    └── ...
```

**帧采样算法**：
```python
# batch_extract_frames.py:75-77
for i in range(actual_frames):
    frame_index = int(i * total_frames / actual_frames)  # 均匀采样
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
```

#### 4.5 目录结构适配

**问题**：不同来源的视频可能有不同的目录层级，需要统一转换为AI模型期望的目录结构。

**解决方案**：上传时自动解析路径层级并重组。

```python
# routes_oridata.py:350-420
# CASE A: 无路径 → 生成默认病例名
if not anySlash:
    row = { caseName: f"patient {len(staging)+1}", files: [...] }

# CASE B1: 3层及以上路径 (TaskName/patient001/video.mp4)
# 提取第一层作为任务名，第二层作为病例名

# CASE B2: 2层路径 (patient001/video.mp4)
# 提取第一层作为病例名
```

### 核心流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      视频上传与处理流程                               │
├─────────────────────────────────────────────────────────────────────┤
│  dashboard.html                                                     │
│      │                                                              │
│      ├── 选择文件/拖拽文件夹                                          │
│      └── POST /api/users/{username}/upload-oridata                   │
│               │                                                      │
│               ├── 保存原始视频到 oridata/{submission_id}/           │
│               │    ↓                                                  │
│               ├── 触发后台任务: run_model_inference_wrapper          │
│               │    │                                                  │
│               │    ├── AI模型推理                                    │
│               │    ├── 生成热力图/边界框/关键帧                       │
│               │    ├── 转码为H.264 (transcode_to_h264)              │
│               │    └── 更新索引状态 (is_cmp=True)                    │
│               │                                                      │
│               └── 返回上传结果                                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      帧提取与3D渲染流程                               │
├─────────────────────────────────────────────────────────────────────┤
│  batch_extract_frames.py                                            │
│      │                                                              │
│      ├── 读取视频获取帧数/FPS/分辨率                                │
│      ├── 均匀采样提取帧 (frame_0000.png - frame_XXXX.png)           │
│      └── 保存到 {video_name}/frames/                                │
│               │                                                      │
│               ▼                                                      │
│  diagnosis.html                                                     │
│      │                                                              │
│      ├── 点击视频 → openVideoModal()                                │
│      ├── detectFrameCount() 动态检测实际帧数                         │
│      ├── loadFrames() 加载所有帧图片                                 │
│      └── 创建3D切片堆叠                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键数据结构

#### 帧文件命名规范
```
frames/
├── frame_0000.png   ← 4位数字补零
├── frame_0001.png
├── frame_0002.png
└── ...
```

#### metadata.json结构 (AI模型输出)
```json
{
  "key_frame_index": 10,
  "roi_crop_box_original_space": [50, 80, 300, 200],
  "width": 512,
  "height": 288,
  "confidence_scores": {
    "Normal": 0.85,
    "VSD": 0.12,
    "ASD": 0.02,
    "PDA": 0.01
  }
}
```

### 模块交互关系

```
┌─────────────┐     ┌─────────────────────────────┐     ┌─────────────┐
│ dashboard   │ ←→  │ routes_oridata.py            │ ←→  │ AI模型     │
│ .html       │     │                             │     │ run_diagn  │
│             │     │ - upload-oridata            │     │ esis       │
│ 前端上传    │     │ - transcode_to_h264         │     │            │
│             │     │ - generate_thumbnail_ffmpeg  │     └─────────────┘
└─────────────┘     └─────────────────────────────┘
                            │
                            │
                            ▼
                    ┌─────────────┐
                    │ 视频文件     │
                    │ data_batch  │
                    │ _storage/   │
                    └─────────────┘

┌─────────────┐     ┌─────────────────────────────┐     ┌─────────────┐
│ batch_     │ ←→  │ diagnosis.html              │ ←→  │ video_3d   │
│ extract    │     │                             │     │ _modal.js  │
│ _frames.py │     │ - detectFrameCount()        │     │            │
│             │     │ - loadFrames()             │     │ Three.js   │
│ 帧提取工具  │     │ - 动态检测+加载            │     │ 3D渲染     │
└─────────────┘     └─────────────────────────────┘     └─────────────┘
```

---

## 5. 诊断数据流与状态管理

### 设计难点与解决方案

#### 5.1 查看时间追踪

**问题**：需要记录医师对每个病例的查看时间，用于分析诊断效率和行为模式。  

**解决方案**：在前端使用计时器追踪每个病例的查看时长，提交时一并发送到后端。

```javascript
// diagnosis.html
let patientViewTimes = {};  // {patientId: seconds}
let currentPatientStartTime = null;

// 切换病例时保存上一个病例的时间
if (currentPatientStartTime !== null) {
    const viewTime = Math.floor((Date.now() - currentPatientStartTime) / 1000);
    patientViewTimes[prevPatientId] = (patientViewTimes[prevPatientId] || 0) + viewTime;
}

// 提交时包含查看时间
const diagnosisData = {
    totalTime: { seconds: timerSeconds, formatted: "MM:SS" },
    patientCount: Object.keys(diagnosisRecords).length,
    records: Object.values(diagnosisRecords).map(record => ({
        patientId: record.patientId,
        diagnosis: record.diagnosis,
        severity: record.severity,
        viewTime: patientViewTimes[record.patientId] || 0  // 查看时间(秒)
    }))
};
```

#### 5.2 草稿缓存机制

**问题**：医师填写大量诊断信息后，如果浏览器意外关闭或刷新，数据会丢失。

**解决方案**：使用sessionStorage保存诊断草稿，页面加载时自动恢复。

```javascript
// diagnosis.html
function saveDraft(reason = '') {
    const draft = {
        v: 1,
        username: taskInfo.username,
        taskFolder: taskInfo.taskFolder,
        savedAt: Date.now(),
        timerSeconds,
        timerPaused,
        currentPatientIndex,
        diagnosisRecords,
        inProgress: getCurrentUiSelection(),
        reason
    };
    sessionStorage.setItem(getDraftKey(), JSON.stringify(draft));
}

function loadDraft() {
    const raw = sessionStorage.getItem(getDraftKey());
    if (!raw) return null;
    const draft = JSON.parse(raw);
    // 验证草稿属于当前任务
    if (draft.username !== taskInfo.username || draft.taskFolder !== taskInfo.taskFolder) return null;
    return draft;
}

// 页面加载时
const draft = loadDraft();
if (draft) {
    // 恢复草稿状态
    timerSeconds = draft.timerSeconds;
    currentPatientIndex = draft.currentPatientIndex;
    diagnosisRecords = draft.diagnosisRecords;
}
```

**特点**：
- `sessionStorage`：仅在当前浏览器会话有效，关闭标签页或浏览器后清除
- 包含版本号`v`便于后续兼容
- 记录保存原因便于调试

#### 5.3 诊断数据提交流程

**问题**：诊断数据需要同时支持判读模式和教育模式，且教育模式需要关联到具体的eduSubMode。

**解决方案**：统一的数据结构，通过`mode`字段区分不同模式。

```javascript
// diagnosis.html 提交数据结构
const diagnosisData = {
    username: taskInfo.username,
    taskFolder: taskInfo.taskFolder,
    mode: currentMode,  // "diag" 或 "edu"
    eduSubMode: currentMode === 'edu' ? eduSubMode : null,  // "single" 或 "assist"
    submittedAt: new Date().toISOString(),
    totalTime: { seconds: timerSeconds, formatted: formattedTime },
    patientCount: Object.keys(diagnosisRecords).length,
    records: [...]
};
```

```python
# main.py 接收后根据mode分流处理
if request.mode == "edu":
    # 教育模式：计算准确率、敏感度、特异性，保存到成绩单
    # 触发大模型分析
else:
    # 判读模式：直接保存诊断报告到文件
    json_filename = f"final_diagnosis_report_{timestamp}.json"
    with open(task_path / json_filename, 'w') as f:
        json.dump(data_to_save, f)
```

#### 5.4 任务状态轮询

**问题**：后台AI处理任务需要较长时间，前端需要知道任务何时完成。

**解决方案**：前端定期轮询任务状态。

```javascript
// edu_status.html
async function pollTaskCompletion(submissionId, maxAttempts = 60) {
    const response = await fetch(`/api/edu/user/tasks/${username}`);
    const { tasks } = await response.json();
    const task = tasks.find(t => t.submission_id === submissionId);
    
    if (task?.is_completed) {
        // 任务完成，刷新列表
        fetchEduTasks();
    } else if (attempts < maxAttempts) {
        // 继续轮询
        setTimeout(() => pollTaskCompletion(submissionId, attempts + 1), 2000);
    }
}
```

### 核心流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      诊断提交数据流                                   │
├─────────────────────────────────────────────────────────────────────┤
│  diagnosis.html                                                     │
│      │                                                              │
│      ├── 医师查看病例、选择诊断结果                                   │
│      ├── 记录每个病例的查看时间 (patientViewTimes)                   │
│      └── 点击"提交诊断" → POST /api/diagnosis/submit-json           │
│               │                                                      │
│               ├── mode="edu" → 教育模式                              │
│               │    │                                                │
│               │    ├── 计算准确率/敏感度/特异性                      │
│               │    ├── 分析AI依赖性                                  │
│               │    ├── 保存到 Doctor_Diag_Result/{user}.json        │
│               │    │    └── 键名: taskId + "_SINGLE"/"_AI-ASSIST"  │
│               │    └── 触发异步大模型分析                            │
│               │                                                      │
│               └── mode="diag" → 判读模式                             │
│                    │                                                │
│                    └── 保存到 final_diagnosis_report_{timestamp}.json
└─────────────────────────────────────────────────────────────────────┘
```

### 关键数据结构

#### 诊断提交请求
```python
class DiagnosisSubmitJsonRequest(BaseModel):
    username: str
    taskFolder: str
    mode: str = "diag"  # diag | edu
    eduSubMode: Optional[str] = None  # single | assist (仅教育模式)
    submittedAt: str
    totalTime: Dict[str, Any]  # {seconds: int, formatted: "MM:SS"}
    patientCount: int
    records: List[DiagnosisRecordSimple]
```

#### 诊断记录
```python
class DiagnosisRecordSimple(BaseModel):
    patientId: str
    diagnosis: Optional[str]  # Normal/VSD/ASD/PDA
    severity: Optional[str]   # 严重程度
    viewTime: int = 0  # 查看时间(秒)
```

#### 教育成绩单
```json
{
  "task123_SINGLE": {
    "accuracy": 0.75,
    "sensitivity": 0.80,
    "specificity": 0.70,
    "totalTime": {"seconds": 300, "formatted": "5:00"},
    "patientCount": 10,
    "category_stats": {...},
    "ai_dependency": {...},
    "view_times": [30, 45, 20, ...],
    "edu_sub_mode": "single",
    "llm_analysis_status": "pending"
  },
  "task123_AI-ASSIST": {...}
}
```

### 模块交互关系

```
┌─────────────┐     ┌─────────────────────────────┐     ┌─────────────┐
│ diagnosis   │ ←→  │ main.py                     │ ←→  │ 成绩单文件  │
│ .html      │     │                             │     │ Doctor_Diag │
│             │     │ /api/diagnosis/submit-json  │     │ _Result/   │
│ 诊断数据    │     │   → 计算统计指标            │     │             │
│ 提交+时间   │     │   → 保存到对应位置         │     └─────────────┘
│ 追踪        │     └─────────────────────────────┘
└─────────────┘              │
                            │
                            ▼
                    ┌─────────────────────────────┐
                    │ llm_analyzer.py             │
                    │                             │
                    │ asyncio.create_task()       │
                    │   → 异步调用大模型API        │
                    └─────────────────────────────┘
```

---

## 6. 用户权限与数据隔离

### 设计难点与解决方案

#### 6.1 保留用户名保护（SYSTEM）

**问题**：系统使用"SYSTEM"作为保留用户名用于教育模式数据存储，如果允许普通用户注册SYSTEM账户，会导致数据混乱和安全问题。

**解决方案**：在用户创建和检查时拦截SYSTEM用户名。

```python
# database.py
def create_user(username: str, password: str, ...):
    # [核心加固] 禁止注册系统保留名
    if username.upper() == "SYSTEM":
        print(f"🚫 拦截：禁止创建保留用户名 {username}")
        return False

def check_username_exists(username: str, ...):
    # 大小写不敏感，阻止任何变体
    if username.upper() == "SYSTEM":
        return True  # 假装已存在，阻止注册
```

**效果**：
- `system` / `System` / `SYSTEM` / `SyStEm` 都无法注册
- 任何登录验证都会失败（因为数据库中不存在）

#### 6.2 Cookie-based认证

**问题**：HTTP是无状态协议，需要在多次请求间保持用户登录状态。

**解决方案**：使用Cookie存储用户名，后端通过Cookie获取当前用户。

```python
# main.py:89-122
@app.post("/api/login")
async def login(request: LoginRequest, response: Response):
    user = verify_user(request.username, request.password)
    if user:
        response.set_cookie(key="username", value=user["username"], httponly=True)
        return {"data": user}
    raise HTTPException(status_code=401)

@app.get("/api/users/current")
async def get_current_user(request: Request):
    username = request.cookies.get("username")
    if not username:
        raise HTTPException(status_code=401)
    user = get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=401)
    return {"data": user}
```

**安全特性**：
- `httponly=True`：防止JavaScript访问Cookie（XSS攻击防护）
- 密码不在Cookie中存储
- 每次请求验证用户存在性

#### 6.3 角色权限管理

**问题**：需要区分普通用户和管理员，管理员可以访问用户管理、教育管理等功能。

**解决方案**：数据库增加`is_admin`字段，前端根据用户角色显示/隐藏功能。

```python
# database.py - 用户表结构
cursor.execute('''
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        doctor TEXT NOT NULL,
        organization TEXT NOT NULL,
        is_admin BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
```

```javascript
// flow.html - 前端权限控制
const isAdmin = u.is_admin === true;
// 根据is_admin动态显示管理按钮
if (isAdmin) {
    document.getElementById('adminBtn').style.display = '';
}
```

#### 6.4 用户私有空间隔离

**问题**：每个用户只能访问自己的任务数据和上传文件，不能访问其他用户的数据。

**解决方案**：数据按用户目录隔离，API路径包含用户名。

```
data_batch_storage/
├── admin/              ← 管理员用户
│   ├── oridata/
│   ├── processed/
│   └── data.json
├── doctor1/           ← 普通用户
│   ├── oridata/
│   ├── processed/
│   └── data.json
└── doctor2/           ← 普通用户
    └── ...
```

```python
# routes_oridata.py - 路径隔离
user_root = BASE_DATA_DIR / username  # 基于Cookie中的用户名
# 所有文件操作都在 user_root 下进行
```

#### 6.5 教育数据隔离

**问题**：教育模式的任务数据存放在SYSTEM保留空间中，需要确保：
- 管理员可以发布/管理教育任务
- 普通用户只能看到分配给自己的任务
- 其他用户看不到别人的成绩

**解决方案**：使用`target_users`列表控制任务分发。

```python
# routes_oridata.py - 发布任务
def publish_task(submission_id: str, target_users: List[str]):
    # 更新任务索引，添加目标用户列表
    update_edu_task_index({
        "submission_id": submission_id,
        "status": "published",
        "target_users": target_users
    })

# 获取用户任务时过滤
def get_user_tasks(username: str):
    all_tasks = load_edu_tasks()
    # 只返回 target_users 包含当前用户的任务
    return [t for t in all_tasks if username in t.get("target_users", [])]
```

### 核心流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户认证流程                                   │
├─────────────────────────────────────────────────────────────────────┤
│  login.html                                                         │
│      │                                                              │
│      ├── POST /api/login {username, password}                       │
│      │    │                                                         │
│      │    └── verify_user() → 设置Cookie: username                  │
│      │                                                               │
│      └── 跳转 /flow                                                  │
│               │                                                       │
│               ├── GET /api/users/current → 获取当前用户信息          │
│               │    │                                                │
│               │    └── 返回 {username, is_admin, organization, ...} │
│               │                                                      │
│               └── 根据 is_admin 显示不同功能入口                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         数据访问控制流程                               │
├─────────────────────────────────────────────────────────────────────┤
│  用户请求 /api/users/{username}/...                                  │
│      │                                                              │
│      ├── 从Cookie获取当前用户                                         │
│      ├── 比较 username 参数 vs Cookie中的用户                         │
│      │    │                                                         │
│      │    ├── 不一致 → 403 Forbidden                                │
│      │    └── 一致 → 允许访问                                       │
│      │                                                               │
│      └── 文件路径: data_batch_storage/{username}/...                  │
│               │                                                      │
│               └── 自动隔离，不存在越权访问                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键数据结构

#### 用户表结构
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,           -- SHA-256哈希
    doctor TEXT NOT NULL,             -- 医生姓名
    organization TEXT NOT NULL,       -- 机构名称
    is_admin BOOLEAN DEFAULT 0,       -- 是否管理员
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

#### 教育任务索引
```json
{
  "tasks": [
    {
      "submission_id": "edu_xxx",
      "request_name": "教育批次A",
      "status": "published",
      "target_users": ["doctor1", "doctor2"],
      "request_case_cnt": 10,
      "request_video_cnt": 30
    }
  ]
}
```

### 模块交互关系

```
┌─────────────┐     ┌─────────────────────────────┐     ┌─────────────┐
│ 前端页面    │ ←→  │ main.py                     │ ←→  │ database.py │
│             │     │                             │     │             │
│ login.html │     │ /api/login → 设置Cookie    │     │ verify_user │
│ flow.html  │     │ /api/users/current          │     │ create_user │
│ admin.html │     │ /api/users/{id}              │     │ update_user │
│             │     │ /api/users/*                 │     │ is_admin    │
└─────────────┘     └─────────────────────────────┘     └─────────────┘
                            │
                            │
                            ▼
                    ┌─────────────┐
                    │ 用户目录     │
                    │ data_batch  │
                    │ _storage/   │
                    │ {username}/ │
                    └─────────────┘
```

---

## 文档完成

所有模块已完成整理。
