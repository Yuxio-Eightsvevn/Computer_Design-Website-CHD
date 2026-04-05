# 儿童先心病超声诊断平台 - 部署指南

## 一、环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10+ |
| 运行环境 | Conda (推荐) 或 Virtualenv |
| 系统依赖 | ffmpeg (视频处理) |
| 磁盘空间 | ≥5GB (含模型和视频数据) |

---

## 二、克隆与初始化

### 2.1 克隆仓库

```bash
git clone <仓库地址>
cd Diagnostic_Platform_Stable_Version_C3D
```

### 2.2 创建Python环境

```bash
# 使用Conda创建环境
conda create -n challengeserver python=3.10
conda activate challengeserver

# 或使用venv
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

### 2.3 安装依赖

```bash
pip install -r requirements.txt

# 安装模型额外依赖
cd model
pip install -r requirements.txt
cd ..
```

### 2.4 安装系统依赖

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
```bash
# 下载 ffmpeg: https://ffmpeg.org/download.html
# 添加到系统PATH环境变量
```

**macOS:**
```bash
brew install ffmpeg
```

---

## 三、AI模型配置

### 3.1 下载模型权重

从原项目获取以下模型文件，放入对应目录：

```
model/
├── singleview/           # 单视角模型 (4个)
│   ├── spatial_size_4.pth
│   ├── spatial_size_5.pth
│   ├── spatial_size_6.pth
│   └── spatial_size_7.pth
│
├── multiviews/           # 多视角模型 (1个)
│   └── epoch_97.98295452219057.pth
│
└── Baseline_4size/      # 基线特征 (4个)
    ├── final_baseline_features_4x4_512dim_1101.pt
    ├── final_baseline_features_5x5_512dim_1101.pt
    ├── final_baseline_features_6x6_512dim_1101.pt
    └── final_baseline_features_7x7_512dim_1101.pt
```

**模型文件约 1GB**，可从以下途径获取：
- 原项目部署机器的 `model/` 目录
- 训练好的模型文件

### 3.2 配置大模型API (可选)

编辑 `config/llm_config.json`:

```json
{
    "base_url": "https://open.bigmodel.cn/api",
    "api_key": "your-api-key-here",
    "model": "glm-4"
}
```

如不使用AI分析报告功能，可跳过此步骤。

---

## 四、目录结构说明

### 4.1 系统自动创建的目录

运行 `python main.py` 时，系统会自动创建以下目录：

```
data_batch_storage/
├── SYSTEM/
│   └── edu_data/          # 教育模式数据
├── {username}/
│   ├── oridata/           # 用户上传的原始视频
│   └── processed/         # AI处理结果
└── ...
```

**注意**: `data_batch_storage/` 目录会被 `.gitignore` 忽略，不会提交到仓库。

### 4.2 Git追踪的文件

| 文件/目录 | 说明 |
|-----------|------|
| `users.db` | SQLite用户数据库 |
| `config/llm_config.json` | 大模型配置 |
| `model/Codes/` | 模型代码 |
| `model/README.md` | 模型说明 |
| `model/*.py` | 模型入口脚本 |
| `model/singleview/.gitkeep` | 占位文件 |
| `model/multiviews/.gitkeep` | 占位文件 |
| `model/Baseline_4size/.gitkeep` | 占位文件 |

### 4.3 Git忽略的文件

| 文件/目录 | 说明 |
|-----------|------|
| `__pycache__/` | Python缓存 |
| `*.pyc` | 编译文件 |
| `data_batch_storage/` | 用户数据 |
| `model/*.pth` | 模型权重 |
| `model/*.pt` | 基线特征 |
| `.venv/` | 虚拟环境 |
| `.idea/` / `.vscode/` | IDE配置 |

---

## 五、启动服务

### 5.1 开发模式

```bash
python main.py
```

服务启动后访问: **http://localhost:11000/**

### 5.2 生产模式

```bash
# 后台运行
nohup uvicorn main:app --host 0.0.0.0 --port 11000 --workers 4 > app.log 2>&1 &

# 或使用screen
screen -S server
uvicorn main:app --host 0.0.0.0 --port 11000 --workers 4
# Ctrl+A+D 退出screen
```

---

## 六、验证部署

### 6.1 默认账户

| 角色 | 用户名 | 密码 |
|------|--------|------|
| 管理员 | admin | 123456 |
| 普通用户 | doctor1 | 123456 |

### 6.2 检查清单

- [ ] Python环境激活成功
- [ ] 依赖安装完成 (无报错)
- [ ] ffmpeg可用 (`ffmpeg -version`)
- [ ] 模型权重文件存在
- [ ] 服务启动成功 (无端口冲突)
- [ ] 登录页可访问 (http://localhost:11000/)
- [ ] 默认账户可登录
- [ ] 上传视频功能正常
- [ ] AI处理功能正常 (如配置了模型)

### 6.3 常见问题

| 问题 | 解决方案 |
|------|----------|
| 端口被占用 | 修改main.py中的端口号或杀死占用进程 |
| 模型文件缺失 | 检查model/目录下的权重文件是否完整 |
| ffmpeg不可用 | 确认已安装并加入PATH |
| 视频无法播放 | 检查ffmpeg是否正确转码 |
| 数据库错误 | 删除users.db后重启，会自动重建 |

---

## 七、备份与恢复

### 7.1 需要备份的数据

如需保留用户数据和配置：

```bash
# 备份
cp users.db users.db.backup
cp config/llm_config.json config/llm_config.json.backup
cp -r data_batch_storage data_batch_storage.backup
```

### 7.2 恢复数据

```bash
# 恢复
cp users.db.backup users.db
cp config/llm_config.json.backup config/llm_config.json
cp -r data_batch_storage.backup data_batch_storage
```

---

## 八、更新部署

如需更新到最新代码：

```bash
# 拉取更新
git pull origin master

# 重启服务
# 先停止旧进程，再重新启动
pkill -f uvicorn
python main.py
```

**注意**: 更新代码不会影响 `data_batch_storage/` 中的用户数据和 `users.db`。

---

*文档版本: 2026-04-05*
