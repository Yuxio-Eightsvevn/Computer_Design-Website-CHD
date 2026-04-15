from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from routes_oridata import router as oridata_router
import database
import json
from datetime import datetime
import shutil
import zipfile
import re  # 确保正则模块可用
from fastapi.responses import JSONResponse
from llm_analyzer import LLMAnalyzer, test_connection
import asyncio


# 数据存储目录
DATA_BATCH_STORAGE = "data_batch_storage"

# [新增] 教育模式系统根目录
SYSTEM_EDU_DIR = Path(DATA_BATCH_STORAGE) / "SYSTEM" / "edu_data"
RESULT_STORAGE_DIR = SYSTEM_EDU_DIR / "Doctor_Diag_Result"

# [新增] 大模型配置文件路径
LLM_CONFIG_FILE = Path("config/llm_config.json")  # 旧配置文件（保留用于迁移）
LLM_MODELS_FILE = Path("config/llm_models.json")  # 新配置文件


# [新增] 大模型配置请求模型
class LLMConfigRequest(BaseModel):
    base_url: str
    api_key: str
    model: str = "glm-4"

# [新增] 单个模型配置模型
class LLMModelConfig(BaseModel):
    display_name: str
    base_url: str
    api_key: str
    model: str
    description: Optional[str] = ""


# [新增] 配置迁移函数
def migrate_llm_config():
    """迁移旧的LLM配置到新格式"""
    try:
        if not LLM_CONFIG_FILE.exists():
            return
        
        with open(LLM_CONFIG_FILE, "r", encoding="utf-8") as f:
            old_config = json.load(f)
        
        # 只在有有效配置时才迁移
        if old_config.get("base_url") and old_config.get("api_key"):
            new_config = {
                "models": [{
                    "id": "migrated_model",
                    "display_name": f"迁移的模型 ({old_config.get('model', 'unknown')})",
                    "base_url": old_config.get("base_url", ""),
                    "api_key": old_config.get("api_key", ""),
                    "model": old_config.get("model", ""),
                    "description": "从旧配置自动迁移",
                    "created_at": datetime.now().isoformat()
                }],
                "selected_model_id": "migrated_model"
            }
            
            with open(LLM_MODELS_FILE, "w", encoding="utf-8") as f:
                json.dump(new_config, f, ensure_ascii=False, indent=2)
            
            # 备份旧配置
            import shutil
            backup_path = LLM_CONFIG_FILE.with_suffix(".json.bak")
            shutil.move(str(LLM_CONFIG_FILE), str(backup_path))
            print(f"✅ LLM配置已迁移到新格式，旧配置已备份到: {backup_path}")
        else:
            # 旧配置无效，创建空配置
            default_config = {
                "models": [],
                "selected_model_id": None
            }
            with open(LLM_MODELS_FILE, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            print(f"🤖 LLM模型配置文件已创建: {LLM_MODELS_FILE}")
            
    except Exception as e:
        print(f"⚠️ LLM配置迁移失败: {e}")
        # 创建空配置
        default_config = {
            "models": [],
            "selected_model_id": None
        }
        with open(LLM_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)

# 系统初始化
def init_system():
    """初始化系统"""
    # 初始化数据库
    database.init_database()
    
    # 创建主存储目录
    storage_path = Path(DATA_BATCH_STORAGE)
    storage_path.mkdir(exist_ok=True)

    # 2. [重点新增] 初始化教育模式空间
    RESULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    edu_index = SYSTEM_EDU_DIR / "data.json"
    if not edu_index.exists():
        with open(edu_index, "w", encoding="utf-8") as f:
            json.dump({"tasks": []}, f, ensure_ascii=False, indent=2)
    print(f"🏛️ 系统空间已就绪: {SYSTEM_EDU_DIR}")
    
    # 3. [新增] 初始化LLM配置
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # 检查新配置文件是否存在
    if not LLM_MODELS_FILE.exists():
        # 检查是否有旧配置需要迁移
        if LLM_CONFIG_FILE.exists():
            migrate_llm_config()
        else:
            # 创建空配置
            default_config = {
                "models": [],
                "selected_model_id": None
            }
            with open(LLM_MODELS_FILE, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            print(f"🤖 LLM模型配置文件已创建: {LLM_MODELS_FILE}")
    
    # 为所有用户创建文件夹
    users = database.get_all_users()
    for user in users:
        doctor_folder = storage_path / user['username']
        if not doctor_folder.exists():
            doctor_folder.mkdir(parents=True)
            print(f"✅ 创建文件夹: {doctor_folder}")
        else:
            print(f"📁 文件夹已存在: {doctor_folder}")

# 使用新的 lifespan 事件处理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    print("🚀 系统启动中...")
    init_system()
    print("✅ 系统初始化完成！")
    yield
    # 关闭时执行（如有需要）
    print("👋 系统关闭")

app = FastAPI(lifespan=lifespan)

app.include_router(oridata_router)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class LoginRequest(BaseModel):
    username: str
    password: str

# 响应模型
class LoginResponse(BaseModel):
    data: Dict[str, Any]

# 登录接口
@app.post("/api/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    username = request.username
    password = request.password
    
    # 从数据库验证用户
    user = database.verify_user(username, password)
    
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    # --- [新增：data.json 自动初始化钩子] ---
    user_root = Path(DATA_BATCH_STORAGE) / username
    user_root.mkdir(parents=True, exist_ok=True)
    index_path = user_root / "data.json"
    
    if not index_path.exists():
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump({"tasks": []}, f, ensure_ascii=False, indent=2)
            print(f"📄 已为用户 {username} 初始化任务索引文件")
        except Exception as e:
            print(f"⚠️ 初始化索引文件失败: {e}")
    # ---------------------------------------

    resp = JSONResponse({"data": user})
    resp.set_cookie(
        key="username",
        value=user["username"],
        httponly=False,   # 最小改动，前端可读取 document.cookie
        samesite="Lax",
        path="/"
    )
    return resp

@app.post("/api/logout")
async def logout():
    resp = JSONResponse({"message": "已登出"})
    resp.delete_cookie(key="username", path="/")
    return resp

# 用户管理相关的请求模型
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

# 获取所有用户列表
@app.get("/api/users")
async def get_users():
    users = database.get_all_users()
    return {"data": users}

@app.get("/api/users/current")
async def get_current_user(request: Request):
    """
    静态 endpoint，优先于 /api/users/{user_id} 被匹配。
    优先查 cookie/header 中的 username，然后尝试通过 database.get_user_by_username 查找用户。
    如果未提供 username，则返回第一个用户（如果存在）。
    返回格式与 get_users 保持一致： {"data": user}
    """
    # 1) 从 cookie 或 header 尝试读取用户名
    cookie_un = None
    try:
        cookie_un = request.cookies.get("username") or request.cookies.get("user")
    except Exception:
        cookie_un = None

    header_un = request.headers.get("x-username") or request.headers.get("X-User")

    lookup = cookie_un or header_un
    if lookup:
        # 优先使用 database 中现有的按用户名查找函数
        try:
            if hasattr(database, "get_user_by_username"):
                user = database.get_user_by_username(lookup)
                if user:
                    return {"data": user}
        except Exception:
            # 忽略查找时的异常，继续降级查找
            pass

        # 降级实现：遍历所有用户按常见字段匹配
        try:
            all_users = database.get_all_users() or []
        except Exception:
            all_users = []

        lower_lookup = str(lookup).lower()
        for u in all_users:
            # u 预期为 dict
            for k in ("username", "user", "login", "name", "full_name", "email", "id"):
                # 兼容 dict 或对象属性
                val = u.get(k) if isinstance(u, dict) else getattr(u, k, None)
                if val is None:
                    continue
                if str(val).lower() == lower_lookup:
                    return {"data": u}

    # 3) 未提供 username 或未匹配到：返回第一个用户（若有）
    try:
        users = database.get_all_users() or []
    except Exception:
        users = []
    if users:
        return {"data": users[0]}

    # 4) 没有任何用户
    raise HTTPException(status_code=404, detail="no users in database")


# 获取单个用户信息
@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    user = database.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"data": user}

# 创建新用户
@app.post("/api/users")
async def create_user(request: UserCreateRequest):
    # 检查用户名是否已存在
    if database.check_username_exists(request.username):
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    success = database.create_user(
        request.username,
        request.password,
        request.doctor,
        request.organization
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="创建用户失败")
    
    return {"message": "用户创建成功"}

# 更新用户信息
@app.put("/api/users/{user_id}")
async def update_user(user_id: int, request: UserUpdateRequest):
    user = database.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    success = database.update_user(
        user_id,
        request.doctor,
        request.organization,
        request.password,
        request.is_admin
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="更新用户失败")
    
    return {"message": "用户更新成功"}

# 删除用户
@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int):
    user = database.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    success = database.delete_user(user_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="删除用户失败")
    
    return {"message": "用户删除成功"}

# 获取用户的任务列表
@app.get("/api/tasks/{username}")
async def get_user_tasks(username: str):
    """
    获取指定用户的所有任务
    遍历用户文件夹下的所有子文件夹，读取其中的tasks.json文件
    检查是否存在CSV文件来判断任务是否完成
    """
    user_folder = Path(DATA_BATCH_STORAGE) / username
    
    # 检查用户文件夹是否存在
    if not user_folder.exists():
        raise HTTPException(status_code=404, detail="用户文件夹不存在")
    
    tasks = []
    
    # 遍历用户文件夹下的所有子文件夹
    for task_folder in user_folder.iterdir():
        if task_folder.is_dir():
            # 查找tasks.json文件
            task_json_path = task_folder / "tasks.json"
            
            if task_json_path.exists():
                try:
                    # 读取JSON文件
                    with open(task_json_path, 'r', encoding='utf-8') as f:
                        task_data = json.load(f)
                    
                    # 检查该任务文件夹下是否有JSON结果文件（判断任务是否完成）
                    json_files = list(task_folder.glob("diagnosis_results_*.json"))
                    is_completed = len(json_files) > 0
                    
                    # 如果有JSON文件，获取最新的完成时间
                    completion_time = None
                    if json_files:
                        # 按修改时间排序，取最新的
                        latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
                        completion_time = datetime.fromtimestamp(latest_json.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 提取任务信息
                    if 'tasks' in task_data and len(task_data['tasks']) > 0:
                        for task in task_data['tasks']:
                            tasks.append({
                                'folder': task_folder.name,  # 任务文件夹名称
                                'name': task.get('name', '未命名任务'),
                                'description': task.get('description', '暂无描述'),
                                'completed': is_completed,  # 是否完成
                                'completionTime': completion_time,  # 完成时间
                                'resultCount': len(json_files)  # JSON结果文件数量
                            })
                except Exception as e:
                    print(f"读取任务文件失败 {task_json_path}: {e}")
                    continue
    
    return {"data": tasks}

# --- [重点重构：智慧适配、归组逻辑与元数据对齐] ---

@app.get("/api/tasks/{username}/{task_folder}/patients")
async def get_task_patients(username: str, task_folder: str):
    """
    [重构版] 获取病例列表。
    适配逻辑：支持模糊匹配，并增加 SYSTEM 空间搜索以支持教育模式。
    """
    # 1. 定义所有可能的物理搜索根路径
    user_proc_root = Path(DATA_BATCH_STORAGE) / username / "processed"
    user_root_root = Path(DATA_BATCH_STORAGE) / username
    system_edu_root = Path(DATA_BATCH_STORAGE) / "SYSTEM" / "edu_data" / "processed"

    task_path = None
    final_task_id = task_folder 
    # 用于动态构造 URL 的路径前缀
    base_prefix = "" 

    # 智慧寻址优先级：私有结果 > 私有根目录 > 系统教育目录
    if (user_proc_root / task_folder).exists():
        task_path = user_proc_root / task_folder
        base_prefix = f"{username}/processed"
    elif (user_root_root / task_folder).exists():
        task_path = user_root_root / task_folder
        base_prefix = f"{username}"
    elif (system_edu_root / task_folder).exists():
        task_path = system_edu_root / task_folder
        base_prefix = "SYSTEM/edu_data/processed"
        print(f"🎓 教育模式：已定位到系统公共资源库 {task_folder}")
    # 模糊匹配逻辑 (仅针对私有目录，保持原文)
    elif user_proc_root.exists():
        print(f"🔍 正在执行长短名模糊匹配: 目标是 {task_folder}")
        for sub_dir in user_proc_root.iterdir():
            if sub_dir.is_dir() and (sub_dir.name in task_folder or task_folder in sub_dir.name):
                task_path = sub_dir
                final_task_id = sub_dir.name
                base_prefix = f"{username}/processed"
                print(f"🎯 模糊匹配成功：找到物理文件夹 {sub_dir.name}")
                break

    if not task_path:
        print(f"❌ 404 彻底找不到：{task_folder}")
        raise HTTPException(status_code=404, detail="找不到任务文件夹")

    print(f"✅ 找到任务路径: {task_path}")
    print(f"📁 任务目录内容: {list(task_path.iterdir())}")
    
    patients = []
    
    # 2. 扫描病例文件夹
    for case_folder in task_path.iterdir():
        if case_folder.is_dir() and not case_folder.name.startswith('.'):
            print(f"  ✅ 发现病例: {case_folder.name}")
            video_groups = {}  # {base_name: {modality: path}}
            
            # 进入 output_videos 扫描 MP4 并使用正则归组
            videos_dir = case_folder / "output_videos"
            if videos_dir.exists():
                for v_file in videos_dir.iterdir():
                    if v_file.is_file() and v_file.suffix.lower() == '.mp4':
                        fn = v_file.name
                        import re
                        m_match = re.search(r'_(heatmap|bbox|original)\.mp4$', fn, re.IGNORECASE)
                        if m_match:
                            modality = m_match.group(1).lower()
                            base_name = fn[:m_match.start()]
                        else:
                            modality = 'original'; base_name = v_file.stem
                        
                        if base_name not in video_groups: video_groups[base_name] = {}
                        
                        # [修改] 使用动态 base_prefix 生成 URL 路径，支持跨空间访问
                        rel_url = f"/data/{base_prefix}/{final_task_id}/{case_folder.name}/output_videos/{fn}".replace("//", "/")
                        video_groups[base_name][modality] = rel_url

            # 3. 关联置信度元数据 (指向动态适配接口)
            videos_list = []
            for bn in sorted(video_groups.keys()):
                abs_conf_path = case_folder / "output_data" / "confidence_scores.json"
                abs_video_json_path = case_folder / "output_data" / f"{bn}.json"
                metadata_url = None
                video_json_url = None
                if abs_conf_path.exists():
                    # [修改] path 参数同步使用动态前缀
                    metadata_url = f"/api/get-metadata?path={base_prefix}/{final_task_id}/{case_folder.name}/output_data/confidence_scores.json"
                    metadata_url = metadata_url.replace("//", "/")
                if abs_video_json_path.exists():
                    video_json_url = f"/api/get-metadata?path={base_prefix}/{final_task_id}/{case_folder.name}/output_data/{bn}.json"
                    video_json_url = video_json_url.replace("//", "/")

                group = video_groups[bn]
                # 质量守卫
                if 'original' not in group and group: group['original'] = list(group.values())[0]
                
                videos_list.append({'baseName': bn, 'modalities': group, 'metadataPath': metadata_url, 'videoJsonUrl': video_json_url})
            
            if videos_list:
                patients.append({'id': case_folder.name, 'videos': videos_list})

    def sort_key(p):
        import re
        match = re.search(r'\d+', p['id'])
        return (0, int(match.group()), p['id']) if match else (1, 0, p['id'])
    
    patients.sort(key=sort_key)
    return {"data": patients}

# --- [新增：元数据格式适配接口] ---
@app.get("/api/get-metadata")
async def get_metadata(path: str):
    """读取原始 JSON 并在内存中包上一层 confidence_scores 外壳以适配前端"""
    file_path = Path(DATA_BATCH_STORAGE) / path
    if not file_path.exists(): raise HTTPException(status_code=404)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        return raw_data if "confidence_scores" in raw_data else {"confidence_scores": raw_data}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))


# 提供资源访问
# 确保data_batch_storage目录存在
Path(DATA_BATCH_STORAGE).mkdir(exist_ok=True)
app.mount("/videos", StaticFiles(directory=DATA_BATCH_STORAGE), name="videos")
app.mount("/data", StaticFiles(directory=DATA_BATCH_STORAGE), name="data")
# 使用绝对路径挂载config目录
import os
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
Path(config_dir).mkdir(exist_ok=True)
app.mount("/config", StaticFiles(directory=config_dir), name="config")
# 挂载UI目录
ui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UI")
Path(ui_dir).mkdir(exist_ok=True)
app.mount("/UI", StaticFiles(directory=ui_dir), name="UI")
# 挂载UI/res目录（静态资源）
res_dir = os.path.join(ui_dir, "res")
Path(res_dir).mkdir(exist_ok=True)
app.mount("/res", StaticFiles(directory=res_dir), name="res")


# JSON格式提交诊断结果的请求模型
class DiagnosisRecordSimple(BaseModel):
    patientId: str
    diagnosis: str
    severity: Optional[int] = None
    viewTime: Optional[int] = 0  # 新增：查看时间（秒）

class DiagnosisSubmitJsonRequest(BaseModel):
    username: str
    taskFolder: str
    mode: Optional[str] = "diag"  # [新增] 默认为判读模式
    eduSubMode: Optional[str] = None  # 教育子模式: single / assist
    submittedAt: str
    totalTime: Dict[str, Any]
    patientCount: int
    records: List[DiagnosisRecordSimple]

# ==================== 大模型配置管理 API（新版） ====================

@app.get("/api/admin/llm-models")
async def get_llm_models(request: Request):
    """获取所有模型列表（API Key脱敏）"""
    # 验证权限
    username = request.cookies.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    
    user = database.get_user_by_username(username)
    if not user or not user.get('is_admin'):
        raise HTTPException(status_code=403, detail="仅管理员可访问")
    
    try:
        if not LLM_MODELS_FILE.exists():
            return {"models": [], "selected_model_id": None}
        
        with open(LLM_MODELS_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 脱敏处理API Key
        models = []
        for model in config.get("models", []):
            api_key = model.get("api_key", "")
            masked_key = ""
            if len(api_key) > 8:
                masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
            elif api_key:
                masked_key = "***"
            
            models.append({
                **model,
                "masked_api_key": masked_key
            })
        
        return {
            "models": models,
            "selected_model_id": config.get("selected_model_id")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取失败: {str(e)}")


@app.post("/api/admin/llm-models")
async def add_llm_model(model: LLMModelConfig, request: Request):
    """添加新模型"""
    # 验证权限
    username = request.cookies.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    
    user = database.get_user_by_username(username)
    if not user or not user.get('is_admin'):
        raise HTTPException(status_code=403, detail="仅管理员可配置")
    
    try:
        # 读取现有配置
        if LLM_MODELS_FILE.exists():
            with open(LLM_MODELS_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {"models": [], "selected_model_id": None}
        
        # 生成唯一ID
        import uuid
        model_id = str(uuid.uuid4())[:8]
        
        # 创建新模型
        new_model = {
            "id": model_id,
            "display_name": model.display_name,
            "base_url": model.base_url,
            "api_key": model.api_key,
            "model": model.model,
            "description": model.description or "",
            "created_at": datetime.now().isoformat()
        }
        
        config["models"].append(new_model)
        
        # 如果是第一个模型，自动选中
        if len(config["models"]) == 1:
            config["selected_model_id"] = model_id
        
        # 保存配置
        with open(LLM_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 新模型已添加: {model.display_name} by {username}")
        return {"message": "模型添加成功", "model_id": model_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加失败: {str(e)}")


@app.put("/api/admin/llm-models/{model_id}")
async def update_llm_model(model_id: str, model: LLMModelConfig, request: Request):
    """更新模型配置"""
    # 验证权限
    username = request.cookies.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    
    user = database.get_user_by_username(username)
    if not user or not user.get('is_admin'):
        raise HTTPException(status_code=403, detail="仅管理员可配置")
    
    try:
        with open(LLM_MODELS_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 查找并更新模型
        for i, m in enumerate(config["models"]):
            if m["id"] == model_id:
                config["models"][i].update({
                    "display_name": model.display_name,
                    "base_url": model.base_url,
                    "model": model.model,
                    "description": model.description or "",
                    "updated_at": datetime.now().isoformat()
                })
                
                # 只在提供了新API Key时才更新
                if model.api_key:
                    config["models"][i]["api_key"] = model.api_key
                
                break
        else:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        # 保存配置
        with open(LLM_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 模型已更新: {model.display_name} by {username}")
        return {"message": "模型更新成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@app.delete("/api/admin/llm-models/{model_id}")
async def delete_llm_model(model_id: str, request: Request):
    """删除模型"""
    # 验证权限
    username = request.cookies.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    
    user = database.get_user_by_username(username)
    if not user or not user.get('is_admin'):
        raise HTTPException(status_code=403, detail="仅管理员可配置")
    
    try:
        with open(LLM_MODELS_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 查找并删除模型
        original_count = len(config["models"])
        config["models"] = [m for m in config["models"] if m["id"] != model_id]
        
        if len(config["models"]) == original_count:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        # 如果删除的是当前选中的模型，清除选择
        if config.get("selected_model_id") == model_id:
            config["selected_model_id"] = config["models"][0]["id"] if config["models"] else None
        
        # 保存配置
        with open(LLM_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 模型已删除: {model_id} by {username}")
        return {"message": "模型删除成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@app.post("/api/admin/llm-models/select")
async def select_llm_model(request: Request, model_id: str = Form(...)):
    """选择模型"""
    # 验证权限
    username = request.cookies.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    
    user = database.get_user_by_username(username)
    if not user or not user.get('is_admin'):
        raise HTTPException(status_code=403, detail="仅管理员可配置")
    
    try:
        with open(LLM_MODELS_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 验证模型是否存在
        model_ids = [m["id"] for m in config.get("models", [])]
        if model_id not in model_ids:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        # 更新选中的模型
        config["selected_model_id"] = model_id
        
        # 保存配置
        with open(LLM_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已选择模型: {model_id} by {username}")
        return {"message": "模型选择成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"选择失败: {str(e)}")


@app.post("/api/admin/llm-models/{model_id}/test")
async def test_llm_model(model_id: str, request: Request):
    """测试模型连接"""
    # 验证权限
    username = request.cookies.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    
    user = database.get_user_by_username(username)
    if not user or not user.get('is_admin'):
        raise HTTPException(status_code=403, detail="仅管理员可测试")
    
    try:
        with open(LLM_MODELS_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 查找模型
        model = None
        for m in config.get("models", []):
            if m["id"] == model_id:
                model = m
                break
        
        if not model:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        # 测试连接
        result = await test_connection(
            model["base_url"],
            model["api_key"],
            model["model"]
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# 提交诊断结果（支持智慧寻址）
@app.post("/api/diagnosis/submit-json")
async def submit_diagnosis_json(request: DiagnosisSubmitJsonRequest):
    """
    提交诊断结论：
    1. 判读模式：存入任务目录。
    2. 教育模式：计算增强统计数据，调用大模型分析，并存入 SYSTEM 成绩单。
    """
    try:
        # 定义诊断名与 Label 的映射
        LABEL_MAP = {"Normal": 0, "VSD": 1, "ASD": 2, "PDA": 3}
        LABEL_REVERSE_MAP = {0: "Normal", 1: "VSD", 2: "ASD", 3: "PDA"}
        
        # --- 情况 A：教育模式 (Assessment) ---
        if request.mode == "edu":
            # 提取原始任务ID（去除双阶段后缀）
            original_task_folder = re.sub(r'_SINGLE|_AI-ASSIST$', '', request.taskFolder)
            
            # 1. 定位标准答案
            gt_path = Path(DATA_BATCH_STORAGE) / "SYSTEM" / "edu_data" / original_task_folder / "epoch_data.json"
            if not gt_path.exists():
                raise HTTPException(status_code=404, detail="教育批次答案文件丢失")
            
            with open(gt_path, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)

            # 2. 初始化统计变量
            tp, tn, fp, fn = 0, 0, 0, 0
            correct_count = 0
            total = len(request.records)
            
            # [新增] 病种统计
            category_stats = {
                "Normal": {"total": 0, "correct": 0, "error": 0},
                "VSD": {"total": 0, "correct": 0, "error": 0},
                "ASD": {"total": 0, "correct": 0, "error": 0},
                "PDA": {"total": 0, "correct": 0, "error": 0}
            }
            
            # [新增] 时间分析数据
            case_time_data = []
            
            # [新增] AI依赖性统计
            ai_dependency = {
                "correct_reliance": 0,      # AI正确且判读正确
                "insufficient_reliance": 0,  # AI正确且判读错误
                "correct_independence": 0,    # AI错误且判读正确
                "over_reliance": 0            # AI错误且判读错误
            }
            
            # [新增] 详细数据列表
            ground_truth_labels = []
            ai_labels = []
            user_labels = []
            view_times = []
            
            # 3. 遍历所有记录，计算统计数据
            for rec in request.records:
                user_diag = rec.diagnosis
                user_label = LABEL_MAP.get(user_diag, -1)
                gt_label = ground_truth.get(rec.patientId, {}).get("label")
                
                # 获取AI预测
                ai_label = None
                ai_confidence_path = Path(DATA_BATCH_STORAGE) / "SYSTEM" / "edu_data" / "processed" / original_task_folder / rec.patientId / "output_data" / "confidence_scores.json"
                if ai_confidence_path.exists():
                    with open(ai_confidence_path, "r", encoding="utf-8") as f:
                        ai_data = json.load(f)
                        # 支持两种格式：直接格式 {Normal:0.85,...} 或 包装格式 {confidence_scores:{Normal:0.85,...}}
                        scores = ai_data.get("confidence_scores", ai_data)
                        if scores:
                            ai_pred_name = max(scores, key=scores.get)
                            ai_label = LABEL_MAP.get(ai_pred_name, -1)
                            print(f"  🤖 AI预测: {ai_pred_name} -> {ai_label}, 置信度: {scores[ai_pred_name]}")
                
                # 收集数据
                ground_truth_labels.append(LABEL_REVERSE_MAP.get(gt_label, "Unknown"))
                ai_labels.append(LABEL_REVERSE_MAP.get(ai_label, "Unknown") if ai_label is not None else "Unknown")
                user_labels.append(user_diag)
                view_times.append(rec.viewTime or 0)
                
                # 时间分析数据
                case_time_data.append({
                    "patientId": rec.patientId,
                    "viewTime": rec.viewTime or 0,
                    "groundTruth": LABEL_REVERSE_MAP.get(gt_label, "Unknown"),
                    "isCorrect": user_label == gt_label
                })
                
                # 病种统计
                if gt_label in [0, 1, 2, 3]:
                    gt_category = LABEL_REVERSE_MAP[gt_label]
                    category_stats[gt_category]["total"] += 1
                    if user_label == gt_label:
                        category_stats[gt_category]["correct"] += 1
                        correct_count += 1
                    else:
                        category_stats[gt_category]["error"] += 1
                
                # 计算敏感度/特异性 (Positive = 非0, Negative = 0)
                is_user_pos = user_label > 0
                is_gt_pos = gt_label > 0

                if is_user_pos and is_gt_pos: tp += 1
                elif not is_user_pos and not is_gt_pos: tn += 1
                elif is_user_pos and not is_gt_pos: fp += 1
                elif not is_user_pos and is_gt_pos: fn += 1
                
                # [新增] AI依赖性分析
                # 添加调试日志
                if ai_label is not None:
                    print(f"🔍 AI依赖分析 - 病例: {rec.patientId}, AI标签: {ai_label}, 真实标签: {gt_label}, 用户标签: {user_label}")
                    
                    ai_correct = (ai_label == gt_label)
                    user_correct = (user_label == gt_label)
                    
                    if ai_correct and user_correct:
                        ai_dependency["correct_reliance"] += 1
                        print(f"  ✅ 正确依赖")
                    elif ai_correct and not user_correct:
                        ai_dependency["insufficient_reliance"] += 1
                        print(f"  ⚠️ 依赖不足")
                    elif not ai_correct and user_correct:
                        ai_dependency["correct_independence"] += 1
                        print(f"  💪 正确独立")
                    elif not ai_correct and not user_correct:
                        ai_dependency["over_reliance"] += 1
                        print(f"  ❌ 过度依赖")
                else:
                    print(f"⚠️ 病例 {rec.patientId} 没有AI预测数据")
            
            print(f"📊 AI依赖性统计结果: {ai_dependency}")
            
            # 4. [新增] 计算时间分析
            case_time_data.sort(key=lambda x: x["viewTime"], reverse=True)
            top3_longest_cases = case_time_data[:3]
            
            # 病种级别时间比较
            category_time_analysis = {}
            for category in ["Normal", "VSD", "ASD", "PDA"]:
                category_cases = [c for c in case_time_data if c["groundTruth"] == category]
                if category_cases:
                    avg_time = sum(c["viewTime"] for c in category_cases) / len(category_cases)
                    correct_cases = [c for c in category_cases if c["isCorrect"]]
                    error_cases = [c for c in category_cases if not c["isCorrect"]]
                    
                    category_time_analysis[category] = {
                        "avg_view_time": round(avg_time, 1),
                        "correct_avg_time": round(sum(c["viewTime"] for c in correct_cases) / len(correct_cases), 1) if correct_cases else 0,
                        "error_avg_time": round(sum(c["viewTime"] for c in error_cases) / len(error_cases), 1) if error_cases else 0
                    }
            
            # 5. 计算基础评估指标
            stats = {
                # 基础指标
                "accuracy": correct_count / total if total > 0 else 0,
                "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 1.0,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 1.0,
                "formatted_duration": request.totalTime.get("formatted", "0:00"),
                "submitted_at": request.submittedAt,
                
                # [新增] 病种统计
                "category_stats": category_stats,
                
                # [新增] 时间分析
                "time_analysis": {
                    "top3_longest_cases": top3_longest_cases,
                    "category_time_analysis": category_time_analysis
                },
                
                # [新增] AI依赖性
                "ai_dependency": ai_dependency,
                
                # [新增] 详细数据
                "ground_truth_labels": ground_truth_labels,
                "ai_labels": ai_labels,
                "user_labels": user_labels,
                "view_times": view_times,
                
                # 大模型分析状态
                "llm_analysis_status": "pending"
            }

            # 6. 存入 SYSTEM 个人成绩单 (Doctor_Diag_Result/{user}.json)
            user_res_path = Path(DATA_BATCH_STORAGE) / "SYSTEM" / "edu_data" / "Doctor_Diag_Result" / f"{request.username}.json"
            user_data = {}
            if user_res_path.exists():
                with open(user_res_path, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
            
            # 根据 eduSubMode 添加后缀（如果 taskFolder 不带后缀则添加）
            save_key = request.taskFolder
            if request.eduSubMode and not save_key.endswith('_SINGLE') and not save_key.endswith('_AI-ASSIST'):
                if request.eduSubMode == 'single':
                    save_key = f"{request.taskFolder}_SINGLE"
                elif request.eduSubMode == 'assist':
                    save_key = f"{request.taskFolder}_AI-ASSIST"
            
            # 在stats中添加模式标识
            stats['edu_sub_mode'] = request.eduSubMode
            user_data[save_key] = stats
            
            with open(user_res_path, "w", encoding="utf-8") as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)

            print(f"📊 教育模式评分完成: {request.username} - 模式: {request.eduSubMode} - 得分: {stats['accuracy']*100}%")
            
            # 7. [新增] 异步调用大模型分析（使用带后缀的save_key确保key匹配）
            print(f"🔔 准备触发AI分析: username={request.username}, save_key={save_key}")
            asyncio.create_task(analyze_with_llm(request.username, save_key, stats))
            print(f"🔔 已创建AI分析任务")
            
            return {"message": "评估完成，AI分析报告生成中", "stats": stats}

        # --- 情况 B：普通判读模式 (保持原有逻辑) ---
        else:
            # 执行智慧寻址找到任务路径
            proc_path = Path(DATA_BATCH_STORAGE) / request.username / "processed" / request.taskFolder
            root_path = Path(DATA_BATCH_STORAGE) / request.username / request.taskFolder
            task_path = proc_path if proc_path.exists() else root_path
            
            if not task_path.exists(): task_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"final_diagnosis_report_{timestamp}.json"
            with open(task_path / json_filename, 'w', encoding='utf-8') as f:
                data_to_save = request.model_dump() if hasattr(request, 'model_dump') else request.dict()
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
            return {"message": "结论已归档", "filename": json_filename}

    except Exception as e:
        print(f"❌ 提交逻辑崩溃: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def analyze_with_llm(username: str, task_folder: str, stats: Dict[str, Any]):
    """
    异步调用大模型分析
    
    Args:
        username: 用户名
        task_folder: 任务文件夹
        stats: 统计数据
    """
    print(f"🔍 [AI分析] 开始执行: username={username}, task_folder={task_folder}")
    try:
        # 1. 读取LLM配置（新格式）
        print(f"🔍 [AI分析] 检查配置文件: {LLM_MODELS_FILE}")
        if not LLM_MODELS_FILE.exists():
            print("⚠️ LLM配置文件不存在，跳过AI分析")
            return
        
        with open(LLM_MODELS_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"🔍 [AI分析] 配置读取成功, selected_model_id={config.get('selected_model_id')}")
        
        # 获取选中的模型
        selected_model_id = config.get("selected_model_id")
        if not selected_model_id:
            print("⚠️ 未选择模型，跳过AI分析")
            return
        
        # 查找选中的模型配置
        selected_model = None
        for model in config.get("models", []):
            if model.get("id") == selected_model_id:
                selected_model = model
                break
        
        if not selected_model:
            print("⚠️ 选中的模型不存在，跳过AI分析")
            return
        
        if not selected_model.get("base_url") or not selected_model.get("api_key"):
            print("⚠️ 模型配置不完整，跳过AI分析")
            return
        
        print(f"🔍 [AI分析] 使用模型: {selected_model.get('display_name')}")
        
        # 2. 调用大模型
        analyzer = LLMAnalyzer(
            base_url=selected_model["base_url"],
            api_key=selected_model["api_key"],
            model=selected_model.get("model", "glm-4")
        )
        
        print(f"🤖 正在调用大模型分析: {username} - {task_folder} (模型: {selected_model.get('display_name', 'unknown')})")
        analysis_result = await analyzer.analyze_performance(stats)
        print(f"🔍 [AI分析] 大模型返回结果: {analysis_result.get('status')}")
        
        # 3. 更新成绩单
        user_res_path = Path(DATA_BATCH_STORAGE) / "SYSTEM" / "edu_data" / "Doctor_Diag_Result" / f"{username}.json"
        print(f"🔍 [AI分析] 成绩单路径: {user_res_path}")
        print(f"🔍 [AI分析] 成绩单存在: {user_res_path.exists()}")
        
        if user_res_path.exists():
            with open(user_res_path, "r", encoding="utf-8") as f:
                user_data = json.load(f)
            
            print(f"🔍 [AI分析] 当前成绩单keys: {list(user_data.keys())}")
            print(f"🔍 [AI分析] 查找key: {task_folder}")
            print(f"🔍 [AI分析] key存在: {task_folder in user_data}")
            
            if task_folder in user_data:
                user_data[task_folder]["llm_analysis_status"] = analysis_result.get("status", "failed")
                user_data[task_folder]["llm_analysis"] = analysis_result
                user_data[task_folder]["llm_model_used"] = selected_model.get("display_name", "unknown")
                
                with open(user_res_path, "w", encoding="utf-8") as f:
                    json.dump(user_data, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 大模型分析完成: {username} - {task_folder}")
            else:
                print(f"⚠️ [AI分析] 成绩单中未找到key: {task_folder}")
        else:
            print(f"⚠️ [AI分析] 成绩单文件不存在")
        
    except Exception as e:
        print(f"❌ 大模型分析失败: {e}")
        
        # 记录失败状态
        try:
            user_res_path = Path(DATA_BATCH_STORAGE) / "SYSTEM" / "edu_data" / "Doctor_Diag_Result" / f"{username}.json"
            if user_res_path.exists():
                with open(user_res_path, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
                
                if task_folder in user_data:
                    user_data[task_folder]["llm_analysis_status"] = "failed"
                    user_data[task_folder]["llm_analysis"] = {"error": str(e)}
                    
                    with open(user_res_path, "w", encoding="utf-8") as f:
                        json.dump(user_data, f, ensure_ascii=False, indent=2)
        except:
            pass

# 上传任务文件夹（接收zip压缩包）
@app.post("/api/users/{username}/upload-task")
async def upload_task(username: str, file: UploadFile = File(...)):
    """
    上传任务文件夹（zip压缩包格式）
    解压到用户的data_batch_storage目录下
    """
    try:
        # 检查用户是否存在
        user = database.get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        
        # 检查文件格式
        if not file.filename or not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="只支持zip格式的压缩包")
        
        user_folder = Path(DATA_BATCH_STORAGE) / username
        user_folder.mkdir(parents=True, exist_ok=True)
        
        # 保存上传的zip文件
        zip_path = user_folder / str(file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 解压文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(user_folder)
        
        # 删除zip文件
        zip_path.unlink()
        
        print(f"✅ 任务上传成功: {username} - {file.filename}")
        
        return {
            "message": "任务上传成功",
            "filename": file.filename
        }
        
    except Exception as e:
        print(f"❌ 任务上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

# 获取用户的所有JSON结果文件
@app.get("/api/users/{username}/result-files")
async def get_user_result_files(username: str):
    """
    获取用户所有任务文件夹中的JSON结果文件
    """
    user_folder = Path(DATA_BATCH_STORAGE) / username
    
    if not user_folder.exists():
        raise HTTPException(status_code=404, detail="用户文件夹不存在")
    
    result_files = []
    
    # 遍历用户文件夹下的所有任务文件夹
    for task_folder in user_folder.iterdir():
        if task_folder.is_dir():
            # 只查找JSON文件
            for json_file in task_folder.glob("diagnosis_results_*.json"):
                result_files.append({
                    'filename': json_file.name,
                    'taskFolder': task_folder.name,
                    'size': json_file.stat().st_size,
                    'modifiedTime': datetime.timestamp(json_file.stat().st_mtime), # 保持原始数值
                    'downloadUrl': f"/api/users/{username}/download-result/{task_folder.name}/{json_file.name}"
                })
    
    # 排序
    result_files.sort(key=lambda x: x['modifiedTime'], reverse=True)
    # 格式化时间供显示
    for r in result_files:
        r['modifiedTime'] = datetime.fromtimestamp(r['modifiedTime']).strftime("%Y-%m-%d %H:%M:%S")
        
    return {"data": result_files}

# 下载JSON结果文件
@app.get("/api/users/{username}/download-result/{task_folder}/{filename}")
async def download_result(username: str, task_folder: str, filename: str):
    """
    下载指定的JSON结果文件
    """
    file_path = Path(DATA_BATCH_STORAGE) / username / task_folder / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/json'
    )

# 批量下载请求模型
class BatchDownloadFile(BaseModel):
    filename: str
    taskFolder: str

class BatchDownloadRequest(BaseModel):
    username: str
    files: List[BatchDownloadFile]

# 批量下载结果文件（ZIP格式）
@app.post("/api/batch-download")
async def batch_download(request: BatchDownloadRequest):
    """
    批量下载多个结果文件，打包成ZIP
    """
    try:
        import io
        import zipfile
        
        # 创建内存中的ZIP文件
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_info in request.files:
                file_path = Path(DATA_BATCH_STORAGE) / request.username / file_info.taskFolder / file_info.filename
                
                if file_path.exists():
                    # 添加文件到ZIP，保留任务文件夹结构
                    arcname = f"{file_info.taskFolder}/{file_info.filename}"
                    zip_file.write(file_path, arcname=arcname)
                    print(f"✅ 添加文件到ZIP: {arcname}")
                else:
                    print(f"⚠️ 文件不存在，跳过: {file_path}")
        
        # 将指针移到开始
        zip_buffer.seek(0)
        
        # 生成ZIP文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{request.username}_diagnosis_results_{timestamp}.zip"
        
        print(f"✅ ZIP文件创建成功: {zip_filename}")
        
        # 返回ZIP文件
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            iter([zip_buffer.getvalue()]),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_filename}"
            }
        )
        
    except Exception as e:
        print(f"❌ 批量下载失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"批量下载失败: {str(e)}")


@app.get("/api/tasks/{username}/{submission_id}/download-info")
async def get_task_download_info(username: str, submission_id: str):
    # 路径锁定：data_batch_storage/{user}/processed/{id}
    task_dir = Path(DATA_BATCH_STORAGE) / username / "processed" / submission_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="任务结果文件夹尚未生成")

    # 1. 搜寻最新的医师判读报告 (支持模糊搜索)
    reports = list(task_dir.glob("final_diagnosis_report_*.json"))
    latest_report = None
    if reports:
        latest_file = max(reports, key=lambda p: p.stat().st_mtime)
        latest_report = {
            "name": latest_file.name,
            "url": f"/data/{username}/processed/{submission_id}/{latest_file.name}",
            "time": datetime.fromtimestamp(latest_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        }

    # 2. 搜寻模型结果压缩包 (case_name/output_data.zip)
    model_zips = []
    for case_folder in task_dir.iterdir():
        if case_folder.is_dir() and not case_folder.name.startswith('.'):
            # [核心修复] 检查 case_folder 根目录下的 output_data.zip
            zip_path = case_folder / "output_data.zip"
            if zip_path.exists():
                model_zips.append({
                    "case_id": case_folder.name,
                    "url": f"/data/{username}/processed/{submission_id}/{case_folder.name}/output_data.zip"
                })
    
    return {"doctor_report": latest_report, "model_results": model_zips}

# 静态文件路由
@app.get("/video_3d_modal.js")
async def serve_video_3d_modal():
    return FileResponse("UI/video_3d_modal.js", media_type="application/javascript")

@app.get("/login")
async def serve_login():
    return FileResponse("UI/login.html")

@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse("UI/dashboard.html")

@app.get("/admin")
async def serve_admin():
    return FileResponse("UI/admin.html")

@app.get("/diagnosis")
async def serve_diagnosis():
    return FileResponse("UI/diagnosis.html")


@app.get("/flow")
async def serve_flow():
    return FileResponse("UI/flow.html")

@app.get("/")
async def root():
    return FileResponse("UI/login.html")

@app.get("/task_status")
async def serve_task_status():
    """返回任务状态进度页面"""
    return FileResponse("UI/task_status.html")

@app.get("/edu_status")
async def serve_edu_status(): return FileResponse("UI/edu_status.html")

@app.get("/edu_admin")
async def serve_edu_admin():
    """返回教育模式管理页面"""
    return FileResponse("UI/edu_admin.html")

@app.get("/edu_admin")
async def serve_edu_admin():
    """返回教育模式管理端页面"""
    return FileResponse("UI/edu_admin.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)