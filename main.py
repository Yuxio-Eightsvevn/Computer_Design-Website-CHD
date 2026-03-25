from fastapi import FastAPI, HTTPException, UploadFile, File, Request
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
from fastapi.responses import JSONResponse


# 数据存储目录
DATA_BATCH_STORAGE = "data_batch_storage"

# 系统初始化
def init_system():
    """初始化系统"""
    # 初始化数据库
    database.init_database()
    
    # 创建主存储目录
    storage_path = Path(DATA_BATCH_STORAGE)
    storage_path.mkdir(exist_ok=True)
    
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
    
    # --- [此处为您手动追加的 data.json 自动初始化钩子] ---
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
    # ------------------------------------------------

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
        request.password
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

# 获取任务下的病人列表
@app.get("/api/tasks/{username}/{task_folder}/patients")
async def get_task_patients(username: str, task_folder: str):
    """
    [重构] 适配模型输出结构。
    深入 processed/{task_folder}/{case_folder}/output_videos/ 和 output_data/
    """
    # 路径解析：统一指向 processed 目录
    task_path = Path(DATA_BATCH_STORAGE) / username / "processed" / task_folder
    
    print(f"📂 正在查找模型结果路径: {task_path}")
    
    # 检查任务文件夹是否存在
    if not task_path.exists():
        print(f"❌ 任务结果文件夹不存在: {task_path}")
        raise HTTPException(status_code=404, detail="任务结果文件夹不存在")
    
    patients = []
    
    # 遍历任务文件夹下的所有病例子文件夹 (case1, case2...)
    for case_folder in task_path.iterdir():
        print(f"  检查文件夹: {case_folder.name}")
        if case_folder.is_dir() and not case_folder.name.startswith('.'):
            print(f"  ✅ 找到病例文件夹: {case_folder.name}")
            
            video_groups = {}  # {base_name: {modality: path}}
            
            # 深入 output_videos 寻找可视化 MP4
            videos_dir = case_folder / "output_videos"
            if videos_dir.exists():
                for video_file in videos_dir.iterdir():
                    if video_file.is_file() and video_file.suffix.lower() == '.mp4':
                        filename = video_file.name
                        print(f"    找到展示视频: {filename}")
                        
                        # 解析模态类型和基础文件名
                        if filename.endswith('_heatmap.mp4'):
                            base_name = filename[:-12]; modality = 'heatmap'
                        elif filename.endswith('_bbox.mp4'):
                            base_name = filename[:-9]; modality = 'bbox'
                        elif filename.endswith('_original.mp4'):
                            base_name = filename[:-13]; modality = 'original'
                        else:
                            base_name = filename[:-4]; modality = 'original'
                        
                        if base_name not in video_groups: video_groups[base_name] = {}
                        
                        # 生成供前端访问的 /data 路径
                        video_groups[base_name][modality] = f"/data/{username}/processed/{task_folder}/{case_folder.name}/output_videos/{filename}"
            
            # 转换为列表格式并绑定 output_data 下的置信度文件
            videos = []
            for base_name in sorted(video_groups.keys()):
                # 寻找该病例的置信度文件 (output_data/confidence_scores.json)
                metadata_path = None
                conf_json = case_folder / "output_data" / "confidence_scores.json"
                if conf_json.exists():
                    metadata_path = f"/data/{username}/processed/{task_folder}/{case_folder.name}/output_data/confidence_scores.json"
                    print(f"      找到置信度数据: {conf_json.name}")
                
                videos.append({
                    'baseName': base_name,
                    'modalities': video_groups[base_name],
                    'metadataPath': metadata_path
                })
            
            if videos:
                patients.append({
                    'id': case_folder.name,
                    'videos': videos
                })
                print(f"  ✅ 病例 {case_folder.name} 添加成功，共 {len(videos)} 个视频组")
            else:
                print(f"  ⚠️ 病例文件夹 {case_folder.name} 没有检测到输出视频")
    
    # 按病例ID智能排序 (保持原文逻辑)
    def sort_key(patient):
        patient_id = patient['id']
        import re
        match = re.search(r'\d+', patient_id)
        if match:
            return (0, int(match.group()), patient_id)
        else:
            return (1, 0, patient_id)
    
    patients.sort(key=sort_key)
    
    print(f"🎬 最终返回 {len(patients)} 个病例对象")
    return {"data": patients}

# 提供视频文件访问
app.mount("/videos", StaticFiles(directory=DATA_BATCH_STORAGE), name="videos")

# [重点新增] 提供全局数据访问权，支持 3D PNG 序列帧的访问
app.mount("/data", StaticFiles(directory=DATA_BATCH_STORAGE), name="data")


# JSON格式提交诊断结果的请求模型
class DiagnosisRecordSimple(BaseModel):
    patientId: str
    diagnosis: str
    severity: int

class DiagnosisSubmitJsonRequest(BaseModel):
    username: str
    taskFolder: str
    submittedAt: str
    totalTime: Dict[str, Any]
    patientCount: int
    records: List[DiagnosisRecordSimple]

# 提交诊断结果（重构保存路径）
@app.post("/api/diagnosis/submit-json")
async def submit_diagnosis_json(request: DiagnosisSubmitJsonRequest):
    """
    提交诊断结论并保存在对应的 processed 结果文件夹中。
    """
    try:
        # 修改保存路径，使其归档在处理后的 processed 文件夹内
        task_path = Path(DATA_BATCH_STORAGE) / request.username / "processed" / request.taskFolder
        
        if not task_path.exists():
            task_path.mkdir(parents=True, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 结果文件名
        json_filename = f"final_diagnosis_report_{timestamp}.json"
        json_path = task_path / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(request.dict(), jsonfile, ensure_ascii=False, indent=2)
        
        print(f"✅ 诊断结论已在结果区归档: {json_path}")
        
        return {
            "message": "诊断结果提交成功",
            "filename": json_filename,
            "path": str(json_path)
        }
        
    except Exception as e:
        print(f"❌ 保存诊断结果失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")

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
                    'modifiedTime': datetime.fromtimestamp(json_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    'downloadUrl': f"/api/users/{username}/download-result/{task_folder.name}/{json_file.name}"
                })
    
    # 按修改时间倒序排列
    result_files.sort(key=lambda x: x['modifiedTime'], reverse=True)
    
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


# 静态文件路由
@app.get("/video_3d_modal.js")
async def serve_video_3d_modal():
    return FileResponse("video_3d_modal.js", media_type="application/javascript")

@app.get("/login")
async def serve_login():
    return FileResponse("login.html")

@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse("dashboard.html")

@app.get("/admin")
async def serve_admin():
    return FileResponse("admin.html")

@app.get("/diagnosis")
async def serve_diagnosis():

    return FileResponse("diagnosis.html")


@app.get("/flow")
async def serve_flow():
    return FileResponse("flow.html")

@app.get("/")
async def root():
    return FileResponse("login.html")

@app.get("/task_status")
async def serve_task_status():
    """返回任务状态进度页面"""
    return FileResponse("task_status.html")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)