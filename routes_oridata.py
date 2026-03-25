# routes_oridata.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from typing import List, Dict
from pathlib import Path, PurePosixPath
import datetime
import random
import string
import json
import aiofiles
import asyncio
import shutil
import subprocess

router = APIRouter()

BASE_DATA_DIR = Path("data_batch_storage")
ORIDATA_DIRNAME = "oridata"
MAX_VIDEOS_PER_CASE = 5
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# --- 基础工具函数 ---

def generate_submission_id() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    rnd = "".join(random.choices("0123456789", k=6))
    return f"{ts}_{rnd}"

def safe_join(base: Path, *paths) -> Path:
    target = (base.joinpath(*paths)).resolve()
    base_resolved = base.resolve()
    if not str(target).startswith(str(base_resolved)):
        raise ValueError("路径校验失败：疑似路径穿越")
    return target

def is_allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

async def generate_thumbnail_ffmpeg(video_path: Path, thumb_path: Path, seek_time: float = 0.05) -> bool:
    """异步调用 ffmpeg 生成缩略图"""
    cmd = [
        "ffmpeg", "-y", "-ss", f"{seek_time}", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "2", str(thumb_path)
    ]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        return proc.returncode == 0
    except Exception:
        return False

# --- 任务索引与模型后台逻辑 ---
def update_user_task_index(username: str, task_update: dict):
    """
    维护用户根目录下的 data.json 任务索引。
    """
    try:
        user_root = BASE_DATA_DIR / username
        user_root.mkdir(parents=True, exist_ok=True)
        index_path = user_root / "data.json"
        
        data = {"tasks": []}
        
        # 1. 安全读取
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
            except Exception as e:
                print(f"⚠️ data.json 解析失败，重新初始化: {e}")

        # 2. 查找并更新
        sub_id = task_update.get("submission_id")
        if not sub_id: return

        found = False
        for t in data.get("tasks", []):
            if t.get("submission_id") == sub_id:
                t.update(task_update)
                found = True
                break
        
        if not found:
            new_entry = {
                "submission_id": sub_id,
                "request_name": task_update.get("request_name", "未命名任务"),
                "request_pos": task_update.get("request_pos", ""),
                "request_case_cnt": task_update.get("request_case_cnt", 0),
                "request_video_cnt": task_update.get("request_video_cnt", 0),
                "is_cmp": False,
                "upload_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cmp_time": None
            }
            data.setdefault("tasks", []).append(new_entry)

        # 3. 安全写入
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"❌ update_user_task_index 彻底失败: {e}")


async def run_model_inference_wrapper(request_path: Path, output_root: Path, request_name: str, username: str, submission_id: str):
    """后台模型处理包装器"""
    target_output_dir = output_root / f"{request_name}_{submission_id}"
    target_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 模拟模型处理耗时
    print(f"🤖 [后台任务] 正在处理: {request_name}...")
    await asyncio.sleep(10) 
    
    # 处理完成后更新索引
    update_user_task_index(username, {
        "submission_id": submission_id,
        "is_cmp": True,
        "cmp_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    print(f"✅ [后台任务] {request_name} 完成")

# --- 核心 API 路由 ---
@router.post("/api/users/{username}/upload-oridata")
async def upload_oridata(
    username: str, 
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    request_name: str = Form("未命名请求"),
    case_names: List[str] = Form(...)  # [新增参数] 接收前端配对的病例名列表
):
    # --- 1. 前置检查 ---
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="未提供任何文件")
    
    if len(files) != len(case_names):
        raise HTTPException(status_code=400, detail="文件与病例元数据不匹配")

    # --- 2. [重构] 预过滤与配对归类 ---
    case_group_map: Dict[str, List[UploadFile]] = {}
    valid_video_total = 0

    # 使用 zip 将文件和它们在前端所属的病例名一一配对
    for f_obj, c_name in zip(files, case_names):
        # 过滤隐藏文件和不支持的格式
        if not Path(f_obj.filename).name.startswith('.') and is_allowed_file(f_obj.filename):
            case_group_map.setdefault(c_name, []).append(f_obj)
    
    if not case_group_map:
        raise HTTPException(status_code=400, detail="未检测到有效的视频文件")

    # --- 3. 截断限额与警告收集 ---
    warnings = []
    final_save_map: Dict[str, List[UploadFile]] = {}
    actual_total_video_cnt = 0

    for c_name, c_files in case_group_map.items():
        if len(c_files) > 5:
            warnings.append(f"病例 '{c_name}' 包含 {len(c_files)} 个视频，系统已自动截取前 5 个。")
            final_save_map[c_name] = c_files[:5]
        else:
            final_save_map[c_name] = c_files
        actual_total_video_cnt += len(final_save_map[c_name])

    submission_id = generate_submission_id()
    submission_base = BASE_DATA_DIR / username / ORIDATA_DIRNAME / submission_id

    try:
        submission_base.mkdir(parents=True, exist_ok=True)

        # --- 4. 物理保存逻辑（按 c_name 强制隔离） ---
        for c_name, c_files in final_save_map.items():
            # 创建物理文件夹
            target_case_dir = safe_join(submission_base, c_name)
            target_case_dir.mkdir(parents=True, exist_ok=True)
            
            for f in c_files:
                # 仅存入纯文件名
                dest_filename = Path(f.filename).name
                dest = target_case_dir / dest_filename
                async with aiofiles.open(dest, 'wb') as out_f:
                    while chunk := await f.read(65536):
                        await out_f.write(chunk)
                
                # 异步生成预览缩略图
                asyncio.create_task(generate_thumbnail_ffmpeg(dest, dest.with_name(dest_filename + ".jpg")))

        # --- 5. 统一收尾逻辑 ---
        # 写入 metadata.json
        meta = {
            "submission_id": submission_id, 
            "request_name": request_name, 
            "username": username,
            "created_at": datetime.datetime.now().isoformat(),
            "total_cases": len(final_save_map),
            "total_videos": actual_total_video_cnt,
            "warnings": warnings
        }
        with open(submission_base / "metadata.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)

        # 更新 data.json 索引
        try:
            update_user_task_index(username, {
                "submission_id": submission_id,
                "request_name": request_name,
                "request_pos": f"processed/{request_name}_{submission_id}",
                "request_case_cnt": len(final_save_map),
                "request_video_cnt": actual_total_video_cnt,
                "is_cmp": False
            })
        except Exception as e:
            print(f"⚠️ Index Update Error: {e}")

        # 启动后台模型处理
        background_tasks.add_task(
            run_model_inference_wrapper, 
            submission_base, 
            BASE_DATA_DIR / username / "processed", 
            request_name, 
            username, 
            submission_id
        )

        return {
            "message": "上传成功", 
            "submissionId": submission_id, 
            "warnings": warnings
        }

    except Exception as e:
        print(f"🚨 Upload Failed: {e}")
        if submission_base.exists():
            shutil.rmtree(submission_base)
        raise HTTPException(status_code=500, detail=str(e))
# --- 附加接口 ---

@router.get("/api/users/{username}/oridata-count")
async def get_oridata_count(username: str):
    user_oridata_path = BASE_DATA_DIR / username / ORIDATA_DIRNAME
    count = len([d for d in user_oridata_path.iterdir() if d.is_dir()]) if user_oridata_path.exists() else 0
    return {"count": count}

@router.get("/api/users/{username}/all-tasks")
async def get_all_tasks(username: str):
    index_path = BASE_DATA_DIR / username / "data.json"
    if not index_path.exists(): return {"tasks": []}
    with open(index_path, "r", encoding="utf-8") as f: data = json.load(f)
    data["tasks"].sort(key=lambda x: x['upload_time'], reverse=True)
    return data

@router.delete("/api/users/{username}/tasks/{submission_id}")
async def delete_task(username: str, submission_id: str):
    index_path = BASE_DATA_DIR / username / "data.json"
    if not index_path.exists(): raise HTTPException(status_code=404)
    with open(index_path, "r", encoding="utf-8") as f: data = json.load(f)
    task = next((t for t in data["tasks"] if t.get("submission_id") == submission_id), None)
    if not task: raise HTTPException(status_code=404)
    
    ori = BASE_DATA_DIR / username / ORIDATA_DIRNAME / submission_id
    if ori.exists(): shutil.rmtree(ori)
    if task.get("request_pos"):
        proc = BASE_DATA_DIR / username / task.get("request_pos")
        if proc.exists(): shutil.rmtree(proc)
        
    data["tasks"] = [t for t in data["tasks"] if t.get("submission_id") != submission_id]
    with open(index_path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    return {"message": "已删除"}


@router.get("/api/users/{username}/re-sync-tasks")
async def re_sync_tasks(username: str):
    """
    [调试工具] 扫描物理目录，强制将所有已上传文件夹同步到 data.json
    """
    user_oridata_path = BASE_DATA_DIR / username / ORIDATA_DIRNAME
    if not user_oridata_path.exists():
        return {"message": "找不到原始数据目录"}

    # 扫描磁盘
    physical_folders = [d.name for d in user_oridata_path.iterdir() if d.is_dir()]
    
    sync_count = 0
    for sub_id in physical_folders:
        # 这里模拟读取 metadata.json 获取当时的任务名
        req_name = "同步补全任务"
        meta_path = user_oridata_path / sub_id / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    req_name = meta.get("request_name", req_name)
            except: pass
            
        update_user_task_index(username, {
            "submission_id": sub_id,
            "request_name": req_name,
            "is_cmp": True # 同步的任务假设已跑完，或者您可以根据 processed 是否有文件夹来判断
        })
        sync_count += 1

    return {"message": f"成功同步了 {sync_count} 个任务到 data.json"}