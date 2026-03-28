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
import os
import sys
import time # [新增] 用于并发锁重试等待

# --- [新增] 动态导入模型路径 ---
CURRENT_ROOT = Path(__file__).parent.resolve()
MODEL_PATH = (CURRENT_ROOT / "model").resolve()

if str(MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(MODEL_PATH))
if str(CURRENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_ROOT))

# --- [核心修复] 环境自愈：确保代码能看到 Conda 环境中的 ffmpeg ---
def patch_ffmpeg_env():
    # 自动探测当前虚拟环境下的工具目录
    paths_to_add = [
        str(Path(sys.prefix) / "Library" / "bin"),
        str(Path(sys.prefix) / "bin"),
        str(Path(sys.prefix) / "Scripts")
    ]
    current_path = os.environ.get("PATH", "")
    new_paths = [p for p in paths_to_add if p not in current_path and os.path.exists(p)]
    if new_paths:
        os.environ["PATH"] = os.pathsep.join(new_paths) + os.pathsep + current_path
        print(f"🔧 已自动挂载环境工具链: {new_paths}")

patch_ffmpeg_env()

try:
    import yaml 
    from model_main import run_diagnosis
    print("✅ AI 模型及依赖库加载成功")
except ImportError as e:
    print(f"⚠️ 模型导入失败: {e}")
    # 打印 sys.path 帮助调试环境
    # print(f"当前 Python 搜索路径: {sys.path}")
    run_diagnosis = None

router = APIRouter()

BASE_DATA_DIR = Path("data_batch_storage")
ORIDATA_DIRNAME = "oridata"
MAX_VIDEOS_PER_CASE = 5
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# --- 基础工具函数 (完全保留) ---

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
    """异步调用 ffmpeg 生成缩略图 (已修正为 Windows Shell 兼容模式)"""
    cmd = f'ffmpeg -y -ss {seek_time} -i "{video_path.absolute()}" -frames:v 1 -q:v 2 "{thumb_path.absolute()}"'
    try:
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        return proc.returncode == 0
    except Exception:
        return False

# --- [重点修复] 浏览器兼容性转码函数 ---
async def transcode_to_h264(video_path: Path):
    """
    将视频转换为 H.264 编码。采用 Shell 模式和绝对路径，彻底解决 WinError 2。
    """
    temp_output = video_path.with_name(f"temp_{video_path.name}")
    
    # 强制使用绝对路径并加双引号，解决 Windows 下的空格和字符集问题
    input_str = f'"{video_path.absolute()}"'
    output_str = f'"{temp_output.absolute()}"'
    
    # 构造标准 H.264 转码命令
    cmd = (
        f'ffmpeg -y -i {input_str} '
        f'-vcodec libx264 -pix_fmt yuv420p '
        f'-preset fast -crf(23) '
        f'-acodec aac {output_str}'
    ).replace("(23)", " 23") # 确保空格正确
    
    try:
        print(f"🎬 正在修复视频编码: {video_path.name}...")
        # 改用 create_subprocess_shell 替代 create_subprocess_exec，确保能读取系统 PATH
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode == 0:
            os.replace(temp_output, video_path)
            print(f"✅ 编码修复完成: {video_path.name}")
            return True
        else:
            if temp_output.exists(): temp_output.unlink()
            err_info = stderr.decode('gbk', errors='ignore')
            print(f"❌ 转码失败: {video_path.name}\n错误详情: {err_info}")
            return False
    except Exception as e:
        print(f"⚠️ 转码异常: {e}")
        return False

# --- [加固版] 任务索引维护逻辑 ---

def update_user_task_index(username: str, task_update: dict):
    """
    维护用户根目录下的 data.json 任务索引。
    [加固] 引入文件排他锁，防止多用户/多任务并发写入冲突。
    """
    user_root = BASE_DATA_DIR / username
    user_root.mkdir(parents=True, exist_ok=True)
    index_path = user_root / "data.json"
    lock_path = user_root / "data.json.lock"
    
    # 1. 获取文件锁 (最长等待 5 秒)
    acquired = False
    for _ in range(50):
        try:
            # 'x' 模式：如果文件已存在则抛出 FileExistsError，具有原子性
            with open(lock_path, "x") as _: pass
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.1) # 等待 100 毫秒重试
    
    if not acquired:
        print(f"⚠️ 无法获取文件锁，放弃更新 {username} 的任务索引")
        return

    try:
        # 2. 安全读取与更新逻辑 (原有逻辑)
        data = {"tasks": []}
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content: data = json.loads(content)
            except Exception as e: print(f"⚠️ data.json 解析失败: {e}")
        
        sub_id = task_update.get("submission_id")
        if not sub_id: return
        
        found = False
        for t in data.get("tasks", []):
            if t.get("submission_id") == sub_id:
                t.update(task_update); found = True; break
        
        if not found:
            new_entry = {
                "submission_id": sub_id, "request_name": task_update.get("request_name", "未命名任务"),
                "request_pos": task_update.get("request_pos", ""), "request_case_cnt": task_update.get("request_case_cnt", 0),
                "request_video_cnt": task_update.get("request_video_cnt", 0), "is_cmp": False,
                "upload_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "cmp_time": None
            }
            data.setdefault("tasks", []).append(new_entry)
        
        # 3. 写入文件
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"❌ update_user_task_index 彻底失败: {e}")
    finally:
        # 4. 无论成功失败，必须释放锁
        if lock_path.exists(): lock_path.unlink()

# --- 后台模型调用包装器 (完全保留) ---

async def run_model_inference_wrapper(request_path: Path, output_root: Path, request_name: str, username: str, submission_id: str):
    print(f"🤖 [AI任务] 启动真实推理流程: {request_name} (ID: {submission_id})")
    if run_diagnosis is None:
        print("❌ [AI任务] 错误：模型函数未导入")
        return
    try:
        # 1. 运行 AI 诊断模型
        await asyncio.to_thread(run_diagnosis, target_dir=str(request_path), output_dir=str(output_root))
        
        # 2. 结果转码：遍历 output_videos 下的所有视频并修复编码
        processed_task_dir = output_root / submission_id
        if processed_task_dir.exists():
            for mp4_file in processed_task_dir.glob("**/output_videos/*.mp4"):
                await transcode_to_h264(mp4_file)
        
        # 3. 更新索引状态
        update_user_task_index(username, {
            "submission_id": submission_id, 
            "is_cmp": True, 
            "cmp_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        print(f"✅ [AI任务] {request_name} 处理与编码修复全部完成。")
        
    except Exception as e:
        print(f"🚨 [AI任务] 流程崩溃: {e}")
        import traceback
        traceback.print_exc()

# --- 核心 API 路由 (保持 100% 原文逻辑) ---

@router.post("/api/users/{username}/upload-oridata")
async def upload_oridata(
    username: str, 
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    request_name: str = Form("未命名请求"),
    case_names: List[str] = Form(...)
):
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="未提供任何文件")
    if len(files) != len(case_names):
        raise HTTPException(status_code=400, detail="文件与病例名清单不匹配")

    case_group_map: Dict[str, List[UploadFile]] = {}
    for f_obj, c_name in zip(files, case_names):
        if not Path(f_obj.filename).name.startswith('.') and is_allowed_file(f_obj.filename):
            case_group_map.setdefault(c_name, []).append(f_obj)
    
    if not case_group_map:
        raise HTTPException(status_code=400, detail="未检测到有效的视频文件")

    warnings = []
    final_save_map: Dict[str, List[UploadFile]] = {}
    actual_total_video_cnt = 0

    for c_name, c_files in case_group_map.items():
        if len(c_files) > MAX_VIDEOS_PER_CASE:
            warnings.append(f"病例 '{c_name}' 包含 {len(c_files)} 个视频，系统已自动截取前 5 个。")
            final_save_map[c_name] = c_files[:MAX_VIDEOS_PER_CASE]
        else:
            final_save_map[c_name] = c_files
        actual_total_video_cnt += len(final_save_map[c_name])

    submission_id = generate_submission_id()
    submission_base = BASE_DATA_DIR / username / ORIDATA_DIRNAME / submission_id

    try:
        submission_base.mkdir(parents=True, exist_ok=True)

        for c_name, c_files in final_save_map.items():
            target_case_dir = safe_join(submission_base, c_name)
            target_case_dir.mkdir(parents=True, exist_ok=True)
            for f in c_files:
                dest_filename = Path(f.filename).name
                dest_path = target_case_dir / dest_filename
                async with aiofiles.open(dest_path, 'wb') as out_f:
                    while chunk := await f.read(65536):
                        await out_f.write(chunk)
                asyncio.create_task(generate_thumbnail_ffmpeg(dest_path, dest_path.with_name(dest_filename + ".jpg")))

        meta = {
            "submission_id": submission_id, "request_name": request_name, "username": username,
            "created_at": datetime.datetime.now().isoformat(), "total_cases": len(final_save_map),
            "total_videos": actual_total_video_cnt, "warnings": warnings
        }
        with open(submission_base / "metadata.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)

        update_user_task_index(username, {
            "submission_id": submission_id,
            "request_name": request_name,
            "request_pos": f"processed/{submission_id}", 
            "request_case_cnt": len(final_save_map),
            "request_video_cnt": actual_total_video_cnt,
            "is_cmp": False
        })

        background_tasks.add_task(
            run_model_inference_wrapper, 
            submission_base, 
            BASE_DATA_DIR / username / "processed", 
            request_name, 
            username, 
            submission_id
        )

        return {"message": "上传成功", "submissionId": submission_id, "warnings": warnings}

    except Exception as e:
        print(f"🚨 Upload Failed: {e}")
        if submission_base.exists(): shutil.rmtree(submission_base)
        raise HTTPException(status_code=500, detail=str(e))

# --- 附加接口 (完全保留) ---

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
    user_oridata_path = BASE_DATA_DIR / username / ORIDATA_DIRNAME
    if not user_oridata_path.exists(): return {"message": "找不到原始数据目录"}
    physical_folders = [d.name for d in user_oridata_path.iterdir() if d.is_dir()]
    for sub_id in physical_folders:
        update_user_task_index(username, {"submission_id": sub_id, "request_name": "同步任务", "is_cmp": True})
    return {"message": "同步完成"}