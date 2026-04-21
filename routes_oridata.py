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
import zipfile
import io

# --- [新增] 动态导入模型路径 ---
CURRENT_ROOT = Path(__file__).parent.resolve()
MODEL_PATH = (CURRENT_ROOT / "model").resolve()

if str(MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(MODEL_PATH))
if str(CURRENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_ROOT))

# 教育模式专用路径
SYSTEM_EDU_DIR = Path("data_batch_storage") / "SYSTEM" / "edu_data"
EDU_RESULTS_DIR = SYSTEM_EDU_DIR / "Doctor_Diag_Result"
BASE_DATA_DIR = Path("data_batch_storage")
INFERENCE_STATS_FILE = SYSTEM_EDU_DIR / "inference_stats.json"

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
    
    # [已修正] 移除括号，确保 -crf 23 参数正确
    cmd = (
        f'ffmpeg -y -i {input_str} '
        f'-vcodec libx264 -pix_fmt yuv420p '
        f'-preset fast -crf 23 '
        f'-acodec aac {output_str}'
    )
    
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

# 教育版
def update_edu_task_index(task_update: dict):
    """维护 SYSTEM/edu_data/data.json 教育任务索引"""
    index_path = SYSTEM_EDU_DIR / "data.json"
    lock_path = SYSTEM_EDU_DIR / "data.json.lock"
    
    # 1. 获取文件锁 (原子安全)
    acquired = False
    for _ in range(50):
        try:
            with open(lock_path, "x") as _: pass
            acquired = True; break
        except FileExistsError: time.sleep(0.1)
    if not acquired: return

    try:
        # 2. 读取现有数据
        data = {"tasks": []}
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content: data = json.loads(content)
        
        sub_id = task_update.get("submission_id")
        found = False
        for t in data.get("tasks", []):
            if t.get("submission_id") == sub_id:
                t.update(task_update); found = True; break
        
        if not found:
            new_entry = {
                "submission_id": sub_id,
                "request_name": task_update.get("request_name", "教育批次"),
                "status": "processing",  # processing | unreleased | published
                "target_users": [],      # 已发布的用户列表
                "request_case_cnt": task_update.get("request_case_cnt", 0),
                "request_video_cnt": task_update.get("request_video_cnt", 0),
                "upload_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_cmp": False
            }
            data["tasks"].append(new_entry)
        
        # 3. 写入文件
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    finally:
        if lock_path.exists(): lock_path.unlink()

# --- [重构] 后台模型调用包装器：支持教育模式 ---

async def run_model_inference_wrapper(request_path: Path, output_root: Path, request_name: str, username: str, submission_id: str, is_system: bool = False):
    """
    后台AI推理任务包装器
    状态追踪：使用 .processing 和 .failed 文件标记任务状态
    """
    print(f"🤖 [AI任务] 启动推理流程: {request_name} (ID: {submission_id}, 系统任务: {is_system})")
    
    if run_diagnosis is None:
        print("❌ [AI任务] 错误：模型函数未导入")
        return
    
    # 确定状态文件路径（用户空间或系统空间）
    if is_system:
        task_root = SYSTEM_EDU_DIR
    else:
        task_root = BASE_DATA_DIR / username
    
    processing_flag = task_root / "oridata" / submission_id / ".processing"
    failed_flag = task_root / "oridata" / submission_id / ".failed"
    
    # 创建处理中标记
    try:
        processing_flag.parent.mkdir(parents=True, exist_ok=True)
        with open(processing_flag, "w", encoding="utf-8") as f:
            f.write(f"started_at: {datetime.datetime.now().isoformat()}\n")
        print(f"📍 [AI任务] 状态标记已创建: {processing_flag}")
    except Exception as e:
        print(f"⚠️ [AI任务] 无法创建状态标记: {e}")
    
    try:
        # 1. 运行 AI 诊断模型（捕获返回的推理时间）
        inference_result = await asyncio.to_thread(run_diagnosis, target_dir=str(request_path), output_dir=str(output_root))
        inference_duration = inference_result.get("duration", 0) if isinstance(inference_result, dict) else 0
        
        # 2. 结果转码：遍历 output_videos 下的所有视频并修复编码
        processed_task_dir = output_root / submission_id
        if processed_task_dir.exists():
            for mp4_file in processed_task_dir.glob("**/output_videos/*.mp4"):
                await transcode_to_h264(mp4_file)
        
        # 3. 更新索引状态 (分流更新用户索引或系统教育索引)
        if is_system:
            update_edu_task_index({
                "submission_id": submission_id, 
                "status": "unreleased", # 处理完默认为未发布
                "is_cmp": True, 
                "cmp_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            update_user_task_index(username, {
                "submission_id": submission_id, 
                "is_cmp": True, 
                "cmp_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # 4.5 将推理时间持久化到独立统计文件（不受任务删除影响）
        try:
            stats_data = {"total_duration": 0, "total_cases": 0}
            if INFERENCE_STATS_FILE.exists():
                with open(INFERENCE_STATS_FILE, "r", encoding="utf-8") as f:
                    stats_data = json.load(f)
            
            stats_data["total_duration"] = stats_data.get("total_duration", 0) + inference_duration
            
            # 获取病例数
            case_count = 0
            if processed_task_dir.exists():
                case_count = sum(1 for d in processed_task_dir.iterdir() if d.is_dir())
            stats_data["total_cases"] = stats_data.get("total_cases", 0) + case_count
            
            with open(INFERENCE_STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 [AI任务] 推理统计已保存: 时间={inference_duration}秒, 病例={case_count}")
        except Exception as e:
            print(f"⚠️ [AI任务] 保存推理统计失败: {e}")
        
        # 4. 成功完成：删除处理中标记
        if processing_flag.exists():
            processing_flag.unlink()
            print(f"✅ [AI任务] {request_name} 任务处理与编码修复全部完成。")
        
    except Exception as e:
        print(f"🚨 [AI任务] 流程崩溃: {e}")
        import traceback
        traceback.print_exc()
        
        # 失败：将 .processing 重命名为 .failed
        if processing_flag.exists():
            try:
                processing_flag.rename(failed_flag)
                with open(failed_flag, "a", encoding="utf-8") as f:
                    f.write(f"failed_at: {datetime.datetime.now().isoformat()}\n")
                    f.write(f"error: {str(e)}\n")
                print(f"❌ [AI任务] 状态已标记为失败: {failed_flag}")
            except Exception as rename_err:
                print(f"⚠️ [AI任务] 无法更新失败标记: {rename_err}")

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

        # 触发模型 (is_system=False)
        background_tasks.add_task(
            run_model_inference_wrapper, 
            submission_base, 
            BASE_DATA_DIR / username / "processed", 
            request_name, 
            username, 
            submission_id,
            False 
        )

        return {"message": "上传成功", "submissionId": submission_id, "warnings": warnings}

    except Exception as e:
        print(f"🚨 Upload Failed: {e}")
        if submission_base.exists(): shutil.rmtree(submission_base)
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
    
    # 为每个任务添加状态检查
    tasks_with_status = []
    for task in data["tasks"]:
        submission_id = task.get("submission_id")
        ori_path = BASE_DATA_DIR / username / ORIDATA_DIRNAME / submission_id
        
        # 检查失败状态
        failed_flag = ori_path / ".failed"
        has_failed = failed_flag.exists()
        
        # 检查处理状态
        processing_flag = ori_path / ".processing"
        is_processing = processing_flag.exists()
        
        # 检查是否已完成
        is_completed = task.get("is_cmp", False)
        
        # 判断任务状态
        task_status = "completed" if is_completed else "processing" if is_processing else "failed" if has_failed else "stuck"
        
        tasks_with_status.append({
            **task,
            "has_failed": has_failed,
            "is_processing": is_processing,
            "task_status": task_status
        })
    
    data["tasks"] = tasks_with_status
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


# --- [新增] 清空未完成且未在运行的任务 ---

@router.delete("/api/users/{username}/clear-stuck-tasks")
async def clear_stuck_tasks(username: str):
    """
    清空当前用户下所有未完成且未在运行的任务（安全删除）
    
    删除条件：
    1. is_cmp = False（未完成）
    2. 不存在 .processing 文件（没有后台任务在运行）
    
    安全机制：
    - 已完成的任务（is_cmp = True）不受影响
    - 正在运行的任务（存在 .processing 文件）不受影响
    - 失败的任务（存在 .failed 文件）会被清理
    """
    user_root = BASE_DATA_DIR / username
    index_path = user_root / "data.json"
    
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="用户任务索引不存在")
    
    # 读取当前任务列表
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tasks = data.get("tasks", [])
    deleted_tasks = []
    kept_tasks = []
    
    for task in tasks:
        submission_id = task.get("submission_id")
        is_cmp = task.get("is_cmp", False)
        
        # 只处理未完成的任务
        if is_cmp:
            kept_tasks.append(task)
            continue
        
        # 检查是否有 .processing 文件（正在运行）
        ori_path = user_root / ORIDATA_DIRNAME / submission_id
        processing_flag = ori_path / ".processing"
        
        if processing_flag.exists():
            # 正在运行，保留
            kept_tasks.append(task)
            print(f"⏭️ 任务 {submission_id} 正在运行，跳过")
            continue
        
        # 满足删除条件
        try:
            # 删除原始数据
            if ori_path.exists():
                shutil.rmtree(ori_path)
                print(f"🗑️ 已删除原始数据: {ori_path}")
            
            # 删除处理结果（如果存在）
            request_pos = task.get("request_pos")
            if request_pos:
                proc_path = user_root / request_pos
                if proc_path.exists():
                    shutil.rmtree(proc_path)
                    print(f"🗑️ 已删除处理结果: {proc_path}")
            
            deleted_tasks.append({
                "submission_id": submission_id,
                "request_name": task.get("request_name", "未命名")
            })
            
        except Exception as e:
            print(f"⚠️ 删除任务 {submission_id} 失败: {e}")
            # 删除失败的任务保留在列表中
            kept_tasks.append(task)
    
    # 更新索引文件
    data["tasks"] = kept_tasks
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return {
        "message": f"已清理 {len(deleted_tasks)} 个卡住的任务",
        "deleted_count": len(deleted_tasks),
        "deleted_tasks": deleted_tasks,
        "kept_count": len(kept_tasks)
    }


# --- [教育模式核心接口：修正自适应路径版] ---

@router.post("/api/edu/parse-zip")
async def parse_edu_zip(file: UploadFile = File(...)):
    """解析教育压缩包统计信息 [自适应层级版]"""
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="仅支持 ZIP 格式")
    try:
        content = await file.read()
        z = zipfile.ZipFile(io.BytesIO(content))
        
        # [重点修复]：在 ZIP 全路径列表中寻找 epoch_data.json
        full_namelist = z.namelist()
        json_inner_path = next((x for x in full_namelist if x.endswith('epoch_data.json')), None)
        
        if not json_inner_path:
            raise HTTPException(status_code=400, detail="压缩包内缺失 epoch_data.json")
            
        with z.open(json_inner_path) as f:
            epoch_data = json.load(f)
            
        case_cnt = len(epoch_data)
        video_cnt = sum(len(c.get("videos", [])) for c in epoch_data.values())
            
        return {
            "suggested_name": Path(file.filename).stem,
            "case_count": case_cnt,
            "video_count": video_cnt,
            "submission_id": generate_submission_id()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"解析失败: {str(e)}")

@router.post("/api/edu/confirm-upload")
async def confirm_edu_upload(
    background_tasks: BackgroundTasks,
    submission_id: str = Form(...),
    request_name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    管理员确认后：
    1. 解压原始文件。
    2. [核心] 根据 epoch_data.json 将视频重组为 case 子目录结构，以适配 AI 模型。
    3. 触发推理。
    """
    submission_base = SYSTEM_EDU_DIR / submission_id
    processed_root = SYSTEM_EDU_DIR / "processed"
    
    try:
        # 1. 物理接收
        submission_base.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            z.extractall(submission_base)
        
        # [智慧平铺] 如果包了一层父目录，先把它提上来
        for sub in list(submission_base.iterdir()):
            if sub.is_dir() and (sub / "epoch_data.json").exists():
                for item in sub.iterdir(): shutil.move(str(item), str(submission_base))
                sub.rmdir()

        # 2. 结构重组 (Structure Restoration)
        json_path = submission_base / "epoch_data.json"
        with open(json_path, "r", encoding="utf-8") as f:
            epoch_data = json.load(f)

        print(f"🛠️ 正在为 AI 模型重组目录结构...")
        for case_id, info in epoch_data.items():
            # 创建病例文件夹 (如 submission_id/case1/)
            case_dir = submission_base / case_id
            case_dir.mkdir(exist_ok=True)
            
            for rel_v_path in info.get("videos", []):
                # 原始位置 (通常在 videos/ 目录下)
                src_path = submission_base / rel_v_path
                # 目标位置 (病例文件夹下)
                dst_path = case_dir / src_path.name
                
                if src_path.exists():
                    shutil.move(str(src_path), str(dst_path))
        
        # 清理掉现在已经没用的原始视频目录 (如果存在)
        orig_v_dir = submission_base / "videos"
        if orig_v_dir.exists() and not any(orig_v_dir.iterdir()):
            orig_v_dir.rmdir()

        # 3. 记录索引与分发任务（单个任务，后端自动处理双阶段）
        update_edu_task_index({
            "submission_id": submission_id,
            "request_name": request_name,
            "is_dual_stage": True,  # 标记为双阶段任务
            "request_pos": f"processed/{submission_id}",
            "request_case_cnt": len(epoch_data),
            "request_video_cnt": sum(len(c.get("videos", [])) for c in epoch_data.values()),
            "status": "processing",
            "is_cmp": False
        })
        
        # 触发模型处理，SYSTEM 空间的输出路径也要对齐
        background_tasks.add_task(run_model_inference_wrapper, submission_base, processed_root, request_name, "SYSTEM", submission_id, True)
        
        return {"message": "结构重组成功，任务已开始判读"}
        
    except Exception as e:
        if submission_base.exists(): shutil.rmtree(submission_base)
        print(f"🚨 教育任务创建失败: {e}")
        raise HTTPException(status_code=500, detail=f"结构重组失败: {str(e)}")


@router.get("/api/edu/admin/tasks")
async def get_edu_admin_tasks():
    index_path = SYSTEM_EDU_DIR / "data.json"
    if not index_path.exists(): 
        return {"processing": [], "unreleased": [], "published": []}
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data.get("tasks", [])
    tasks.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
    return {
        "processing": [t for t in tasks if t.get("status") == "processing"],
        "unreleased": [t for t in tasks if t.get("status") == "unreleased"],
        "published": [t for t in tasks if t.get("status") == "published"]
    }


@router.get("/api/admin/inference-stats")
async def get_inference_stats():
    """获取所有任务的模型推理统计（从独立统计文件读取，不受任务删除影响）"""
    total_duration = 0
    total_cases = 0
    
    # 从独立统计文件读取（推理完成后立即保存）
    if INFERENCE_STATS_FILE.exists():
        try:
            with open(INFERENCE_STATS_FILE, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
            total_duration = stats_data.get("total_duration", 0)
            total_cases = stats_data.get("total_cases", 0)
        except Exception as e:
            print(f"⚠️ 读取推理统计失败: {e}")
    
    # 计算每病例平均时间
    avg_per_case = total_duration / total_cases if total_cases > 0 else 0
    
    return {
        "total_duration": round(total_duration, 2),
        "total_cases": total_cases,
        "avg_per_case": round(avg_per_case, 2)
    }


@router.post("/api/edu/publish-task")
async def publish_edu_task(submission_id: str = Form(...), target_users: str = Form(...), publish_mode: str = Form("dual")):
    try:
        users = json.loads(target_users)
        if not users: raise HTTPException(status_code=400, detail="发布对象不能为空")
        
        # 根据 publish_mode 设置 is_dual_stage 和 edu_sub_mode
        is_dual_stage = (publish_mode == "dual")
        is_triple_stage = (publish_mode == "triple")
        edu_sub_mode = "triple" if is_triple_stage else ("dual" if is_dual_stage else "single")
        
        # 更新任务状态
        update_edu_task_index({
            "submission_id": submission_id,
            "status": "published",
            "target_users": users,
            "is_dual_stage": is_dual_stage,
            "edu_sub_mode": edu_sub_mode
        })
        
        if is_triple_stage:
            mode_msg = "三级模式任务"
        elif is_dual_stage:
            mode_msg = "双阶段任务"
        else:
            mode_msg = "单阶段任务"
        return {"message": f"{mode_msg}已成功发布"}
    except Exception as e: raise HTTPException(status_code=400, detail=f"发布失败: {e}")

@router.get("/api/edu/admin/task-status/{submission_id}")
async def get_edu_task_detail_status(submission_id: str):
    """获取特定教育任务下所有发布对象的实时判读进度"""
    index_path = SYSTEM_EDU_DIR / "data.json"
    if not index_path.exists(): raise HTTPException(status_code=404)
    
    with open(index_path, "r", encoding="utf-8") as f:
        tasks = json.load(f).get("tasks", [])
    
    task = next((t for t in tasks if t["submission_id"] == submission_id), None)
    if not task: raise HTTPException(status_code=404, detail="任务不存在")

    target_users = task.get("target_users", [])
    is_dual_stage = task.get("is_dual_stage", False)
    user_status_list = []
    print(f"🔍 [admin-task-status] submission_id={submission_id}, is_dual_stage={is_dual_stage}")

    # 扫描每个用户的成绩单
    for username in target_users:
        result_file = EDU_RESULTS_DIR / f"{username}.json"
        user_info = {"username": username, "completed": False, "score": None, "details": None}
        
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                res_data = json.load(f)
                
                if is_dual_stage:
                    # 双阶段任务：检查两个子任务是否都提交了诊断结果
                    # 完成条件：两个阶段都提交了诊断结果
                    # 使用嵌套结构: res_data[parent_id]["stages"]["single"]
                    single_data = res_data.get(submission_id, {}).get("stages", {}).get("single")
                    assist_data = res_data.get(submission_id, {}).get("stages", {}).get("assist")
                    single_exists = single_data is not None
                    assist_exists = assist_data is not None
                    
                    print(f"🔍 [admin-task-status] user={username}, single_exists={single_exists}, assist_exists={assist_exists}")
                    
                    if single_exists and assist_exists:
                        user_info["completed"] = True
                        # 取两个阶段的平均分
                        single_score = single_data.get("accuracy", 0)
                        assist_score = assist_data.get("accuracy", 0)
                        user_info["score"] = (single_score + assist_score) / 2
                        user_info["details"] = {
                            "single": single_data,
                            "assist": assist_data
                        }
                else:
                    # 普通任务：检查原始task ID是否已提交诊断结果
                    if submission_id in res_data:
                        print(f"🔍 [admin-task-status] user={username}, exists=True")
                        user_info["completed"] = True
                        user_info["score"] = res_data[submission_id].get("accuracy")
                        user_info["details"] = res_data[submission_id]
        
        user_status_list.append(user_info)
    
    return {"task_name": task["request_name"], "user_statuses": user_status_list}


@router.post("/api/edu/admin/unpublish-task")
async def unpublish_edu_task(submission_id: str = Form(...)):
    """取消发布任务：清空所有用户成绩并更新状态为未发布"""
    index_path = SYSTEM_EDU_DIR / "data.json"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="任务索引文件不存在")
    
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    task = next((t for t in data.get("tasks", []) if t["submission_id"] == submission_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task.get("status") != "published":
        raise HTTPException(status_code=400, detail="任务未发布，无法取消发布")
    
    target_users = task.get("target_users", [])
    is_dual_stage = task.get("is_dual_stage", False)
    
    for username in target_users:
        result_file = EDU_RESULTS_DIR / f"{username}.json"
        if result_file.exists():
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
                # 新格式：直接删除父键（包含stages嵌套结构，三阶段也适用）
                if submission_id in user_data:
                    del user_data[submission_id]
                # 旧格式后缀结构（兼容旧数据）
                if f"{submission_id}_SINGLE" in user_data:
                    del user_data[f"{submission_id}_SINGLE"]
                if f"{submission_id}_AI-ASSIST" in user_data:
                    del user_data[f"{submission_id}_AI-ASSIST"]
                if f"{submission_id}_REVIEW" in user_data:
                    del user_data[f"{submission_id}_REVIEW"]
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(user_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"清空用户 {username} 成绩失败: {e}")
    
    task["status"] = "unreleased"
    if "target_users" in task:
        del task["target_users"]
    
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return {"message": "任务已取消发布，成绩已清空"}


@router.get("/api/edu/user/tasks/{username}")
async def get_user_edu_tasks(username: str):
    index_path = SYSTEM_EDU_DIR / "data.json"
    if not index_path.exists(): return {"tasks": []}
    with open(index_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f).get("tasks", [])
    user_tasks = [t for t in all_tasks if t.get("status") == "published" and username in t.get("target_users", [])]
    result_file = EDU_RESULTS_DIR / f"{username}.json"
    user_results = {}

    # 读取用户已完成的成绩
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            user_results = json.load(f)
    
    # 不再拆分双阶段任务，只标记完成状态
    for t in user_tasks:
        sub_id = t["submission_id"]
        is_dual = t.get("is_dual_stage", False)
        is_triple = t.get("edu_sub_mode") == "triple"
        
        if is_triple:
            # 三阶段任务：检查三个子任务是否都完成
            task_result = user_results.get(sub_id, {})
            stages = task_result.get("stages", {})
            single_data = stages.get("single")
            assist_data = stages.get("assist")
            review_data = stages.get("review")
            single_done = single_data is not None
            assist_done = assist_data is not None
            review_done = review_data is not None
            all_done = single_done and assist_done and review_done
            
            t["is_completed"] = all_done
            t["single_done"] = single_done
            t["assist_done"] = assist_done
            t["review_done"] = review_done
            t["edu_sub_mode"] = "triple"
            
            if all_done:
                t["last_score"] = review_data.get("accuracy", 0)
            else:
                t["last_score"] = None
        elif is_dual:
            # 双阶段任务：检查两个子任务是否都完成
            # 使用嵌套结构: user_results[parent_id]["stages"]["single"]
            task_result = user_results.get(sub_id, {})
            stages = task_result.get("stages", {})
            single_data = stages.get("single")
            assist_data = stages.get("assist")
            single_done = single_data is not None
            assist_done = assist_data is not None
            both_done = single_done and assist_done
            
            t["is_completed"] = both_done
            t["single_done"] = single_done
            t["assist_done"] = assist_done
            t["edu_sub_mode"] = "dual"  # 标记为双阶段任务
            
            if both_done:
                t["last_score"] = max(
                    single_data.get("accuracy", 0),
                    assist_data.get("accuracy", 0)
                )
            else:
                t["last_score"] = None
        else:
            t["is_completed"] = sub_id in user_results
            t["last_score"] = user_results.get(sub_id, {}).get("accuracy") if t["is_completed"] else None

    return {"tasks": user_tasks}


@router.get("/api/edu/check-task-status/{submission_id}")
async def check_task_status(submission_id: str):
    """检查教育任务状态，返回是否有效"""
    # 处理多阶段任务：提取父任务ID
    parent_id = submission_id
    if submission_id.endswith('_SINGLE') or submission_id.endswith('_AI-ASSIST') or submission_id.endswith('_REVIEW'):
        parent_id = submission_id.rsplit('_', 1)[0]  # 去掉后缀
    
    index_path = SYSTEM_EDU_DIR / "data.json"
    if not index_path.exists():
        return {"valid": False, "reason": "任务不存在"}
    
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    task = next((t for t in data.get("tasks", []) if t.get("submission_id") == parent_id), None)
    if not task:
        return {"valid": False, "reason": "任务不存在"}
    
    if task.get("status") != "published":
        return {"valid": False, "reason": f"任务状态为 {task.get('status')}", "status": task.get("status")}
    
    return {"valid": True}


@router.get("/api/edu/user/result/{username}/{submission_id}")
async def get_user_edu_result(username: str, submission_id: str):
    """
    获取用户教育任务成绩
    支持新旧两种格式：
    - 新格式（嵌套结构）: /result/username/taskId?stage=single
    - 旧格式（后缀结构）: /result/username/taskId_SINGLE
    """
    result_file = EDU_RESULTS_DIR / f"{username}.json"
    if not result_file.exists(): raise HTTPException(status_code=404)
    with open(result_file, "r", encoding="utf-8") as f: data = json.load(f)
    
    # 尝试提取stage参数（Query参数）
    from urllib.parse import urlparse, parse_qs
    # 这里stage会从查询参数获取
    
    # 提取父任务ID（去掉后缀）
    parent_id = submission_id
    stage = None
    if submission_id.endswith('_SINGLE'):
        parent_id = submission_id.replace('_SINGLE', '')
        stage = 'single'
    elif submission_id.endswith('_AI-ASSIST'):
        parent_id = submission_id.replace('_AI-ASSIST', '')
        stage = 'assist'
    elif submission_id.endswith('_REVIEW'):
        parent_id = submission_id.replace('_REVIEW', '')
        stage = 'review'
    
    # 尝试新嵌套结构
    if parent_id in data:
        task_data = data[parent_id]
        # 如果指定了stage，返回合并后的数据（包含stages和llm_analysis）
        if stage:
            result = {}
            # 合并stages数据
            if "stages" in task_data and stage in task_data["stages"]:
                result.update(task_data["stages"][stage])
            # 合并llm_analysis数据
            if "llm_analysis" in task_data and stage in task_data["llm_analysis"]:
                result.update(task_data["llm_analysis"][stage])
            result["stage"] = stage
            return result
        else:
            # 返回整个任务数据
            return task_data
    
    # 兼容旧格式：直接用submission_id查找
    if submission_id in data:
        return data[submission_id]
    
    raise HTTPException(status_code=404)


@router.post("/api/edu/trigger-llm-analysis/{username}/{submission_id}")
async def trigger_llm_analysis(username: str, submission_id: str):
    """
    手动触发AI分析（当自动触发失败时使用）
    支持嵌套结构和新旧两种格式
    """
    from main import analyze_with_llm, DATA_BATCH_STORAGE
    from pathlib import Path
    import json
    
    result_file = EDU_RESULTS_DIR / f"{username}.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="成绩文件不存在")
    
    with open(result_file, "r", encoding="utf-8") as f:
        user_data = json.load(f)
    
    # 提取父任务ID和阶段
    parent_id = submission_id
    stage = None
    if submission_id.endswith('_SINGLE'):
        parent_id = submission_id.replace('_SINGLE', '')
        stage = 'single'
    elif submission_id.endswith('_AI-ASSIST'):
        parent_id = submission_id.replace('_AI-ASSIST', '')
        stage = 'assist'
    elif submission_id.endswith('_REVIEW'):
        parent_id = submission_id.replace('_REVIEW', '')
        stage = 'review'
    
    # 尝试嵌套结构
    if parent_id in user_data and stage:
        task_data = user_data[parent_id]
        stages_data = task_data.get("stages", {})
        llm_analysis = task_data.get("llm_analysis", {})
        
        if stage in stages_data:
            stats = stages_data[stage]
        else:
            raise HTTPException(status_code=404, detail=f"阶段 {stage} 的成绩不存在")
        
        # 检查是否已有分析结果
        if llm_analysis.get(stage, {}).get("status") == "completed":
            return {"message": "分析已完成", "status": "completed"}
        if llm_analysis.get(stage, {}).get("status") == "failed":
            # 清空失败状态，允许重新触发
            if "llm_analysis" not in task_data:
                task_data["llm_analysis"] = {}
            task_data["llm_analysis"][stage] = {"status": "pending"}
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
        
        # 异步调用大模型分析
        import asyncio
        asyncio.create_task(analyze_with_llm(username, parent_id, stage, stats))
        
        # 更新状态为pending
        if "llm_analysis" not in task_data:
            task_data["llm_analysis"] = {}
        task_data["llm_analysis"][stage] = {"status": "pending"}
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
        
        return {"message": "已触发AI分析", "status": "pending"}
    
    # 兼容旧格式
    if submission_id not in user_data:
        raise HTTPException(status_code=404, detail="任务记录不存在")
    
    stats = user_data[submission_id]
    
    # 检查是否已有分析结果或错误
    current_status = stats.get("llm_analysis_status", "pending")
    if current_status == "completed":
        return {"message": "分析已完成", "status": "completed"}
    if current_status == "failed":
        # 清空失败状态，允许重新触发
        stats["llm_analysis_status"] = "pending"
        stats.pop("llm_analysis", None)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
    
    # 异步调用大模型分析
    import asyncio
    asyncio.create_task(analyze_with_llm(username, submission_id, None, stats))
    
    # 更新状态为pending
    stats["llm_analysis_status"] = "pending"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)
    
    return {"message": "已触发AI分析", "status": "pending"}



@router.delete("/api/edu/admin/tasks/{submission_id}")
async def delete_edu_task(submission_id: str):
    """
    [管理员特权] 物理删除 SYSTEM 空间下的教育任务及所有关联数据
    """
    index_path = SYSTEM_EDU_DIR / "data.json"
    if not index_path.exists(): raise HTTPException(status_code=404)

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 1. 查找任务
        task = next((t for t in data["tasks"] if t["submission_id"] == submission_id), None)
        if not task: raise HTTPException(status_code=404, detail="任务不存在")

        # 2. 物理删除 (原始包 + 处理结果)
        shutil.rmtree(SYSTEM_EDU_DIR / submission_id, ignore_errors=True)
        shutil.rmtree(SYSTEM_EDU_DIR / "processed" / submission_id, ignore_errors=True)

        # 3. 删除所有用户的成绩记录
        target_users = task.get("target_users", [])
        for username in target_users:
            result_file = EDU_RESULTS_DIR / f"{username}.json"
            if result_file.exists():
                try:
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
                    print(f"🗑️ 已清理用户 {username} 的任务成绩")
                except Exception as e:
                    print(f"清理用户 {username} 成绩失败: {e}")

        # 4. 更新索引
        data["tasks"] = [t for t in data["tasks"] if t["submission_id"] != submission_id]
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"🗑️ 系统教育任务已物理删除: {submission_id}")
        return {"message": "任务已彻底从系统空间删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- [新增] 清空无效的教育任务 ---
@router.delete("/api/edu/admin/clear-stuck-tasks")
async def clear_stuck_edu_tasks():
    """
    清理无效的教育任务（卡住的任务）
    
    清理条件：
    1. status = "processing" (AI处理中)
    2. 不存在 .processing 文件 (没有后台任务在运行)
    
    保护机制：
    - 已发布/未发布的任务不受影响
    - 正在运行的任务不受影响
    """
    index_path = SYSTEM_EDU_DIR / "data.json"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="任务索引不存在")
    
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tasks = data.get("tasks", [])
    deleted_tasks = []
    kept_tasks = []
    
    for task in tasks:
        submission_id = task.get("submission_id")
        status = task.get("status", "")
        
        # 只处理 processing 状态的任务
        if status != "processing":
            kept_tasks.append(task)
            continue
        
        # 检查是否有 .processing 文件（正在运行）
        processed_path = SYSTEM_EDU_DIR / "processed" / submission_id
        processing_flag = processed_path / ".processing"
        
        if processing_flag.exists():
            # 正在运行，保留
            kept_tasks.append(task)
            print(f"⏭️ 任务 {submission_id} 正在运行，跳过")
        else:
            # 无效任务，可以清理
            # 将状态改为 unreleased，而不是物理删除
            task["status"] = "unreleased"
            kept_tasks.append(task)
            deleted_tasks.append(submission_id)
            print(f"🧹 任务 {submission_id} 已标记为无效（无processing文件），已移至未发布")
    
    data["tasks"] = kept_tasks
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return {"message": f"已清理 {len(deleted_tasks)} 个无效任务", "deleted_count": len(deleted_tasks)}