# routes_oridata.py
from fastapi import APIRouter, UploadFile, File, HTTPException
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
    """
    异步尝试用 ffmpeg 生成缩略图（首帧），如果系统没有 ffmpeg 或命令失败则返回 False。
    thumb_path: 完整输出路径（jpg）
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{seek_time}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(thumb_path)
    ]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        return proc.returncode == 0
    except FileNotFoundError:
        # ffmpeg 未安装或不可用
        return False
    except Exception:
        return False

# 如果需要在后台并行处理（不阻塞响应），可以在保存后调用 asyncio.create_task(generate_thumbnail_ffmpeg(...))
# 或者在同步环境中使用 subprocess.run，但这里用 async 子进程更合适。

@router.post("/api/users/{username}/upload-oridata")
async def upload_oridata(username: str, files: List[UploadFile] = File(...)):
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="未提供任何文件")

    normalized_names = []
    for f in files:
        name = f.filename or ""
        name = name.replace("\\", "/")
        normalized_names.append(name)

    submission_id = generate_submission_id()
    submission_base = BASE_DATA_DIR / username / ORIDATA_DIRNAME / submission_id
    try:
        submission_base.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        submission_id = generate_submission_id()
        submission_base = BASE_DATA_DIR / username / ORIDATA_DIRNAME / submission_id
        submission_base.mkdir(parents=True, exist_ok=False)

    try:
        contains_slash = any("/" in n for n in normalized_names)

        # CASE A: No slashes at all -> standalone files (client likely sent plain filenames)
        if not contains_slash:
            if len(files) > MAX_VIDEOS_PER_CASE:
                raise HTTPException(status_code=400, detail=f"一次上传的独立视频数量为 {len(files)}，超过上限 {MAX_VIDEOS_PER_CASE}")
            case_name = "".join(random.choices(string.ascii_letters + string.digits, k=16))
            target_case_dir = safe_join(submission_base, case_name)
            target_case_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                if not is_allowed_file(f.filename):
                    raise HTTPException(status_code=400, detail=f"不支持的文件格式：{f.filename}")
                dest = safe_join(target_case_dir, Path(f.filename).name)
                async with aiofiles.open(dest, 'wb') as out_file:
                    while True:
                        chunk = await f.read(65536)
                        if not chunk:
                            break
                        await out_file.write(chunk)
                try:
                    await f.close()
                except Exception:
                    pass
                # try to async generate thumbnail (background)
                thumb_path = dest.with_name(dest.name + ".jpg")
                asyncio.create_task(generate_thumbnail_ffmpeg(dest, thumb_path))
            meta = {
                "submission_id": submission_id,
                "username": username,
                "created_at": datetime.datetime.now().isoformat(),
                "cases": { case_name: [p.name for p in sorted((submission_base / case_name).iterdir()) if p.is_file()] }
            }
            meta_path = submission_base / "metadata.json"
            async with aiofiles.open(meta_path, "w", encoding="utf-8") as mf:
                await mf.write(json.dumps(meta, ensure_ascii=False, indent=2))
            return {"message": "上传成功", "submissionId": submission_id, "path": str(submission_base)}

        # CASE B: There are slashes -> could be either:
        #   B1) three+ parts: requestDir/caseName/...  (require first component consistent across all files)
        #   B2) two parts: caseName/file  (treat as direct case mapping)
        parts_list = [ [p for p in PurePosixPath(n).parts if p not in ("", ".")] for n in normalized_names ]
        if all(len(p) >= 3 for p in parts_list):
            first_components = set(p[0] for p in parts_list)
            if len(first_components) != 1:
                raise HTTPException(status_code=400, detail="一次上传应来自同一请求目录（请确保上传的文件均来自同一一级目录）")
            case_map: Dict[str, List[Dict]] = {}
            for f_obj, rawname in zip(files, normalized_names):
                parts = [p for p in PurePosixPath(rawname).parts if p not in ("", ".")]
                case_name = parts[1]
                rel_under_case = "/".join(parts[2:])
                if not rel_under_case:
                    raise HTTPException(status_code=400, detail=f"文件路径无效或缺少文件名：{rawname}")
                if not is_allowed_file(rel_under_case):
                    raise HTTPException(status_code=400, detail=f"不支持的文件格式：{rawname}")
                case_map.setdefault(case_name, []).append({"upload": f_obj, "orig_name": Path(rel_under_case).name})
            for case_name, items in case_map.items():
                if len(items) > MAX_VIDEOS_PER_CASE:
                    raise HTTPException(status_code=400, detail=f"病例 '{case_name}' 包含 {len(items)} 个视频，超过上限 {MAX_VIDEOS_PER_CASE}")
            for case_name, items in case_map.items():
                target_case_dir = safe_join(submission_base, case_name)
                target_case_dir.mkdir(parents=True, exist_ok=True)
                for it in items:
                    dest = safe_join(target_case_dir, it["orig_name"])
                    up = it["upload"]
                    async with aiofiles.open(dest, 'wb') as out_file:
                        while True:
                            chunk = await up.read(65536)
                            if not chunk:
                                break
                            await out_file.write(chunk)
                    try:
                        await up.close()
                    except Exception:
                        pass
                    thumb_path = dest.with_name(dest.name + ".jpg")
                    asyncio.create_task(generate_thumbnail_ffmpeg(dest, thumb_path))
            meta = {
                "submission_id": submission_id,
                "username": username,
                "created_at": datetime.datetime.now().isoformat(),
                "cases": {}
            }
            for case_name in case_map.keys():
                case_dir = submission_base / case_name
                files_in_case = [p.name for p in sorted(case_dir.iterdir()) if p.is_file()]
                meta["cases"][case_name] = files_in_case
            meta_path = submission_base / "metadata.json"
            async with aiofiles.open(meta_path, "w", encoding="utf-8") as mf:
                await mf.write(json.dumps(meta, ensure_ascii=False, indent=2))
            return {"message": "上传成功", "submissionId": submission_id, "path": str(submission_base)}

        if all(len(p) == 2 for p in parts_list):
            case_map: Dict[str, List[Dict]] = {}
            for f_obj, rawname in zip(files, normalized_names):
                parts = [p for p in PurePosixPath(rawname).parts if p not in ("", ".")]
                case_name = parts[0]
                file_part = parts[1]
                if not is_allowed_file(file_part):
                    raise HTTPException(status_code=400, detail=f"不支持的文件格式：{rawname}")
                case_map.setdefault(case_name, []).append({"upload": f_obj, "orig_name": file_part})
            for case_name, items in case_map.items():
                if len(items) > MAX_VIDEOS_PER_CASE:
                    raise HTTPException(status_code=400, detail=f"病例 '{case_name}' 包含 {len(items)} 个视频，超过上限 {MAX_VIDEOS_PER_CASE}")
            for case_name, items in case_map.items():
                target_case_dir = safe_join(submission_base, case_name)
                target_case_dir.mkdir(parents=True, exist_ok=True)
                for it in items:
                    dest = safe_join(target_case_dir, it["orig_name"])
                    up = it["upload"]
                    async with aiofiles.open(dest, 'wb') as out_file:
                        while True:
                            chunk = await up.read(65536)
                            if not chunk:
                                break
                            await out_file.write(chunk)
                    try:
                        await up.close()
                    except Exception:
                        pass
                    thumb_path = dest.with_name(dest.name + ".jpg")
                    asyncio.create_task(generate_thumbnail_ffmpeg(dest, thumb_path))
            meta = {
                "submission_id": submission_id,
                "username": username,
                "created_at": datetime.datetime.now().isoformat(),
                "cases": {}
            }
            for case_name in case_map.keys():
                case_dir = submission_base / case_name
                files_in_case = [p.name for p in sorted(case_dir.iterdir()) if p.is_file()]
                meta["cases"][case_name] = files_in_case
            meta_path = submission_base / "metadata.json"
            async with aiofiles.open(meta_path, "w", encoding="utf-8") as mf:
                await mf.write(json.dumps(meta, ensure_ascii=False, indent=2))
            return {"message": "上传成功", "submissionId": submission_id, "path": str(submission_base)}

        raise HTTPException(status_code=400, detail="上传路径结构不被支持（请使用 request/case/file 或 case/file 或 直接上传视频文件）")

    except HTTPException:
        raise
    except Exception as e:
        try:
            if submission_base.exists():
                shutil.rmtree(submission_base)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"服务器错误: {e}")