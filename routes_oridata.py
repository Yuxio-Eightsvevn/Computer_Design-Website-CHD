# routes_oridata.py
# 新增 FastAPI 路由模块，提供 /api/users/{username}/upload-oridata 接口。
# 使用方法：在 main.py 中 include_router:
#   from routes_oridata import router as oridata_router
#   app.include_router(oridata_router, prefix="", tags=["oridata"])
#
# 依赖：FastAPI, typing, pathlib, datetime, random, os, shutil
# 写入路径：data_batch_storage/{username}/oridata/{submission_id}/...

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi import Request
from typing import List, Dict
from pathlib import Path, PurePosixPath
import hashlib
import datetime
import random
import string
import os
import json
import aiofiles

router = APIRouter()

# 配置（如需修改可调整）
BASE_DATA_DIR = Path("data_batch_storage")
ORIDATA_DIRNAME = "oridata"
MAX_VIDEOS_PER_CASE = 5
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}  # 可按需调整

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

@router.post("/api/users/{username}/upload-oridata")
async def upload_oridata(username: str, files: List[UploadFile] = File(...)):
    """
    接收多文件上传（支持来自 <input webkitdirectory> 的相对路径文件名，
    也支持直接上传若干视频文件）。
    约定：
      - 若上传文件名包含路径分隔符 ('/'), 则认为是文件夹上传，且相对路径应为:
            requestDir / caseName / videoFile
        后端将检查所有文件的第一层 requestDir 是否一致（一次上传只允许一个 request）。
      - 若文件名均不包含 '/', 则视为用户直接选择了多个视频文件 -> 这些文件被视为一个病例（case），数量上限 5。
    返回:
      - JSON { message, submissionId, path }
    校验：
      - 每个 case 的视频数量 <= MAX_VIDEOS_PER_CASE
      - 文件扩展名在 ALLOWED_EXTENSIONS 中
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="未提供任何文件")

    # 解析上传文件的路径信息（UploadFile.filename 可能包含用户提供的相对路径）
    # Normalize filename separators
    normalized_names = []
    for f in files:
        name = f.filename or ""
        # Browsers may include full path or relative path with backslashes; unify to POSIX style
        name = name.replace("\\", "/")
        normalized_names.append(name)

    contains_paths = any("/" in n for n in normalized_names)

    submission_id = generate_submission_id()
    submission_base = BASE_DATA_DIR / username / ORIDATA_DIRNAME / submission_id

    # create base dir
    try:
        submission_base.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Extremely unlikely because of timestamp+rand, but handle safely by regenerating
        submission_id = generate_submission_id()
        submission_base = BASE_DATA_DIR / username / ORIDATA_DIRNAME / submission_id
        submission_base.mkdir(parents=True, exist_ok=False)

    # if contains paths -> treat as folder upload
    case_map: Dict[str, List[Dict]] = {}  # caseName -> list of { upload:UploadFile, rel_path:str, orig_name:str }
    try:
        if contains_paths:
            # All files must have at least two separators (requestDir/caseName/file) -> parts >=3
            # Also enforce all files share same first component (requestDir)
            first_components = set()
            for idx, name in enumerate(normalized_names):
                parts = [p for p in PurePosixPath(name).parts if p not in ("", ".")]
                if len(parts) < 3:
                    raise HTTPException(status_code=400, detail=f"上传的路径不符合要求（应为 requestDir/caseName/videoFile）：{name}")
                first_components.add(parts[0])
            if len(first_components) != 1:
                raise HTTPException(status_code=400, detail="一次上传应来自同一请求目录（请确保上传的文件均来自同一一级目录）")

            # Build case map using second component as caseName
            for f, rawname in zip(files, normalized_names):
                parts = [p for p in PurePosixPath(rawname).parts if p not in ("", ".")]
                case_name = parts[1]
                # relative path under case dir: join remaining parts after first two
                rel_under_case = "/".join(parts[2:])
                if not rel_under_case:
                    raise HTTPException(status_code=400, detail=f"文件路径无效或缺少文件名：{rawname}")
                if not is_allowed_file(rel_under_case):
                    raise HTTPException(status_code=400, detail=f"不支持的文件格式：{rawname}")
                case_map.setdefault(case_name, []).append({
                    "upload": f,
                    "rel_path": rel_under_case,
                    "orig_name": Path(rel_under_case).name
                })

            # Validate per-case count
            for case_name, items in case_map.items():
                if len(items) > MAX_VIDEOS_PER_CASE:
                    # 清理已创建的目录
                    raise HTTPException(status_code=400, detail=f"病例 '{case_name}' 包含 {len(items)} 个视频，超过上限 {MAX_VIDEOS_PER_CASE}")

            # Save files preserving case_name folder structure
            for case_name, items in case_map.items():
                target_case_dir = safe_join(submission_base, case_name)
                target_case_dir.mkdir(parents=True, exist_ok=True)
                for it in items:
                    dest_path = safe_join(target_case_dir, it["orig_name"])
                    # write stream
                    upload_file = it["upload"]
                    try:
                        async with aiofiles.open(dest_path, 'wb') as out_file:
                            while True:
                                chunk = await upload_file.read(65536)
                                if not chunk:
                                    break
                                await out_file.write(chunk)
                    finally:
                        try:
                            await upload_file.close()
                        except Exception:
                            pass

        else:
            # No path info -> user uploaded standalone videos; treat them as one case
            if len(files) > MAX_VIDEOS_PER_CASE:
                raise HTTPException(status_code=400, detail=f"一次上传的独立视频文件数量为 {len(files)}，超过上限 {MAX_VIDEOS_PER_CASE}")

            # generate a random case name (16 chars alnum)
            case_name = "".join(random.choices(string.ascii_letters + string.digits, k=16))
            target_case_dir = safe_join(submission_base, case_name)
            target_case_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                if not is_allowed_file(f.filename):
                    raise HTTPException(status_code=400, detail=f"不支持的文件格式：{f.filename}")
                dest_path = safe_join(target_case_dir, Path(f.filename).name)
                try:
                    async with aiofiles.open(dest_path, 'wb') as out_file:
                        while True:
                            chunk = await f.read(65536)
                            if not chunk:
                                break
                            await out_file.write(chunk)
                finally:
                    try:
                        await f.close()
                    except Exception:
                        pass
            # reflect in case_map for metadata
            case_map[case_name] = [{"upload_name": Path(f.filename).name} for f in files]

        # 保存 metadata.json
        meta = {
            "submission_id": submission_id,
            "username": username,
            "created_at": datetime.datetime.now().isoformat(),
            "cases": {}
        }
        # build cases entries
        for case_name in case_map.keys():
            # list files physically in folder
            case_dir = submission_base / case_name
            files_in_case = [p.name for p in sorted(case_dir.iterdir()) if p.is_file()]
            meta["cases"][case_name] = files_in_case

        meta_path = submission_base / "metadata.json"
        async with aiofiles.open(meta_path, "w", encoding="utf-8") as mf:
            await mf.write(json.dumps(meta, ensure_ascii=False, indent=2))

        return {"message": "上传成功", "submissionId": submission_id, "path": str(submission_base)}

    except HTTPException:
        # re-raise HTTP error
        # clean up submission_base if empty
        raise
    except Exception as e:
        # cleanup on unexpected error
        try:
            if submission_base.exists():
                # remove created directory tree
                import shutil
                shutil.rmtree(submission_base)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"服务器错误: {e}")