import os
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "users.db")

def connect_db(path: str = DB_PATH) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise FileNotFoundError(f"database not found: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def get_table_names(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [r["name"] for r in cur.fetchall()]

def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r["name"] for r in cur.fetchall()]

def normalize_user_row(row: sqlite3.Row, cols: List[str]) -> Dict[str, Any]:
    mapping = {
        "username": ["username", "user", "login", "account"],
        "name": ["name", "full_name", "fullname", "display_name", "realname"],
        "organization": ["organization", "org", "company", "institution"],
        "email": ["email", "mail"],
        "id": ["id", "user_id"],
    }
    result: Dict[str, Any] = {"raw": dict(row)}
    for key, candidates in mapping.items():
        value = None
        for c in candidates:
            if c in cols:
                value = row[c]
                if value is not None:
                    break
        result[key] = value
    if not result.get("username"):
        result["username"] = result.get("name") or result.get("email") or result.get("id")
    return result

def find_users(conn: sqlite3.Connection, limit: int = 100) -> List[Dict[str, Any]]:
    tables = get_table_names(conn)
    users_list: List[Dict[str, Any]] = []
    user_table = None
    for t in tables:
        if t.lower() == "users":
            user_table = t
            break
    if not user_table and tables:
        user_table = tables[0]
    if not user_table:
        return users_list
    cols = get_columns(conn, user_table)
    q = f"SELECT * FROM {user_table} LIMIT ?"
    cur = conn.execute(q, (limit,))
    for row in cur.fetchall():
        users_list.append(normalize_user_row(row, cols))
    return users_list

def find_user_by_username(conn: sqlite3.Connection, username: str) -> Optional[Dict[str, Any]]:
    tables = get_table_names(conn)
    ordered = sorted(tables, key=lambda x: 0 if x.lower() == "users" else 1)
    for t in ordered:
        cols = get_columns(conn, t)
        candidates = [c for c in cols if c.lower() in ("username", "user", "login", "account", "name", "email")]
        if not candidates:
            continue
        where_clauses = " OR ".join([f"{c} = ?" for c in candidates])
        q = f"SELECT * FROM {t} WHERE {where_clauses} LIMIT 1"
        params = tuple([username] * len(candidates))
        cur = conn.execute(q, params)
        row = cur.fetchone()
        if row:
            return normalize_user_row(row, cols)
    return None

@router.get("/users", response_class=JSONResponse)
def api_list_users(limit: int = 100):
    try:
        conn = connect_db()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        users = find_users(conn, limit=limit)
        return JSONResponse({"data": users})
    finally:
        conn.close()

# <-- 将 current 路由放在动态路由前，避免被误匹配 -->
@router.get("/users/current", response_class=JSONResponse)
def api_get_current_user(request: Request):
    try:
        conn = connect_db()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        cookie_un = request.cookies.get("username") or request.cookies.get("user")
        if cookie_un:
            u = find_user_by_username(conn, cookie_un)
            if u:
                return JSONResponse({"data": u})

        header_un = request.headers.get("x-username") or request.headers.get("X-User")
        if header_un:
            u = find_user_by_username(conn, header_un)
            if u:
                return JSONResponse({"data": u})

        users = find_users(conn, limit=1)
        if users:
            return JSONResponse({"data": users[0]})
        raise HTTPException(status_code=404, detail="no users in database")
    finally:
        conn.close()

@router.get("/users/{username}", response_class=JSONResponse)
def api_get_user(username: str):
    try:
        conn = connect_db()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        u = find_user_by_username(conn, username)
        if not u:
            raise HTTPException(status_code=404, detail="user not found")
        return JSONResponse({"data": u})
    finally:
        conn.close()