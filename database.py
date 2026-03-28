import sqlite3
from pathlib import Path
from typing import Optional, List, Dict
import hashlib

# 数据库文件路径
DB_PATH = "users.db"

def get_db_connection():
    """获取数据库连接（每次调用都会创建新的连接）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 使结果可以像字典一样访问
    return conn

def hash_password(password: str) -> str:
    """使用 SHA-256 对密码进行哈希（返回十六进制字符串）"""
    if password is None:
        return ""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(stored_password: str, provided_password: str) -> bool:
    """验证密码，兼容明文与哈希"""
    if stored_password is None:
        return False
    # 简单判断是否为 SHA-256 的十六进制表示
    is_hashed = isinstance(stored_password, str) and len(stored_password) == 64 and all(c in "0123456789abcdef" for c in stored_password.lower())
    if is_hashed:
        return stored_password == hash_password(provided_password)
    else:
        return stored_password == provided_password

def init_database():
    """初始化数据库和表结构（包含自动迁移逻辑）"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # [改动] 建表时增加 is_admin 字段，默认值为 0 (False)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
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
    conn.commit()

    # [新增] 数据库迁移逻辑：检查是否已有 is_admin 字段，如果没有则添加
    cursor.execute("PRAGMA table_info(users)")
    columns = [info['name'] for info in cursor.fetchall()]
    if 'is_admin' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0")
        # 将已有的 admin 账户强行提升为管理员
        cursor.execute("UPDATE users SET is_admin = 1 WHERE username = 'admin'")
        conn.commit()
        print("🔄 数据库已迁移: 成功添加 is_admin 字段并更新管理员权限")

    # 检查是否有初始数据
    cursor.execute("SELECT COUNT(*) as count FROM users")
    count = cursor.fetchone()['count']

    # 如果没有数据，插入初始账号
    if count == 0:
        # [改动] 初始化时，明确指定 is_admin 的值 (1 为是，0 为否)
        initial_users = [
            ('admin', hash_password('123456'), '管理员', '深圳大学', 1),
            ('doctor1', hash_password('123456'), '张医生', '深圳大学附属医院', 0)
        ]

        cursor.executemany('''
            INSERT INTO users (username, password, doctor, organization, is_admin)
            VALUES (?, ?, ?, ?, ?)
        ''', initial_users)

        conn.commit()
        print(f"✅ 初始化数据库，创建了 {len(initial_users)} 个默认账户")

    conn.close()

def verify_user(username: str, password: str) -> Optional[Dict]:
    """验证用户登录并返回用户信息"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # [改动] SELECT 语句中加入 is_admin
    cursor.execute('''
        SELECT id, username, password, doctor, organization, is_admin 
        FROM users 
        WHERE username = ?
    ''', (username,))

    row = cursor.fetchone()

    if not row:
        conn.close()
        return None

    stored_password = row['password']

    if verify_password(stored_password, password):
        user = dict(row)
        is_hashed = isinstance(stored_password, str) and len(stored_password) == 64 and all(c in "0123456789abcdef" for c in stored_password.lower())
        if not is_hashed:
            try:
                new_hashed = hash_password(password)
                cursor.execute('UPDATE users SET password = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?', (new_hashed, row['id']))
                conn.commit()
                print(f"🔐 已将用户 {username} 的密码迁移为哈希存储")
            except Exception as e:
                print(f"⚠️ 更新哈希密码失败: {e}")
        user.pop('password', None)
        # 确保布尔值格式规范
        user['is_admin'] = bool(user['is_admin'])
        conn.close()
        return user

    conn.close()
    return None

def get_all_users() -> List[Dict]:
    """获取所有用户列表（不包含密码）"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # [改动] SELECT 语句中加入 is_admin
    cursor.execute('''
        SELECT id, username, doctor, organization, is_admin, created_at, updated_at
        FROM users
        ORDER BY id
    ''')

    users = []
    for row in cursor.fetchall():
        user_dict = dict(row)
        user_dict['is_admin'] = bool(user_dict['is_admin'])
        users.append(user_dict)
        
    conn.close()
    return users

def get_user_by_id(user_id: int) -> Optional[Dict]:
    """根据ID获取用户信息"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # [改动] SELECT 语句中加入 is_admin
    cursor.execute('''
        SELECT id, username, doctor, organization, is_admin, created_at, updated_at
        FROM users
        WHERE id = ?
    ''', (user_id,))

    user = cursor.fetchone()
    conn.close()

    if user:
        user_dict = dict(user)
        user_dict['is_admin'] = bool(user_dict['is_admin'])
        return user_dict
    return None

def get_user_by_username(username: str) -> Optional[Dict]:
    """根据用户名获取用户信息（不包含密码）"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # [改动] SELECT 语句中加入 is_admin
    cursor.execute('''
        SELECT id, username, doctor, organization, is_admin, created_at
        FROM users
        WHERE username = ?
    ''', (username,))

    user = cursor.fetchone()
    conn.close()

    if user:
        user_dict = dict(user)
        user_dict['is_admin'] = bool(user_dict['is_admin'])
        return user_dict
    return None

# [改动] 增加 is_admin 参数，默认为 False (即 0)
def create_user(username: str, password: str, doctor: str, organization: str, is_admin: bool = False) -> bool:
    """创建新用户"""
    # [核心加固] 禁止注册系统保留名
    if username.upper() == "SYSTEM":
        print(f"🚫 拦截：禁止创建保留用户名 {username}")
        return False
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        hashed = hash_password(password)
        admin_value = 1 if is_admin else 0
        
        # [改动] 插入语句加入 is_admin
        cursor.execute('''
            INSERT INTO users (username, password, doctor, organization, is_admin)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, hashed, doctor, organization, admin_value))

        conn.commit()

        doctor_folder = Path("data_batch_storage") / username
        doctor_folder.mkdir(parents=True, exist_ok=True)

        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        print(f"创建用户失败: {e}")
        return False

def update_user(user_id: int, doctor: str, organization: str, password: Optional[str] = None) -> bool:
    """更新用户信息"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if password:
            hashed = hash_password(password)
            cursor.execute('''
                UPDATE users 
                SET doctor = ?, organization = ?, password = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (doctor, organization, hashed, user_id))
        else:
            cursor.execute('''
                UPDATE users 
                SET doctor = ?, organization = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (doctor, organization, user_id))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"更新用户失败: {e}")
        return False

def delete_user(user_id: int) -> bool:
    """删除用户"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT username FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()

        if not user:
            conn.close()
            return False

        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()

        return True
    except Exception as e:
        print(f"删除用户失败: {e}")
        return False

def check_username_exists(username: str, exclude_id: Optional[int] = None) -> bool:
    """检查用户名是否已存在"""
    if username.upper() == "SYSTEM":
        return True
    conn = get_db_connection()
    cursor = conn.cursor()

    if exclude_id:
        cursor.execute('SELECT COUNT(*) as count FROM users WHERE username = ? AND id != ?', (username, exclude_id))
    else:
        cursor.execute('SELECT COUNT(*) as count FROM users WHERE username = ?', (username,))

    count = cursor.fetchone()['count']
    conn.close()

    return count > 0