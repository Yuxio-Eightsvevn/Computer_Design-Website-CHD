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
    """
    验证密码。
    支持两种情况：
      - 如果 stored_password 看起来像 SHA-256（长度为64且为十六进制），则对provided_password哈希后比较。
      - 否则，把 stored_password 当作明文，直接比较（用于向后兼容旧数据）。
    如果检测到旧明文并匹配，将触发调用方更新数据库以存储哈希后的密码。
    """
    if stored_password is None:
        return False

    # 简单判断是否为 SHA-256 的十六进制表示
    is_hashed = isinstance(stored_password, str) and len(stored_password) == 64 and all(c in "0123456789abcdef" for c in stored_password.lower())

    if is_hashed:
        return stored_password == hash_password(provided_password)
    else:
        # 明文兼容分支
        return stored_password == provided_password

def init_database():
    """初始化数据库和表结构（并插入初始账户，使用哈希存储密码）"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            doctor TEXT NOT NULL,
            organization TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()

    # 检查是否有初始数据
    cursor.execute("SELECT COUNT(*) as count FROM users")
    count = cursor.fetchone()['count']

    # 如果没有数据，插入初始医生账号（密码以哈希存储）
    if count == 0:
        initial_users = [
            ('admin', hash_password('123456'), '管理员', '深圳大学'),
            ('doctor1', hash_password('123456'), '张医生', '深圳大学附属医院')
        ]

        cursor.executemany('''
            INSERT INTO users (username, password, doctor, organization)
            VALUES (?, ?, ?, ?)
        ''', initial_users)

        conn.commit()
        print(f"✅ 初始化数据库，创建了 {len(initial_users)} 个默认账户")

    conn.close()

def verify_user(username: str, password: str) -> Optional[Dict]:
    """验证用户登录。若数据库中存储的是明文密码且匹配，将把密码更新为哈希值（迁移）"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, username, password, doctor, organization 
        FROM users 
        WHERE username = ?
    ''', (username,))

    row = cursor.fetchone()

    if not row:
        conn.close()
        return None

    stored_password = row['password']

    # 验证密码，支持明文兼容
    if verify_password(stored_password, password):
        user = dict(row)
        # 如果原来是明文（即不是哈希格式），则更新为哈希值（迁移）
        is_hashed = isinstance(stored_password, str) and len(stored_password) == 64 and all(c in "0123456789abcdef" for c in stored_password.lower())
        if not is_hashed:
            try:
                new_hashed = hash_password(password)
                cursor.execute('UPDATE users SET password = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?', (new_hashed, row['id']))
                conn.commit()
                print(f"🔐 已将用户 {username} 的密码迁移为哈希存储")
            except Exception as e:
                print(f"⚠️ 更新哈希密码失败: {e}")
        # 不返回密码字段
        user.pop('password', None)
        conn.close()
        return user

    conn.close()
    return None

def get_all_users() -> List[Dict]:
    """获取所有用户列表（不包含密码）"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, username, doctor, organization, created_at, updated_at
        FROM users
        ORDER BY id
    ''')

    users = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return users

def get_user_by_id(user_id: int) -> Optional[Dict]:
    """根据ID获取用户信息"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, username, doctor, organization, created_at, updated_at
        FROM users
        WHERE id = ?
    ''', (user_id,))

    user = cursor.fetchone()
    conn.close()

    if user:
        return dict(user)
    return None

def create_user(username: str, password: str, doctor: str, organization: str) -> bool:
    """创建新用户（密码以哈希形式存储）"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        hashed = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, password, doctor, organization)
            VALUES (?, ?, ?, ?)
        ''', (username, hashed, doctor, organization))

        conn.commit()

        # 为新用户创建文件夹
        doctor_folder = Path("data_batch_storage") / username
        doctor_folder.mkdir(parents=True, exist_ok=True)

        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False  # 用户名已存在
    except Exception as e:
        print(f"创建用户失败: {e}")
        return False

def update_user(user_id: int, doctor: str, organization: str, password: Optional[str] = None) -> bool:
    """更新用户信息；如果提供了 password，则用哈希存储"""
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

        # 先获取用户名，用于删除文件夹
        cursor.execute('SELECT username FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()

        if not user:
            conn.close()
            return False

        username = user['username']

        # 删除数据库记录
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()

        # 删除用户文件夹（可选）
        # import shutil
        # doctor_folder = Path("data_batch_storage") / username
        # if doctor_folder.exists():
        #     shutil.rmtree(doctor_folder)

        return True
    except Exception as e:
        print(f"删除用户失败: {e}")
        return False

def check_username_exists(username: str, exclude_id: Optional[int] = None) -> bool:
    """检查用户名是否已存在"""
    conn = get_db_connection()
    cursor = conn.cursor()

    if exclude_id:
        cursor.execute('SELECT COUNT(*) as count FROM users WHERE username = ? AND id != ?', (username, exclude_id))
    else:
        cursor.execute('SELECT COUNT(*) as count FROM users WHERE username = ?', (username,))

    count = cursor.fetchone()['count']
    conn.close()

    return count > 0

def get_user_by_username(username: str) -> Optional[Dict]:
    """根据用户名获取用户信息（不包含密码）"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, username, doctor, organization, created_at
        FROM users
        WHERE username = ?
    ''', (username,))

    user = cursor.fetchone()
    conn.close()

    if user:
        return dict(user)
    return None