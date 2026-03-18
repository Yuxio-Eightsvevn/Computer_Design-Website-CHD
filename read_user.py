import sqlite3
from pathlib import Path
from typing import List, Dict

# 数据库文件路径
DB_PATH = "users.db"

def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 使结果可以像字典一样访问
    return conn

def get_all_users() -> List[Dict]:
    """获取所有用户列表（包含密码）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 这里我修改了SQL，包含密码字段
    cursor.execute('''
        SELECT id, username, password, doctor, organization, created_at, updated_at
        FROM users
        ORDER BY id
    ''')
    
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return users

if __name__ == "__main__":
    print("=== 账号数据列表 ===")
    users = get_all_users()
    
    if users:
        for user in users:
            print(f"\nID: {user['id']}")
            print(f"用户名: {user['username']}")
            print(f"密码: {user['password']}")
            print(f"医生姓名: {user['doctor']}")
            print(f"所属机构: {user['organization']}")
            print(f"创建时间: {user['created_at']}")
            if 'updated_at' in user:
                print(f"更新时间: {user['updated_at']}")
    else:
        print("没有账号数据")
    
    print(f"\n总计: {len(users)} 个账号")
