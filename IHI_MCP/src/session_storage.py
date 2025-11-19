"""
会话存储后端
提供会话数据的持久化存储功能
"""

import json
import os
import duckdb
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging


class SessionStorageBackend:
    """
    会话存储后端基类
    定义会话存储的通用接口
    """
    
    def save_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        保存会话数据
        
        Args:
            session_id: 会话ID
            session_data: 会话数据
            
        Returns:
            bool: 保存是否成功
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        加载会话数据
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 会话数据，如果不存在则返回None
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话数据
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 删除是否成功
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """
        列出会话ID
        
        Args:
            user_id: 用户ID，如果为None则列出所有会话
            
        Returns:
            List[str]: 会话ID列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def cleanup_expired_sessions(self, expire_time: int) -> int:
        """
        清理过期会话
        
        Args:
            expire_time: 过期时间（秒）
            
        Returns:
            int: 清理的会话数量
        """
        raise NotImplementedError("子类必须实现此方法")


class FileSessionStorage(SessionStorageBackend):
    """
    基于文件系统的会话存储后端
    """
    
    def __init__(self, storage_dir: str = "data/sessions"):
        """
        初始化文件存储后端
        
        Args:
            storage_dir: 存储目录
        """
        self.logger = logging.getLogger(f"IHI_detection.{__name__}")
        self.storage_dir = storage_dir
        self._lock = threading.RLock()
        
        # 确保存储目录存在
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.logger.info(f"文件会话存储初始化完成，存储目录: {self.storage_dir}")
    
    def _get_session_file_path(self, session_id: str) -> str:
        """获取会话文件路径"""
        return os.path.join(self.storage_dir, f"{session_id}.json")
    
    def save_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """保存会话数据到文件"""
        try:
            with self._lock:
                file_path = self._get_session_file_path(session_id)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, ensure_ascii=False, indent=2)
                return True
        except Exception as e:
            self.logger.error(f"保存会话 {session_id} 失败: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """从文件加载会话数据"""
        try:
            with self._lock:
                file_path = self._get_session_file_path(session_id)
                if not os.path.exists(file_path):
                    return None
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"加载会话 {session_id} 失败: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话文件"""
        try:
            with self._lock:
                file_path = self._get_session_file_path(session_id)
                if os.path.exists(file_path):
                    os.remove(file_path)
                return True
        except Exception as e:
            self.logger.error(f"删除会话 {session_id} 失败: {e}")
            return False
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """列出会话ID"""
        try:
            with self._lock:
                sessions = []
                for filename in os.listdir(self.storage_dir):
                    if filename.endswith('.json'):
                        session_id = filename[:-5]  # 去掉.json后缀
                        
                        # 如果指定了用户ID，检查会话是否属于该用户
                        if user_id is not None:
                            session_data = self.load_session(session_id)
                            if session_data and session_data.get('user_id') == user_id:
                                sessions.append(session_id)
                        else:
                            sessions.append(session_id)
                
                return sessions
        except Exception as e:
            self.logger.error(f"列出会话失败: {e}")
            return []
    
    def cleanup_expired_sessions(self, expire_time: int) -> int:
        """清理过期会话"""
        try:
            with self._lock:
                now = datetime.now().timestamp()
                expired_count = 0
                
                for session_id in self.list_sessions():
                    session_data = self.load_session(session_id)
                    if session_data:
                        last_accessed = session_data.get('last_accessed', 0)
                        if now - last_accessed > expire_time:
                            if self.delete_session(session_id):
                                expired_count += 1
                
                self.logger.info(f"清理了 {expired_count} 个过期会话")
                return expired_count
        except Exception as e:
            self.logger.error(f"清理过期会话失败: {e}")
            return 0


# class SQLiteSessionStorage(SessionStorageBackend):
#     """
#     基于SQLite的会话存储后端
#     """
    
#     def __init__(self, db_path: str = "data/database/sessions.db"):
#         """
#         初始化SQLite存储后端
        
#         Args:
#             db_path: 数据库文件路径
#         """
#         self.logger = logging.getLogger(f"IHI_detection.{__name__}")
#         self.db_path = db_path
#         self._lock = threading.RLock()
        
#         # 确保数据库目录存在
#         os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
#         # 初始化数据库
#         self._init_db()
        
#         self.logger.info(f"SQLite会话存储初始化完成，数据库路径: {self.db_path}")
    
#     def _init_db(self):
#         """初始化数据库表"""
#         with self._lock:
#             conn = sqlite3.connect(self.db_path)
#             try:
#                 cursor = conn.cursor()
#                 cursor.execute('''
#                     CREATE TABLE IF NOT EXISTS sessions (
#                         session_id TEXT PRIMARY KEY,
#                         user_id TEXT NOT NULL,
#                         created_at REAL NOT NULL,
#                         last_accessed REAL NOT NULL,
#                         session_data TEXT NOT NULL
#                     )
#                 ''')
                
#                 # 创建索引以提高查询性能
#                 cursor.execute('''
#                     CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)
#                 ''')
                
#                 cursor.execute('''
#                     CREATE INDEX IF NOT EXISTS idx_sessions_last_accessed ON sessions(last_accessed)
#                 ''')
                
#                 conn.commit()
#             finally:
#                 conn.close()
    
#     def save_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
#         """保存会话数据到数据库"""
#         try:
#             with self._lock:
#                 conn = sqlite3.connect(self.db_path)
#                 try:
#                     cursor = conn.cursor()
#                     cursor.execute('''
#                         INSERT OR REPLACE INTO sessions 
#                         (session_id, user_id, created_at, last_accessed, session_data)
#                         VALUES (?, ?, ?, ?, ?)
#                     ''', (
#                         session_id,
#                         session_data.get('user_id', ''),
#                         session_data.get('created_at', 0),
#                         session_data.get('last_accessed', 0),
#                         json.dumps(session_data, ensure_ascii=False)
#                     ))
#                     conn.commit()
#                     return True
#                 finally:
#                     conn.close()
#         except Exception as e:
#             self.logger.error(f"保存会话 {session_id} 失败: {e}")
#             return False
    
#     def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
#         """从数据库加载会话数据"""
#         try:
#             with self._lock:
#                 conn = sqlite3.connect(self.db_path)
#                 try:
#                     cursor = conn.cursor()
#                     cursor.execute('''
#                         SELECT session_data FROM sessions WHERE session_id = ?
#                     ''', (session_id,))
                    
#                     row = cursor.fetchone()
#                     if row:
#                         return json.loads(row[0])
#                     return None
#                 finally:
#                     conn.close()
#         except Exception as e:
#             self.logger.error(f"加载会话 {session_id} 失败: {e}")
#             return None
    
#     def delete_session(self, session_id: str) -> bool:
#         """从数据库删除会话"""
#         try:
#             with self._lock:
#                 conn = sqlite3.connect(self.db_path)
#                 try:
#                     cursor = conn.cursor()
#                     cursor.execute('''
#                         DELETE FROM sessions WHERE session_id = ?
#                     ''', (session_id,))
#                     conn.commit()
#                     return True
#                 finally:
#                     conn.close()
#         except Exception as e:
#             self.logger.error(f"删除会话 {session_id} 失败: {e}")
#             return False
    
#     def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
#         """列出会话ID"""
#         try:
#             with self._lock:
#                 conn = sqlite3.connect(self.db_path)
#                 try:
#                     cursor = conn.cursor()
                    
#                     if user_id is not None:
#                         cursor.execute('''
#                             SELECT session_id FROM sessions WHERE user_id = ?
#                         ''', (user_id,))
#                     else:
#                         cursor.execute('''
#                             SELECT session_id FROM sessions
#                         ''')
                    
#                     rows = cursor.fetchall()
#                     return [row[0] for row in rows]
#                 finally:
#                     conn.close()
#         except Exception as e:
#             self.logger.error(f"列出会话失败: {e}")
#             return []
    
#     def cleanup_expired_sessions(self, expire_time: int) -> int:
#         """清理过期会话"""
#         try:
#             with self._lock:
#                 conn = sqlite3.connect(self.db_path)
#                 try:
#                     cursor = conn.cursor()
#                     now = datetime.now().timestamp()
                    
#                     cursor.execute('''
#                         DELETE FROM sessions WHERE ? - last_accessed > ?
#                     ''', (now, expire_time))
                    
#                     deleted_count = cursor.rowcount
#                     conn.commit()
                    
#                     self.logger.info(f"清理了 {deleted_count} 个过期会话")
#                     return deleted_count
#                 finally:
#                     conn.close()
#         except Exception as e:
#             self.logger.error(f"清理过期会话失败: {e}")
#             return 0


class DuckDBSessionStorage(SessionStorageBackend):
    """
    基于DuckDB的会话存储后端
    """
    
    def __init__(self, db_path: str = "data/database/sessions.db"):
        """
        初始化DuckDB存储后端
        
        Args:
            db_path: 数据库文件路径
        """
        self.logger = logging.getLogger(f"IHI_detection.{__name__}")
        self.db_path = db_path
        self._lock = threading.RLock()
        
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_db()
        
        self.logger.info(f"DuckDB会话存储初始化完成，数据库路径: {self.db_path}")
    
    def _init_db(self):
        """初始化数据库表"""
        with self._lock:
            conn = duckdb.connect(self.db_path)
            try:
                # 创建会话表
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id VARCHAR PRIMARY KEY,
                        user_id VARCHAR NOT NULL,
                        created_at DOUBLE NOT NULL,
                        last_accessed DOUBLE NOT NULL,
                        session_data VARCHAR NOT NULL
                    )
                ''')
                
                # 创建索引以提高查询性能
                try:
                    conn.execute('''
                        CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)
                    ''')
                except Exception as e:
                    # DuckDB可能不支持某些索引创建语法，记录但不中断
                    self.logger.warning(f"创建用户ID索引失败: {e}")
                
                try:
                    conn.execute('''
                        CREATE INDEX IF NOT EXISTS idx_sessions_last_accessed ON sessions(last_accessed)
                    ''')
                except Exception as e:
                    # DuckDB可能不支持某些索引创建语法，记录但不中断
                    self.logger.warning(f"创建最后访问时间索引失败: {e}")
                
                conn.commit()
            finally:
                conn.close()
    
    def save_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """保存会话数据到数据库"""
        try:
            with self._lock:
                conn = duckdb.connect(self.db_path)
                try:
                    conn.execute('''
                        INSERT OR REPLACE INTO sessions 
                        (session_id, user_id, created_at, last_accessed, session_data)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        session_id,
                        session_data.get('user_id', ''),
                        session_data.get('created_at', 0),
                        session_data.get('last_accessed', 0),
                        json.dumps(session_data, ensure_ascii=False)
                    ))
                    conn.commit()
                    return True
                finally:
                    conn.close()
        except Exception as e:
            self.logger.error(f"保存会话 {session_id} 失败: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """从数据库加载会话数据"""
        try:
            with self._lock:
                conn = duckdb.connect(self.db_path)
                try:
                    result = conn.execute('''
                        SELECT session_data FROM sessions WHERE session_id = ?
                    ''', (session_id,)).fetchone()
                    
                    if result:
                        return json.loads(result[0])
                    return None
                finally:
                    conn.close()
        except Exception as e:
            self.logger.error(f"加载会话 {session_id} 失败: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """从数据库删除会话"""
        try:
            with self._lock:
                conn = duckdb.connect(self.db_path)
                try:
                    conn.execute('''
                        DELETE FROM sessions WHERE session_id = ?
                    ''', (session_id,))
                    conn.commit()
                    return True
                finally:
                    conn.close()
        except Exception as e:
            self.logger.error(f"删除会话 {session_id} 失败: {e}")
            return False
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """列出会话ID"""
        try:
            with self._lock:
                conn = duckdb.connect(self.db_path)
                try:
                    if user_id is not None:
                        result = conn.execute('''
                            SELECT session_id FROM sessions WHERE user_id = ?
                        ''', (user_id,)).fetchall()
                    else:
                        result = conn.execute('''
                            SELECT session_id FROM sessions
                        ''').fetchall()
                    
                    return [row[0] for row in result]
                finally:
                    conn.close()
        except Exception as e:
            self.logger.error(f"列出会话失败: {e}")
            return []
    
    def cleanup_expired_sessions(self, expire_time: int) -> int:
        """清理过期会话"""
        try:
            with self._lock:
                conn = duckdb.connect(self.db_path)
                if conn is None:
                    raise ConnectionError("无法连接到DuckDB数据库")
                try:
                    now = datetime.now().timestamp()
                    
                    # DuckDB使用不同的语法进行时间差计算
                    result = conn.execute('''
                        DELETE FROM sessions WHERE (? - last_accessed) > ?
                    ''', (now, expire_time))
                    
                    deleted_count = result.rowcount if hasattr(result, 'rowcount') else conn.execute('SELECT changes()').fetchone()[0]
                    conn.commit()
                    
                    self.logger.info(f"清理了 {deleted_count} 个过期会话")
                    return deleted_count
                finally:
                    conn.close()
        except Exception as e:
            self.logger.error(f"清理过期会话失败: {e}")
            return 0


def get_session_storage_backend(backend_type: str = "sqlite", **kwargs) -> SessionStorageBackend:
    """
    获取会话存储后端实例
    
    Args:
        backend_type: 后端类型，支持 "file" 或 "duckdb"
        **kwargs: 后端特定的参数
        
    Returns:
        SessionStorageBackend: 会话存储后端实例
    """
    if backend_type.lower() == "file":
        storage_dir = kwargs.get('storage_dir', 'data/sessions')
        return FileSessionStorage(storage_dir)
    elif backend_type.lower() == "duckdb":
        db_path = kwargs.get('db_path', 'data/database/sessions.db')
        return DuckDBSessionStorage(db_path)
    else:
        raise ValueError(f"不支持的存储后端类型: {backend_type}")
