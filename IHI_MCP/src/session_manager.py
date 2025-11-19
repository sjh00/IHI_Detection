"""
多用户会话管理器
用于管理多用户、多会话的全局变量和状态
"""

import json
import time
import uuid
import threading
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from .session_storage import get_session_storage_backend, SessionStorageBackend


class SessionManager:
    """
    多用户会话管理器
    为每个用户和会话提供独立的全局变量存储空间
    """
    
    def __init__(self, config: dict = None):
        """
        初始化会话管理器
        
        Args:
            config: 配置字典，包含会话相关配置
        """
        self.logger = logging.getLogger(f"IHI_detection.{__name__}")
        
        # 默认配置
        self.config = config or {}
        self.session_timeout = self.config.get('session_timeout', 3600)  # 默认1小时超时
        self.max_sessions = self.config.get('max_sessions', 1000)  # 最大会话数
        self.cleanup_interval = self.config.get('cleanup_interval', 300)  # 清理间隔(秒)
        
        # 会话存储
        self._sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> session_data
        self._user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
        self._lock = threading.RLock()  # 线程锁
        
        # 为每个会话创建独立的MemorySaver
        self._memory_savers: Dict[str, MemorySaver] = {}
        
        # 初始化存储后端
        backend_type = self.config.get('storage_backend', 'duckdb')
        storage_path = self.config.get('storage_path', None)
        
        if storage_path and backend_type.lower() == 'duckdb':
            self.storage_backend = get_session_storage_backend(backend_type, db_path=storage_path)
        else:
            self.storage_backend = get_session_storage_backend(backend_type)
        
        # 启动清理线程
        self._start_cleanup_thread()
        
        self.logger.info("会话管理器初始化完成")
    
    def create_session(self, user_id: str, initial_data: Dict[str, Any] = None) -> str:
        """
        为用户创建新会话
        
        Args:
            user_id: 用户ID
            initial_data: 初始化数据
            
        Returns:
            str: 会话ID
        """
        with self._lock:
            # 检查会话数限制
            if len(self._sessions) >= self.max_sessions:
                self._cleanup_expired_sessions()
                if len(self._sessions) >= self.max_sessions:
                    self.logger.warning("达到最大会话数限制，无法创建新会话")
                    raise Exception("达到最大会话数限制")
            
            # 生成会话ID
            session_id = str(uuid.uuid4())
            now = time.time()
            
            # 创建会话数据
            session_data = {
                'session_id': session_id,
                'user_id': user_id,
                'created_at': now,
                'last_accessed': now,
                'global_vars': initial_data or {},  # 该会话的全局变量
                'metadata': {}  # 其他元数据
            }
            
            # 存储会话
            self._sessions[session_id] = session_data
            
            # 更新用户会话列表
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session_id)
            
            # 为会话创建独立的MemorySaver
            self._memory_savers[session_id] = MemorySaver()
            
            # 保存会话到存储后端
            self._save_session_to_storage(session_id)
            
            self.logger.debug(f"为用户 {user_id} 创建会话 {session_id}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话数据
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 会话数据，如果不存在则返回None
        """
        with self._lock:
            # 先从内存中查找
            session = self._sessions.get(session_id)
            if session:
                # 更新最后访问时间
                session['last_accessed'] = time.time()
                self._save_session_to_storage(session_id)
                return session.copy()  # 返回副本避免外部修改
            
            # 如果内存中没有，尝试从存储后端加载
            session_data = self.storage_backend.load_session(session_id)
            if session_data:
                # 重新创建MemorySaver对象
                self._memory_savers[session_id] = MemorySaver()
                self._sessions[session_id] = session_data
                
                # 更新用户会话列表
                user_id = session_data['user_id']
                if user_id not in self._user_sessions:
                    self._user_sessions[user_id] = []
                if session_id not in self._user_sessions[user_id]:
                    self._user_sessions[user_id].append(session_id)
                
                return session_data.copy()
            
            return None
    
    def get_memory_saver(self, session_id: str) -> Optional[MemorySaver]:
        """
        获取会话的MemorySaver实例
        
        Args:
            session_id: 会话ID
            
        Returns:
            MemorySaver: MemorySaver实例，如果不存在则返回None
        """
        with self._lock:
            return self._memory_savers.get(session_id)
    
    def update_session_vars(self, session_id: str, vars_dict: Dict[str, Any]) -> bool:
        """
        更新会话的全局变量
        
        Args:
            session_id: 会话ID
            vars_dict: 要更新的变量字典
            
        Returns:
            bool: 更新是否成功
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                # 尝试从存储后端加载
                session_data = self.storage_backend.load_session(session_id)
                if not session_data:
                    return False
                # 重新创建MemorySaver对象
                self._memory_savers[session_id] = MemorySaver()
                self._sessions[session_id] = session_data
                session = session_data
            
            # 更新全局变量
            session['global_vars'].update(vars_dict)
            session['last_accessed'] = time.time()
            
            # 保存到存储后端
            self._save_session_to_storage(session_id)
            return True
    
    def get_session_vars(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话的全局变量
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 全局变量字典，如果会话不存在则返回空字典
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session['last_accessed'] = time.time()
                return session['global_vars'].copy()  # 返回副本避免外部修改
            return {}
    
    def set_session_var(self, session_id: str, key: str, value: Any) -> bool:
        """
        设置会话的单个全局变量
        
        Args:
            session_id: 会话ID
            key: 变量键
            value: 变量值
            
        Returns:
            bool: 设置是否成功
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            session['global_vars'][key] = value
            session['last_accessed'] = time.time()
            return True
    
    def get_session_var(self, session_id: str, key: str, default: Any = None) -> Any:
        """
        获取会话的单个全局变量
        
        Args:
            session_id: 会话ID
            key: 变量键
            default: 默认值
            
        Returns:
            Any: 变量值，如果不存在则返回默认值
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session['last_accessed'] = time.time()
                return session['global_vars'].get(key, default)
            return default
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """
        获取用户的所有会话ID
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[str]: 会话ID列表
        """
        with self._lock:
            memory_sessions = self._user_sessions.get(user_id, []).copy()
            # 合并存储后端中的会话
            storage_sessions = self.storage_backend.list_sessions(user_id)
            all_sessions = list(set(memory_sessions + storage_sessions))
            return all_sessions
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 删除是否成功
        """
        with self._lock:
            session = self._sessions.get(session_id)
            user_id = session['user_id'] if session else None
            
            # 从会话存储中删除
            if session_id in self._sessions:
                del self._sessions[session_id]
            
            # 从用户会话列表中删除
            if user_id and user_id in self._user_sessions:
                if session_id in self._user_sessions[user_id]:
                    self._user_sessions[user_id].remove(session_id)
                
                # 如果用户没有其他会话，删除用户记录
                if not self._user_sessions[user_id]:
                    del self._user_sessions[user_id]
            
            # 删除MemorySaver
            if session_id in self._memory_savers:
                del self._memory_savers[session_id]
            
            # 从存储后端删除
            storage_result = self.storage_backend.delete_session(session_id)
            
            self.logger.debug(f"删除会话 {session_id}")
            return storage_result
    
    def cleanup_expired_sessions(self) -> int:
        """
        清理过期会话
        
        Returns:
            int: 清理的会话数量
        """
        return self._cleanup_expired_sessions()
    
    def _cleanup_expired_sessions(self) -> int:
        """
        内部方法：清理过期会话
        
        Returns:
            int: 清理的会话数量
        """
        with self._lock:
            now = time.time()
            expired_sessions = []
            
            # 查找过期会话
            for session_id, session in self._sessions.items():
                if now - session['last_accessed'] > self.session_timeout:
                    expired_sessions.append(session_id)
            
            # 删除过期会话
            for session_id in expired_sessions:
                self.delete_session(session_id)
            
            # 清理存储后端中的过期会话
            storage_expired_count = self.storage_backend.cleanup_expired_sessions(self.session_timeout)
            
            total_expired = len(expired_sessions) + storage_expired_count
            if total_expired > 0:
                self.logger.debug(f"清理了 {total_expired} 个过期会话")
            
            return total_expired
    
    def _save_session_to_storage(self, session_id: str):
        """
        保存会话到存储后端
        
        Args:
            session_id: 会话ID
        """
        if session_id in self._sessions:
            # 创建可序列化的会话数据副本
            session_data = self._sessions[session_id].copy()
            # MemorySaver对象不可序列化，所以不保存
            session_data.pop('checkpointer', None)
            
            self.storage_backend.save_session(session_id, session_data)
    
    def _start_cleanup_thread(self):
        """启动清理线程"""
        def cleanup_task():
            while True:
                time.sleep(300)  # 每5分钟检查一次
                try:
                    self.cleanup_expired_sessions()
                except Exception as e:
                    self.logger.error(f"清理过期会话时出错: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
        self.logger.debug("会话清理线程已启动")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        获取会话统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self._lock:
            now = time.time()
            active_sessions = 0
            expired_sessions = 0
            
            for session in self._sessions.values():
                if now - session['last_accessed'] <= self.session_timeout:
                    active_sessions += 1
                else:
                    expired_sessions += 1
            
            return {
                'total_sessions': len(self._sessions),
                'total_users': len(self._user_sessions),
                'active_sessions': active_sessions,
                'expired_sessions': expired_sessions,
                'max_sessions': self.max_sessions,
                'session_timeout': self.session_timeout
            }
    
    def export_session_data(self, session_id: str) -> Optional[str]:
        """
        导出会话数据为JSON字符串
        
        Args:
            session_id: 会话ID
            
        Returns:
            str: JSON格式的会话数据，如果会话不存在则返回None
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            
            # 创建导出数据（不包含敏感信息）
            export_data = {
                'session_id': session['session_id'],
                'user_id': session['user_id'],
                'created_at': session['created_at'],
                'last_accessed': session['last_accessed'],
                'global_vars': session['global_vars'],
                'metadata': session['metadata']
            }
            
            return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def import_session_data(self, json_data: str, new_session_id: str = None) -> Optional[str]:
        """
        从JSON字符串导入会话数据
        
        Args:
            json_data: JSON格式的会话数据
            new_session_id: 新会话ID，如果为None则生成新ID
            
        Returns:
            str: 会话ID，如果导入失败则返回None
        """
        try:
            with self._lock:
                data = json.loads(json_data)
                user_id = data.get('user_id')
                
                if not user_id:
                    self.logger.error("导入会话数据失败：缺少用户ID")
                    return None
                
                # 生成或使用提供的会话ID
                session_id = new_session_id or str(uuid.uuid4())
                now = time.time()
                
                # 创建会话数据
                session_data = {
                    'session_id': session_id,
                    'user_id': user_id,
                    'created_at': now,
                    'last_accessed': now,
                    'global_vars': data.get('global_vars', {}),
                    'metadata': data.get('metadata', {})
                }
                
                # 存储会话
                self._sessions[session_id] = session_data
                
                # 更新用户会话列表
                if user_id not in self._user_sessions:
                    self._user_sessions[user_id] = []
                self._user_sessions[user_id].append(session_id)
                
                # 为会话创建独立的MemorySaver
                self._memory_savers[session_id] = MemorySaver()
                
                self.logger.debug(f"导入会话数据成功，会话ID: {session_id}")
                return session_id
                
        except Exception as e:
            self.logger.error(f"导入会话数据失败: {e}")
            return None


# 全局会话管理器实例
_global_session_manager = None


def get_session_manager(config: dict = None) -> SessionManager:
    """
    获取全局会话管理器实例
    
    Args:
        config: 配置字典
        
    Returns:
        SessionManager: 会话管理器实例
    """
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager(config)
    return _global_session_manager


def reset_session_manager():
    """重置全局会话管理器（主要用于测试）"""
    global _global_session_manager
    _global_session_manager = None