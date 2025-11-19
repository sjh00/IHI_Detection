"""
会话清理服务
提供定期清理过期会话的功能
"""

import time
import threading
import logging
import schedule
from typing import Optional, Callable
from datetime import datetime, timedelta

from .session_manager import get_session_manager
from .session_storage import get_session_storage_backend


class SessionCleanupService:
    """
    会话清理服务类
    负责定期清理过期会话
    """
    
    def __init__(self, cleanup_interval: int = 3600, session_timeout: int = 86400):
        """
        初始化会话清理服务
        
        Args:
            cleanup_interval: 清理间隔（秒），默认为1小时
            session_timeout: 会话超时时间（秒），默认为24小时
        """
        self.logger = logging.getLogger(f"IHI_detection.{__name__}")
        self.cleanup_interval = cleanup_interval
        self.session_timeout = session_timeout
        self.running = False
        self.cleanup_thread = None
        self.session_manager = get_session_manager()
        
        self.logger.info(f"会话清理服务初始化完成，清理间隔: {cleanup_interval}秒，会话超时: {session_timeout}秒")
    
    def start(self):
        """启动清理服务"""
        if self.running:
            self.logger.warning("会话清理服务已在运行")
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        self.logger.info("会话清理服务已启动")
    
    def stop(self):
        """停止清理服务"""
        if not self.running:
            self.logger.warning("会话清理服务未在运行")
            return
        
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        self.logger.info("会话清理服务已停止")
    
    def cleanup_once(self) -> int:
        """
        执行一次清理操作
        
        Returns:
            int: 清理的会话数量
        """
        try:
            # 使用会话管理器的清理方法
            cleaned_count = self.session_manager.cleanup_expired_sessions()
            
            if cleaned_count > 0:
                self.logger.info(f"清理了 {cleaned_count} 个过期会话")
            else:
                self.logger.debug("没有需要清理的过期会话")
            
            return cleaned_count
        except Exception as e:
            self.logger.error(f"清理过期会话时出错: {e}")
            return 0
    
    def _cleanup_loop(self):
        """清理循环"""
        self.logger.info(f"会话清理线程已启动，每 {self.cleanup_interval} 秒执行一次清理")
        
        while self.running:
            try:
                # 执行清理
                self.cleanup_once()
                
                # 等待下次清理
                for _ in range(self.cleanup_interval):
                    if not self.running:
                        break
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"清理循环出错: {e}")
                time.sleep(60)  # 出错后等待1分钟再重试


class ScheduledSessionCleanupService:
    """
    基于schedule库的定时清理服务
    支持更灵活的定时清理配置
    """
    
    def __init__(self, session_timeout: int = 86400):
        """
        初始化定时清理服务
        
        Args:
            session_timeout: 会话超时时间（秒），默认为24小时
        """
        self.logger = logging.getLogger(f"IHI_detection.{__name__}")
        self.session_timeout = session_timeout
        self.running = False
        self.schedule_thread = None
        self.session_manager = get_session_manager()
        
        self.logger.info(f"定时会话清理服务初始化完成，会话超时: {session_timeout}秒")
    
    def schedule_cleanup(self, time_pattern: str, job_func: Optional[Callable] = None):
        """
        安排清理任务
        
        Args:
            time_pattern: 时间模式，支持schedule库的所有模式，如 "daily", "hourly", "every 10 minutes" 等
            job_func: 自定义清理函数，如果为None则使用默认清理函数
        """
        if job_func is None:
            job_func = self._default_cleanup_job
        
        try:
            # 解析时间模式并安排任务
            if time_pattern.lower() == "daily":
                schedule.every().day.do(job_func)
            elif time_pattern.lower() == "hourly":
                schedule.every().hour.do(job_func)
            elif time_pattern.lower().startswith("every"):
                # 解析 "every X minutes/hours/seconds" 模式
                parts = time_pattern.lower().split()
                if len(parts) >= 3 and parts[1].isdigit():
                    interval = int(parts[1])
                    unit = parts[2]
                    
                    if unit.startswith("minute"):
                        schedule.every(interval).minutes.do(job_func)
                    elif unit.startswith("hour"):
                        schedule.every(interval).hours.do(job_func)
                    elif unit.startswith("second"):
                        schedule.every(interval).seconds.do(job_func)
                    else:
                        self.logger.error(f"不支持的时间单位: {unit}")
                        return False
                else:
                    self.logger.error(f"无法解析的时间模式: {time_pattern}")
                    return False
            else:
                self.logger.error(f"不支持的时间模式: {time_pattern}")
                return False
            
            self.logger.info(f"已安排清理任务: {time_pattern}")
            return True
        except Exception as e:
            self.logger.error(f"安排清理任务时出错: {e}")
            return False
    
    def start(self):
        """启动定时服务"""
        if self.running:
            self.logger.warning("定时会话清理服务已在运行")
            return
        
        self.running = True
        self.schedule_thread = threading.Thread(target=self._schedule_loop, daemon=True)
        self.schedule_thread.start()
        
        self.logger.info("定时会话清理服务已启动")
    
    def stop(self):
        """停止定时服务"""
        if not self.running:
            self.logger.warning("定时会话清理服务未在运行")
            return
        
        self.running = False
        schedule.clear()
        
        if self.schedule_thread and self.schedule_thread.is_alive():
            self.schedule_thread.join(timeout=5)
        
        self.logger.info("定时会话清理服务已停止")
    
    def _default_cleanup_job(self):
        """默认清理任务"""
        try:
            cleaned_count = self.session_manager.cleanup_expired_sessions()
            
            if cleaned_count > 0:
                self.logger.info(f"[定时清理] 清理了 {cleaned_count} 个过期会话")
            else:
                self.logger.debug("[定时清理] 没有需要清理的过期会话")
        except Exception as e:
            self.logger.error(f"[定时清理] 清理过期会话时出错: {e}")
    
    def _schedule_loop(self):
        """定时循环"""
        self.logger.info("定时清理线程已启动")
        
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"定时循环出错: {e}")
                time.sleep(60)  # 出错后等待1分钟再重试


# 全局清理服务实例
_cleanup_service = None
_scheduled_cleanup_service = None


def get_cleanup_service(cleanup_interval: int = 3600, session_timeout: int = 86400) -> SessionCleanupService:
    """
    获取全局清理服务实例
    
    Args:
        cleanup_interval: 清理间隔（秒）
        session_timeout: 会话超时时间（秒）
        
    Returns:
        SessionCleanupService: 清理服务实例
    """
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = SessionCleanupService(cleanup_interval, session_timeout)
    return _cleanup_service


def get_scheduled_cleanup_service(session_timeout: int = 86400) -> ScheduledSessionCleanupService:
    """
    获取全局定时清理服务实例
    
    Args:
        session_timeout: 会话超时时间（秒）
        
    Returns:
        ScheduledSessionCleanupService: 定时清理服务实例
    """
    global _scheduled_cleanup_service
    if _scheduled_cleanup_service is None:
        _scheduled_cleanup_service = ScheduledSessionCleanupService(session_timeout)
    return _scheduled_cleanup_service


def start_cleanup_service(cleanup_interval: int = 3600, session_timeout: int = 86400):
    """
    启动清理服务
    
    Args:
        cleanup_interval: 清理间隔（秒）
        session_timeout: 会话超时时间（秒）
    """
    service = get_cleanup_service(cleanup_interval, session_timeout)
    service.start()


def stop_cleanup_service():
    """停止清理服务"""
    global _cleanup_service
    if _cleanup_service:
        _cleanup_service.stop()


def setup_scheduled_cleanup(time_patterns: list, session_timeout: int = 86400):
    """
    设置定时清理
    
    Args:
        time_patterns: 时间模式列表，如 ["daily", "every 6 hours"]
        session_timeout: 会话超时时间（秒）
    """
    service = get_scheduled_cleanup_service(session_timeout)
    
    for pattern in time_patterns:
        service.schedule_cleanup(pattern)
    
    service.start()