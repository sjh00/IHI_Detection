"""
浏览器队列管理器
用于管理多用户并发使用browser Client的排队机制
"""

import asyncio
import logging
import time
from uuid import uuid4
from typing import Any, Callable, Dict, Optional, Tuple
from fastmcp import Client

class BrowserQueue:
    """
    浏览器队列管理器
    实现单实例browser Client的排队调用机制
    """
    
    def __init__(self, browser_mcp_server: Dict[str, Any], timeout: int = 120):
        """
        初始化浏览器队列管理器
        
        Args:
            browser_mcp_server: 浏览器MCP服务器配置
            timeout: 单个任务的超时时间，默认120秒（2分钟）
        """
        self.logger = logging.getLogger(f"IHI_detection.{__name__}")
        self.browser_mcp_server = browser_mcp_server
        self.timeout = timeout
        self.queue = asyncio.Queue()  # 请求队列
        self.is_processing = False  # 是否正在处理请求
        self.current_client = None  # 当前活跃的客户端
        self.current_task_id = None  # 当前任务ID
        self.lock = asyncio.Lock()  # 异步锁，确保线程安全
        self.client_lock = asyncio.Lock()  # 客户端锁，确保客户端创建和销毁的线程安全
        
    async def execute_with_client(self, func: Callable, *args, **kwargs) -> Any:
        """
        使用browser Client执行函数
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        # 获取任务ID
        task_id = await self._get_task_id()
        start_time = time.time()
        
        # 创建任务元组
        task = (task_id, func, args, kwargs, asyncio.Future())
        
        # 将任务加入队列
        await self.queue.put(task)
        self.logger.debug(f"任务 {task_id} 已加入队列，当前队列长度: {self.queue.qsize()}")
        
        # 等待任务完成
        try:
            result = await task[4]
            elapsed_time = time.time() - start_time
            self.logger.debug(f"任务 {task_id} 完成，耗时: {elapsed_time:.2f}秒")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"任务 {task_id} 失败，耗时: {elapsed_time:.2f}秒，错误: {str(e)}")
            raise
    
    async def _get_task_id(self) -> str:
        """获取唯一任务ID"""
        async with self.lock:
            return str(uuid4())
    
    async def _process_queue(self):
        """处理队列中的任务"""
        # 创建一个共享的浏览器客户端实例
        async with self.client_lock:
            client = Client(self.browser_mcp_server, timeout=self.timeout)
            await client.__aenter__()
            
        try:
            while True:
                try:
                    # 从队列中获取任务
                    task = await self.queue.get()
                    task_id, func, args, kwargs, future = task
                    
                    # 设置当前任务信息
                    async with self.lock:
                        self.current_task_id = task_id
                        self.is_processing = True
                        self.current_client = client
                    
                    self.logger.debug(f"开始处理任务 {task_id}")
                    
                    try:
                        # 执行任务，设置超时
                        try:
                            result = await asyncio.wait_for(
                                func(client, *args, **kwargs), 
                                timeout=self.timeout
                            )
                            future.set_result(result)
                        except asyncio.TimeoutError:
                            error_msg = f"任务 {task_id} 超时（{self.timeout}秒）"
                            self.logger.warning(error_msg)
                            future.set_exception(TimeoutError(error_msg))
                        except Exception as e:
                            self.logger.error(f"任务 {task_id} 执行失败: {str(e)}")
                            future.set_exception(e)
                    
                    finally:
                        # 清理当前任务信息
                        async with self.lock:
                            self.current_client = None
                            self.current_task_id = None
                            self.is_processing = False
                        
                        # 标记任务完成
                        self.queue.task_done()
                        self.logger.debug(f"任务 {task_id} 处理完成")
                
                except Exception as e:
                    self.logger.error(f"处理队列时发生错误: {str(e)}")
        finally:
            # 确保客户端被正确关闭
            async with self.client_lock:
                await client.__aexit__(None, None, None)
    
    async def start(self):
        """启动队列处理器"""
        if not self.is_processing:
            self.logger.debug("启动浏览器队列处理器")
            asyncio.create_task(self._process_queue())
    
    async def stop(self):
        """停止队列处理器"""
        self.logger.debug("停止浏览器队列处理器")
        # 清空队列
        while not self.queue.empty():
            task = await self.queue.get()
            task_id, _, _, _, future = task
            error_msg = f"任务 {task_id} 因队列停止而被取消"
            future.set_exception(RuntimeError(error_msg))
            self.queue.task_done()
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        async with self.lock:
            return {
                "queue_size": self.queue.qsize(),
                "is_processing": self.is_processing,
                "current_task_id": self.current_task_id,
                "timeout": self.timeout
            }
    
    def is_client_available(self) -> bool:
        """检查browser Client是否可用"""
        return self.current_client is not None
    
    async def get_current_client(self) -> Optional[Client]:
        """获取当前活跃的客户端"""
        async with self.lock:
            return self.current_client