import asyncio
import logging
import os
import sys
import signal
import atexit
from fastmcp import FastMCP
from src.utils import setup_logging, config, ai_helper
from mcp_tools import register_all_tools

# 初始化日志系统
logger = setup_logging(f"IHI_detection.{__name__}", logging.DEBUG)

# 创建行动者MCP实例
mcp = FastMCP(name="IHI Detection actor MCP Server")
# 注册行动者MCP工具模块
logger.debug("🔧 注册行动者MCP工具模块...")
register_all_tools(mcp, config=config, ai_helper=ai_helper)
logger.debug("✅ 行动者MCP工具模块注册完成")

async def run_server():
    """启动 MCP 服务器集群"""
    config_mcp = config['mcp']
    logger.info("🚀 启动 IHI Detection MCP Server")
    await mcp.run_async(transport=config_mcp['transport'], show_banner=False, host=config_mcp['host'], port=config_mcp['port'])


# 信号处理优化，避免Intel Fortran运行时库冲突
def setup_signal_handlers():
    """设置信号处理器以避免Intel Fortran运行时库冲突"""
    def signal_handler(signum, frame):
        """优雅的信号处理器"""
        logger.info("👋 MCP服务器集群已关闭")
        # 设置退出标志而不是直接退出
        os._exit(0)
    
    # 注册信号处理器
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    # 注册退出处理器
    def cleanup():
        logger = logging.getLogger(f"IHI_detection.{__name__}")
        logger.info("👋 MCP服务器集群已关闭")
    
    atexit.register(cleanup)

if __name__ == "__main__":
    # 设置信号处理器以避免Intel Fortran运行时库冲突
    setup_signal_handlers()
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.debug(f"🛑 正在关闭 MCP 服务器集群...")
    except Exception as e:
        logger.error(f"❌ 启动 MCP 服务器集群失败: {e}")
    finally: 
        logger.info("👋 MCP服务器集群已关闭")
