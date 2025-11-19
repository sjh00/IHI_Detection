"""
工具模块包
包含所有MCP工具的分类模块
"""

from src.db_handler import DuckDBHandler
from .history import register_history_tools
from .browser import register_browser_tools
from .file import register_file_tools

def register_all_tools(mcp, config: dict, ai_helper):
    """注册所有工具到主FastMCP实例"""
    try:
        # db
        db_handler = DuckDBHandler(config)
        if not db_handler:
            raise Exception("数据库处理器未初始化")

        # 注册工具
        register_history_tools(mcp, config=config, db_handler=db_handler)
        register_browser_tools(mcp, config=config, db_handler=db_handler, ai_helper=ai_helper)
        register_file_tools(mcp)
    except Exception as e:
        raise Exception(f"注册主MCP工具失败: {e}")
