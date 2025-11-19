"""
检测历史记录相关工具模块
包含记录检测历史、查询检测历史、跟踪预警处置情况等功能
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from fastmcp import FastMCP
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.history_manager import HistoryManager

logger = logging.getLogger(f"IHI_detection.tools.{__name__}")

def register_history_tools(mcp: FastMCP, config:dict, db_handler):
    """注册检测历史相关工具"""

    history_manager = HistoryManager(config=config, db_handler=db_handler)
    
    @mcp.tool()
    def history_insert(data: dict) -> bool:
        """
        插入检测历史记录
        
        Args:
            data: 检测历史记录数据，包含所有字段
            
        Returns:
            插入的结果，成功时返回true，失败时返回false
        """
        try:
            if 'content' in data and 'disposal_status' in data and data['content']:
                return history_manager.insert_detection_history(**data)
        except Exception as e:
            logger.error(f'插入检测历史记录失败: {str(e)}')
        return False
    
    @mcp.tool()
    def history_update_analytical_notes(id: str, data: dict) -> bool:
        """
        更新检测历史记录的分析笔记
        :param id: 检测历史记录ID
        :param data: 分析笔记字段和值的字典
        :return: 更新结果，成功时返回true，失败时返回false
        """
        try:
            history_manager.update_analytical_notes(id=id, data=data)
            return True
        except Exception as e:
            logger.error(f'更新ID为{id}的检测历史记录分析笔记失败: {str(e)}')
            return False

    @mcp.tool()
    def history_update_disposal_status(id: str, disposal_status: int) -> bool:
        """
        更新检测历史记录的处置状态
        :param id: 检测历史记录ID
        :param disposal_status: 处置状态
        :return: 更新结果，成功时返回true，失败时返回false
        """
        try:
            history_manager.update_disposal_status(id=id, disposal_status=disposal_status)
            return True
        except Exception as e:
            logger.error(f'更新ID为{id}的检测历史记录处置状态失败: {str(e)}')
            return False

    @mcp.tool()
    def history_get_with_id(id: str, flatten_nested: bool = False) -> Dict[str, Any] | bool:
        """
        根据ID查询检测历史记录
        :param id: 检测历史记录ID
        :param flatten_nested: 是否将嵌套字段展平，默认False
        :return: 匹配的检测历史记录，未找到返回false
        """
        try:
            res = history_manager.get_detection_history_with_id(id, flatten_nested=flatten_nested)
            return res or False
        except Exception as e:
            logger.error(f'查询ID为{id}的检测历史记录失败: {str(e)}')
            return False

    @mcp.tool()
    def history_get_with_content_and_url(content: str = '', source_url: str = '', flatten_nested: bool = False) -> Dict[str, Any] | bool:
        """
        根据内容查询历史检测记录
        :param content: 内容，可选
        :param source_url: 内容来源URL，可选
        :param flatten_nested: 是否将嵌套字段展平，默认False
        :return: 匹配的历史检测记录，未找到返回false
        """
        if not content and not source_url:
            return False
        try:
            res = history_manager.get_detection_history_with_content_and_url(content, source_url, flatten_nested=flatten_nested)
            return res or False
        except Exception as e:
            logger.error(f'查询内容为{content}，URL为{source_url}的历史检测记录失败: {str(e)}')
            return False
    
    @mcp.tool()
    def history_get_with_url(source_url: str, flatten_nested: bool = False) -> Dict[str, Any] | bool:
        """
        根据URL查询历史检测记录
        :param source_url: 内容来源URL
        :param flatten_nested: 是否将嵌套字段展平，默认False
        :return: 匹配的历史检测记录，未找到返回false
        """
        try:
            res = history_manager.get_detection_history_with_url(source_url, flatten_nested=flatten_nested)
            return res or False
        except Exception as e:
            logger.error(f'查询URL为{source_url}的历史检测记录失败: {str(e)}')
            return False

    @mcp.tool()
    def history_get_with_disposal_status(disposal_status: int, flatten_nested: bool = False) -> list[Dict[str, Any]] | bool:
        """
        根据处置状态查询历史检测记录
        :param disposal_status: 处置状态
        :param flatten_nested: 是否将嵌套字段展平，默认False
        :return: 匹配的历史检测记录列表，未找到返回false
        """
        try:
            res_list = history_manager.get_detection_history_with_disposal_status(disposal_status, flatten_nested=flatten_nested)
            return res_list or False
        except Exception as e:
            logger.error(f'查询处置状态为{disposal_status}的历史检测记录失败: {str(e)}')
            return False

