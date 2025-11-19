"""
浏览器工具模块
"""

import logging
from typing import Dict, Any
from fastmcp import FastMCP
from src.browser_handler import BrowserHandler

logger = logging.getLogger(f"IHI_detection.tools.{__name__}")

def register_browser_tools(mcp: FastMCP, config: dict, db_handler, ai_helper):

    browser = BrowserHandler(config=config, db_handler=db_handler, ai_helper=ai_helper)

    @mcp.tool()
    async def browser_get_struct_document_with_url(url: str) -> Dict[str, Any]:
        """
        获取指定URL的结构化文档内容
        :param url: 要获取内容的URL
        :return: 包含结构化文档内容的字典
        """
        document = await browser.get_struct_document_with_url(url)
        return document

    @mcp.tool()
    async def browser_get_struct_document_with_webpage(webpage: str, url: str = '', screenshot: str = '') -> Dict[str, Any]:
        """
        在已有网页内容的基础上获取该网页结构化文档内容
        :param webpage: 要处理的网页内容
        :param url: 网页的URL，用于辅助识别网页平台，可选
        :param screenshot: 网页截图的base64编码字符串，可选
        :return: 包含结构化文档内容的字典
        """
        document = await browser.get_struct_document_with_webpage(webpage, url, screenshot)
        return document
    
    @mcp.tool()
    async def browser_navigate_and_get_snapshots(url_list: list[str]) -> str:
        """
        导航到指定URL并获取其markdown快照
        :param url_list: 要导航到的URL列表，最多5个URL
        :return: 所有页面的markdown快照内容
        """
        snapshots = await browser.navigate_and_get_snapshots(url_list, ascending=False)
        return '\n\n'.join(snapshots)
