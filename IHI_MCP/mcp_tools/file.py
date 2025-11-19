"""
文件工具
"""

import logging
import os
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from src.utils import get_screenshot, ext2mime_type, get_url_hash, pack_files_with_data, caption_multimodel

logger = logging.getLogger(f"IHI_detection.tools.{__name__}")

def register_file_tools(mcp: FastMCP):
    """
    注册文件工具到FastMCP实例
    :param mcp: FastMCP实例
    """

    @mcp.tool()
    def file_get_screenshot(url: str, include_invalid: bool = False) -> Image | None:
        """
        获取URL网页的已存档的截图
        :param url: URL网址
        :param include_invalid: 是否包含无效截图，默认为False
        :return: 网页截图字节流
        """
        filename = get_url_hash(url) + '.webp'
        data = get_screenshot(filename, include_invalid)
        if data:
            try:
                return Image(data=data, format='webp')
            except Exception:
                pass
        return None

    @mcp.tool()
    async def file_packfiles_with_data(data: dict) -> Image | None:
        """
        给出分析结果数据，其已存档的附件连同信息表打包成zip文件
        :param data: 分析结果数据
        :return: zip压缩文件
        """
        zip_data = await pack_files_with_data(data)
        if zip_data:
            # 创建Image对象并返回
            try:
                res = Image(data=zip_data, format='zip')
                res._mime_type = 'application/zip'
                logger.debug("Image对象创建成功，返回客户端")
                return res
            except Exception as img_error:
                logger.error(f"创建Image对象时出错: {img_error}")
        return None
    
    @mcp.tool()
    async def files_caption(file_path_list: list[str], max_length: int = 0) -> dict[str, str]:
        """
        获取文件描述、提取文本、语音转录文本
        :param file_path_list: 附件url、本地路径、附件名列表或dify文件json列表，必须要有扩展名
        :param max_length: 最大长度（不得小于500），默认0无限制
        :return: 文件描述、提取文本、语音转录文本的字典
        """
        res = {"multimodel_desc": "", "multimodel_text": "", "speech_text": ""}
        if not file_path_list:
            return res
        multimodel_desc, multimodel_text, speech_text = await caption_multimodel(file_path_list)
        if max_length > 0:
            if max_length < 500:
                max_length = 500
            if len(multimodel_desc) > max_length:
                multimodel_desc = multimodel_desc[:max_length-3]+'...'
            if len(multimodel_text) > max_length:
                multimodel_text = multimodel_text[:max_length-3]+'...'
            if len(speech_text) > max_length:
                speech_text = speech_text[:max_length-3]+'...'
        res["multimodel_desc"] = multimodel_desc
        res["multimodel_text"] = multimodel_text
        res["speech_text"] = speech_text
        return res
