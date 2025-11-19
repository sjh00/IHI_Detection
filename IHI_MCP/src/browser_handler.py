"""
浏览器处理器
"""

import json
import logging
import re
import asyncio
from pathlib import Path
import shutil
import time
from typing import Dict, List, Optional, Tuple, Union, Any, cast
from urllib.parse import unquote, quote, urlparse
from fastmcp import Client
import src.utils as utils
from src.ai_helper import AIHelper
from src.history_manager import HistoryManager
from src.browser_queue import BrowserQueue

class BrowserHandler:
    """
    浏览器处理类
    """

    JSON_CODE_COMPILE = re.compile(r'```json\n+(.*?)\n+```', re.DOTALL)
    JSON_PROCESS_COMPILE = re.compile(r'\\(?=")')
    JSON_UNICODE_COMPILE = re.compile(r'\\u([0-9a-fA-F]{4})')

    def __init__(self, config: dict, db_handler, ai_helper: AIHelper):
        self.logger = logging.getLogger(f"IHI_detection.{__name__}")
        self.config = config['browser']
        self.browser_mcp_server = self.config['browser_mcp_server']
        self.screenshots_dir = utils.screenshots_dir.absolute()
        self.screenshots_orig_dir = utils.screenshots_orig_dir.absolute()
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
        self.ai_helper = ai_helper
        self.history_manager = HistoryManager(config=config, db_handler=db_handler) # 初始化历史记录管理器
        self.browser_queue = BrowserQueue(self.browser_mcp_server, timeout=self.config.get('timeout', 240)) # 初始化浏览器队列管理器，设置2分钟超时
        self._queue_started = False # 标记队列是否已启动
        self.domain_lasttime = {} # 域名上次访问时间，用于避免频繁访问同一域名
        self.platform_map = self.config.get('platform_map', {}) # 发布平台名映射表

    def process_coroutine(self, coro) -> str:
        """
        处理Coroutine对象
        :param coro: Coroutine对象
        :return: 处理后的文字内容
        """
        if coro:
            return '\n'.join([cont.text for cont in coro.content])
        return ''
    
    async def execute_with_browser_queue(self, func, *args, **kwargs):
        """
        使用BrowserQueue执行browser操作
        :param func: 要执行的函数
        :param args: 函数参数
        :param kwargs: 函数关键字参数
        :return: 函数执行结果
        """
        # 如果队列还未启动，则启动队列
        if not self._queue_started:
            await self.browser_queue.start()
            self._queue_started = True
        
        return await self.browser_queue.execute_with_client(func, *args, **kwargs)
    
    async def mcp_text_process(self, client: Client, tool: str, param: dict = {}, json_process: bool = False) -> str:
        """
        调用MCP服务器的工具后处理输出
        :param client: 客户端实例
        :param tool: 工具名称
        :param param: 工具参数
        :param json_process: 是否期望处理JSON字符串
        :return: 处理好的字符串，或空字符串
        """
        try:
            ct = await client.call_tool(tool, param)
            if ct:
                res = '\n'.join([cont.text for cont in ct.content])
                res = cast(str, utils.sanitize_text(res))
                if res and (json_process or tool == "evaluate_script"):
                    json_match = self.JSON_CODE_COMPILE.search(res)
                    if json_match:
                        json_str = json_match.group(1).strip().strip('"')
                        # 只处理Unicode转义序列，不影响正常中文字符
                        json_str = self.JSON_UNICODE_COMPILE.sub(lambda m: chr(int(m.group(1), 16)), json_str)
                        res = json_str
                return res
        except Exception as e:
            if tool != "evaluate_script" or "(reading 'remove')" not in str(e):
                self.logger.warning(f"调用MCP工具 {tool} 时发生错误: {e}")
            # 忽略超时错误
            pass
        return ""
    
    def parse_network_request(self, markdown: str) -> tuple[dict, str]:
        """
        从MCP返回的Markdown结果中提取headers和body内容
        :param markdown: 包含headers和body的Markdown字符串
        :return: 包含headers和body内容的元组
        """
        if not markdown:
            return {}, ""
        headers = {}
        body = ""
        cur_is = ''
        for line in markdown.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line[0] == '#':
                if cur_is == 'body' and body:
                    break
                elif line == '### Request Headers':
                    cur_is = 'headers'
                    continue
                elif line == '### Response Body':
                    cur_is = 'body'
                    continue
                else:
                    cur_is = ''
                    continue
            elif cur_is == 'headers':
                if line.startswith('- '):
                    line = line[2:]
                kvp = line.split(':', 1)
                if len(kvp) == 2:
                    key = kvp[0].strip()
                    value = kvp[1].strip()
                    headers[key] = value
            elif cur_is == 'body':
                body += line + '\n'
        return headers, body.strip()

    async def _get_tab_idx(self, client: Client, url: str, ifnotfoundopenit: bool = False) -> tuple[int, bool, str]:
        """
        获取浏览器已打开的标签页中指定url的索引和是否已选中
        :param client: 客户端实例
        :param url: 检查过的要查找的URL
        :param ifnotfoundopenit: 如果未找到是否打开新标签页
        :return: 包含标签页索引和是否已选中的元组，或(-1, False, "")表示未找到
        """
        navigate_list = await self.mcp_text_process(client, "list_pages")
        if not navigate_list:
            self.logger.error("获取标签页列表失败")
            return -1, False, ""
        tab_lines = [line.strip() for line in navigate_list.split('\n') if line.strip() and line.strip()[0] != '#']
        pageIdx = -1
        selected = False
        for idx, line in enumerate(tab_lines):
            try:
                line_data = line.split(' ')
                if len(line_data) > 1 and line_data[1].strip() == url:
                    pageIdx = idx
                    selected = "[selected]" in line_data
                    break
            except:
                pass
        else:
            if ifnotfoundopenit:
                pageIdx, realurl = await self.navigate_page(client, url)
                if pageIdx < 0:
                    return -1, False, realurl
                return pageIdx, True, realurl
            else:
                self.logger.warning(f"未找到URL {url} 的页标签索引")
                return -1, False, ""
        return pageIdx, selected, url
    
    async def get_reqids(self, client: Client, url_startswith: str, resource_types: list = []) -> list[int]:
        """
        获取浏览器已打开的标签页中指定url前缀的所有XHR请求ID
        :param client: 客户端实例
        :param url_startswith: 要查找的URL前缀
        :param resource_types: 资源类型列表，默认[]
        :return: 包含所有请求ID的列表，或空列表表示未找到
        """
        if resource_types:
            navigate_list = await self.mcp_text_process(client, "list_network_requests", {"resourceTypes": resource_types})
        else:
            navigate_list = await self.mcp_text_process(client, "list_network_requests")
        if not navigate_list:
            self.logger.error("获取网络请求列表失败")
            return []
        reqids = []
        for line in navigate_list.split('\n'):
            if ' ' + url_startswith in line:
                reqid = line.split(' ', 1)[0]
                if '=' in reqid:
                    reqid = reqid.split('=', 1)[1]
                    if reqid.isdigit():
                        reqids.append(int(reqid))
        return reqids
    
    async def get_network_request_with_id(self, client: Client, reqid: int) -> str:
        """
        获取浏览器已打开的标签页中指定请求ID的详细信息
        :param client: 客户端实例
        :param reqid: 要查找的请求ID
        :return: 包含请求详细信息的字符串，或空字符串表示未找到
        """
        network_request = ""
        if reqid:
            network_request = await self.mcp_text_process(client, "get_network_request", {"reqid": reqid})
        return network_request
    
    async def get_network_request_with_url(self, client: Client, url_startswith: str, resource_types: list = [], index: int = 0) -> str:
        """
        获取浏览器已打开的标签页中指定URL前缀的指定请求ID的详细信息
        :param client: 客户端实例
        :param url_startswith: 要查找的URL前缀
        :param resource_types: 资源类型列表，默认[]
        :param index: 要获取的请求索引，默认0
        :return: 包含请求详细信息的字符串，或空字符串表示未找到
        """
        network_request = ""
        reqids = await self.get_reqids(client, url_startswith, resource_types)
        if reqids:
            reqid = reqids[index]
            network_request = await self.get_network_request_with_id(client, reqid)
        return network_request
    
    async def get_all_network_requests_with_url(self, client: Client, url_startswith: str, resource_types: list = []) -> list[str]:
        """
        获取浏览器已打开的标签页中指定URL前缀的所有请求ID的详细信息
        :param client: 客户端实例
        :param url_startswith: 要查找的URL前缀
        :param resource_types: 资源类型列表，默认[]
        :return: 包含所有请求详细信息的字符串列表，或空列表表示未找到
        """
        network_requests = []
        reqids = await self.get_reqids(client, url_startswith, resource_types)
        if reqids:
            for reqid in reqids:
                network_request = await self.get_network_request_with_id(client, reqid)
                network_requests.append(network_request)
        return network_requests

    def get_realurl(self, navigate_tabs: str) -> tuple[int, str]:
        """
        获取实际URL
        :param navigate_tabs: 导航标签字符串
        :return: 返回标签id和跳转后的实际url，失败返回(-1, "")
        """
        if navigate_tabs:
            tab_lines = [line.strip() for line in navigate_tabs.split('\n') if line.strip() and line.strip()[0] != '#']
            for idx, line in enumerate(tab_lines):
                try:
                    line_data = line.split(' ')
                    if len(line_data) > 1 and "[selected]" in line_data:
                        realurl = line_data[1].strip()
                        return idx, realurl
                except Exception as e:
                    self.logger.error(f"解析导航标签行失败：{line}，错误：{e}")
        return -1, ""

    async def simple_navigate_page(self, client: Client, url: str, timeout: int = 30) -> tuple[int, str]:
        """
        简单导航到指定URL，返回导航结果
        :param client: 客户端实例
        :param url: 要导航的URL
        :param timeout: 超时时间，默认30秒
        :return: 成功返回标签id和跳转后的实际url，失败返回(-1, "")
        """
        navigate_tabs = await self.mcp_text_process(client, "navigate_page", {"url": url, "timeout": timeout * 1000})
        return self.get_realurl(navigate_tabs)

    async def navigate_page(self, client: Client, url: str, need_colddown: bool = True, first_time: bool = False, is_ref_url: bool = False, timeout: int = 30) -> tuple[int, str]:
        """
        控制浏览器导航到指定URL
        :param client: 客户端实例
        :param url: 检查过的要导航的URL
        :param need_colddown: 是否需要等待域名冷却时间，默认True
        :param first_time: 是否是第一次访问，默认False
        :param is_ref_url: 是否是参考URL，默认False
        :param timeout: 超时时间，默认30秒
        :return: 成功返回标签id和跳转后的实际url，失败返回(-1, "错误信息")，
        """

        try:
            domain = urlparse(url).netloc
            if not domain:
                self.logger.error(f"URL域名解析失败：{url}")
                return -1, ""
            # 检查域名是否上次访问过，若访问过短则等待
            if need_colddown:
                lasttime = self.domain_lasttime.get(domain, 0)
                timediff = time.time() - lasttime
                if timediff <= 3:
                    await asyncio.sleep(3 - timediff)

            site_name = utils.get_site_name(url)
                
            # 链接需要确认的情况 - 访问前
            if first_time and not is_ref_url:
                if site_name == "抖音":
                    if 'modal_id=' in url:
                        modal_id = url.split("modal_id=")[1].split("&", 1)[0]
                        if modal_id:
                            url = 'https://www.douyin.com/video/' + modal_id
                    url = url.split("#", 1)[0].split("?", 1)[0].split("&", 1)[0]
                elif site_name == "快手":
                    url = url.split("#", 1)[0].split("?", 1)[0]
                    if '/f/' not in url:
                        sourceId = url.split("/")[-1]
                        if not sourceId or sourceId == 'short-video' or sourceId.endswith('kuaishou.com'):
                            self.logger.warning(f"快手链接格式错误：{url}")
                            return -1, "快手链接格式错误"
                        if not url.startswith('https://www.kuaishou.com/short-video/'):
                            url = 'https://www.kuaishou.com/short-video/' + sourceId
                elif site_name == "小红书":
                    if "&xsec_source" in url:
                        url = url.split("&xsec_source")[0]

            self.domain_lasttime[domain] = time.time()
            idx, realurl = await self.simple_navigate_page(client, url, timeout)
            if idx == -1:
                return -1, ""
            
            await asyncio.sleep(0.5) # 等待页面反应一会
            navigate_tabs = await self.mcp_text_process(client, "list_pages")
            idx, realurl = self.get_realurl(navigate_tabs)
            if idx == -1:
                return -1, ""
            
            site_name = utils.get_site_name(realurl)
            if not is_ref_url:
                if site_name in ["抖音", "快手"]:
                    realurl = realurl.split("#", 1)[0].split("?", 1)[0]
                    # 检测是否是验证码页面，若是刷新一次
                    snapshot = await self.mcp_text_process(client, "take_snapshot")
                    if '"captcha"' in snapshot:
                        await asyncio.sleep(1)
                        await self.mcp_text_process(client, "navigate_page", {"url": realurl, "timeout": timeout * 1000})
                elif site_name == "小红书":
                    # 检查是否跳转到了登录页
                    if realurl.startswith('https://www.xiaohongshu.com/login'):
                        self.logger.warning(f"页面为登录页面，需通过小红书App扫码验证")
                        return -1, "页面为登录页面，需通过小红书App扫码验证"
                
                # 链接需要确认的情况 - 访问后
                if site_name == "抖音":
                    await asyncio.sleep(2) # 等待页面反应一会
                    await self.mcp_text_process(client, "evaluate_script", {"function": "() => {let loginPanel = document.querySelector('[id^=\"login-full-panel\"]'); if (loginPanel) {loginPanel.remove();}}"})
                elif site_name == "快手":
                    await asyncio.sleep(0.7) # 等待页面反应一会
                    await self.mcp_text_process(client, "evaluate_script", {"function": "() => {let retryBtn = document.querySelector('[class=\"retry-btn\"]'); if (retryBtn) {retryBtn.click();}}"})
                elif site_name == "小红书":
                    await self.mcp_text_process(client, "evaluate_script", {"function": "() => {let loginModal = document.querySelector('[class$=\"login-modal\"]'); if (loginModal) {loginModal.remove();}}"})
                elif site_name == "微博":
                    await self.mcp_text_process(client, "wait_for", {"text": '评论', "timeout": 5000})
                    await self.mcp_text_process(client, "evaluate_script", {"function": "() => {let body = document.body; if (body) {body.style.zoom = 1.5;}}"})
                elif site_name == "今日头条":
                    await self.mcp_text_process(client, "wait_for", {"text": '评论', "timeout": 5000})
                elif site_name == "知乎":
                    await self.mcp_text_process(client, "evaluate_script", {"function": "() => {let modal = document.querySelector('[class^=\"Modal-wrapper\"]'); if (modal) {modal.remove();}}"})

                # 视频类加载完成判断
                if site_name in ["抖音", "快手", "小红书", "微博", "今日头条"]:
                    readyState = 0
                    retry_count = 0
                    # 减少重试次数和等待时间，优化性能
                    while readyState >= 0 and readyState < 4 and retry_count < 3:
                        retry_count += 1
                        readyState = int(await self.mcp_text_process(client, "evaluate_script", {"function": "() => {let video = document.querySelector('video'); return video ? video.readyState : '-1';}"}))
                        if readyState >= 0 and readyState < 4:
                            self.logger.debug(f"视频未加载完成，readyState：{readyState}")
                            await asyncio.sleep(1) # 等待页面反应一会

            return idx, realurl
        except Exception as e:
            self.logger.error(f"导航失败：{url}，错误信息：{e}")
        return -1, "网页访问失败"

    def get_history_with_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        获取URL的检测历史
        :param url: 要查询的URL
        :return: 包含检测历史的字典，若不存在则返回None
        """
        return self.history_manager.get_detection_history_with_url(url.strip(), flatten_nested=True)

    async def get_struct_document_with_url(self, url: str) -> Dict[str, Any]:
        """
        控制浏览器获取网页内容结构化数据
        :param url: 要分析的已打开的URL
        :return: 包含网页内容的字典，失败返回包含error的字典，例如{"error": "URL不能为空或不合法"}
        """
        url = utils.url_check(url)
        if not url:
            return {"error": "URL不能为空或不合法"}
        
        publish_platform = utils.get_site_name(url, self.platform_map)
        if not publish_platform:
            # 判断给定的链接是否为内容发布平台的内容页链接
            res = await self.ai_helper.chat(
                prompt=f"判断给定的链接是否为内容发布平台的链接，并且是内容页链接（而不是主页、异常状态、风险链接、木马链接、虚假链接、访问验证等非内容页链接）。链接为：{url}",
                json_schema={
                    "type": "object",
                    "properties": {
                        "is_content_page": {
                            "type": "string",
                            "description": "是否为内容发布平台的内容页链接",
                            "enum": ["是", "否"]
                        }
                    },
                    "required": ["is_content_page"]
                },
            )
            if 'error' in res or "is_content_page" not in res['content'] or not res['content']["is_content_page"] or res['content']["is_content_page"] == "否":
                return {"error": "链接无可检测内容，或访问过于频繁已被暂时阻止访问"}

        snapshot, screenshot_path, orig_screenshot, screenshot_orig_path, screenshot_invalid_path = "", Path(""), False, Path(""), Path("")
        # 使用BrowserQueue执行browser操作
        async def _get_struct_document(client: Client) -> Dict[str, Any]:
            nonlocal url, publish_platform, snapshot, screenshot_path, orig_screenshot, screenshot_orig_path, screenshot_invalid_path
            pageIdx, realurl = await self.navigate_page(client, url, need_colddown=True, first_time=True)
            if pageIdx < 0:
                if realurl:
                    return {"error": realurl} # 此时realurl实际是错误信息
                return {"error": "当前网站因访问频繁或未登录，已被暂时阻止访问，请稍后再试"}
            if realurl and realurl != url:
                url = realurl
                publish_platform = utils.get_site_name(url, self.platform_map)

            # 获取页面快照
            snapshot = await self.mcp_text_process(client, "take_snapshot")
            url_hash = utils.get_url_hash(url)
            # 若没有验证码，截图
            orig_screenshot = False # 是否为初始截图
            screenshot_path = self.screenshots_dir / f"{url_hash}.webp"
            screenshot_orig_path = self.screenshots_orig_dir / f"{url_hash}.webp"
            screenshot_invalid_path = self.screenshots_dir / f"invalid_{url_hash}.webp" # 无效截图
            if '"captcha"' not in snapshot:
                await self.mcp_text_process(client, "take_screenshot", {"filePath": screenshot_path, "format": "webp", "fullPage": True, "quality": 80})
                # 复制原始截图到orig目录
                if not screenshot_orig_path.exists():
                    shutil.copy(screenshot_path, screenshot_orig_path)
                    orig_screenshot = True

            document = {
                "source_url": "", # 此字段不可为空
                "content": "", # 此字段为空且附件为空说明抓取失败
                "attachment_count": 0, # 附件数量
                "multimodel_desc": "", # 多模态描述内容
                "multimodel_text": "", # 多模态文本内容
                "speech_text": "", # 语音转录文本内容
                "presentation_form": "",
                "is_aigc": False,
                "scrape_time": int(time.time()), # 爬取时间戳，秒级时间戳
                "publish_time": 0, # 被检测的原始内容的发布时间戳，注意是秒级时间戳，爬取时注意换算
                "publish_platform": "",
                "publisher_id": "",
                "publisher_nickname": "",
                "ip_region": "",
                "publisher_page_url": "", # 该用户的个人主页URL
                "verified_desc": "", # 该用户的认证描述
                "publisher_signature": "",
                "publisher_followers": "", # 该用户的粉丝数
                "publisher_likes": "", # 该用户的获赞数
                "comment_wordcloud": "", # 评论词云
                "content_views": "", # 该内容的浏览数
                "content_likes": "", # 该内容的点赞数
                "content_comments": "", # 该内容的评论数
                "content_collections": "", # 该内容的收藏数
                "content_shares": "", # 该内容的分享数
                "content_forwards": "", # 该内容的转发数
            }

            history = self.get_history_with_url(url)
            if history:
                self.logger.debug(f"存在历史记录，URL: {url}")
                document.update(history)

            document["source_url"] = url
            document["publish_platform"] = publish_platform
            return document
        
        # 使用BrowserQueue执行browser操作
        document = await self.execute_with_browser_queue(_get_struct_document)
        if 'error' in document:
            return document
        
        try:
            # 调用大模型分析
            self.logger.debug(f"调用大模型分析：{document['source_url']}")
            pure_snapshot = snapshot.split('\n', 2)[-1]
            if screenshot_path.exists():
                screenshot_url = utils.file_to_url(screenshot_path)
                page_document = await self.get_struct_document_with_webpage(pure_snapshot, document['source_url'], screenshot_url)
            else:
                page_document = await self.get_struct_document_with_webpage(pure_snapshot, document['source_url'])
            self.logger.debug(f"大模型分析结果中content预览: {page_document.get("content","")[:200]}")
            if 'error' in page_document:
                if orig_screenshot:
                    screenshot_orig_path.unlink(missing_ok=True)
                if screenshot_invalid_path.exists():
                    screenshot_path.unlink(missing_ok=True)
                else:
                    screenshot_path.rename(screenshot_invalid_path)
                return {"error": page_document['error']}
            else:
                if screenshot_invalid_path.exists():
                    screenshot_invalid_path.unlink(missing_ok=True)
            if 'is_aigc' in page_document:
                document['is_aigc'] = document['is_aigc'] or page_document['is_aigc']
                del page_document['is_aigc']
            if 'publish_time' in page_document:
                if len(str(document['publish_time'])) < 10 and isinstance(page_document['publish_time'], str) and len(page_document['publish_time']) > 9:
                    page_document['publish_time'] = page_document['publish_time'].strip()
                    if len(page_document['publish_time']) < 12:
                        page_document['publish_time'] += ' 00:00:00'
                    elif len(page_document['publish_time']) < 14:
                        page_document['publish_time'] += ':00:00'
                    elif len(page_document['publish_time']) < 17:
                        page_document['publish_time'] += ':00'
                    # 转换为时间戳（page_document['publish_time']是东八区时间）
                    document['publish_time'] = int(time.mktime(time.strptime(page_document['publish_time'], '%Y-%m-%d %H:%M:%S')))
                del page_document['publish_time']
            none_words = ['未知', '无', '不详', '无信息', '无数据', '无描述', '无内容', '无链接', '无图片', '无视频', '无音频', '无文件', '无处置建议', '无建议', 'None', 'none', 'null', 'NA', 'na']
            for k, v in page_document.items():
                if v and v not in none_words:
                    if k not in document:
                        document[k] = v
                    elif not document[k]:
                        document[k] = v
            # 标准化内容
            if document["content"]:
                document["content"] = utils.process_sample_for_prompt(document["content"])

            if not document["content"]:
                return {"error": "页面内容为空"}
            else:
                # 标准化用户签名
                if document["publisher_signature"]:
                    document["publisher_signature"] = utils.process_sample_for_prompt(document["publisher_signature"])
                # 标准化用户ID和用户昵称
                if document["publisher_id"]:
                    document["publisher_id"] = document["publisher_id"].strip()
                if document["publisher_nickname"]:
                    document["publisher_nickname"] = document["publisher_nickname"].strip()
                # 标准化评论词云
                if document["comment_wordcloud"]:
                    if isinstance(document["comment_wordcloud"], str):
                        document["comment_wordcloud"] = document["comment_wordcloud"].strip()
                    elif isinstance(document["comment_wordcloud"], list):
                        document["comment_wordcloud"] = ','.join([word.strip() for word in document["comment_wordcloud"]])
                    else:
                        document["comment_wordcloud"] = ''
                # 标准化发布者统计
                publisher_stats = ''
                if document["publisher_followers"]:
                    publisher_stats += f'粉丝:{document["publisher_followers"]},'
                del document["publisher_followers"]
                if document["publisher_likes"]:
                    publisher_stats += f'获赞:{document["publisher_likes"]},'
                del document["publisher_likes"]
                publisher_stats = publisher_stats.strip(',')
                if publisher_stats:
                    document["publisher_stats"] = publisher_stats
                #标准化内容统计
                content_stats = ''
                if document["content_views"]:
                    content_stats += f'浏览:{document["content_views"]},'
                del document["content_views"]
                if document["content_likes"]:
                    content_stats += f'点赞:{document["content_likes"]},'
                del document["content_likes"]
                if document["content_comments"]:
                    content_stats += f'评论:{document["content_comments"]},'
                del document["content_comments"]
                if document["content_collections"]:
                    content_stats += f'收藏:{document["content_collections"]},'
                del document["content_collections"]
                if document["content_shares"]:
                    content_stats += f'分享:{document["content_shares"]},'
                del document["content_shares"]
                if document["content_forwards"]:
                    content_stats += f'转发:{document["content_forwards"]},'
                del document["content_forwards"]
                content_stats = content_stats.strip(',')
                if content_stats:
                    document["content_stats"] = content_stats
                
                return document
    
        except Exception as e:
            self.logger.error(f"页面大模型识别出错 {document['source_url']}: {e}")
    
        return {"error": "当前站点因频繁访问或未登录暂时拒绝访问，请稍后再试"}


    async def get_struct_document_with_webpage(self, webpage: str, url: str = '', screenshot: str = '') -> Dict[str, Any]:
        """
        在已有网页内容的基础上获取该网页结构化文档内容
        :param webpage: 要处理的网页内容
        :param url: 网页的URL，用于辅助识别网页平台，可选
        :param screenshot: 网页截图的标准base64编码字符串，可选
        :return: 包含结构化文档内容的字典
        """
        prompt = ''
        
        images = [screenshot] if screenshot else None
        if images:
            vres = await self.ai_helper.chat(
                prompt=f"""你是擅长分析违法有害信息的人民警察，请分析并描述页面内容，按照以下字段提取页面内主要内容，返回提取结果（字段内容未找到的留空，今天日期：{utils.now2str()}）：
content: 正文内容
presentation_form: 正文内容的呈现形式，视频内容则为视频，图片内容则为图文，只有文本则为文本
is_aigc: 正文内容是否疑似AI生成，根据页面内文字内容、是否有AI生成提示、水印判断
publish_time: 正文内容的发布时间，格式为YYYY-MM-DD HH:MM:SS
publish_platform: 正文内容的发布平台，如抖音、快手、B站等
publisher_nickname: 正文内容的发布者昵称
ip_region: 正文内容发布者的IP所属区域
verified_desc: 正文内容发布者的认证描述，如抖音认证、快手认证、央视认证等
publisher_attribute: 正文内容发布者的属性，如个人、企业、机构、媒体等
publisher_signature: 正文内容发布者的个人签名，一般在发布者昵称附近
publisher_followers: 正文内容发布者的粉丝数量
publisher_likes: 正文内容发布者的获赞数量
content_views: 正文内容的浏览量
content_likes: 正文内容的点赞量
content_comments: 正文内容的评论量
content_collections: 正文内容的收藏量
content_shares: 正文内容的分享量
content_forwards: 正文内容的转发量
comment_wordcloud: 正文内容的评论词云图，展示评论中出现频率较高的词语
""",
                images=images
            )
            if 'error' not in vres:
                page_desc = vres['content'].strip()
                if page_desc:
                    prompt = f'## 网页截图描述\n{page_desc}'
        else:
            prompt = f'## 网页代码转化文本\n{webpage}'

        if url:
            checked_url = utils.url_check(url)
            if checked_url:
                url = checked_url
                prompt = f'## 链接\n{url}\n\n{prompt}'
        
        res = await self.ai_helper.chat(
            prompt=prompt,
            system=f"你是擅长分析违法有害信息的人民警察，请分析页面内容，判断若页面为登录页面、无法访问、已删除、跳转回了平台主页等不包含要检测的内容，则使用error字段填写具体情况（不要笼统），其他字段全空；没有前述情况则不要有error字段并按照以下字段提取页面内主要内容，返回提取结果（字段内容未找到的留空，今天日期：{utils.now2str()}）。",
            json_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "提取正文内容"
                    },
                    "presentation_form": {
                        "type": "string",
                        "description": "正文内容的呈现形式，视频内容则为视频，图片内容则为图文，只有文本则为文本",
                        "enum": ["视频", "图文", "文本"]
                    },
                    "is_aigc": {
                        "type": "string",
                        "description": "正文内容是否疑似AI生成，根据页面内文字内容、是否有AI生成提示、水印判断，无法判断写否",
                        "enum": ["是", "否"]
                    },
                    "publish_time": {
                        "type": "string",
                        "description": "正文内容发布时间，格式为yyyy-MM-dd HH:mm:ss，严格按照格式输出，如果缺少年份填今年，如果缺少时间用00:00:00补上"
                    },
                    "publish_platform": {
                        "type": "string",
                        "description": "当前页面的平台名称，如抖音、快手、微博、小红书、哔哩哔哩、今日头条、百度贴吧、知乎、网易、搜狐、腾讯网、凤凰网、百度百家号等"
                    },
                    "publisher_nickname": {
                        "type": "string",
                        "description": "发布者的昵称"
                    },
                    "ip_region": {
                        "type": "string",
                        "description": "内容发布时的IP属地，于哪个省市发布"
                    },
                    "verified_desc": {
                        "type": "string",
                        "description": "如果发布者有认证信息，则填写其认证描述"
                    },
                    "publisher_attribute": {
                        "type": "string",
                        "description": "发布者的属性",
                        "enum": ["个人", "企业", "机构", "媒体"]
                    },
                    "publisher_signature": {
                        "type": "string",
                        "description": "发布者的个人签名，一般在发布者昵称附近"
                    },
                    "publisher_followers": {
                        "type": "string",
                        "description": "发布者的粉丝数"
                    },
                    "publisher_likes": {
                        "type": "string",
                        "description": "发布者的获赞数"
                    },
                    "content_views": {
                        "type": "string",
                        "description": "内容的浏览量"
                    },
                    "content_likes": {
                        "type": "string",
                        "description": "内容的点赞数"
                    },
                    "content_comments": {
                        "type": "string",
                        "description": "内容的评论数"
                    },
                    "content_collections": {
                        "type": "string",
                        "description": "内容的收藏数"
                    },
                    "content_shares": {
                        "type": "string",
                        "description": "内容的分享数"
                    },
                    "content_forwards": {
                        "type": "string",
                        "description": "内容的转发数"
                    },
                    "comment_wordcloud": {
                        "type": "array",
                        "description": "评论区可见的评论信息的词云，重要的词放前面",
                        "items": {
                            "type": "string",
                            "minLength": 0,
                            "maxLength": 10
                        },
                        "minItems": 0,
                        "maxItems": 10
                    }
                },
	             "required": ["content", "presentation_form", "publish_platform", "publisher_nickname", "publisher_attribute"]
            }
        )
        if 'error' in res:
            return {"error": res["error"]}
        elif "content" in res and 'error' in res['content']:
            if res['content']["error"]:
                return {"error": res['content']["error"]}
            else:
                del res['content']['error']
        if "content" not in res['content'] or not res['content']["content"]:
            return {"error": "未提取到可检测内容"}
        res['content'] = cast(dict, utils.sanitize_text(res['content']))
        if "ip_region" in res['content'] and '中国' in res['content']["ip_region"] and len(res['content']["ip_region"]) > 3:
            res['content']["ip_region"] = res['content']["ip_region"][2:]
        if 'is_aigc' in res['content'] and res['content']["is_aigc"] == "是":
            res['content']["is_aigc"] = True
        else:
            res['content']["is_aigc"] = False
        return res['content']

    async def navigate_and_get_snapshots(self, url_list: list[str], ascending: bool = False) -> list[str]:
        """
        导航到指定URL并获取其markdown快照
        :param url_list: 要导航到的URL列表，最多返回5个URL的快照
        :param ascending: 返回快照时是否重新计算URL列表顺序，默认False
        :return: 所有页面的markdown快照内容列表
        """
        if not url_list or not isinstance(url_list, list):
            return []
        
        # 使用BrowserQueue执行browser操作
        async def _navigate_and_get_snapshots(client: Client) -> list[str]:
            snapshots = []
            jishu = 1
            max_url_num = 5
            # 为每个URL创建独立的客户端连接，确保一个URL失败不影响其他URL
            for idx, url in enumerate(url_list, 1):
                try:
                    reqid, realurl = await self.navigate_page(client, url, need_colddown=False, is_ref_url=True, timeout=5)
                    if reqid < 0:
                        continue
                    
                    snapshot = await self.mcp_text_process(client, "take_snapshot")
                    if snapshot:
                        if ascending:
                            urldesc = f"{jishu}: {url}"
                            jishu += 1
                        else:
                            urldesc = f"{idx}: {url}"
                        if url != realurl:
                            urldesc += f" (跳转后URL: {realurl})"
                        urldesc += " 页面内容"
                        snapshot = '## ' + urldesc + '\n' + snapshot.split('\n', 2)[2]
                    
                        snapshots.append(snapshot)
                        max_url_num -= 1
                        if max_url_num <= 0:
                            break
                except Exception as e:
                    self.logger.error(f"处理URL {url} 时出错：{e}")
                
            return snapshots
        
        return await self.execute_with_browser_queue(_navigate_and_get_snapshots)
    