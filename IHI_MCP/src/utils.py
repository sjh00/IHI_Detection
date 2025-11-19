import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import md5
from base64 import b64decode, b64encode
from io import BytesIO
import logging
import os
import random
import re
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union, List, Any, cast
from urllib.parse import unquote, urlparse
from uuid import uuid4
import zipfile
import pandas as pd
import httpx
from trafilatura import extract as trafilatura_extract
import chardet

from fake_useragent import UserAgent
import requests
import cv2
from PIL import Image
from src.ai_helper import AIHelper

screenshots_dir = Path("data") / "screenshots"
screenshots_dir.mkdir(parents=True, exist_ok=True)
screenshots_orig_dir = Path("data") / "screenshots" / "orig" # 初始截图目录，用于在内容下架后回看
screenshots_orig_dir.mkdir(parents=True, exist_ok=True)
temp_dir = Path("data") / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)
thirdparty_dir = Path("thirdparty")
thirdparty_dir.mkdir(parents=True, exist_ok=True)
ffmpeg_dir = thirdparty_dir / "ffmpeg"
ffmpeg_dir.mkdir(parents=True, exist_ok=True)
ffmpeg_path = ffmpeg_dir / "ffmpeg.exe"
logger = logging.getLogger(f"IHI_detection.{__name__}")
fake_ua = UserAgent(os='Windows')

DATE4URL_COMPILE = re.compile(r'\/(\d{4})[\/\-]?(\d{2})[\/\-]?(\d{2})[\/_]') # 支持多种常见的URL日期格式，/YYYY/MM/DD /YYYY-MM-DD /YYYY-MM/DD /YYYY/MM-DD /YYYYMMDD等
RE_PROXY_CHARS = re.compile(r'[\uD800-\uDFFF]')  # UTF-16代理字符
ALLOWED_CONTROLS = {'\t', '\n', '\r'}  # JSON允许的控制字符
HTML_SCRIPT_COMPILE = re.compile(r'<script[^>]*>.*?</script>', re.DOTALL | re.IGNORECASE)
HTML_STYLE_COMPILE = re.compile(r'<style[^>]*>.*?</style>', re.DOTALL | re.IGNORECASE)
HTML_TAGS_COMPILE = re.compile(r'<[^>]+>')
WHITESPACE_COMPILE = re.compile(r'\s+')
SOGOU_URL_PATTERNS = [
    re.compile(r'window\.location\.replace\("([^"]+)"\)'),
    re.compile(r'content="0;URL=\'([^\']+)\'"'),
    re.compile(r'window\.location\.replace\(\'([^\']+)\'\)'),
    re.compile(r'content="0;URL="([^"]+)"'),
    re.compile(r'https?://[^\s"\'<>]+')
]

# 配置统一的日志系统
def setup_logging(logger_name, level=logging.INFO):
    """配置统一的日志系统，确保与 FastMCP 兼容"""
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s] %(lineno)d %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(console_handler)
    
    # 设置 FastMCP 相关日志级别
    logging.getLogger("fastmcp").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sse_starlette").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.CRITICAL)
    
    return logging.getLogger(logger_name)

def load_config(config_path: str):
    """
    加载配置文件
    """
    # 如果是相对路径，确保相对于脚本所在目录
    _config_path = Path(config_path)
    if not _config_path.is_absolute():
        # 获取当前脚本的目录
        current_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _config_path = current_dir / _config_path
    
    if not _config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {_config_path}")
    
    config_text = _config_path.read_text(encoding='utf-8').strip()
    if not config_text:
        return {}
    config = json.loads(config_text)
    return config

# 加载配置
config = load_config('config.json')
config['auth'] = load_config('auth.json')

config_browser = config.get('browser', {})
platform_map = config_browser.get('platform_map', {}) # 发布平台名映射表

# 检查FFmpeg是否已安装
if not ffmpeg_path.exists():
    logger.error(f"FFmpeg未安装，请先安装至: {ffmpeg_dir}")
    raise FileNotFoundError(f"FFmpeg未安装，请先安装至: {ffmpeg_dir}")

ai_helper = AIHelper(config) # 初始化AI助手

def extract_valid_urls(inputs: Union[str, List[str]]) -> Union[str, List[str], None]:
    """从输入中提取有效的URL (Extract valid URLs from input)

    Args:
        inputs (Union[str, list[str]]): 输入的字符串或字符串列表 (Input string or list of strings)

    Returns:
        Union[str, list[str]]: 提取出的有效URL或URL列表 (Extracted valid URL or list of URLs)
    """
    url_pattern = re.compile(r"https?://\S+")

    # 如果输入是单个字符串
    if isinstance(inputs, str):
        match = url_pattern.search(inputs)
        return match.group(0) if match else None

    # 如果输入是字符串列表
    elif isinstance(inputs, list):
        valid_urls = []

        for input_str in inputs:
            matches = url_pattern.findall(input_str)
            if matches:
                valid_urls.extend(matches)

        return valid_urls

def normalizeUrl(url: Optional[str], base_url: str) -> str:
    '''处理并规范化URL
    :param url: 要处理的URL
    :param base_url: 基础URL用于相对路径处理
    :return: 规范化后的完整URL
    '''
    if not url or url == 'None' or url == 'null' or url.startswith('data:'):
        return ''

    # 解码URL，如果包含?（即没有被编码）则无须解码，且避免强制解码会破坏查询参数
    try:
        if '?' not in url:
            url = unquote(url)
        base_url = unquote(base_url)
    except:
        pass

    # 尝试处理base_url传错的情况
    if base_url == url or '?' in base_url:
        base_url = '/'.join(base_url.split('/')[:-1])

    # 处理相对路径中的多个../情况
    while url.startswith('../'):
        url = url[3:]
        base_url = '/'.join(base_url.split('/')[:-1])
    
    # 处理不同格式的URL
    if url.startswith('http://') or url.startswith('https://'):
        return url
    elif url.startswith('//'):
        return f"{urlparse(base_url).scheme}:{url}"
    elif url.startswith('/'):
        return f"{base_url.rstrip('/')}{url}"
    elif not url.startswith(('http', 'www.')):
        return f"{base_url.rstrip('/')}/{url.lstrip('/')}"
    
    return url

def sanitize_text(
    data: Union[str, int, float, bool, bytes, None, Dict[Any, Any], List[Any]]
) -> Union[str, int, float, bool, Dict[Any, Any], List[Any], None]:
    """
    清理文本中的无效UTF-8字符，确保文本可以安全地序列化为JSON
    支持递归处理字典和列表中的所有字符串
    """
    # 栈的类型：三元组为 (当前数据, 父容器, 父容器的键/索引)
    stack: List[
        Tuple[
            Union[str, int, float, bool, bytes, None, Dict[Any, Any], List[Any]],
            Union[Dict[Any, Any], List[Any], None],  # 父容器可为dict/list/None
            Union[str, int, None]  # 键可为str（dict）/int（list）/None
        ]
    ] = [(data, None, None)]  # 初始元素：(根数据, 无父容器, 无键)
    result = None

    while stack:
        current, parent, key = stack.pop()
        processed: Union[str, int, float, bool, Dict[Any, Any], List[Any], None] = None

        if isinstance(current, dict):
            processed = {}
            # 逆序入栈，保证处理顺序与原字典一致
            for k in reversed(current.keys()):
                stack.append((current[k], processed, k))  # 父容器是dict，键是str
        
        elif isinstance(current, list):
            processed = []
            # 逆序入栈，保证列表元素顺序
            for i in reversed(range(len(current))):
                stack.append((current[i], processed, i))  # 父容器是list，键是int
        
        elif current is None:
            processed = None
        
        elif isinstance(current, bytes):
            try:
                processed = current.decode('utf-8')
            except UnicodeDecodeError:
                processed = current.decode('utf-8', errors='ignore')
            stack.append((processed, parent, key))  # 重新入栈处理字符串
            continue
        
        elif isinstance(current, str):
            s = RE_PROXY_CHARS.sub('', current)
            s = ''.join(char for char in s if ord(char) >= 32 or char in ALLOWED_CONTROLS)
            try:
                processed = s.encode('utf-8', errors='ignore').decode('utf-8')
            except:
                processed = ""
        
        else:
            processed = current  # int/float/bool直接返回

        # 将处理结果赋值给父容器
        if parent is not None:
            if isinstance(parent, dict) and isinstance(key, str):
                parent[key] = processed
            elif isinstance(parent, list) and isinstance(key, int):
                parent.insert(key, processed)  # 列表用insert保持顺序
        else:
            result = processed

    return result

def get_date_from_url(url: str) -> str:
    '''从url中提取日期
    :param url: 要处理的URL
    :return: 提取的日期字符串(YYYY-MM-DD格式)或空字符串
    '''
    if url:
        # 移除URL中的协议和域名部分，只保留路径
        parsed_url = urlparse(url).path
        match = DATE4URL_COMPILE.search(parsed_url)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"
    return ''

def url_check(url: str) -> str:
    """
    检查URL是否合法
    :param url: 要检查的URL
    :return: 如果URL合法则返回处理后的url，否则返回空字符串
    """
    try:
        url = url.strip()
        if '\n' in url:
            url = url.split('\n')[0].strip()
        if ' ' in url:
            url = url.split(' ')[0]
        if not url or len(url) < 6:
            logger.error("URL不能为空或不合法")
            return ""
        if url.endswith('#ocr'):
            url = url[:-4]
            # 确保移除#ocr后URL仍然有效
            if not url or len(url) < 6:
                logger.error("URL在移除#ocr后不合法")
                return ""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # 检查域名是否为空
        if not domain:
            logger.error("URL缺少域名部分")
            return ""
        domain_split = domain.split('.')
        if len(domain_split) < 2:
            logger.error("域名格式错误")
            return ""
        return url
    except Exception as e:
        logger.error(f"URL检查时出错: {url}, 错误: {e}")
        return ""

def get_url_hash(url: str) -> str:
    """
    获取URL的哈希值，自动处理特殊url（如小红书链接含xsec_token参数）
    :param url: URL网址
    :return: URL的哈希值（MD5）
    """
    if "xsec_token" in url:
        url_hash = md5(url.split("?")[0].encode()).hexdigest()
    else:
        url_hash = md5(url.encode()).hexdigest()
    return url_hash

def timestamp_to_datetime(timestamp: int) -> str:
    '''将时间戳转换为标准日期时间格式
    :param timestamp: Unix时间戳（秒或毫秒）
    :return: 标准日期时间格式字符串(YYYY-MM-DD HH:MM:SS)
    '''
    if not timestamp or timestamp < 100:
        return ''
        
    # 判断是秒级还是毫秒级时间戳
    if len(str(timestamp)) > 10:  # 毫秒级时间戳
        timestamp = int(str(timestamp)[:10])
    
    # 转换为datetime对象
    try:
        dt = time.localtime(timestamp)
        return time.strftime("%Y-%m-%d %H:%M:%S", dt)
    except:
        return ''

def now2str() -> str:
    # 获取当前日期时间
    current_time = int(time.time())

    # 获取星期几
    weekdays = ['周日', '周一', '周二', '周三', '周四', '周五', '周六']
    weekday_number = int(time.strftime('%w', time.localtime(current_time)))  # 返回 0-6，对应周日到周六

    # 跨平台方式：先获取数值再转为字符串（避免系统差异）
    year = int(time.strftime('%Y', time.localtime(current_time)))
    month = int(time.strftime('%m', time.localtime(current_time)))  # 月份（1-12，无前导零）
    day = int(time.strftime('%d', time.localtime(current_time)))      # 日期（1-31，无前导零）

    # 格式化日期字符串
    formatted_date = f'{year}年{month}月{day}日 {weekdays[weekday_number]} {time.strftime("%H:%M:%S", time.localtime(current_time))}'

    return formatted_date

def get_random_headers(ua: str = '', origin: str = '', referer: str = '', cookie: str = '', keep_alive: bool = False) -> dict[str, str]:
    """
    获取随机请求头
    :param ua: 用户代理字符串
    :param origin: 来源域名
    :param referer: 来源页面URL
    :param cookie: Cookie字符串
    :param keep_alive: 是否保持连接
    :return: 请求头字典
    """
    try:
        user_agent = ua if ua else fake_ua.random
    except Exception:
        # 若获取随机用户代理失败，使用默认值
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0'
    headers = {
        'User-Agent': user_agent,
        'Accept': 'application/json, text/html, application/x-www-form-urlencoded, application/xhtml+xml, application/xml;q=0.9, image/webp, image/*, */*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2'
    }
    if origin:
        headers['Origin'] = origin
    if referer:
        headers['Referer'] = referer
    if cookie:
        headers['Cookie'] = cookie
    if keep_alive:
        headers['Connection'] = 'keep-alive'
    return headers

def get_site_name(url: str, unmap: dict = {}) -> str:
    """
    获取网站本项目中定义的名称
    :param url: 要分析的URL
    :param unmap: 网站名称映射表
    :return: 网站名称，失败返回空字符串
    """
    domain = urlparse(url).netloc
    if not unmap:
        unmap = platform_map
    site_name = unmap.get(domain, "")
    if site_name:
        return site_name
    domain_split = domain.split('.')
    if len(domain_split) < 3:
        return ""
    for i in range(len(domain_split)-1, 1, -1):
        domain_last_seconds = '.'.join(domain_split[-i:])
        site_name = unmap.get(domain_last_seconds, "")
        if site_name:
            return site_name
    return ""

def fetch_json_data(url, headers=None):
    if not headers:
        origin = urlparse(url).netloc
        headers = get_random_headers(origin=origin, referer=origin)
    try:
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
        json_data = response.json()
        json_data = cast(dict, sanitize_text(json_data))
        return json_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching JSON data: {e}")
    return None

def request_with_retry(url: str, post_data: dict|None = None, headers=None, encoding: str = 'utf-8', isjson: bool = False, max_retries: int = 3, detect_encoding: bool = True) -> Optional[dict[str, Any]]:
    """
    带重试和防封策略的请求方法
    :param url: 请求的URL
    :param post_data: POST请求的数据，默认为None
    :param headers: 请求头，默认为None
    :param encoding: 响应编码，默认为'utf-8'
    :param isjson: 是否返回json格式，默认为False
    :param max_retries: 最大重试次数，默认为3
    :param detect_encoding: 是否自动检测编码，默认为True
    :return: 访问后的url和响应文本的dict，或None
    """
    for attempt in range(max_retries):
        try:
            if not headers:
                headers = get_random_headers(referer=f'{urlparse(url).scheme}://{urlparse(url).netloc}/', keep_alive=True)
            if post_data:
                response = requests.post(url, headers=headers, data=post_data, timeout=20, allow_redirects=True, verify=False)
            else:
                response = requests.get(url, headers=headers, timeout=20, allow_redirects=True, verify=False)
            
            if response.status_code == 404:
                logger.debug(f"Page not found: {url}")
                return None
            
            response.raise_for_status()
            
            # 智能编码处理
            restext = None
            if detect_encoding:
                try:
                    # 尝试自动检测编码
                    detected = chardet.detect(response.content)
                    if detected['encoding']:
                        response.encoding = detected['encoding']
                        restext = response.text.strip()
                except Exception as e:
                    logger.warning(f"编码检测失败，使用默认编码: {e}")
            
            # 如果检测失败，使用默认编码
            if not restext:
                response.encoding = encoding
                restext = response.text.strip()
            
            if not restext:
                logger.warning(f"Empty response content: {url}")
                continue
            if isjson:
                res = response.json()
            else:
                res = restext
            return {
                "url": response.url,
                "data": sanitize_text(res)
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                # 指数退避策略
                sleep_time = random.uniform(2, 5) * (attempt + 1)
                time.sleep(sleep_time)
    logger.warning(f"Max retries reached for {url}")
    return None

def get_screenshot(filename: str, include_invalid: bool = False) -> bytes:
    """
    获取已存档的截图
    :param filename: 截图文件名
    :param include_invalid: 是否包含无效截图，默认为False
    :return: 截图字节流
    """
    if include_invalid:
        invalid_filename = f"invalid_{filename}"
        if (filepath := screenshots_dir / invalid_filename).exists():
            return filepath.read_bytes()
    if (filepath := screenshots_dir / filename).exists():
        return filepath.read_bytes()
    elif (filepath := screenshots_orig_dir / filename).exists():
        return filepath.read_bytes()
    else:
        return b''

def get_file_type(file_path: Union[str, Path]) -> str:
    """
    根据文件名后缀或data:base64格式字符串判断文件类型
    
    Args:
        file_path: 文件路径、文件名或data:base64格式字符串
        
    Returns:
        str: 文件类型，可能的值包括：
            - "image": 图片文件
            - "video": 视频文件
            - "audio": 音频文件
            - "document": 文档文件
            - "spreadsheet": 表格文件
            - "presentation": 演示文稿文件
            - "archive": 压缩文件
            - "code": 代码文件
            - "text": 文本文件
            - "unknown": 未知类型
            
    Examples:
        >>> get_file_type("image.jpg")
        'image'
        >>> get_file_type("video.mp4")
        'video'
        >>> get_file_type("document.pdf")
        'document'
        >>> get_file_type("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...")
        'image'
    """
    if not file_path:
        return "unknown"
    
    # 处理data:base64格式字符串
    if isinstance(file_path, str) and file_path.startswith("data:"):
        # 提取MIME类型
        # 格式: data:[<mediatype>][;base64],<data>
        try:
            # 找到MIME类型的结束位置
            comma_index = file_path.find(',')
            if comma_index == -1:
                return "unknown"
            
            # 提取MIME类型部分
            mime_part = file_path[5:comma_index]  # 去掉"data:"前缀
            
            # 检查是否包含base64标记
            if ';base64' in mime_part:
                mime_type = mime_part.split(';base64')[0]
            else:
                mime_type = mime_part
            
            # 根据MIME类型判断文件类型
            if mime_type.startswith('image/'):
                return "image"
            elif mime_type.startswith('video/'):
                return "video"
            elif mime_type.startswith('audio/'):
                return "audio"
            elif mime_type in ['application/pdf', 'application/msword', 
                              'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                              'application/rtf', 'text/rtf', 'application/vnd.oasis.opendocument.text']:
                return "document"
            elif mime_type in ['application/vnd.ms-excel', 
                              'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                              'text/csv', 'application/vnd.oasis.opendocument.spreadsheet']:
                return "spreadsheet"
            elif mime_type in ['application/vnd.ms-powerpoint', 
                              'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                              'application/vnd.oasis.opendocument.presentation']:
                return "presentation"
            elif mime_type in ['application/zip', 'application/x-rar-compressed', 
                              'application/x-7z-compressed', 'application/gzip', 
                              'application/x-tar', 'application/x-bzip2']:
                return "archive"
            elif mime_type.startswith('text/'):
                return "text"
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    # 获取文件扩展名
    path = Path(file_path) if isinstance(file_path, str) else file_path
    extension = path.suffix.lower()
    
    # 定义各类型文件的扩展名
    image_extensions = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', 
        '.tiff', '.tif', '.ico', '.psd', '.raw', '.heic', '.avif'
    }
    
    video_extensions = {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', 
        '.m4v', '.3gp', '.ogv', '.ts', '.mts', '.m2ts'
    }
    
    audio_extensions = {
        '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', 
        '.opus', '.aiff', '.au', '.ra', '.amr'
    }
    
    document_extensions = {
        '.pdf', '.doc', '.docx', '.rtf', '.odt', '.txt', '.md', 
        '.tex', '.pages', '.epub', '.mobi', '.azw', '.azw3'
    }
    
    spreadsheet_extensions = {
        '.xls', '.xlsx', '.csv', '.ods', '.numbers', '.tsv'
    }
    
    presentation_extensions = {
        '.ppt', '.pptx', '.odp', '.key', '.pps', '.ppsx'
    }
    
    archive_extensions = {
        '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', 
        '.tar.gz', '.tar.bz2', '.tar.xz', '.deb', '.rpm', '.dmg'
    }
    
    code_extensions = {
        '.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', 
        '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', 
        '.sh', '.bat', '.ps1', '.sql', '.xml', '.json', '.yaml', '.yml'
    }
    
    text_extensions = {
        '.txt', '.log', '.ini', '.cfg', '.conf', '.toml', '.readme'
    }
    
    # 判断文件类型
    if extension in image_extensions:
        return "image"
    elif extension in video_extensions:
        return "video"
    elif extension in audio_extensions:
        return "audio"
    elif extension in document_extensions:
        return "document"
    elif extension in spreadsheet_extensions:
        return "spreadsheet"
    elif extension in presentation_extensions:
        return "presentation"
    elif extension in archive_extensions:
        return "archive"
    elif extension in code_extensions:
        return "code"
    elif extension in text_extensions:
        return "text"
    else:
        return "unknown"

def _detect_mime_type_from_bytes(file_bytes: bytes) -> Optional[str]:
    """
    从文件字节中检测MIME类型
    
    Args:
        file_bytes: 文件字节数据
        
    Returns:
        Optional[str]: 检测到的MIME类型，如果无法检测则返回None
    """
    if not file_bytes or len(file_bytes) < 12:
        return None
    
    # 常见文件头的魔数
    file_signatures = {
        b'\xFF\xD8\xFF': 'image/jpeg',  # JPEG
        b'\x89PNG\r\n\x1a\n': 'image/png',  # PNG
        b'GIF87a': 'image/gif',  # GIF87a
        b'GIF89a': 'image/gif',  # GIF89a
        b'WEBP': 'image/webp',  # WebP
        b'RIFF': 'video/webm',  # WebM (需要进一步检查)
        b'\x00\x00\x00\x18ftypmp4': 'video/mp4',  # MP4
        b'\x00\x00\x00\x1Cftypisom': 'video/mp4',  # MP4
        b'ftyp': 'video/mp4',  # MP4 (简化检查)
        b'FLV': 'video/x-flv',  # FLV
        b'\x1A\x45\xDF\xA3': 'video/x-matroska',  # MKV
        b'RIFF\x00\x00\x00\x00AVI ': 'video/x-msvideo',  # AVI
        b'ftypqt': 'video/quicktime',  # MOV
    }
    
    # 检查文件头
    for signature, mime_type in file_signatures.items():
        if file_bytes.startswith(signature):
            return mime_type
    
    return None

def mime_type2ext(mime_type: str) -> str:
    '''根据MIME类型获取文件扩展名
    :param mime_type: MIME类型
    :return: 文件扩展名
    '''
    # MIME类型到扩展名的映射表
    mime_to_ext = {
        # 图片类型
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp',
        'image/svg+xml': '.svg',
        'image/bmp': '.bmp',
        'image/tiff': '.tiff',
        'image/x-icon': '.ico',
        'image/vnd.microsoft.icon': '.ico',
        'image/x-jpeg': '.jpg',
        'image/pjpeg': '.jpg',
        'image/jfif': '.jpg',
        'image/x-png': '.png',
        'image/x-ms-bmp': '.bmp',
        'image/heic': '.heic',
        'image/heif': '.heif',
        
        # 音频类型
        'audio/mpeg': '.mp3',
        'audio/mp3': '.mp3',
        'audio/wav': '.wav',
        'audio/x-wav': '.wav',
        'audio/ogg': '.ogg',
        'audio/midi': '.mid',
        'audio/x-midi': '.mid',
        'audio/aac': '.aac',
        'audio/flac': '.flac',
        'audio/x-flac': '.flac',
        'audio/mp4': '.m4a',
        'audio/x-m4a': '.m4a',
        
        # 视频类型
        'video/mp4': '.mp4',
        'video/mpeg': '.mpeg',
        'video/quicktime': '.mov',
        'video/x-msvideo': '.avi',
        'video/x-ms-wmv': '.wmv',
        'video/webm': '.webm',
        'video/3gpp': '.3gp',
        'video/x-flv': '.flv',
        'video/x-matroska': '.mkv',
        
        # 文档类型
        'application/pdf': '.pdf',
        'application/msword': '.doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.ms-excel': '.xls',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'application/vnd.ms-powerpoint': '.ppt',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
        'application/rtf': '.rtf',
        'text/rtf': '.rtf',
        
        # 压缩文件
        'application/zip': '.zip',
        'application/x-rar-compressed': '.rar',
        'application/x-7z-compressed': '.7z',
        'application/x-tar': '.tar',
        'application/gzip': '.gz',
        'application/x-gzip': '.gz',
        
        # 文本类型
        'text/plain': '.txt',
        'text/html': '.html',
        'text/css': '.css',
        'text/javascript': '.js',
        'text/csv': '.csv',
        'text/xml': '.xml',
        'text/markdown': '.md',
        
        # 应用程序类型
        'application/javascript': '.js',
        'application/x-javascript': '.js',
        'application/json': '.json',
        'application/xml': '.xml',
        'application/atom+xml': '.atom',
        'application/rss+xml': '.rss',
        'application/xhtml+xml': '.xhtml',
        'application/octet-stream': '.bin',
        
        # 字体类型
        'font/woff': '.woff',
        'font/woff2': '.woff2',
        'font/ttf': '.ttf',
        'font/otf': '.otf',
        'application/font-woff': '.woff',
        'application/font-woff2': '.woff2',
        'application/x-font-ttf': '.ttf',
        'application/x-font-otf': '.otf',
    }
    
    mime_type = mime_type.lower().strip()
    
    # 直接查找映射表
    if mime_type in mime_to_ext:
        return mime_to_ext[mime_type]
    
    # 如果没找到，尝试简单的处理
    if '/' in mime_type:
        ext = '.' + mime_type.split('/')[-1]
    else:
        ext = '.' + mime_type
    
    # 处理一些特殊的MIME类型
    if ext in ['.jpeg', '.jfif']:
        ext = '.jpg'
    elif ext == '.x-javascript':
        ext = '.js'
    elif ext == '.plain':
        ext = '.txt'
    
    return ext

def ext2mime_type(ext: str) -> str:
    '''根据文件扩展名获取MIME类型
    :param ext: 文件扩展名（带点或不带点均可）
    :return: MIME类型
    '''
    # 扩展名到MIME类型的映射表
    ext_to_mime = {
        # 图片类型
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.jfif': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.ico': 'image/x-icon',
        
        # 音频类型
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
        '.mid': 'audio/midi',
        '.midi': 'audio/midi',
        '.aac': 'audio/aac',
        '.flac': 'audio/flac',
        '.m4a': 'audio/mp4',
        
        # 视频类型
        '.mp4': 'video/mp4',
        '.mpeg': 'video/mpeg',
        '.mpg': 'video/mpeg',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.wmv': 'video/x-ms-wmv',
        '.webm': 'video/webm',
        '.3gp': 'video/3gpp',
        '.flv': 'video/x-flv',
        '.mkv': 'video/x-matroska',
        
        # 文档类型
        '.pdf': 'application/pdf',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xls': 'application/vnd.ms-excel',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.ppt': 'application/vnd.ms-powerpoint',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.rtf': 'application/rtf',
        
        # 压缩文件
        '.zip': 'application/zip',
        '.rar': 'application/x-rar-compressed',
        '.7z': 'application/x-7z-compressed',
        '.tar': 'application/x-tar',
        '.gz': 'application/gzip',
        
        # 文本类型
        '.txt': 'text/plain',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.css': 'text/css',
        '.js': 'text/javascript',
        '.csv': 'text/csv',
        '.xml': 'text/xml',
        '.md': 'text/markdown',
        
        # 应用程序类型
        '.json': 'application/json',
        '.atom': 'application/atom+xml',
        '.rss': 'application/rss+xml',
        '.xhtml': 'application/xhtml+xml',
        '.bin': 'application/octet-stream',
        
        # 字体类型
        '.woff': 'font/woff',
        '.woff2': 'font/woff2',
        '.ttf': 'font/ttf',
        '.otf': 'font/otf',
    }
    
    ext = ext.lower().strip()
    
    # 确保扩展名以点开头
    if not ext.startswith('.'):
        ext = '.' + ext
    
    # 查找映射表
    if ext in ext_to_mime:
        return ext_to_mime[ext]
    
    # 如果没找到，返回默认值
    return 'application/octet-stream'

def file_to_base64(file_input: Union[str, Path, bytes], mime_type: Optional[str] = None) -> str:
    """
    将图片或视频的Path或bytes转换为标准的base64格式字符串
    
    Args:
        file_input: 文件输入，可以是文件路径(str/Path)或字节数据(bytes)
        mime_type: 可选的MIME类型，如果不提供则根据文件扩展名自动推断
        
    Returns:
        str: 标准base64格式字符串，格式为 "data:[mime_type];base64,[base64_data]"
        
    Examples:
        >>> # 从文件路径转换
        >>> base64_str = file_to_base64("path/to/image.jpg")
        >>> # 从Path对象转换
        >>> base64_str = file_to_base64(Path("path/to/video.mp4"))
        >>> # 从字节数据转换
        >>> base64_str = file_to_base64(image_bytes, "image/jpeg")
    """
    try:
        # 处理输入数据
        if isinstance(file_input, (str, Path)):
            # 如果是文件路径，读取文件内容
            file_path = Path(file_input)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            file_path = compress_media_file(file_path)
            
            file_bytes = file_path.read_bytes()
            
            # 如果没有提供MIME类型，根据文件扩展名推断
            if mime_type is None:
                mime_type = ext2mime_type(file_path.suffix)

        elif isinstance(file_input, bytes):
            # 如果是字节数据，直接使用
            file_bytes = file_input
            
            # 如果没有提供MIME类型，尝试通过文件头推断
            if mime_type is None:
                mime_type = _detect_mime_type_from_bytes(file_bytes)
        else:
            raise TypeError(f"不支持的输入类型: {type(file_input)}")
        
        # 如果仍然没有MIME类型，使用通用二进制类型
        if mime_type is None:
            mime_type = "application/octet-stream"
        
        # 编码为base64
        base64_data = b64encode(file_bytes).decode('utf-8')
        
        # 返回标准base64格式字符串
        return f"data:{mime_type};base64,{base64_data}"
        
    except Exception as e:
        logger.error(f"转换文件为base64失败: {e}")
        raise

def file_to_url(attachment: Union[str, Path, bytes], mime_type: Optional[str] = None) -> str:
    """
    将文件路径转换为URL
    :param attachment: 本地文件路径或URL
    :param mime_type: 可选的MIME类型，如果不提供则根据文件扩展名自动推断
    :return: URL
    """
    if isinstance(attachment, str) and (attachment.startswith("http") or attachment.startswith("data:")):
        return attachment
    else:
        return file_to_base64(attachment, mime_type)

def file_download_temp(file_url: str, file_name: str='') -> Union[Path, None]:
    """
    下载文件到临时目录
    :param file_url: 文件URL
    :param file_name: 可选，文件名
    :return: 临时文件路径或None
    """
    try:
        if file_url.startswith("data:"):
            # 处理base64编码的文件
            ext = file_url.split(";")[0].split("/")[-1]
            base64_data = file_url.split(",")[1]
            file_bytes = b64decode(base64_data)
            temp_file_path = temp_dir / file_name or f"{uuid4()}.{ext}"
            with open(temp_file_path, 'wb') as f:
                f.write(file_bytes)
            return temp_file_path
        else:
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            # 获取文件名
            file_name = file_name or os.path.basename(file_url)
            temp_file_path = temp_dir / file_name
            # 写入文件
            with open(temp_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return temp_file_path
    except Exception as e:
        logger.error(f"下载文件到临时目录时出错: {e}")
        return None

async def pack_files_with_data(data: dict) -> bytes | None:
    """
    给出分析结果数据，其已存档的附件连同信息表打包成zip文件
    :param data: 分析结果数据
    :return: zip压缩文件字节流
    """
    # 添加超时保护，限制处理时间
    start_time = time.time()
    max_processing_time = 30  # 30秒超时
    try:
        if not data:
            logger.warning(f"分析结果数据为空")
            return None
        
        source_url = data.get('source_url', '')
        
        # 将检测记录生成Excel表格
        # 处理JSON字段
        excel_map = {
            "source_url": "URL",
            "publisher_id": "发布者ID",
            "publisher_nickname": "发布者昵称",
            "publish_platform": "发布平台",
            "publish_time": "发布时间",
            "content": "信息内容",
            "presentation_form": "内容形式",
            "multimodel_desc": "多模态描述内容",
            "multimodel_text": "多模态文本内容",
            "speech_text": "语音转录文本内容",
            "is_aigc": "疑似AI生成",
            "ip_region": "IP属地",
            "publisher_attribute": "发布者属性",
            "publisher_signature": "发布者签名",
            "publisher_stats": "发布者热度",
            "content_stats": "内容热度",
            "labels": "标签",
            "grading": "分级",
            "classification": "归类",
            "is_illegal": "是否违法",
            "reasoning": "分析依据",
            "suggestions": "处置建议",
            "law_ref": "法律文书、案例参考",
            "rumor_detection_conclusion": "谣言检测结论",
            "ref_urls": "谣言检测参考链接",
            "scrape_time": "检测时间",
        }
        _record = cast(dict, sanitize_text(data))
        processed_record = {}
        # 优化Excel字段处理 - 减少嵌套循环
        for key, value in excel_map.items():
            if time.time() - start_time > max_processing_time:
                logger.warning(f"处理超时，跳过剩余字段处理")
                break
                
            if key in _record and _record[key] is not None:
                if key == 'publish_time':
                    processed_record['文本类型'] = '正文'
                
                if key in ["is_aigc", "is_illegal"]:
                    processed_record[value] = "是" if _record[key] else "否"
                elif key in ['publish_time', 'scrape_time']:
                    # 处理时间数据
                    if isinstance(_record[key], (int, float)):
                        processed_record[value] = datetime.fromtimestamp(_record[key]).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        processed_record[value] = str(_record[key])
                else:
                    processed_record[value] = _record[key]

        # 获取URL相关的截图和附件
        url_hash = get_url_hash(source_url)
        logger.debug(f"开始处理URL {source_url} 的截图和附件，哈希: {url_hash}")
        
        # 检查处理时间
        if time.time() - start_time > max_processing_time:
            logger.warning(f"处理超时，跳过截图和附件处理")
            return None
            
        # 获取截图
        screenshot_filename = url_hash + '.webp'
        screenshot_data = get_screenshot(screenshot_filename, include_invalid=False)
        publisher_screenshot_filename = ''
        publisher_screenshot_data = None
        
        if 'publisher_page_url' in _record and _record['publisher_page_url']:
            # 检查处理时间
            if time.time() - start_time > max_processing_time:
                logger.warning(f"处理超时，跳过发布者截图处理")
            else:
                author_url_hash = get_url_hash(_record['publisher_page_url'])
                publisher_screenshot_filename = author_url_hash + '.webp'
                publisher_screenshot_data = get_screenshot(publisher_screenshot_filename, include_invalid=False)
        
        # 检查处理时间
        if time.time() - start_time > max_processing_time:
            logger.warning(f"处理超时，跳过ZIP打包")
            return None
            
        # 将Excel表格、截图和附件打包成ZIP文件
        logger.debug(f"开始创建ZIP文件...")
        zip_buffer = BytesIO()
        
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # 添加截图（如果存在）
                if screenshot_data:
                    zip_file.writestr(screenshot_filename, screenshot_data)
                    processed_record['网页截图名'] = url_hash + '.webp'
                    logger.debug(f"添加网页截图: {screenshot_filename}")
                    
                if publisher_screenshot_data:
                    zip_file.writestr(publisher_screenshot_filename, publisher_screenshot_data)
                    processed_record['账号截图名'] = publisher_screenshot_filename
                    logger.debug(f"添加账号截图: {publisher_screenshot_filename}")
                
                # 检查处理时间
                if time.time() - start_time > max_processing_time:
                    logger.warning(f"处理超时，跳过Excel生成")
                    return None
                    
                # 创建DataFrame
                logger.debug(f"开始生成Excel文件...")
                df = pd.DataFrame([processed_record])
                
                # 将DataFrame保存到内存中的Excel文件
                excel_buffer = BytesIO()
                
                try:
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='检测记录', index=False)
                        logger.debug("Excel数据写入完成")
                        
                        # 获取工作簿和工作表对象
                        workbook = writer.book
                        worksheet = writer.sheets['检测记录']
                        
                        # 设置列宽
                        column_widths = {
                            'A': 30,  # URL
                            'B': 15,  # 发布者ID
                            'C': 20,  # 发布者昵称
                            'D': 15,  # 发布平台
                            'E': 20,  # 发布时间
                            'F': 50,  # 信息内容
                            'G': 15,  # 内容形式
                            'H': 50,  # 多模态分析结果
                            'I': 15,  # 疑似AI生成
                            'J': 15,  # IP属地
                            'K': 15,  # 发布者属性
                            'L': 30,  # 发布者签名
                            'M': 20,  # 发布者热度
                            'N': 20,  # 内容热度
                            'O': 20,  # 标签
                            'P': 10,  # 分级
                            'Q': 15,  # 归类
                            'R': 15,  # 是否违法
                            'S': 50,  # 分析依据
                            'T': 50,  # 处置建议
                            'U': 30,  # 法律性参考
                            'V': 30,  # 谣言检测结论
                            'W': 50,  # 谣言检测参考链接
                            'X': 20,  # 检测时间
                            'Y': 30,  # 网页截图名
                            'Z': 30,  # 账号截图名
                            'AA': 50, # 媒体文件名
                        }
                        
                        # 应用列宽设置
                        for col, width in column_widths.items():
                            worksheet.set_column(f'{col}:{col}', width)
                        
                        # 设置标题行格式
                        header_format = workbook.add_format({
                            'bold': True,
                            'text_wrap': True,
                            'valign': 'top',
                            'fg_color': '#D7E4BC',
                            'border': 1
                        })
                        
                        # 应用标题行格式
                        for col_num, value in enumerate(df.columns.values):
                            worksheet.write(0, col_num, value, header_format)
                            
                    excel_buffer.seek(0)
                    # 添加Excel文件
                    zip_file.writestr(f'检测记录_{url_hash}.xlsx', excel_buffer.getvalue())
                    logger.debug(f"Excel文件已添加到ZIP: 检测记录_{url_hash}.xlsx")
                    
                except Exception as excel_error:
                    logger.error(f"生成Excel文件时出错: {excel_error}")
                    # 即使Excel生成失败，也继续处理其他文件
                    pass
        
        except Exception as zip_error:
            logger.error(f"创建ZIP文件时出错: {zip_error}")
            return None
        
        zip_buffer.seek(0)
        
        # 检查最终处理时间
        total_time = time.time() - start_time
        logger.debug(f"ZIP文件创建完成，总处理时间: {total_time:.2f}秒")
        
        # 如果处理时间超过警告阈值，记录警告
        if total_time > 20:  # 超过20秒
            logger.warning(f"处理时间较长: {total_time:.2f}秒，数据量可能较大")
        
        # 获取ZIP文件大小并记录
        zip_data = zip_buffer.getvalue()
        zip_size = len(zip_data)
        logger.debug(f"ZIP文件大小: {zip_size} 字节 ({zip_size/1024/1024:.2f} MB)")
        
        # 检查ZIP文件大小是否过大
        if zip_size > 100 * 1024 * 1024:  # 超过100MB
            logger.warning(f"ZIP文件过大: {zip_size/1024/1024:.2f} MB，可能导致返回问题")
        
        return zip_data
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"打包文件时发生错误: {e}，处理时间: {total_time:.2f}秒")
        return None

def compress_media_file(file_path: Union[str, Path]) -> Path:
    """
    压缩媒体文件以适应大模型分析需求
    
    Args:
        file_path (Union[str, Path]): 媒体文件路径
        
    Returns:
        Path: 压缩后的文件路径，如果不需要压缩或压缩失败则返回原路径
        
    Note:
        - 图片超过5MB将压缩至5MB以内
        - 视频超过50MB将通过降低帧率(1fps)、分辨率(≤720p)等方式压缩至50MB以内
        - 音频超过5MB将通过调整比特率、采样率等方式压缩
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
        
    if not file_path.exists():
        logger.warning(f"文件不存在: {file_path}")
        return file_path
        
    # 获取文件类型
    file_type = get_file_type(file_path)
    
    # 获取文件大小(MB)
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    
    # 根据文件类型和大小决定是否压缩
    if file_type == "image" and file_size_mb > 5:
        return _compress_image(file_path)
    elif file_type == "video" and file_size_mb > 50:
        return _compress_video(file_path)
    elif file_type == "audio" and file_size_mb > 5:
        return _compress_audio(file_path)
    else:
        # 不需要压缩
        return file_path

def _compress_image(image_path: Path) -> Path:
    """
    压缩图片至5MB以内
    
    Args:
        image_path (Path): 图片文件路径
        
    Returns:
        Path: 压缩后的图片路径
    """
    try:
        # 创建临时文件路径
        temp_path = temp_dir / f"compressed_{image_path.stem}.webp"
        
        # 使用PIL打开图片
        with Image.open(image_path) as img:
            # 保存为WebP格式，初始质量设为80
            quality = 80
            img.save(temp_path, "WEBP", quality=quality)
            
            # 检查压缩后大小
            temp_size_mb = temp_path.stat().st_size / (1024 * 1024)
            
            # 如果仍然大于5MB，继续降低质量
            while temp_size_mb > 5 and quality > 10:
                quality -= 10
                img.save(temp_path, "WEBP", quality=quality)
                temp_size_mb = temp_path.stat().st_size / (1024 * 1024)
                
            # 如果质量降到最低仍然大于5MB，尝试缩小尺寸
            if temp_size_mb > 5:
                # 计算缩放比例
                width, height = img.size
                scale_factor = min(5.0 / temp_size_mb, 0.8)  # 最多缩小到80%
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # 缩放并保存
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_img.save(temp_path, "WEBP", quality=80)
                
        logger.info(f"图片压缩完成: {image_path} -> {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"压缩图片失败: {e}")
        return image_path

def _compress_video(video_path: Path) -> Path:
    """
    压缩视频至50MB以内
    
    Args:
        video_path (Path): 视频文件路径
        
    Returns:
        Path: 压缩后的视频路径
    """
    try:
        # 创建临时文件路径
        temp_path = temp_dir / f"compressed_{video_path.stem}.mp4"
        
        # 第一次压缩：降低帧率到1fps，分辨率限制在720p
        cmd = [
            ffmpeg_path, "-i", str(video_path.absolute()),
            "-vf", "scale=min(iw\\,1280):min(ih\\,720)",
            "-r", "1",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-c:a", "aac",
            "-b:a", "128k",
            "-y", str(temp_path.absolute())
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode != 0:
            logger.error(f"视频压缩失败: {result.stderr}")
            return video_path
            
        # 检查压缩后大小
        temp_size_mb = temp_path.stat().st_size / (1024 * 1024)
        
        # 如果仍然大于50MB，进一步压缩
        if temp_size_mb > 50:
            # 提高CRF值(降低质量)
            for crf in [32, 36, 40]:
                temp_path2 = temp_dir / f"compressed_{video_path.stem}_crf{crf}.mp4"
                cmd = [
                    ffmpeg_path, "-i", str(video_path.absolute()),
                    "-vf", "scale=min(iw\\,1280):min(ih\\,720)",
                    "-r", "1",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", str(crf),
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-y", str(temp_path2.absolute())
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                if result.returncode == 0:
                    temp_size_mb = temp_path2.stat().st_size / (1024 * 1024)
                    if temp_size_mb <= 50:
                        temp_path = temp_path2
                        break
                        
            # 如果仍然大于50MB，尝试裁剪视频
            if temp_size_mb > 50:
                # 获取视频时长
                cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                if result.returncode == 0:
                    duration = float(result.stdout.strip())
                    # 只保留前30秒
                    if duration > 30:
                        temp_path3 = temp_dir / f"compressed_{video_path.stem}_trim.mp4"
                        cmd = [
                            ffmpeg_path, "-i", str(video_path.absolute()),
                            "-t", "30",
                            "-vf", "scale=min(iw\\,1280):min(ih\\,720)",
                            "-r", "1",
                            "-c:v", "libx264",
                            "-preset", "fast",
                            "-crf", "28",
                            "-c:a", "aac",
                            "-b:a", "128k",
                            "-y", str(temp_path3.absolute())
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                        if result.returncode == 0:
                            temp_size_mb = temp_path3.stat().st_size / (1024 * 1024)
                            if temp_size_mb <= 50:
                                temp_path = temp_path3
        
        logger.info(f"视频压缩完成: {video_path} -> {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"压缩视频失败: {e}")
        return video_path

def _compress_audio(audio_path: Path) -> Path:
    """
    压缩音频至5MB以内
    
    Args:
        audio_path (Path): 音频文件路径
        
    Returns:
        Path: 压缩后的音频路径
    """
    original_audio_path = audio_path  # 保存原始路径，用于清理
    webm_temp_path = None  # 用于跟踪临时webm文件
    temp_files_to_delete = []  # 记录需要删除的临时文件
    
    try:
        # 检查原始文件是否为webm格式
        is_webm = audio_path.suffix.lower() == '.webm'
        
        # 如果不是webm格式，先转换为webm格式
        if not is_webm:
            webm_temp_path = temp_dir / f"webm_{audio_path.stem}.webm"
            temp_files_to_delete.append(webm_temp_path)  # 记录临时文件
            cmd = [
                ffmpeg_path, "-i", str(audio_path.absolute()),
                "-c:a", "libopus",  # 使用opus编码器，webm的推荐音频编码
                "-y", str(webm_temp_path.absolute())
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                logger.error(f"音频转换为webm格式失败: {result.stderr}")
                # 转换失败
                webm_temp_path = None
            else:
                # 转换成功，使用webm文件作为输入
                audio_path = webm_temp_path
        
        # 检查当前文件大小
        current_size_mb = audio_path.stat().st_size / (1024 * 1024)
        
        # 如果已经小于5MB，直接返回
        if current_size_mb <= 5:
            logger.info(f"音频大小符合要求({current_size_mb:.2f}MB)，无需压缩: {audio_path}")
            # 清理临时文件（除了返回的文件）
            for temp_file in temp_files_to_delete:
                if temp_file and temp_file != audio_path and temp_file.exists():
                    temp_file.unlink(True)
            return audio_path
        
        # 创建压缩后的临时文件路径
        temp_path = temp_dir / f"compressed_{audio_path.stem}.webm"
        temp_files_to_delete.append(temp_path)  # 记录临时文件
        
        # 第一次压缩：降低比特率到64kbps
        cmd = [
            ffmpeg_path, "-i", str(audio_path.absolute()),
            "-c:a", "libopus",  # 使用opus编码器
            "-b:a", "64k",
            "-ar", "16000",
            "-ac", "1",
            "-y", str(temp_path.absolute())
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode != 0:
            logger.error(f"音频压缩失败: {result.stderr}")
            # 清理临时文件
            for temp_file in temp_files_to_delete:
                if temp_file and temp_file.exists():
                    temp_file.unlink(True)
            return original_audio_path
            
        # 检查压缩后大小
        temp_size_mb = temp_path.stat().st_size / (1024 * 1024)
        
        # 如果仍然大于5MB，进一步降低比特率
        if temp_size_mb > 5:
            for bitrate in ["48k", "32k", "24k", "16k"]:
                temp_path2 = temp_dir / f"compressed_{audio_path.stem}_{bitrate}.webm"
                temp_files_to_delete.append(temp_path2)  # 记录临时文件
                cmd = [
                    ffmpeg_path, "-i", str(audio_path.absolute()),
                    "-c:a", "libopus",  # 使用opus编码器
                    "-b:a", bitrate,
                    "-ar", "16000",
                    "-ac", "1",
                    "-y", str(temp_path2.absolute())
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                if result.returncode == 0:
                    temp_size_mb = temp_path2.stat().st_size / (1024 * 1024)
                    if temp_size_mb <= 5:
                        temp_path = temp_path2
                        break
                        
            # 如果仍然大于5MB，尝试裁剪音频
            if temp_size_mb > 5:
                # 获取音频时长
                cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                if result.returncode == 0:
                    duration = float(result.stdout.strip())
                    # 只保留前60秒
                    if duration > 60:
                        temp_path3 = temp_dir / f"compressed_{audio_path.stem}_trim.webm"
                        temp_files_to_delete.append(temp_path3)  # 记录临时文件
                        cmd = [
                            ffmpeg_path, "-i", str(audio_path.absolute()),
                            "-t", "60",
                            "-c:a", "libopus",  # 使用opus编码器
                            "-b:a", "64k",
                            "-ar", "16000",
                            "-ac", "1",
                            "-y", str(temp_path3.absolute())
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                        if result.returncode == 0:
                            temp_size_mb = temp_path3.stat().st_size / (1024 * 1024)
                            if temp_size_mb <= 5:
                                temp_path = temp_path3
        
        logger.info(f"音频压缩完成: {original_audio_path} -> {temp_path}")
        
        # 清理临时文件（除了返回的文件）
        for temp_file in temp_files_to_delete:
            if temp_file and temp_file != temp_path and temp_file.exists():
                temp_file.unlink(True)
                
        return temp_path
        
    except Exception as e:
        logger.error(f"压缩音频失败: {e}")
        # 清理所有临时文件
        for temp_file in temp_files_to_delete:
            if temp_file and temp_file.exists():
                temp_file.unlink(True)
        return original_audio_path

def process_sample_for_prompt(sample_text: str, escape_n: bool = True) -> str:
    """
    处理样本文本，根据参数决定是否转义换行符
    
    Args:
        sample_text (str): 原始样本文本
        escape_n (bool): 是否转义换行符，默认为True
        
    Returns:
        str: 处理后的文本
        
    Examples:
        >>> process_sample_for_prompt("这是第一行\n这是第二行")
        '这是第一行\\n这是第二行'
    """
    if not sample_text:
        return ""
    
    processed_text = sample_text.strip()
    processed_text = processed_text.replace('\r\n', '\n')
    processed_text = processed_text.replace('\r', '\n')
    processed_text = processed_text.replace('\n\n', '\n')
    processed_text = processed_text.replace('🎼', ' ')
    if escape_n:
        processed_text = processed_text.replace('\n', '\\n')
    
    return processed_text.strip()

async def get_images_description(image_urls: List[str]) -> str:
    """
    获取图片的描述信息
    :param image_urls: 图片URL列表
    :return: 图片描述文本
    """
    try:
        # 使用AI助手获取图片描述
        description = process_sample_for_prompt(str(await ai_helper.media_captioning(image_urls)), escape_n=False)
        return description if description and len(description) > 5 else ""
    except Exception as e:
        logger.error(f"获取图片描述失败: {e}")
        return ""

async def get_video_description(video_url: str) -> str:
    """
    获取视频的描述信息
    :param video_url: 视频URL
    :return: 视频描述文本
    """
    try:
        # 使用AI助手获取视频描述
        description = process_sample_for_prompt(str(await ai_helper.media_captioning([video_url])), escape_n=False)
        return description if description and len(description) > 5 else ""
    except Exception as e:
        logger.error(f"获取视频描述失败: {e}")
        return ""

async def images_ocr(image_urls: Union[List[str], set[str]], only_article: bool = False) -> str:
    '''图片OCR
    :param image_urls: 图片文件列表（网络url/标准base64编码）
    :param only_article: 是否只提取正文内容
    :return: OCR结果
    '''
    if not image_urls:
        return ''
    if isinstance(image_urls, set):
        image_urls = list(image_urls)
    else:
        image_urls = list(set(image_urls))
    ocr_result = process_sample_for_prompt(str(await ai_helper.ocr(image_urls=image_urls, only_article=only_article)), escape_n=only_article)
    logger.debug(f"图片OCR结果预览: {ocr_result[:50]}")
    return ocr_result

async def video_ocr(video_url: str) -> str:
    '''视频OCR
    :param video_url: 视频文件（网络url/标准base64编码）
    :return: OCR结果
    '''
    if not video_url:
        return ''
    ocr_result = process_sample_for_prompt(str(await ai_helper.ocr(video_url=video_url, only_article=True)), escape_n=True)
    logger.debug(f"视频OCR结果预览: {ocr_result[:50]}")
    return ocr_result

async def transcribe_audio(audio: Union[str, Path]) -> str:
    """
    将语音文件转写为文本
    :param audio: 语音文件路径、URL（支持wav/mp3/pcm/opus/webm格式），或base64编码
    :return: 转写后的文本, 若无文本或转写失败则返回空字符串
    """
    if not audio:
        logger.warning("音频参数为空")
        return ""
    
    # 记录需要清理的临时文件
    temp_files_to_delete = []
    
    try:
        audio_path = Path(audio)
        if isinstance(audio, str):
            if audio.startswith("http"):
                # 网络链接，直接下载
                temp_file_path = file_download_temp(audio)
                if temp_file_path:
                    audio_path = temp_file_path
                    temp_files_to_delete.append(audio_path)  # 记录临时文件
                else:
                    logger.warning(f"音频文件下载失败: {audio}")
                    return ""
            elif audio.startswith("data:audio"):
                # base64编码
                _ext = audio.split(';',1)[0].split('/',1)[-1]
                _audio_base64 = audio.split(',')[1]
                temp_file_path = temp_dir / f"{uuid4()}.{_ext}"
                temp_file_path.write_bytes(b64decode(_audio_base64))
                audio_path = temp_file_path
                temp_files_to_delete.append(audio_path)  # 记录临时文件
        
        if not audio_path or not audio_path.exists():
            logger.warning(f"音频文件不存在: {audio_path}")
            # 清理临时文件
            for temp_file in temp_files_to_delete:
                if temp_file and temp_file.exists():
                    temp_file.unlink(True)
            return ""
        
        # 压缩音频
        compressed_audio = _compress_audio(audio_path)
        temp_files_to_delete.append(compressed_audio)  # 记录压缩后的文件
        _audio = compressed_audio.read_bytes()

    except Exception as e:
        logger.error(f"音频格式处理失败: {e}")
        # 清理临时文件
        for temp_file in temp_files_to_delete:
            if temp_file and temp_file.exists():
                temp_file.unlink(True)
        return ""

    res = process_sample_for_prompt(str(await ai_helper.audio_transcriptions(_audio)), escape_n=True)
    
    # 清理临时文件
    for temp_file in temp_files_to_delete:
        if temp_file and temp_file.exists():
            temp_file.unlink(True)
    
    return res

async def transcribe_video_audio(video_path: Union[str, Path]) -> str:
    """
    将视频文件中的音频转写为文本
    :param video_path: 视频文件路径（支持mp4、mkv、mov等格式）或base64编码、url
    :return: 转写后的文本, 若无文本或转写失败则返回空字符串
    """
    if not video_path:
        return ""
    transcribed_text = ""
    
    # 记录需要清理的临时文件
    temp_files_to_delete = []
    
    if isinstance(video_path, str):
        if video_path.startswith("http"):
            # 网络链接，直接下载
            _video_url = video_path
            temp_file_path = file_download_temp(_video_url)
            if temp_file_path:
                video_path = temp_file_path
                temp_files_to_delete.append(video_path)  # 记录临时视频文件
            else:
                logger.warning(f"视频文件下载失败: {video_path}")
                return ""
        elif video_path.startswith("data:video"):
            # base64编码
            _ext = video_path.split(';',1)[0].split('/',1)[-1]
            _video_base64 = video_path.split(',')[1]
            temp_file_path = temp_dir / f"{uuid4()}.{_ext}"
            temp_file_path.write_bytes(b64decode(_video_base64))
            video_path = temp_file_path
            temp_files_to_delete.append(video_path)  # 记录临时视频文件
    
    video_path = Path(video_path)
    if not video_path.exists():
        logger.warning(f"视频文件不存在: {video_path}")
        # 清理临时文件
        for temp_file in temp_files_to_delete:
            if temp_file and temp_file.exists():
                temp_file.unlink(True)
        return ""
    
    try:
        # 调用FFmpeg提取音频为webm格式
        audio_path = temp_dir / (video_path.stem + ".webm")
        temp_files_to_delete.append(audio_path)  # 记录临时音频文件
        
        # 直接提取并压缩音频到合适的大小和格式
        ffmpeg_command = f"{ffmpeg_path} -i \"{video_path.absolute()}\" -c:a libopus \"{audio_path.absolute()}\""
        ffmpeg_result = subprocess.run(ffmpeg_command, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if ffmpeg_result.returncode != 0: 
            logger.error(f"视频提取音频失败: {ffmpeg_result.stderr}")
            # 清理临时文件
            for temp_file in temp_files_to_delete:
                if temp_file and temp_file.exists():
                    temp_file.unlink(True)
            return ""
        
        # 调用transcribe_audio函数转写音频
        if audio_path and audio_path.exists():
            transcribed_text = await transcribe_audio(audio_path)
        else:
            logger.warning(f"提取的音频文件不存在: {audio_path}")

    except Exception as e:
        logger.error(f"视频转写音频失败: {e}")
    
    # 清理临时文件
    for temp_file in temp_files_to_delete:
        if temp_file and temp_file.exists():
            temp_file.unlink(True)
    
    return transcribed_text

async def caption_multimodel(attachment_list: Union[list[str], list[Path]]) -> tuple[str, str, str]:
    """
    获取多模态描述、多模态提取文本、语音转录文本
    :param attachment_list: 附件url、本地路径、附件名列表、文件base64或dify文件json列表，必须要有扩展名
    :return: 多模态描述, 多模态提取文本, 语音转录文本
    """
    if not attachment_list:
        return "", "", ""
    file_type_dict = {}
    # 记录需要清理的临时文件
    temp_files_to_delete: list[Path] = []
    for attachment in attachment_list:
        # 处理dify文件格式
        if isinstance(attachment, str) and '__dify__file__' in attachment:
            try:
                _file_data = json.loads(attachment)
                filename = _file_data.get("filename", '')
                url = _file_data.get("url", '')
                if not filename or not url:
                    logger.warning(f"dify文件格式错误: {attachment}")
                    continue
                file_type = get_file_type(filename)
                attachment = file_download_temp(url, filename)
                if attachment:
                    temp_files_to_delete.append(attachment)  # 记录临时文件
                else:
                    logger.warning(f"dify文件下载失败: {url}，文件名: {filename}")
                    continue
            except Exception as e:
                logger.warning(f"dify文件格式错误: {e}，内容: {attachment}")
                continue
        else:
            file_type = get_file_type(attachment)
            if not(isinstance(attachment, str) and (attachment.startswith("http") or attachment.startswith("data:"))):
                attachment = Path(attachment)
        if not file_type or file_type in ("unknown", "document", "spreadsheet", "presentation", "text"):
            continue
        file_type_dict.setdefault(file_type, set()).add(attachment)
    if not file_type_dict:
        return "", "", ""
    
    multimodel_desc, multimodel_text, speech_text = "", "", ""
    for file_type, attachment_set in file_type_dict.items():
        if file_type == "image": # 图片
            image_set = set()
            for attachment in attachment_set:
                atturl = file_to_url(attachment)
                if atturl:
                    image_set.add(atturl)
            if image_set:
                img_desc = await get_images_description(list(image_set))
                if img_desc and len(img_desc) > 5:
                    multimodel_desc += f"## 图片描述\n{img_desc}\n\n"
                article_ocr = await images_ocr(image_set, only_article=True)
                if article_ocr and len(article_ocr) > 2:
                    multimodel_text += f"{article_ocr}\n"
            
        elif file_type == "video": # 视频
            video_captioning_dict = {}
            attachment_list = list(attachment_set)
            
            # 使用多线程处理视频，提高处理速度
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                
                for idx, attachment in enumerate(attachment_list):
                    # 提交音频转录任务
                    futures[f"speech_{idx}"] = executor.submit(
                        lambda a=attachment: asyncio.run(transcribe_video_audio(a))
                    )

                    atturl = file_to_url(attachment)
                    if atturl:
                        # 提交视频描述任务
                        futures[f"video_desc_{idx}"] = executor.submit(
                            lambda u=atturl: asyncio.run(get_video_description(u))
                        )
                
                # 收集结果
                for future_key, future in futures.items():
                    try:
                        result = future.result()
                        if not result:
                            continue
                            
                        # 解析任务类型和索引
                        parts = future_key.split('_')
                        task_type = '_'.join(parts[:-1])
                        idx = int(parts[-1]) + 1
                        
                        # 根据任务类型存储结果
                        if task_type == "video_desc" and len(result) > 5:
                            video_captioning_dict.setdefault(idx, {})['desc'] = result
                        elif task_type == "speech" and len(result) > 2:
                            video_captioning_dict.setdefault(idx, {})['speech'] = result
                            
                    except Exception as e:
                        logger.error(f"处理视频任务 {future_key} 失败: {e}")
            
            # 合并结果
            if video_captioning_dict:
                _desc = ''
                for idx, video_dict in video_captioning_dict.items():
                    if 'desc' in video_dict and video_dict['desc'] and len(video_dict['desc']) > 5:
                        idextext = ""
                        if idx > 1 or len(video_captioning_dict) > 1:
                            idextext = f"视频 {idx}：\n"
                        _desc += f"{idextext}{video_dict['desc']}\n"
                    if 'speech' in video_dict and video_dict['speech'] and len(video_dict['speech']) > 2:
                        speech_text += f"{video_dict['speech']}\n"

                if _desc:
                    multimodel_desc += "## 视频描述\n" + _desc

        elif file_type == "audio": # 音频
            for attachment in attachment_set:
                _speechtext = await transcribe_audio(attachment)
                if _speechtext and len(_speechtext) > 2:
                    speech_text += f"{_speechtext}\n"
            
        else:
            continue
    
    # 清理临时文件
    for temp_file in temp_files_to_delete:
        try:
            temp_file.unlink(True)
        except Exception as e:
            logger.warning(f"临时文件删除失败: {temp_file}, 错误: {e}")

    return multimodel_desc.strip(), multimodel_text.strip(), speech_text.strip()
