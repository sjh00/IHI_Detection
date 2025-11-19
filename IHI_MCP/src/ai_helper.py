"""
AI助手模块
支持多种大模型调用接口，并集成MCP工具调用功能
"""

import re
import json
import logging
import aiohttp
from typing import Literal, Dict, List, Optional, Union, Any, Callable
from abc import ABC, abstractmethod

import requests

# 配置日志
logger = logging.getLogger(f"IHI_detection.{__name__}")

COMMAND_PLACEHOLDER_COMPILE = re.compile(r'<\|[a-zA-Z0-9_\-]+\|>')


class AIModelProvider(ABC):
    """AI模型提供商抽象基类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"IHI_detection.{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> Dict:
        """聊天完成接口"""
        pass


class OllamaProvider(AIModelProvider):
    """Ollama模型提供商"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.api_key = config.get("api_key", "")
        self.timeout = config.get("timeout", 60)
    
    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> Dict:
        """Ollama聊天完成接口"""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "max_tokens": kwargs.get("max_tokens", 2048)
            }
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "content": result.get("message", {}).get("content", ""),
                        "model": result.get("model", model),
                        "usage": {
                            "prompt_tokens": result.get("prompt_eval_count", 0),
                            "completion_tokens": result.get("eval_count", 0),
                            "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                        }
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(f"Ollama API错误: {response.status} - {error_text}")
                    return {"error": f"API错误: {response.status} - {error_text}"}


class ZhipuProvider(AIModelProvider):
    """智谱AI模型提供商"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://open.bigmodel.cn/api/paas/v4")
        self.api_key = config.get("api_key", "")
        self.timeout = config.get("timeout", 60)
    
    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> Dict:
        """智谱AI聊天完成接口"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    choice = result.get("choices", [{}])[0]
                    return {
                        "content": choice.get("message", {}).get("content", ""),
                        "model": result.get("model", model),
                        "usage": result.get("usage", {})
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(f"智谱AI API错误: {response.status} - {error_text}")
                    return {"error": f"API错误: {response.status} - {error_text}"}


class SiliconFlowProvider(AIModelProvider):
    """硅基流动模型提供商"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.siliconflow.cn/v1")
        self.api_key = config.get("api_key", "")
        self.timeout = config.get("timeout", 60)
    
    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> Dict:
        """硅基流动聊天完成接口"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    choice = result.get("choices", [{}])[0]
                    return {
                        "content": choice.get("message", {}).get("content", ""),
                        "model": result.get("model", model),
                        "usage": result.get("usage", {})
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(f"硅基流动API错误: {response.status} - {error_text}")
                    return {"error": f"API错误: {response.status} - {error_text}"}


class NewAPIProvider(AIModelProvider):
    """NewAPI模型提供商"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "")
        self.api_key = config.get("api_key", "")
        self.timeout = config.get("timeout", 60)
    
    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> Dict:
        """NewAPI聊天完成接口"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    choice = result.get("choices", [{}])[0]
                    return {
                        "content": choice.get("message", {}).get("content", ""),
                        "model": result.get("model", model),
                        "usage": result.get("usage", {})
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(f"NewAPI错误: {response.status} - {error_text}")
                    return {"error": f"API错误: {response.status} - {error_text}"}


class AIHelper:
    """AI助手类，支持多种大模型调用接口"""
    
    def __init__(self, config: Dict):
        self.config = config.get("ai_helper", {})
        self.api_keys = config.get("auth", {}).get("ai_api_key", {})
        self.logger = logging.getLogger(f"IHI_detection.{__name__}")
        self.providers = {}
        self.mcp_server_url = config.get("mcp", {}).get("server_url", "http://127.0.0.1:8001/sse")
        self.system_prompt = self.config.get("system_prompt", "你是一个专业的AI助手，请严格按照要求进行回答")
        self.models_config = self.config.get("models", {})
        
        # 初始化所有配置的提供商
        self._init_providers()
    
    def _init_providers(self):
        """初始化所有配置的AI模型提供商"""
        providers_config = self.config.get("providers", {})
        for provider_name, provider_config in providers_config.items():
            provider_config["api_key"] = self.api_keys.get(provider_name, "")
        
        # 初始化Ollama提供商
        if "ollama" in providers_config:
            self.providers["ollama"] = OllamaProvider(providers_config["ollama"])
        
        # 初始化智谱提供商
        if "zhipu" in providers_config:
            self.providers["zhipu"] = ZhipuProvider(providers_config["zhipu"])
        
        # 初始化硅基流动提供商
        if "silicon_flow" in providers_config:
            self.providers["silicon_flow"] = SiliconFlowProvider(providers_config["silicon_flow"])
        
        # 初始化NewAPI提供商
        if "newapi" in providers_config:
            self.providers["newapi"] = NewAPIProvider(providers_config["newapi"])
        
        self.logger.info(f"已初始化的AI提供商: {list(self.providers.keys())}")
    
    async def chat_completion(self, messages: List[Dict], model_type: Literal["llm", "lvm", "ocr"], **kwargs) -> Dict:
        """聊天完成接口
        
        Args:
            messages: 消息列表，支持包含图像内容
            model_type: 模型类型，llm、lvm、ocr
            **kwargs: 其他传递给chat_completion的参数
                temperature: 生成文本的随机性
                top_p: 控制多样性
                max_tokens: 最大生成token数
                stream: 是否流式输出
        
        Returns:
            包含AI回复的字典
        """
        model_conf = self.models_config.get(model_type, {})
        if not model_conf:
            self.logger.error(f"未找到模型配置，模型类型: {model_type}")
            return {"error": "未找到模型配置"}
        
        model = model_conf.get("name", "")
        if not model:
            self.logger.error("未配置要使用的模型")
            return {"error": "未配置要使用的模型"}
        
        provider = model_conf.get("provider", "")
        if not provider or provider not in self.providers:
            error_msg = f"未知的提供商: {provider}，可用的提供商: {list(self.providers.keys())}"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        # 设置默认参数
        if 'stream' not in kwargs:
            kwargs['stream'] = False
        if 'enable_thinking' not in kwargs:
            kwargs['enable_thinking'] = False

        try:
            return await self.providers[provider].chat_completion(messages, model, **kwargs)
        except Exception as e:
            self.logger.error(f"调用提供商 {provider} 失败: {str(e)}")
            
            # 如果是主要LLM失败，尝试使用备用LLM
            if model_type == "llm":
                self.logger.warning("主要LLM失败，尝试使用备用LLM")
                try:
                    fallback_model_conf = model_conf.get("fallback_model", {})
                    fallback_provider = fallback_model_conf.get("provider", "")
                    fallback_model = fallback_model_conf.get("name", "")
                    if fallback_model and fallback_provider and fallback_provider in self.providers:
                        return await self.providers[fallback_provider].chat_completion(messages, fallback_model, **kwargs)
                except Exception as e2:
                    return {"error": f"调用提供商 {provider} 失败: {str(e2)}"}
            return {"error": f"调用提供商 {provider} 失败: {str(e)}"}
    
    async def chat(self, prompt: str, system: str = "", json_schema: Optional[Dict] = None, validate_json: bool = True, max_retries: int = 3, images: Optional[List[str]] = None, video: Optional[str] = None, files: Optional[List[str]] = None, **kwargs) -> Dict:
        """快捷聊天接口，总是返回符合JSON模式的回复
        
        Args:
            prompt: 用户提问
            system: 系统提示词，默认使用配置中的系统提示词
            json_schema: JSON模式字典，指定返回结果的格式
            validate_json: 是否验证返回结果是否符合JSON模式，如果内含error字段且不为空则直接返回，默认为True
            max_retries: 验证失败时的最大重试次数，默认为3
            images: 图像内容列表，每个图像包含url或base64编码
            video: 视频的url或base64编码
            files: 文件内容列表，每个文件包含url或base64编码
            **kwargs: 其他传递给chat_completion的参数
                temperature: 生成文本的随机性，默认为0.1
                top_p: 控制多样性，默认为0.9
                max_tokens: 最大生成token数，默认为2048
                stop: 停止词列表
        
        Returns:
            包含AI回复的字典，如果指定了JSON模式，则确保返回符合该模式的结果
        """
        _system = system if system else self.system_prompt
        retry_count = 0
        
        # 解析JSON模式（如果提供了）
        json_schema_str = ''
        if json_schema:
            json_schema_str = json.dumps(json_schema, ensure_ascii=False)
            # 添加JSON模式到系统提示
            _system += f"\n请确保回复严格符合Json Schema:\n{json_schema_str}\n只返回JSON数据，不要包含任何其他文本。"
        
        # 准备消息
        messages = [
            {"role": "system", "content": _system}
        ]
        
        # 构建用户消息，支持视觉输入（图像、视频、文件）
        user_message = {"role": "user", "content": []}
        use_vision = False
        if images or video or files:
            # 多模态消息格式
            if video:
                user_message["content"].append({"type": "video_url", "video_url": {"url": video}})
                use_vision = True
            elif images:
                # 添加图像内容
                for image in images:
                    image = image.strip()
                    if image:
                        user_message["content"].append({"type": "image_url", "image_url": {"url": image}})
                        use_vision = True
            elif files:
                # 添加文件内容
                for file in files:
                    file = file.strip()
                    if file:
                        user_message["content"].append({"type": "file_url", "file_url": {"url": file}})
                        use_vision = True
            if prompt:
                user_message["content"].append({"type": "text", "text": prompt})
        else:
            # 普通文本消息
            user_message["content"] = prompt
        
        messages.append(user_message)
        
        # 设置默认参数
        temperature = kwargs.pop("temperature", 0.1)
        top_p = kwargs.pop("top_p", 0.9)
        max_tokens = kwargs.pop("max_tokens", 4096)
        
        while retry_count < max_retries:
            try:
                # 调用聊天完成接口
                result = await self.chat_completion(
                    messages,
                    model_type="lvm" if use_vision else "llm",
                    temperature=temperature, 
                    top_p=top_p, 
                    max_tokens=max_tokens, 
                    **kwargs
                )
                
                # 处理错误结果
                if "error" in result:
                    self.logger.error(f"聊天完成失败: {result['error']}, 提示词字数: {len(prompt)}")
                    retry_count += 1
                    continue
                
                content = result.get("content", "")
                if content.strip() == "<think>\n用户用户":
                    retry_count += 1
                    continue
                
                self.logger.debug(f"原始回复预览: {content[:200]}")
                
                # 如果不需要验证JSON或者没有提供JSON模式，直接返回结果
                if not validate_json or not json_schema:
                    return {
                        "content": content,
                        "model": result.get("model", ""),
                        "usage": result.get("usage", {})
                    }
                # 尝试从内容中提取JSON（可能包含在代码块中）
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    json_content = content.strip()
                
                # 验证JSON格式
                try:
                    response_json = json.loads(json_content)
                    if 'error' in response_json and response_json['error']:
                        return {
                            "content": response_json,  # 返回解析后的JSON对象
                            "model": result.get("model", ""),
                            "usage": result.get("usage", {}),
                            "validated": True
                        }
                    
                    # 验证JSON结构
                    if self._validate_json_structure(response_json, json_schema):
                        return {
                            "content": response_json,  # 返回解析后的JSON对象
                            "model": result.get("model", ""),
                            "usage": result.get("usage", {}),
                            "validated": True
                        }
                    else:
                        self.logger.warning(f"返回结果不符合JSON模式，尝试重试 ({retry_count + 1}/{max_retries})")
                        retry_count += 1
                        # 向消息中添加重试提示
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": f"你返回的结果不符合要求的JSON模式，请严格按照以下模式返回:\n{json_schema_str}\n只返回JSON数据，不要包含任何其他文本。"})
                        
                except json.JSONDecodeError:
                    self.logger.warning(f"返回结果不是有效的JSON，尝试重试 ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    # 向消息中添加重试提示
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"你返回的不是有效的JSON格式，请严格按照以下模式返回JSON数据:\n{json_schema_str}\n只返回JSON数据，不要包含任何其他文本。"})
            
            except Exception as e:
                self.logger.error(f"聊天过程中发生异常: {str(e)}")
                return {"error": f"聊天异常: {str(e)}"}
        
        # 达到最大重试次数仍然失败
        self.logger.error("达到最大重试次数，无法获取成功结果")
        return {"error": "达到最大重试次数，无法获取成功结果"}
        
    def _validate_json_structure(self, data: Any, schema: Dict) -> bool:
        """验证JSON数据是否符合给定的模式
        
        支持验证:
        - 对象类型及其必需字段
        - 数组类型及其元素
        - 基本类型
        - 字符串枚举值和长度范围
        - 数值范围
        - 列表元素类型和数量范围
        
        Args:
            data: 要验证的JSON数据
            schema: JSON模式
            
        Returns:
            如果数据符合模式，返回True；否则返回False
        """
        try:
            # 如果模式包含type字段，验证类型
            if "type" in schema:
                schema_type = schema["type"]
                
                # 验证基本类型
                if schema_type == "object":
                    if not isinstance(data, dict):
                        self.logger.warning(f"期望对象类型，但得到了 {type(data).__name__}")
                        return False
                    
                    # 验证必需字段
                    required_fields = schema.get("required", [])
                    is_in_properties = False
                    for field in required_fields:
                        if field not in data:
                            if not is_in_properties and "properties" in data and field in data["properties"]:
                                data = data["properties"]
                                is_in_properties = True
                                continue
                            self.logger.warning(f"缺少必需字段: {field}")
                            return False
                    
                    # 验证属性
                    properties = schema.get("properties", {})
                    for field_name, field_schema in properties.items():
                        if field_name in data:
                            if not self._validate_json_structure(data[field_name], field_schema):
                                self.logger.warning(f"字段 {field_name} 不符合其模式")
                                return False
                elif schema_type == "array":
                    if not isinstance(data, list):
                        self.logger.warning(f"期望数组类型，但得到了 {type(data).__name__}")
                        return False
                    
                    # 验证数组长度范围
                    min_items = schema.get("minItems", 0)
                    max_items = schema.get("maxItems", float("inf"))
                    if len(data) < min_items:
                        self.logger.warning(f"数组长度 ({len(data)}) 小于最小要求 ({min_items})")
                        return False
                    if len(data) > max_items:
                        self.logger.warning(f"数组长度 ({len(data)}) 大于最大要求 ({max_items})")
                        return False
                    
                    # 验证数组元素
                    items_schema = schema.get("items", {})
                    if items_schema:
                        for item in data:
                            if not self._validate_json_structure(item, items_schema):
                                self.logger.warning(f"数组元素不符合其模式: {item}")
                                return False
                elif schema_type == "string":
                    if not isinstance(data, str):
                        self.logger.warning(f"期望字符串类型，但得到了 {type(data).__name__}")
                        return False
                    
                    # 验证枚举值
                    enum_values = schema.get("enum", [])
                    if enum_values and data not in enum_values:
                        self.logger.warning(f"字符串值 '{data}' 不在枚举列表中: {enum_values}")
                        return False
                    
                    # 验证字符串长度范围
                    min_length = schema.get("minLength", 0)
                    max_length = schema.get("maxLength", float("inf"))
                    if len(data) < min_length:
                        self.logger.warning(f"字符串长度 ({len(data)}) 小于最小要求 ({min_length})")
                        return False
                    if len(data) > max_length:
                        self.logger.warning(f"字符串长度 ({len(data)}) 大于最大要求 ({max_length})")
                        return False
                elif schema_type == "number" or schema_type == "integer":
                    if not isinstance(data, (int, float)):
                        self.logger.warning(f"期望数值类型，但得到了 {type(data).__name__}")
                        return False
                    
                    # 验证数值范围
                    minimum = schema.get("minimum", -float("inf"))
                    maximum = schema.get("maximum", float("inf"))
                    if data < minimum:
                        self.logger.warning(f"数值 ({data}) 小于最小值 ({minimum})")
                        return False
                    if data > maximum:
                        self.logger.warning(f"数值 ({data}) 大于最大值 ({maximum})")
                        return False
                elif schema_type == "boolean":
                    if not isinstance(data, bool):
                        self.logger.warning(f"期望布尔类型，但得到了 {type(data).__name__}")
                        return False
                elif schema_type == "null":
                    if data is not None:
                        self.logger.warning(f"期望null类型，但得到了 {type(data).__name__}")
                        return False
            
            return True
        except Exception as e:
            self.logger.error(f"验证JSON结构时出错: {str(e)}")
            return False

    async def image_to_markdown(self, image_urls: Optional[List[str]] = None) -> str:
        """
        文档图像转换为Markdown格式
        :param image_urls: 图像URL或base64编码列表
        :return: Markdown格式的字符串
        """
        if not image_urls:
            self.logger.warning("未提供文档图片的URL或base64编码")
            return ""
        prompt = "Convert the document to markdown. 输出内容控制在3000字以内。"
        res = await self.chat(prompt=prompt, images=image_urls, temperature=0, max_tokens=4096)
        if 'error' in res or "content" not in res or not res['content']:
            self.logger.warning(f"文档图像转Markdown出错: {res.get('error', '无错误信息')}")
        else:
            result = res['content'].strip()
            result_lines = result.split('\n')
            if result_lines[-1].startswith('（注：'):
                result = '\n'.join(result_lines[:-1])
            return result
        return ""

    async def ocr(self, image_urls: Optional[List[str]] = None, video_url: Optional[str] = None, file_urls: Optional[List[str]] = None, only_article: bool = False) -> str:
        """OCR识别
        
        Args:
            image_urls: 图像URL或base64编码列表
            video_url: 视频URL或base64编码
            file_urls: 文件URL或base64编码列表
            only_article: 是否只返回正文内容
            
        Returns:
            识别出的文本字符串
        """
        if not image_urls and not video_url and not file_urls:
            self.logger.warning("未提供图片、视频或文件的URL或base64编码")
            return ""
        elif video_url and not only_article:
            only_article = True
            self.logger.warning("视频不支持默认的OCR识别，已自动设置为只返回正文内容")
        elif file_urls and not only_article:
            only_article = True
            self.logger.warning("文件不支持默认的OCR识别，已自动设置为只返回正文内容")
            
        if only_article:
            if file_urls:
                prompt = "你是专业的文件内文本提取助手，提取并直接输出文件内主要内容的文本，不要有任何其他内容。输出内容控制在3000字以内。"
            else:
                prompt = "你是专业OCR，提取并直接输出画面正文的文本，不要有任何其他内容。输出内容控制在3000字以内。"
            res = await self.chat(prompt=prompt, images=image_urls, video=video_url, files=file_urls, temperature=0, max_tokens=4096)
            if 'error' in res or "content" not in res or not res['content']:
                self.logger.warning(f"OCR识别出错: {res.get('error', '无错误信息')}")
            else:
                result = COMMAND_PLACEHOLDER_COMPILE.sub('', res['content']).strip()
                result_lines = result.split('\n')
                if result_lines[-1].startswith('（注：'):
                    result_lines = [r.strip() for r in result_lines[:-1] if r.strip()]
                    result = '\n'.join(result_lines)
                return result
            return ""
        elif image_urls:
            # 尝试使用deepseek-ocr
            res = ''
            for image_url in image_urls:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "<image>\n详细描述这张图片."}
                    ]
                }]
                result = await self.chat_completion(
                    messages=messages,
                    model_type="ocr",
                    temperature=0,
                    top_p=0.7,
                    max_tokens=4096
                )
                content = result.get("content", "")
                res += content + '\n\n'
            return res.strip()
        return ""
    
    async def audio_transcriptions(self, audio: bytes) -> str:
        """音频转文字
        
        Args:
            audio: 音频文件的字节流，webm格式
            
        Returns:
            识别出的文本字符串
        """
        model_conf = self.models_config.get('speech2text', {})
        provider = model_conf.get("provider", "")
        model = model_conf.get("name", "")
        if not model or not provider or provider not in self.providers:
            self.logger.error("未正确配置语音转文字模型")
            return ""
        
        try:
            provider = self.providers[provider]
            post_url = provider.base_url.strip('/') + "/audio/transcriptions"
            api_key = provider.api_key
            # 检测音频格式并设置正确的MIME类型
            files = { "file": ("audio.webm", audio, "audio/webm") }
            payload = { "model": model }
            headers = {"Authorization": "Bearer " + api_key}

            response = requests.post(post_url, data=payload, files=files, headers=headers)

            res = response.json()
            if res:
                if 'text' in res:
                    text = res['text'].strip()
                    self.logger.debug(f"音频转写预览: {text[:100]}")
                    return text
                else:
                    self.logger.warning(f"音频转写失败: {json.dumps(res, ensure_ascii=False)}")
        except Exception as e:
            self.logger.error(f"音频转写失败: {e}")
        return ""

    async def media_captioning(self, media_urls: list[str]) -> str:
        """媒体内容描述
        
        Args:
            media_urls: 视频或图片的URL或base64编码列表，视频仅支持单个，图片支持多个，不支持视频图片混合
            
        Returns:
            媒体描述字符串
        """
        if not media_urls:
            self.logger.warning("未提供媒体URL或base64编码")
            return ""
        if media_urls[0].startswith('http'):
            media_type = '视频' if media_urls[0].lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')) else '图片'
        elif media_urls[0].startswith('data:'):
            mime_type = media_urls[0].split(';', 1)[0].split(':', 1)[1]
            media_type = '图片' if mime_type.startswith('image') else '视频'
        else:
            self.logger.warning(f"不支持的媒体格式: {media_urls}")
            return ""
        prompt = f"你是擅长分析违法有害信息的民警，请认真分析给出的{media_type}文件，描述主要内容，以便于辅助分析违法有害信息。输出内容控制在3000字以内。"
        if media_type == '视频':
            res = await self.chat(prompt=prompt, video=media_urls[0], temperature=0, max_tokens=4096)
        else:
            res = await self.chat(prompt=prompt, images=media_urls, temperature=0, max_tokens=4096)
        if 'error' in res or "content" not in res or not res['content']:
            self.logger.warning(f"{media_type}内容描述时未提取到可检测内容: {res.get('error', '无错误信息')}")
        else:
            result = res['content'].strip()
            result_lines = result.split('\n')
            if result_lines[-1].startswith('（注：'):
                result_lines = [r.strip() for r in result_lines[:-1] if r.strip()]
                result = '\n'.join(result_lines)
            return result
        return ""

