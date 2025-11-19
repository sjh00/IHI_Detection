# IHI_Detection - 违法有害信息检测智能体

IHI_Detection是一个基于FastMCP框架开发的违法有害信息检测智能体，旨在帮助识别和分析网络上的违法有害信息。

## 项目概述

该系统提供了一套完整的工具链，用于从互联网上采集、分析和检测违法有害信息。它结合了浏览器自动化、AI分析和数据库管理功能，支持多种主流互联网平台。

## 核心功能

1. **浏览器自动化工具**
   - 获取指定URL的结构化文档内容
   - 基于已有网页内容的结构化文档分析
   - 导航到多个URL并获取其markdown快照

2. **历史记录管理**
   - 保存检测历史记录到数据库
   - 检索历史记录

3. **文件处理**
   - 文件读取和写入功能

4. **AI辅助分析**
   - 违法有害信息识别
   - 多模态内容分析（文本、视频、图像、音频）

## 技术栈

- **FastMCP**: MCP框架，提供工具注册和调用机制
- **Python**: 主要开发语言
- **Asyncio**: 异步编程支持
- **DuckDB**: 轻量级嵌入式数据库
- **Chrome DevTools Protocol**: 浏览器自动化
- **Ollama/硅基流动/智谱**: AI模型提供商支持

## 项目结构

```
IHI_Detection/
├── Dify_DSL/           # Dify工作流文件
├── IHI_MCP/            # 主MCP服务器
│   ├── data/           # 数据存储目录
│   ├── mcp_tools/      # MCP工具模块
│   ├── src/            # 核心功能实现
│   ├── config.json     # 项目配置文件
│   ├── auth.json       # 认证配置文件
│   └── mcp_server.py   # 服务器入口文件
└── LICENSE             # 许可证文件
```

## 安装

1. 克隆项目到本地
```bash
git clone https://github.com/sjh00/IHI_Detection.git
cd IHI_Detection
```

2. 安装依赖（建议使用虚拟环境）
```bash
cd IHI_MCP
pip install -r requirements.txt
```

## 配置

1. 修改配置文件 `IHI_MCP/config.json`
   - 配置AI模型等设置项
2. 配置认证文件 `IHI_MCP/auth.json`
   - 配置AI模型的认证信息
3. 确保所有敏感信息已正确配置

## 使用

1. 启动MCP服务器
```bash
cd IHI_MCP
python mcp_server.py
```

2. 通过MCP客户端调用系统提供的工具

## Dify_DSL目录说明

Dify_DSL目录包含了该系统在Dify平台上的工作流文件，可导入Dify平台进行使用。

## 贡献

欢迎提交Issue和Pull Request。请阅读[CONTRIBUTING.md](CONTRIBUTING.md)了解贡献指南。

## 许可证

该项目采用GPLv3许可证，详情请见[LICENSE](LICENSE)文件。