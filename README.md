# 🦋 逆蝶AI

破茧而出的智能AI助手，支持智能对话、网页爬虫、图片识别。

## ✨ 功能

- **💬 智能对话** — 多轮对话，流式回复，localStorage持久化
- **🕷️ 网页爬虫** — 输入网址抓取网页内容，AI自动分析总结
- **📷 图片识别** — 拍照或上传图片，AI识别图片内容
- **🔒 安全防护** — XSS防护、SSRF防护、输入校验、错误兜底

## 🚀 启动

```bash
cd ai-chat-standalone
node server.js
```

访问 `http://localhost:3000`

## 🛠️ 技术栈

- 零依赖：仅用 Node.js 内置模块，无需 npm install
- 前端：HTML + CSS + JavaScript（单文件）
- 后端：Node.js 原生 http 模块

## ⚙️ 配置（可选）

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| OPENAI_API_KEY | LLM API密钥，配置后接入真实AI | 无（演示模式）|
| OPENAI_API_BASE | API地址 | https://api.openai.com/v1 |
| OPENAI_MODEL | 模型名 | gpt-3.5-turbo |
| PORT | 端口号 | 3000 |

## 📋 功能说明

### 对话
直接在输入框输入消息，回车发送。支持多轮对话、Shift+回车换行。

### 爬虫
点击输入框左侧🕷️按钮，输入网页地址，逆蝶AI会抓取网页内容并自动分析。
内置SSRF防护，禁止访问内网地址。

### 图片识别
点击输入框左侧📷按钮，可以：
- 拍照（需要摄像头权限）
- 切换前后摄像头
- 上传本地图片

配置API Key后可接入GPT-4o等视觉模型进行真实图片识别。

## 📁 文件结构

```
ai-chat-standalone/
├── index.html   # 前端（HTML+CSS+JS全内联）
├── server.js    # 后端服务器
└── README.md    # 说明文档
```

## ✅ 自查结果

14项测试全部通过：首页加载、对话API、身份验证、功能列表、数学计算、空消息防护、无效JSON防护、404处理、爬虫正常URL、爬虫空URL防护、爬虫SSRF防护、图片识别空数据防护、图片识别格式校验、图片识别演示模式。
