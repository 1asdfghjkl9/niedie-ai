# 🦋 逆蝶AI v4.0

浏览器内运行真实大语言模型的 AI 对话应用。

## ✨ v4.0 重大升级

### 🧠 WebLLM — 浏览器内真实AI
- 使用 WebLLM + WebGPU 在浏览器中运行真实大语言模型
- 支持 Qwen2.5、Llama 3.2、Phi 3.5、Gemma 2 等模型
- 数据完全不出本地，零隐私泄露
- 首次加载下载模型后缓存，后续零延迟

### 💎 积分系统
- 初始 1000 积分，每日签到 +200
- 对话消耗 10 / 爬虫消耗 50 / 图片识别消耗 100
- 积分上限 5000

### 📸 多功能
- 智能对话（流式输出）
- 网页爬虫
- 拍照识别 / 图片上传
- 多会话管理
- 模型切换

## 🚀 使用

### 在线访问
https://1asdfghjkl9.github.io/niedie-ai/

> 需要 Chrome 113+ 或 Edge 113+（支持 WebGPU）

### 本地运行
```bash
node server.js
```
访问 http://localhost:3000

## 🛠 技术栈
- 前端：单 HTML 文件（内联 CSS + JS）
- AI引擎：WebLLM (@mlc-ai/web-llm)
- 后端：零依赖 Node.js（可选，用于爬虫和图片识别）
- 部署：GitHub Pages

## 📦 可选模型
| 模型 | 大小 | 特点 |
|------|------|------|
| Qwen2.5 0.5B | 278MB | 极速加载 |
| Qwen2.5 1.5B | 868MB | 中文最佳 |
| Llama 3.2 1B | 712MB | 均衡 |
| Llama 3.2 3B | 1.76GB | 更强能力 |
| Qwen2.5 3B | 1.7GB | 高质量 |
| Phi 3.5 mini | 2.1GB | 推理强 |
| Gemma 2 2B | 1.44GB | Google |

## License
MIT
