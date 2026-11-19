/**
 * 逆蝶AI - 零依赖 Node.js 服务器
 * 功能：智能对话 + 网页爬虫 + 图片识别
 *
 * 启动: node server.js
 * 访问: http://localhost:3000
 *
 * 可选环境变量:
 *   OPENAI_API_KEY  - 配置后接入真实 LLM
 *   OPENAI_API_BASE - API 地址 (默认 https://api.openai.com/v1)
 *   OPENAI_MODEL    - 模型名 (默认 gpt-3.5-turbo)
 *   PORT            - 端口号 (默认 3000)
 */

const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const { URL } = require("url");

const PORT = process.env.PORT || 3000;
const INDEX_HTML = path.join(__dirname, "index.html");

// ============================================================
//  Smart Reply Engine (fallback when no API key)
// ============================================================
function buildSmartReply(message, history) {
  const msg = message.toLowerCase().trim();

  if (/^(你好|嗨|hi|hello|hey|哈喽|在吗)/i.test(msg)) {
    return "你好！👋 我是逆蝶AI，很高兴见到你。\n\n我可以帮你：\n- 回答问题\n- 闲聊解闷\n- 头脑风暴\n- 文字创作\n- 网页爬虫（点击🕷️按钮输入网址）\n- 图片识别（点击📷按钮拍照或上传）\n\n有什么我可以帮你的吗？";
  }
  if (/(你是谁|你叫什么|自我介绍|你是什么|逆蝶)/i.test(msg)) {
    return "我是逆蝶AI 🦋\n\n一只破茧而出的智能蝴蝶，擅长：\n1. **多轮对话** - 我能记住我们的对话上下文\n2. **知识问答** - 各类常识和知识问题\n3. **创意写作** - 帮你写文案、故事等\n4. **网页爬虫** - 输入网址，我帮你抓取分析网页内容\n5. **图片识别** - 拍照或上传图片，我帮你识别分析\n\n目前我运行在演示模式下，如需接入完整AI能力，请联系管理员配置API密钥。";
  }
  if (/(谢谢|感谢|thanks|thank you|多谢)/i.test(msg)) {
    return "不客气！😊 能帮到你是我的荣幸。如果还有其他问题，随时问我～";
  }
  if (/(再见|拜拜|bye|走了|晚安)/i.test(msg)) {
    return "再见！👋 逆蝶AI随时等你回来～";
  }
  if (/(几点|时间|日期|今天|现在)/i.test(msg)) {
    const now = new Date();
    const dateStr = now.toLocaleDateString("zh-CN", { year: "numeric", month: "long", day: "numeric", weekday: "long" });
    const timeStr = now.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    return `现在是 ${dateStr}，${timeStr}。`;
  }
  if (/(能做什么|功能|会什么|帮什么)/i.test(msg)) {
    return "逆蝶AI能做这些事情：\n\n📝 **问答对话** - 各种知识问答\n💡 **创意灵感** - 帮你头脑风暴\n✍️ **文字创作** - 文案、故事、诗歌\n🕷️ **网页爬虫** - 输入网址抓取网页内容\n📷 **图片识别** - 拍照或上传图片进行识别\n🤖 **闲聊陪伴** - 随时陪你聊天\n\n试试问我一个问题吧！";
  }

  // 数学计算
  const mathMatch = msg.match(/(\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(\d+(?:\.\d+)?)/);
  if (mathMatch) {
    const a = parseFloat(mathMatch[1]);
    const op = mathMatch[2];
    const b = parseFloat(mathMatch[3]);
    let result = "?";
    switch (op) {
      case "+": result = a + b; break;
      case "-": result = a - b; break;
      case "*": case "×": result = a * b; break;
      case "/": case "÷": result = b === 0 ? "除数不能为零" : a / b; break;
    }
    return `${a} ${op} ${b} = ${result}`;
  }

  // 写诗
  if (/(写.*诗|作诗|诗歌)/i.test(msg)) {
    return "好的，为你写一首小诗：\n\n《逆蝶》\n\n破茧成蝶逆风飞，\n振翅穿云不知归。\n纵使前路千万里，\n我心自在向天辉。\n\n希望你喜欢这首诗！如果想要其他主题或风格的诗，随时告诉我。";
  }

  // 对话历史感知
  if (history && history.length > 0) {
    const lastAssistant = [...history].reverse().find((h) => h.role === "assistant");
    if (lastAssistant && /继续|接着|然后呢|还有呢/i.test(msg)) {
      return "好的，让我继续补充。如果你想了解更多细节，可以告诉我具体感兴趣的方面，我会针对性地展开说明。";
    }
  }

  // 默认回复
  const replies = [
    `这是一个很有意思的问题！关于「${message.trim()}」，我的理解是：\n\n这个问题涉及多个方面，在演示模式下我只能给出基础回复。配置完整的AI模型后，我能提供更深入、更准确的分析。\n\n你能再具体描述一下你的需求吗？这样我能更好地帮助你。`,
    `我收到了你的消息：「${message.trim()}」\n\n目前我运行在演示模式，回复能力有限。不过我很乐意继续和你聊这个话题！你可以告诉我更多细节，或者尝试问我其他问题。`,
    `关于「${message.trim()}」，我想说：\n\n每个问题都有不同的角度可以探讨。你是在寻找具体的答案，还是想一起讨论这个话题呢？告诉我你的想法，我们一起深入聊聊。`,
  ];
  return replies[message.length % replies.length];
}

// ============================================================
//  Stream smart reply character by character
// ============================================================
function streamSmartReply(reply, res) {
  const chars = Array.from(reply);
  let index = 0;

  const interval = setInterval(() => {
    if (index < chars.length) {
      const chunk = chars.slice(index, index + 2).join("");
      res.write(chunk);
      index += 2;
    } else {
      clearInterval(interval);
      try { res.end(); } catch {}
    }
  }, 30);

  // Safety timeout
  setTimeout(() => {
    clearInterval(interval);
    try { res.end(); } catch {}
  }, 15000);
}

// ============================================================
//  Call LLM API (OpenAI-compatible)
// ============================================================
async function callLLM(message, history, apiKey, apiBase, model, res) {
  const messages = [
    {
      role: "system",
      content: "你是逆蝶AI，一个友好、智能的AI助手。请用简洁、清晰的中文回答用户的问题。如果不确定，请如实说明。你可以进行网页爬虫分析、图片识别等多种任务。",
    },
    ...history.slice(-10).map((h) => ({ role: h.role, content: h.content })),
    { role: "user", content: message },
  ];

  const response = await fetch(`${apiBase}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages,
      stream: true,
      max_tokens: 2000,
      temperature: 0.7,
    }),
  });

  if (!response.ok) {
    throw new Error(`LLM API returned ${response.status}`);
  }

  if (!response.body) {
    throw new Error("LLM API returned no body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith("data: ")) continue;

      const data = trimmed.slice(6);
      if (data === "[DONE]") {
        try { res.end(); } catch {}
        return;
      }

      try {
        const parsed = JSON.parse(data);
        const content = parsed.choices?.[0]?.delta?.content;
        if (content) {
          res.write(content);
        }
      } catch {
        // Skip unparseable lines
      }
    }
  }
  try { res.end(); } catch {}
}

// ============================================================
//  Crawl webpage - fetch URL and extract text content
// ============================================================
function crawlUrl(targetUrl) {
  return new Promise((resolve, reject) => {
    let parsedUrl;
    try {
      parsedUrl = new URL(targetUrl);
    } catch {
      return reject(new Error("无效的URL格式"));
    }

    if (!parsedUrl.protocol || (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:")) {
      return reject(new Error("仅支持 http 和 https 协议"));
    }

    const lib = parsedUrl.protocol === "https:" ? https : http;

    const options = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (parsedUrl.protocol === "https:" ? 443 : 80),
      path: parsedUrl.pathname + parsedUrl.search,
      method: "GET",
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
      },
      timeout: 15000,
    };

    const req = lib.request(options, (response) => {
      // Handle redirects
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        const redirectUrl = new URL(response.headers.location, targetUrl).href;
        return resolve(crawlUrl(redirectUrl));
      }

      if (response.statusCode !== 200) {
        return reject(new Error(`网页返回状态码 ${response.statusCode}`));
      }

      let data = "";
      const contentType = response.headers["content-type"] || "";

      response.setEncoding("utf-8");
      response.on("data", (chunk) => {
        data += chunk;
        if (data.length > 500000) {
          req.destroy();
          resolve(extractTextFromHtml(data.slice(0, 500000), targetUrl));
        }
      });
      response.on("end", () => {
        resolve(extractTextFromHtml(data, targetUrl));
      });
    });

    req.on("error", (err) => {
      reject(new Error("请求失败: " + err.message));
    });

    req.on("timeout", () => {
      req.destroy();
      reject(new Error("请求超时（15秒）"));
    });

    req.end();
  });
}

// Extract readable text from HTML
function extractTextFromHtml(html, sourceUrl) {
  // Extract title
  let title = "";
  const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  if (titleMatch) {
    title = titleMatch[1].trim().replace(/&[a-z]+;/gi, " ");
  }

  // Remove scripts, styles, and other non-content elements
  let cleaned = html
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<noscript[\s\S]*?<\/noscript>/gi, "")
    .replace(/<iframe[\s\S]*?<\/iframe>/gi, "")
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<nav[\s\S]*?<\/nav>/gi, "")
    .replace(/<footer[\s\S]*?<\/footer>/gi, "")
    .replace(/<header[\s\S]*?<\/header>/gi, "");

  // Extract meta description
  let description = "";
  const descMatch = html.match(/<meta[^>]+name=["']description["'][^>]+content=["']([^"']+)["']/i);
  if (descMatch) {
    description = descMatch[1].trim();
  }

  // Extract meta keywords
  let keywords = "";
  const kwMatch = html.match(/<meta[^>]+name=["']keywords["'][^>]+content=["']([^"']+)["']/i);
  if (kwMatch) {
    keywords = kwMatch[1].trim();
  }

  // Extract all text from paragraphs, headings, list items, etc.
  const texts = [];
  const tagRegex = /<(p|h[1-6]|li|td|th|div|span|article|section|blockquote|pre|code)([^>]*)>([\s\S]*?)<\/\1>/gi;
  let match;
  while ((match = tagRegex.exec(cleaned)) !== null) {
    let text = match[3]
      .replace(/<[^>]+>/g, " ")
      .replace(/&nbsp;/gi, " ")
      .replace(/&amp;/gi, "&")
      .replace(/&lt;/gi, "<")
      .replace(/&gt;/gi, ">")
      .replace(/&quot;/gi, '"')
      .replace(/&#39;/gi, "'")
      .replace(/&[a-z]+;/gi, " ")
      .replace(/\s+/g, " ")
      .trim();
    if (text.length > 5) {
      texts.push(text);
    }
  }

  // Deduplicate
  const uniqueTexts = [...new Set(texts)];

  // Limit total length
  let bodyText = uniqueTexts.join("\n").slice(0, 10000);

  return {
    url: sourceUrl,
    title: title || "无标题",
    description: description,
    keywords: keywords,
    content: bodyText,
    contentLength: bodyText.length,
    crawledAt: new Date().toISOString(),
  };
}

// ============================================================
//  Image Analysis (vision)
// ============================================================
async function analyzeImage(imageBase64, apiKey, apiBase, model) {
  // If we have an API key, try to use vision model
  if (apiKey) {
    try {
      const response = await fetch(`${apiBase}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: model.includes("gpt-4") ? model : "gpt-4o-mini",
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "请详细描述这张图片的内容，包括主要物体、场景、颜色、文字等信息。用中文回答。" },
                { type: "image_url", image_url: { url: imageBase64 } },
              ],
            },
          ],
          max_tokens: 1000,
        }),
      });

      if (!response.ok) throw new Error(`Vision API returned ${response.status}`);

      const data = await response.json();
      return data.choices?.[0]?.message?.content || "未能识别图片内容";
    } catch (err) {
      // Fall through to demo mode
    }
  }

  // Demo mode: simulate image analysis based on image size and basic patterns
  return simulateImageAnalysis(imageBase64);
}

// Simulate basic image analysis (demo mode)
function simulateImageAnalysis(imageBase64) {
  const header = imageBase64.substring(0, 50);
  let analysis = "📷 图片识别结果（演示模式）\n\n";

  // Detect image type
  if (header.includes("iVBORw0KGgo")) {
    analysis += "📷 图片格式：PNG\n";
  } else if (header.includes("/9j/")) {
    analysis += "📷 图片格式：JPEG\n";
  } else if (header.includes("R0lGOD")) {
    analysis += "📷 图片格式：GIF\n";
  } else if (header.includes("UklGR")) {
    analysis += "📷 图片格式：WebP\n";
  } else {
    analysis += "📷 图片格式：未知\n";
  }

  // Estimate file size
  const sizeBytes = Math.floor(imageBase64.length * 0.75);
  const sizeKB = (sizeBytes / 1024).toFixed(1);
  analysis += `📊 图片大小：约 ${sizeKB} KB\n\n`;

  analysis += "⚠️ 当前为演示模式，无法进行真正的图片内容识别。\n\n";
  analysis += "要启用完整的图片识别功能，需要配置 OPENAI_API_KEY 环境变量，接入支持视觉的AI模型（如 GPT-4o）。\n\n";
  analysis += "配置后，逆蝶AI将能够：\n- 识别图片中的物体和场景\n- 读取图片中的文字（OCR）\n- 描述图片内容和细节\n- 分析图片中的图表数据";

  return analysis;
}

// ============================================================
//  Read request body
// ============================================================
function readBody(req, maxSize) {
  return new Promise((resolve, reject) => {
    let body = "";
    let size = 0;
    const limit = maxSize || 1e6;

    req.on("data", (chunk) => {
      body += chunk;
      size += chunk.length;
      if (size > limit) {
        req.destroy();
        reject(new Error("请求体过大"));
      }
    });

    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

// ============================================================
//  HTTP Server
// ============================================================
const server = http.createServer(async (req, res) => {
  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  // Serve index.html
  if (req.method === "GET" && (req.url === "/" || req.url === "/index.html")) {
    try {
      const html = fs.readFileSync(INDEX_HTML, "utf-8");
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(html);
    } catch {
      res.writeHead(500);
      res.end("Internal Server Error");
    }
    return;
  }

  // API: /api/chat
  if (req.method === "POST" && req.url === "/api/chat") {
    try {
      const body = await readBody(req, 1e6);
      let parsed;
      try {
        parsed = JSON.parse(body);
      } catch {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "无效的请求格式" }));
        return;
      }

      const { message, history } = parsed;

      if (!message || typeof message !== "string" || message.trim().length === 0) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "消息内容不能为空" }));
        return;
      }

      if (message.length > 4000) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "消息内容过长，请控制在4000字以内" }));
        return;
      }

      const sanitizedMessage = message.replace(/<script[^>]*>.*?<\/script>/gi, "");

      const apiKey = process.env.OPENAI_API_KEY;
      const apiBase = process.env.OPENAI_API_BASE || "https://api.openai.com/v1";
      const model = process.env.OPENAI_MODEL || "gpt-3.5-turbo";

      res.writeHead(200, {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-cache",
        "Transfer-Encoding": "chunked",
      });

      if (apiKey) {
        try {
          await callLLM(sanitizedMessage, history || [], apiKey, apiBase, model, res);
        } catch (error) {
          console.error("LLM API error:", error);
          const reply = buildSmartReply(sanitizedMessage, history || []);
          streamSmartReply(reply, res);
        }
      } else {
        const reply = buildSmartReply(sanitizedMessage, history || []);
        streamSmartReply(reply, res);
      }
    } catch (err) {
      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "服务器内部错误" }));
      }
    }
    return;
  }

  // API: /api/crawl - 爬虫
  if (req.method === "POST" && req.url === "/api/crawl") {
    try {
      const body = await readBody(req, 1e6);
      let parsed;
      try {
        parsed = JSON.parse(body);
      } catch {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "无效的请求格式" }));
        return;
      }

      const { url } = parsed;

      if (!url || typeof url !== "string" || url.trim().length === 0) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "URL不能为空" }));
        return;
      }

      // Basic URL validation
      let targetUrl = url.trim();
      if (!targetUrl.startsWith("http://") && !targetUrl.startsWith("https://")) {
        targetUrl = "https://" + targetUrl;
      }

      // Prevent internal network access (SSRF protection)
      try {
        const parsed = new URL(targetUrl);
        const hostname = parsed.hostname;
        if (hostname === "localhost" || hostname === "127.0.0.1" || hostname === "0.0.0.0" ||
            hostname.startsWith("192.168.") || hostname.startsWith("10.") ||
            hostname.startsWith("172.16.") || hostname === "::1") {
          res.writeHead(403, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "出于安全考虑，不允许访问内网地址" }));
          return;
        }
      } catch {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "无效的URL格式" }));
        return;
      }

      const result = await crawlUrl(targetUrl);

      res.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
      res.end(JSON.stringify(result));
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json; charset=utf-8" });
      res.end(JSON.stringify({ error: err.message || "爬虫请求失败" }));
    }
    return;
  }

  // API: /api/vision - 图片识别
  if (req.method === "POST" && req.url === "/api/vision") {
    try {
      const body = await readBody(req, 20e6); // 允许20MB用于图片
      let parsed;
      try {
        parsed = JSON.parse(body);
      } catch {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "无效的请求格式" }));
        return;
      }

      const { image } = parsed;

      if (!image || typeof image !== "string" || image.trim().length === 0) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "图片数据不能为空" }));
        return;
      }

      // Validate it's a data URL
      if (!image.startsWith("data:image/")) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "图片格式无效，仅支持 data:image/ 格式" }));
        return;
      }

      const apiKey = process.env.OPENAI_API_KEY;
      const apiBase = process.env.OPENAI_API_BASE || "https://api.openai.com/v1";
      const model = process.env.OPENAI_MODEL || "gpt-3.5-turbo";

      const analysis = await analyzeImage(image, apiKey, apiBase, model);

      res.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
      res.end(JSON.stringify({ result: analysis }));
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json; charset=utf-8" });
      res.end(JSON.stringify({ error: err.message || "图片识别失败" }));
    }
    return;
  }

  // Serve favicon (empty 1x1 to prevent 404)
  if (req.method === "GET" && req.url === "/favicon.ico") {
    res.writeHead(204);
    res.end();
    return;
  }

  // 404
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Not Found" }));
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n❌ 端口 ${PORT} 已被占用，请使用 PORT=xxxx node server.js 指定其他端口`);
  } else {
    console.error('\n❌ 服务器错误:', err.message);
  }
  process.exit(1);
});

server.listen(PORT, () => {
  console.log(`\n🦋 逆蝶AI 已启动！`);
  console.log(`   访问地址: http://localhost:${PORT}`);
  console.log(`   功能: 智能对话 | 网页爬虫 | 图片识别`);
  console.log(`   模式: ${process.env.OPENAI_API_KEY ? "LLM API" : "智能回复引擎（演示模式）"}`);
  if (!process.env.OPENAI_API_KEY) {
    console.log(`   提示: 设置 OPENAI_API_KEY 环境变量可接入真实 AI\n`);
  } else {
    console.log(`   API: ${process.env.OPENAI_API_BASE || "https://api.openai.com/v1"}`);
    console.log(`   模型: ${process.env.OPENAI_MODEL || "gpt-3.5-turbo"}\n`);
  }
});
