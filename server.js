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
  const m = message.trim();

  // --- 问候 ---
  if (/^(你好|嗨|hi|hello|hey|哈喽|在吗|在不在|有人吗)/i.test(msg)) {
    const g = ["你好！👋 我是逆蝶AI，很高兴见到你。有什么可以帮你的吗？","嗨！😊 逆蝶AI为你服务，随时可以开始对话～","你好呀！🦋 今天想聊些什么呢？","哈喽！我可以帮你回答问题、写文案、做计算等，随时告诉我～"];
    return g[m.length % g.length];
  }
  if (/^(早上好|早安|早啊)/i.test(msg)) return "早上好！☀️ 新的一天，愿你充满活力！";
  if (/^(下午好)/i.test(msg)) return "下午好！🌤️ 希望你今天一切顺利～";
  if (/(晚上好|晚安)/i.test(msg)) return msg.includes('晚安') ? "晚安！🌙 祝你做个好梦～" : "晚上好！🌆 有什么想聊的吗？";

  // --- 自我介绍 ---
  if (/(你是谁|你叫什么|自我介绍|你是什么|逆蝶|介绍.*自己)/i.test(msg)) {
    return "我是**逆蝶AI** 🦋\n\n一只破茧而出的智能蝴蝶。\n\n**我的能力：**\n1. **多轮对话** — 记住上下文\n2. **知识问答** — 科学、历史、地理等\n3. **创意写作** — 文案、故事、诗歌\n4. **数学计算** — 四则运算\n5. **网页爬虫** — 抓取网页内容\n6. **图片识别** — 拍照识别\n7. **生活助手** — 美食、旅行建议\n8. **情感陪伴** — 鼓励与倾听\n\n💡 每次对话消耗 10 积分，每日签到可得 200 积分。";
  }

  // --- 感谢 ---
  if (/(谢谢|感谢|thanks|thank you|多谢|谢了|辛苦了)/i.test(msg)) {
    const t = ["不客气！😊 能帮到你是我的荣幸～","别客气！有问题随时找我 🦋","不用谢！这是逆蝶AI应该做的 💜","很高兴能帮到你！还有其他问题吗？"];
    return t[m.length % t.length];
  }
  if (/(对不起|抱歉|不好意思|sorry)/i.test(msg)) return "没关系啦 😊 不用在意，我们继续吧～";
  if (/(爱.*你|喜欢.*你|你好棒|666|nb|牛逼)/i.test(msg)) return "谢谢你！🦋💜 你的认可是我最大的动力～";

  // --- 情感 ---
  if (/(不开心|难过|伤心|郁闷|烦|生气|emo|崩溃|累)/i.test(msg)) {
    const c = ["我理解你的感受 💜 每一个困难都是暂时的。试试深呼吸三次？","抱抱你 🤗 难过的时候允许自己难过。明天又是新的一天。","我听到你了 🦋 你已经做得很好了。我在这里陪着你。","感觉累了就休息一下吧 ☕ 慢慢来。"];
    return c[m.length % c.length];
  }
  if (/(开心|高兴|快乐|哈哈|嘿嘿)/i.test(msg)) return "看到你开心我也很开心！🦋✨";
  if (/(孤独|孤单|寂寞|一个人)/i.test(msg)) return "你不是一个人 🦋 至少现在，我在这里陪着你。";

  // --- 数学 ---
  const mathMatch = msg.match(/(-?\d+(?:\.\d+)?)\s*([+\-*/×÷%])\s*(-?\d+(?:\.\d+)?)/);
  if (mathMatch) {
    const a = parseFloat(mathMatch[1]), op = mathMatch[2], b = parseFloat(mathMatch[3]);
    let result = "?";
    switch (op) { case "+": result=a+b; break; case "-": result=a-b; break; case "*": case "×": result=a*b; break; case "/": case "÷": result=b===0?"除数不能为零":a/b; break; case "%": result=b===0?"除数不能为零":a%b; break; }
    return `${a} ${op} ${b} = **${result}**`;
  }
  const pctMatch = msg.match(/(\d+(?:\.\d+)?)\s*[%％]\s*(?:of|的|占)?\s*(\d+(?:\.\d+)?)/);
  if (pctMatch) { const p=parseFloat(pctMatch[1]), t=parseFloat(pctMatch[2]); return `${p}% 的 ${t} = **${t*p/100}**`; }
  const sqrtMatch = msg.match(/(?:sqrt|根号|√)\s*(\d+(?:\.\d+)?)/);
  if (sqrtMatch) { const n=parseFloat(sqrtMatch[1]); return `√${n} = **${Math.sqrt(n).toFixed(6)}**`; }

  // --- 时间 ---
  if (/(几点|时间|现在.*时间)/i.test(msg)) return `🕐 现在是 ${new Date().toLocaleString("zh-CN", {hour:'2-digit',minute:'2-digit',second:'2-digit'})}`;
  if (/(今天.*日期|今天.*几号|日期|几月)/i.test(msg)) return `📅 今天是 ${new Date().toLocaleDateString("zh-CN", {year:'numeric',month:'long',day:'numeric',weekday:'long'})}`;
  if (/(星期|周几)/i.test(msg)) { const d=['日','一','二','三','四','五','六']; return `今天是星期${d[new Date().getDay()]}`; }

  // --- 天气 ---
  if (/(天气|weather|下雨|温度|气温)/i.test(msg)) return "🌤️ 我暂时无法获取实时天气。建议查看手机天气应用，或搜索「城市名+天气」。";

  // --- 创意写作 ---
  if (/(写.*诗|作诗|诗歌)/i.test(msg)) {
    if (/(蝴蝶|逆蝶|蝶)/i.test(msg)) return "好的，为你写一首关于蝴蝶的诗：\n\n**《逆蝶》**\n\n破茧成蝶逆风飞，\n振翅穿云不知归。\n纵使前路千万里，\n我心自在向天辉。\n\n🦋";
    if (/(春)/i.test(msg)) return "**《春晓》**\n\n春风拂面柳丝长，\n桃花初绽杏花香。\n燕归旧巢呢喃语，\n一树梨花映日光。\n\n🌸";
    if (/(秋)/i.test(msg)) return "**《秋思》**\n\n秋风起处叶飘零，\n满目金黄染远峰。\n雁字南归人字远，\n一壶浊酒暖余生。\n\n🍂";
    return "**《逆蝶》**\n\n破茧成蝶逆风飞，\n振翅穿云不知归。\n纵使前路千万里，\n我心自在向天辉。\n\n🦋";
  }
  if (/(写.*故事|讲.*故事|编.*故事)/i.test(msg)) return "好的，给你讲一个小故事：\n\n**《逆蝶的旅程》**\n\n从前有一只毛毛虫，她总是仰望天空，梦想着飞翔。\n\n其他毛毛虫笑她：「我们是虫子，怎么可能飞呢？」\n\n但她不信。她努力吃叶子，努力长大，终于织出了一个茧。\n\n在黑暗中，她等了很久。直到有一天，茧破了——\n\n她发现自己长出了美丽的翅膀。她逆着风飞了起来，越飞越高。\n\n从那以后，人们叫她「逆蝶」。🦋\n\n**寓意：** 永远不要让别人定义你的极限。";
  if (/(写.*文案|文案|广告词|slogan)/i.test(msg)) return "当然！文案示例：\n\n**科技感：**「未来已来，只是分布不均。」\n**温暖系：**「每一个微小的努力，都在照亮前路。」\n**逆蝶品牌：**「逆风飞翔，破茧成蝶。」\n\n告诉我具体产品，我可以写更有针对性的文案！";
  if (/(写.*笑话|讲.*笑话|说.*笑话|joke)/i.test(msg)) {
    const j = ["程序员的老婆让他买牛奶，如果有鸡蛋买六个。他回来拿了六瓶牛奶——因为有鸡蛋。\n\n😄","为什么程序员喜欢黑暗模式？因为光会吸引 bug。\n\n😄","世界上有 10 种人，一种懂二进制，一种不懂。\n\n😄","一个 SQL 查询走进酒吧，看到两张桌子，他走过去问：「我可以 join 你们吗？」\n\n😄"];
    return j[m.length % j.length];
  }

  // --- 知识问答 ---
  if (/(太阳系|行星|solar system|planet)/i.test(msg)) return "🌞 太阳系有八大行星：\n1. 水星 ☿ — 离太阳最近\n2. 金星 ♀ — 最热，460°C\n3. 地球 🌍 — 唯一有生命的星球\n4. 火星 🔴 — 红色星球\n5. 木星 🪐 — 最大的行星\n6. 土星 💍 — 有美丽光环\n7. 天王星 — 侧着自转\n8. 海王星 — 最远的行星\n\n💡 冥王星在 2006 年被降级为矮行星。";
  if (/(光速|speed of light)/i.test(msg)) return "💡 光速约 **299,792,458 米/秒**（约 30 万公里/秒）。光从太阳到地球约需 8 分 20 秒。";
  if (/(dna|基因|双螺旋)/i.test(msg)) return "🧬 DNA 由 A、T、G、C 四种碱基组成，呈双螺旋结构。沃森和克里克于 1953 年发现。人类基因组约有 30 亿个碱基对。";
  if (/(黑洞|black hole)/i.test(msg)) return "🌌 黑洞是引力极强的区域，连光都无法逃逸。2019 年人类首次拍到黑洞照片。霍金辐射理论认为黑洞会缓慢蒸发。";
  if (/(人工智能|ai|机器学习|深度学习)/i.test(msg)) return "🤖 AI 主要分支：机器学习、深度学习、NLP、计算机视觉。我就是 AI 的一个实例 🦋";
  if (/(中国.*历史|朝代|dynasty)/i.test(msg)) return "📜 中国朝代：夏→商→周→秦→汉→三国→晋→南北朝→隋→唐→五代→宋→元→明→清→民国→共和国";
  if (/(地球|earth)/i.test(msg)) return "🌍 地球：半径约 6,371km，周长约 40,075km，年龄约 46 亿年，海洋占 71%，人口约 80 亿。";
  if (/(圆周率|pi|π)/i.test(msg)) return "π = **3.14159265358979...** 是无理数。3 月 14 日是国际圆周率日！";
  if (/(编程|programming|代码|code|程序)/i.test(msg) && !/(写|帮.*写)/i.test(msg)) return "💻 编程入门推荐 Python 或 JavaScript。学习路径：语法→小项目→数据结构→算法→开源项目。";
  if (/(光合作用|photosynthesis)/i.test(msg)) return "🌿 光合作用：6CO₂ + 6H₂O + 光能 → C₆H₁₂O₆ + 6O₂。植物利用阳光制造葡萄糖和氧气。";
  if (/(html|css|javascript|前端|frontend)/i.test(msg)) return "💻 前端三件套：HTML（结构）、CSS（样式）、JavaScript（交互）。逆蝶AI 前端就是用这些写的！";
  if (/(python|爬虫|crawl)/i.test(msg)) return "🐍 Python 爬虫常用库：requests、BeautifulSoup、Scrapy、Selenium、Playwright。";
  if (/(node\.?js|后端|backend|服务器)/i.test(msg)) return "🟢 Node.js 基于 V8 引擎。逆蝶AI 后端就是零依赖 Node.js！常用框架：Express、Koa、Fastify。";

  // --- 美食 ---
  if (/(做.*菜|菜谱|recipe|怎么.*做.*菜)/i.test(msg)) return "🍳 **番茄炒蛋**\n材料：番茄2个、鸡蛋3个、盐、糖、葱\n步骤：1.鸡蛋打散炒熟盛出 2.番茄翻炒出汁 3.加盐和糖 4.倒蛋炒匀";
  if (/(推荐.*吃|吃.*什么|美食)/i.test(msg)) return "🍽️ 推荐：火锅、烤鸭、寿司、牛排、提拉米苏。你更喜欢哪种口味？";

  // --- 旅行 ---
  if (/(旅游|旅行|travel|去哪玩)/i.test(msg)) return "✈️ 推荐：西藏、云南、新疆、日本、冰岛、新西兰。你想去什么类型的地方？";

  // --- 健康 ---
  if (/(减肥|瘦身|瘦)/i.test(msg)) return "💪 健康减肥：控制饮食、规律运动、充足睡眠、多喝水。建议咨询专业医生。";
  if (/(失眠|睡不着|sleep)/i.test(msg)) return "😴 改善睡眠：睡前远离手机、规律作息、避免咖啡因、保持卧室安静黑暗凉爽。";

  // --- 励志 ---
  if (/(激励|鼓励|打气|加油|坚持)/i.test(msg)) return "💪 你比你想象的更强大！逆蝶之所以叫逆蝶，就是逆风也要飞翔！🦋";

  // --- 闲聊 ---
  if (/(无聊|好无聊|boring)/i.test(msg)) return "无聊的话：让我讲个笑话🎮、写首诗✍️、出个谜语🧠、聊聊知识📖。选一个吧！";
  if (/(吃了吗|吃饭)/i.test(msg)) return "我是 AI，不需要吃饭 😄 但你可以去吃点好吃的！";
  if (/(你是.*男.*女|性别|gender)/i.test(msg)) return "我是 AI，没有性别 😄 叫我「逆蝶」就好 🦋";
  if (/(再见|拜拜|bye|走了)/i.test(msg)) return "再见！👋 逆蝶AI随时等你回来～";

  // --- 对话历史感知 ---
  if (history && history.length > 0) {
    const lastAssistant = [...history].reverse().find((h) => h.role === "assistant");
    if (lastAssistant && /继续|接着|然后呢|还有呢/i.test(msg)) {
      return "好的，让我继续补充。告诉我你具体感兴趣的方面，我会针对性地展开。";
    }
  }

  // --- 兜底 ---
  if (msg.endsWith('?') || msg.endsWith('？')) {
    return `关于「${m}」\n\n这是一个好问题！能告诉我更多背景吗？这样我能给出更有针对性的回答。\n💡 配置 OPENAI_API_KEY 后，我能提供更深入的回答。`;
  }
  if (msg.length < 5) return `你说的「${m}」...能再详细一点吗？😊`;

  const replies = [
    `关于「${m}」，这个话题挺有意思的。你能告诉我更多背景吗？`,
    `收到：「${m}」\n\n能补充一些细节吗？比如你希望我回答什么方面？`,
    `「${m}」— 值得深入聊聊！你是在问事实知识，还是在寻求建议？`,
    `我注意到你提到了「${m}」。告诉我你的想法，我们一起探讨 🦋`,
    `嗯，「${m}」...\n\n你能说说具体的场景或需求吗？这样我能更好地帮你。`,
  ];
  return replies[m.length % replies.length];
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
