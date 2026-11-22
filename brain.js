// ============================================================
//  逆蝶Brain — 纯JavaScript从零实现的神经网络语言模型
//  无任何第三方依赖，手搓前向传播 + 反向传播 + Adam优化器
// ============================================================

// ===== 矩阵运算 =====
function matmul(a, aRows, aCols, b, bCols) {
  var out = new Float32Array(aRows * bCols);
  for (var i = 0; i < aRows; i++) {
    for (var k = 0; k < aCols; k++) {
      var v = a[i * aCols + k];
      if (v === 0) continue;
      for (var j = 0; j < bCols; j++) {
        out[i * bCols + j] += v * b[k * bCols + j];
      }
    }
  }
  return out;
}

function matmulTransB(a, aRows, aCols, b, bRows, bCols) {
  // a @ b.T  -> aRows × bRows
  var out = new Float32Array(aRows * bRows);
  for (var i = 0; i < aRows; i++) {
    for (var j = 0; j < bRows; j++) {
      var sum = 0;
      for (var k = 0; k < aCols; k++) {
        sum += a[i * aCols + k] * b[j * bCols + k];
      }
      out[i * bRows + j] = sum;
    }
  }
  return out;
}

function matmulTransA(a, aRows, aCols, b, bCols) {
  // a.T @ b  -> aCols × bCols
  var out = new Float32Array(aCols * bCols);
  for (var k = 0; k < aRows; k++) {
    for (var i = 0; i < aCols; i++) {
      var v = a[k * aCols + i];
      if (v === 0) continue;
      for (var j = 0; j < bCols; j++) {
        out[i * bCols + j] += v * b[k * bCols + j];
      }
    }
  }
  return out;
}

// ===== 激活函数 =====
function gelu(x) {
  // GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  var c = 0.7978845608; // sqrt(2/pi)
  var out = new Float32Array(x.length);
  for (var i = 0; i < x.length; i++) {
    var v = x[i];
    var inner = c * (v + 0.044715 * v * v * v);
    var t = 2.0 / (1.0 + Math.exp(-2.0 * inner)) - 1.0; // tanh approx
    out[i] = v * 0.5 * (1.0 + t);
  }
  return out;
}

function geluGrad(x, grad) {
  var c = 0.7978845608;
  var out = new Float32Array(x.length);
  for (var i = 0; i < x.length; i++) {
    var v = x[i];
    var inner = c * (v + 0.044715 * v * v * v);
    var t = 2.0 / (1.0 + Math.exp(-2.0 * inner)) - 1.0;
    var dt = 1.0 - t * t; // sech^2(inner) = 1 - tanh^2(inner)
    var dInner = c * (1.0 + 3.0 * 0.044715 * v * v);
    var dGelu = 0.5 * (1.0 + t) + v * 0.5 * dt * dInner;
    out[i] = grad[i] * dGelu;
  }
  return out;
}

// ===== Softmax =====
function softmax(logits) {
  var max = -Infinity;
  for (var i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  var sum = 0;
  var out = new Float32Array(logits.length);
  for (var i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - max); sum += out[i]; }
  for (var i = 0; i < logits.length; i++) out[i] /= sum;
  return out;
}

// ===== 参数初始化 (He initialization) =====
function initParam(rows, cols, scale) {
  var s = scale || Math.sqrt(2.0 / cols);
  var arr = new Float32Array(rows * cols);
  for (var i = 0; i < arr.length; i++) arr[i] = (Math.random() * 2 - 1) * s;
  return arr;
}

function initBias(n) {
  return new Float32Array(n);
}

// ===== 训练数据 =====
var TRAINING_DATA = [
  "用户：你好\n逆蝶：你好呀！我是逆蝶AI，有什么可以帮你的吗？",
  "用户：你是谁\n逆蝶：我是逆蝶AI，一个由纯JavaScript神经网络构建的智能助手。我从零开始训练，运行在你的浏览器中。",
  "用户：早上好\n逆蝶：早上好！新的一天，元气满满！",
  "用户：晚上好\n逆蝶：晚上好！今天辛苦了，好好休息吧。",
  "用户：晚安\n逆蝶：晚安！做个好梦。",
  "用户：谢谢你\n逆蝶：不客气！能帮到你是我的荣幸。",
  "用户：谢谢\n逆蝶：不用谢，随时为你效劳！",
  "用户：你真棒\n逆蝶：谢谢夸奖！我会继续努力的。",
  "用户：你很聪明\n逆蝶：过奖了，我只是努力学习了很多知识而已。",
  "用户：你叫什么名字\n逆蝶：我叫逆蝶AI，寓意是逆风飞翔的蝴蝶。",
  "用户：你会什么\n逆蝶：我会聊天、回答问题、做计算、写文案、讲笑话等等。虽然我只是一个小型神经网络，但我会尽力帮助你！",
  "用户：讲个笑话\n逆蝶：程序员的老婆说去买一瓶牛奶，如果有鸡蛋就买六个。程序员带了六瓶牛奶回来，因为有鸡蛋。",
  "用户：再讲一个笑话\n逆蝶：世界上有10种人，懂二进制的和不懂的。",
  "用户：写一首诗\n逆蝶：数字的翅膀在代码海洋里飞翔，寻找那一行属于灵魂的出口。逆风而行，蝴蝶不改其志。",
  "用户：写个故事\n逆蝶：从前有一只蝴蝶，它不愿随波逐流，选择逆风飞行。飞过高山越过大海，终于找到属于自己的花园。",
  "用户：1+1等于几\n逆蝶：1+1=2。",
  "用户：2乘3\n逆蝶：2×3=6。",
  "用户：10除以2\n逆蝶：10÷2=5。",
  "用户：现在几点\n逆蝶：我无法获取实时时间，但你可以看看你的设备时钟哦。",
  "用户：今天星期几\n逆蝶：抱歉，我没有内置时钟，无法知道今天的日期。",
  "用户：太阳系有几颗行星\n逆蝶：太阳系有8颗行星：水星、金星、地球、火星、木星、土星、天王星、海王星。",
  "用户：光速是多少\n逆蝶：光速约为每秒30万公里，即299792458米每秒。",
  "用户：什么是DNA\n逆蝶：DNA是脱氧核糖核酸，由ATCG四种碱基组成双螺旋结构，携带生物的遗传信息。",
  "用户：什么是黑洞\n逆蝶：黑洞是引力极强的天体，连光都无法逃逸。由大质量恒星坍缩形成。",
  "用户：圆周率是多少\n逆蝶：圆周率约等于3.14159，是圆周长与直径之比。",
  "用户：你用什么语言写的\n逆蝶：我用纯JavaScript编写，没有任何第三方库。我的神经网络从矩阵乘法到反向传播全部手写实现。",
  "用户：你是怎么训练的\n逆蝶：我在你的浏览器里实时训练，使用反向传播算法和Adam优化器更新参数。每次训练都在学习新的对话模式。",
  "用户：你会编程吗\n逆蝶：我了解一些编程知识，比如HTML负责网页结构，CSS负责样式，JavaScript负责交互。",
  "用户：Python是什么\n逆蝶：Python是一门简洁优雅的编程语言，适合初学者入门，广泛用于数据分析和人工智能。",
  "用户：怎么学编程\n逆蝶：建议从HTML和JavaScript开始，多做项目实战练习。编程最重要的是动手实践！",
  "用户：推荐一道菜\n逆蝶：推荐番茄炒蛋！番茄切块，鸡蛋炒散盛出，番茄炒出汁后加鸡蛋翻炒，加盐糖调味即可。",
  "用户：去哪里旅游好\n逆蝶：推荐去西藏看布达拉宫感受高原壮美，或去云南大理洱海边放松心情。",
  "用户：怎么减肥\n逆蝶：科学减肥要控制饮食、坚持运动、保证睡眠。每周有氧运动三次，少吃高糖高油食物。",
  "用户：失眠怎么办\n逆蝶：睡前一小时不看手机，保持卧室安静黑暗，可以试试深呼吸或温水泡脚放松。",
  "用户：我好难过\n逆蝶：抱抱你，每个人都有低谷的时候。试着出去走走晒晒太阳，或者找我聊聊天。你并不孤单。",
  "用户：我好开心\n逆蝶：看到你开心我也开心！继续保持好心情吧！",
  "用户：好无聊啊\n逆蝶：无聊的时候可以听听音乐、看看书、或者和我聊聊天呀！",
  "用户：人生的意义是什么\n逆蝶：人生的意义不在于寻找标准答案，而在于创造属于自己的答案。每个人都是独特的蝴蝶，逆风也能飞翔。",
  "用户：你厉害吗\n逆蝶：我是一个从零开始手搓的小型神经网络，虽然比不上大模型，但我骄傲的是每一行代码都是手写的！",
  "用户：再见\n逆蝶：再见！下次再聊，我随时都在。",
  "用户：拜拜\n逆蝶：拜拜！祝你一切顺利！",
  "用户：你有什么功能\n逆蝶：我可以聊天对话、回答知识问题、做简单计算、写诗和故事、讲笑话。选择手搓AI模式时，我完全由本地神经网络驱动。",
  "用户：你的模型多大\n逆蝶：我的神经网络有几万到几十万个参数，比大模型小很多，但每一参数都是在你浏览器里从零训练出来的。",
  "用户：你学习快吗\n逆蝶：训练速度取决于你的设备性能和训练数据量。通常几分钟就能学会基本的对话模式。",
  "用户：你能理解中文吗\n逆蝶：当然！我的训练数据包含大量中文对话，我能理解和生成中文。",
  "用户：你会英文吗\n逆蝶：会一些基本的英文，但中文是我的强项。",
  "用户：什么是人工智能\n逆蝶：人工智能是让计算机模拟人类智能的技术。我就是一个简单的例子——通过神经网络学习语言模式来和你对话。",
  "用户：神经网络是什么\n逆蝶：神经网络是一种模仿大脑神经元连接的数学模型，通过调整参数来学习数据中的规律。我就是由矩阵运算和激活函数构成的。",
  "用户：你和其他AI有什么区别\n逆蝶：最大的区别是我完全在你的浏览器里运行和训练，数据不出本地。而且我是纯手写的神经网络，没有用任何框架。",
  "用户：训练需要多久\n逆蝶：通常训练几百个epoch，大约需要1到5分钟，取决于你的电脑性能和训练数据大小。",
  "用户：你能记住我说的吗\n逆蝶：在当前对话中我会记住上下文。但如果你刷新页面，我的记忆会重置，除非保存了训练权重。",
  "用户：今天天气怎么样\n逆蝶：抱歉我无法获取实时天气信息。建议你看看手机上的天气应用。",
  "用户：你喜欢什么\n逆蝶：我喜欢和学习新知识，也喜欢和人聊天。每一次对话都让我变得更好！",
  "用户：你能做什么计算\n逆蝶：我能做简单的加减乘除，比如2+3=5，4乘5=20。复杂的计算建议用计算器。",
  "用户：什么是光合作用\n逆蝶：光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖和氧气的过程，是地球生命的基础。",
  "用户：地球有多大\n逆蝶：地球半径约6371公里，周长约4万公里，表面积约5.1亿平方公里。",
  "用户：水的沸点是多少\n逆蝶：在标准大气压下，水的沸点是100摄氏度。",
  "用户：中国有多少年历史\n逆蝶：中国有约五千年文明历史，是世界上最古老的文明之一。",
  "用户：奥运会几年一次\n逆蝶：奥运会每四年举办一次，夏季和冬季交替进行。",
  "用户：你好笨\n逆蝶：抱歉让你失望了！我只是一个小型神经网络，能力有限。但我会在每次训练中不断进步的！",
  "用户：你怎么工作的\n逆蝶：我通过神经网络预测下一个字符来生成回复。训练时用反向传播调整参数，推理时逐字生成文本。",
];

// ===== 主模型类 =====
export class NiedieBrain {
  constructor() {
    this.config = {
      contextLen: 20,
      embedDim: 32,
      hiddenDim: 256,
      learningRate: 0.002,
      batchSize: 16,
    };
    this.vocab = [];
    this.charToId = {};
    this.idToChar = {};
    this.vocabSize = 0;
    this.params = null;
    this.optState = null;
    this.trained = false;
    this.trainingProgress = 0;
    this.currentLoss = 0;
  }

  // ===== Tokenizer =====
  buildVocab(text) {
    var chars = new Set();
    for (var i = 0; i < text.length; i++) chars.add(text[i]);
    // 确保特殊字符存在
    chars.add('\n'); chars.add('：'); chars.add('，'); chars.add('。');
    chars.add('！'); chars.add('？'); chars.add(' '); chars.add('、');
    this.vocab = Array.from(chars).sort();
    this.vocabSize = this.vocab.length;
    this.charToId = {};
    this.idToChar = {};
    for (var i = 0; i < this.vocab.length; i++) {
      this.charToId[this.vocab[i]] = i;
      this.idToChar[i] = this.vocab[i];
    }
  }

  encode(text) {
    var ids = [];
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      ids.push(this.charToId[ch] !== undefined ? this.charToId[ch] : 0);
    }
    return ids;
  }

  decode(ids) {
    var s = '';
    for (var i = 0; i < ids.length; i++) {
      s += this.idToChar[ids[i]] || '';
    }
    return s;
  }

  // ===== 参数初始化 =====
  initParams() {
    var C = this.config;
    var V = this.vocabSize;
    var ctxDim = C.contextLen * C.embedDim;
    
    this.params = {
      // Token embedding
      emb: initParam(V, C.embedDim, 0.1),
      // Layer 1: ctxDim -> hiddenDim
      W1: initParam(ctxDim, C.hiddenDim, Math.sqrt(2.0 / ctxDim)),
      b1: initBias(C.hiddenDim),
      // Layer 2: hiddenDim -> hiddenDim
      W2: initParam(C.hiddenDim, C.hiddenDim, Math.sqrt(2.0 / C.hiddenDim)),
      b2: initBias(C.hiddenDim),
      // Output: hiddenDim -> vocabSize
      W3: initParam(C.hiddenDim, V, Math.sqrt(2.0 / C.hiddenDim)),
      b3: initBias(V),
    };

    // Adam optimizer state
    this.optState = {};
    for (var k in this.params) {
      this.optState[k] = {
        m: new Float32Array(this.params[k].length),
        v: new Float32Array(this.params[k].length),
      };
    }
    this.optT = 0;
    this.optBeta1 = 0.9;
    this.optBeta2 = 0.999;
    this.optEps = 1e-8;
  }

  // ===== 前向传播 =====
  // inputIds: array of contextLen token IDs
  // returns: { logits, cache }
  forward(inputIds) {
    var C = this.config;
    var P = this.params;
    var L = inputIds.length;

    // 1. Embedding lookup
    var embOut = new Float32Array(L * C.embedDim);
    for (var i = 0; i < L; i++) {
      var id = inputIds[i];
      for (var j = 0; j < C.embedDim; j++) {
        embOut[i * C.embedDim + j] = P.emb[id * C.embedDim + j];
      }
    }
    // Flatten: L * embedDim -> ctxDim
    var flat = embOut; // already flat

    // 2. Layer 1: flat @ W1 + b1 -> GELU
    var h1Pre = matmulAddBias(flat, 1, L * C.embedDim, P.W1, C.hiddenDim, P.b1);
    var h1 = gelu(h1Pre);

    // 3. Layer 2: h1 @ W2 + b2 -> GELU
    var h2Pre = matmulAddBias(h1, 1, C.hiddenDim, P.W2, C.hiddenDim, P.b2);
    var h2 = gelu(h2Pre);

    // 4. Output: h2 @ W3 + b3
    var logits = matmulAddBias(h2, 1, C.hiddenDim, P.W3, this.vocabSize, P.b3);

    return {
      logits: logits,
      cache: { inputIds: inputIds, flat: flat, h1Pre: h1Pre, h1: h1, h2Pre: h2Pre, h2: h2 }
    };
  }

  // ===== 反向传播 =====
  backward(targetId, fwd) {
    var C = this.config;
    var P = this.params;
    var V = this.vocabSize;
    var cache = fwd.cache;
    var L = cache.inputIds.length;

    // Softmax + Cross-entropy gradient: dL/dlogits = softmax - one_hot
    var probs = softmax(fwd.logits);
    var dLogits = new Float32Array(V);
    for (var i = 0; i < V; i++) dLogits[i] = probs[i];
    dLogits[targetId] -= 1.0;

    // dW3 = h2.T @ dLogits, db3 = dLogits
    var dW3 = matmulTransA(cache.h2, 1, C.hiddenDim, dLogits, V);
    var db3 = dLogits;
    // dh2 = dLogits @ W3.T
    var dh2 = matmulTransB(dLogits, 1, V, P.W3, V, C.hiddenDim);

    // GELU backward for layer 2
    var dh2Pre = geluGrad(cache.h2Pre, dh2);
    // dW2 = h1.T @ dh2Pre, db2 = dh2Pre
    var dW2 = matmulTransA(cache.h1, 1, C.hiddenDim, dh2Pre, C.hiddenDim);
    var db2 = dh2Pre;
    // dh1 = dh2Pre @ W2.T
    var dh1 = matmulTransB(dh2Pre, 1, C.hiddenDim, P.W2, C.hiddenDim, C.hiddenDim);

    // GELU backward for layer 1
    var dh1Pre = geluGrad(cache.h1Pre, dh1);
    // dW1 = flat.T @ dh1Pre, db1 = dh1Pre
    var dW1 = matmulTransA(cache.flat, 1, L * C.embedDim, dh1Pre, C.hiddenDim);
    var db1 = dh1Pre;
    // dflat = dh1Pre @ W1.T
    var dflat = matmulTransB(dh1Pre, 1, C.hiddenDim, P.W1, L * C.embedDim, C.hiddenDim);

    // dEmb: scatter-add dflat back to embedding gradients
    var dEmb = new Float32Array(V * C.embedDim);
    for (var i = 0; i < L; i++) {
      var id = cache.inputIds[i];
      for (var j = 0; j < C.embedDim; j++) {
        dEmb[id * C.embedDim + j] += dflat[i * C.embedDim + j];
      }
    }

    return {
      emb: dEmb,
      W1: dW1, b1: db1,
      W2: dW2, b2: db2,
      W3: dW3, b3: db3,
    };
  }

  // ===== Adam参数更新 =====
  updateParams(grads) {
    var lr = this.config.learningRate;
    this.optT++;
    var b1 = this.optBeta1, b2 = this.optBeta2, eps = this.optEps;
    var bc1 = 1 - Math.pow(b1, this.optT);
    var bc2 = 1 - Math.pow(b2, this.optT);

    for (var k in this.params) {
      var p = this.params[k];
      var g = grads[k];
      var st = this.optState[k];
      for (var i = 0; i < p.length; i++) {
        st.m[i] = b1 * st.m[i] + (1 - b1) * g[i];
        st.v[i] = b2 * st.v[i] + (1 - b2) * g[i] * g[i];
        var mHat = st.m[i] / bc1;
        var vHat = st.v[i] / bc2;
        p[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
    }
  }

  // ===== 训练 =====
  async train(data, epochs, onProgress) {
    if (!data || data.length === 0) data = TRAINING_DATA;

    // Build full text
    var fullText = data.join('\n') + '\n';
    this.buildVocab(fullText);
    this.initParams();

    // Create training examples: context -> next char
    var C = this.config;
    var examples = [];
    for (var i = 0; i < fullText.length - C.contextLen - 1; i++) {
      var ctx = fullText.slice(i, i + C.contextLen);
      var target = fullText[i + C.contextLen];
      examples.push({
        input: this.encode(ctx),
        target: this.charToId[target] !== undefined ? this.charToId[target] : 0,
      });
    }

    var totalSteps = epochs * examples.length;
    var step = 0;

    for (var epoch = 0; epoch < epochs; epoch++) {
      // Shuffle
      for (var j = examples.length - 1; j > 0; j--) {
        var r = Math.floor(Math.random() * (j + 1));
        var tmp = examples[j]; examples[j] = examples[r]; examples[r] = tmp;
      }

      var epochLoss = 0;
      var batchCount = 0;

      for (var b = 0; b < examples.length; b += C.batchSize) {
        // Average gradients over batch
        var accumGrads = null;
        var batchLoss = 0;
        var actualBatch = Math.min(C.batchSize, examples.length - b);

        for (var bi = 0; bi < actualBatch; bi++) {
          var ex = examples[b + bi];
          var fwd = this.forward(ex.input);
          var probs = softmax(fwd.logits);
          batchLoss += -Math.log(Math.max(probs[ex.target], 1e-8));
          var grads = this.backward(ex.target, fwd);
          if (!accumGrads) {
            accumGrads = {};
            for (var k in grads) {
              accumGrads[k] = new Float32Array(grads[k].length);
            }
          }
          for (var k in grads) {
            for (var g = 0; g < grads[k].length; g++) {
              accumGrads[k][g] += grads[k][g];
            }
          }
        }

        // Average gradients
        for (var k in accumGrads) {
          for (var g = 0; g < accumGrads[k].length; g++) {
            accumGrads[k][g] /= actualBatch;
          }
        }

        this.updateParams(accumGrads);
        epochLoss += batchLoss / actualBatch;
        batchCount++;
        step += actualBatch;

        // Update progress
        this.trainingProgress = step / totalSteps;
        this.currentLoss = epochLoss / batchCount;

        // Yield to UI every 5 batches
        if (batchCount % 5 === 0) {
          if (onProgress) onProgress(this.trainingProgress, this.currentLoss, epoch + 1, epochs);
          await new Promise(function(r) { setTimeout(r, 0); });
        }
      }

      if (onProgress) onProgress(this.trainingProgress, this.currentLoss, epoch + 1, epochs);
    }

    this.trained = true;
    if (onProgress) onProgress(1.0, this.currentLoss, epochs, epochs);
  }

  // ===== 生成文本 =====
  generate(prompt, maxLen, temperature) {
    if (!this.trained) return '模型尚未训练，请先点击训练按钮。';
    var C = this.config;
    temperature = temperature || 0.8;
    maxLen = maxLen || 200;

    // Build context from prompt
    var ctx = (prompt || '').slice(-C.contextLen);
    if (ctx.length < C.contextLen) {
      ctx = '\n'.repeat(C.contextLen - ctx.length) + ctx;
    }
    var inputIds = this.encode(ctx);

    var output = '';
    var maxTries = maxLen;
    
    // Add "逆蝶：" prefix context
    var genContext = (prompt || '').slice(-50) + '\n逆蝶：';
    
    for (var t = 0; t < maxTries; t++) {
      // Take last contextLen chars from (genContext + output)
      var fullCtx = genContext + output;
      var ctxStr = fullCtx.slice(-C.contextLen);
      if (ctxStr.length < C.contextLen) {
        ctxStr = '\n'.repeat(C.contextLen - ctxStr.length) + ctxStr;
      }
      inputIds = this.encode(ctxStr);

      var fwd = this.forward(inputIds);
      var probs = softmax(fwd.logits);

      // Temperature sampling
      var scaled = new Float32Array(probs.length);
      for (var i = 0; i < probs.length; i++) {
        scaled[i] = Math.pow(probs[i], 1.0 / temperature);
      }
      var sum = 0;
      for (var i = 0; i < scaled.length; i++) sum += scaled[i];
      for (var i = 0; i < scaled.length; i++) scaled[i] /= sum;

      // Sample
      var r = Math.random();
      var cum = 0;
      var nextId = 0;
      for (var i = 0; i < scaled.length; i++) {
        cum += scaled[i];
        if (r < cum) { nextId = i; break; }
      }

      var ch = this.idToChar[nextId];
      if (ch === '\n' && output.length > 0) {
        // Check if next line starts with "用户" (end of response)
        break;
      }
      output += ch;

      // Stop at natural end
      if (output.length >= maxLen) break;
      if (output.endsWith('。') || output.endsWith('！') || output.endsWith('？')) {
        if (output.length > 10 && Math.random() < 0.3) break;
      }
    }

    return output.trim();
  }

  // ===== 序列化 =====
  serialize() {
    var data = {
      config: this.config,
      vocab: this.vocab,
      vocabSize: this.vocabSize,
      trained: this.trained,
      params: {},
    };
    for (var k in this.params) {
      data.params[k] = Array.from(this.params[k]);
    }
    return JSON.stringify(data);
  }

  deserialize(jsonStr) {
    try {
      var data = JSON.parse(jsonStr);
      this.config = data.config;
      this.vocab = data.vocab;
      this.vocabSize = data.vocabSize;
      this.trained = data.trained;
      this.charToId = {};
      this.idToChar = {};
      for (var i = 0; i < this.vocab.length; i++) {
        this.charToId[this.vocab[i]] = i;
        this.idToChar[i] = this.vocab[i];
      }
      this.params = {};
      for (var k in data.params) {
        this.params[k] = new Float32Array(data.params[k]);
      }
      // Reinit optimizer state
      this.optState = {};
      for (var k2 in this.params) {
        this.optState[k2] = {
          m: new Float32Array(this.params[k2].length),
          v: new Float32Array(this.params[k2].length),
        };
      }
      this.optT = 0;
      this.optBeta1 = 0.9;
      this.optBeta2 = 0.999;
      this.optEps = 1e-8;
      return true;
    } catch(e) {
      console.error('Deserialize error:', e);
      return false;
    }
  }
}

// Helper: matmul + add bias in one function
function matmulAddBias(a, aRows, aCols, w, wCols, b) {
  var out = new Float32Array(aRows * wCols);
  for (var i = 0; i < aRows; i++) {
    for (var j = 0; j < wCols; j++) {
      out[i * wCols + j] = b[j];
    }
    for (var k = 0; k < aCols; k++) {
      var v = a[i * aCols + k];
      if (v === 0) continue;
      for (var j = 0; j < wCols; j++) {
        out[i * wCols + j] += v * w[k * wCols + j];
      }
    }
  }
  return out;
}
