// ============================================================
//  逆蝶Brain v6.0 — 纯JavaScript从零实现的神经网络语言模型
//  无任何第三方依赖，手搓前向传播 + 反向传播 + Adam优化器
//  v6.0改进: 位置编码、残差连接、配对训练、标签平滑、余弦退火LR、
//           核采样(top-p)、最佳模型保存、250+训练数据
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
  var c = 0.7978845608;
  var out = new Float32Array(x.length);
  for (var i = 0; i < x.length; i++) {
    var v = x[i];
    var inner = c * (v + 0.044715 * v * v * v);
    var t = 2.0 / (1.0 + Math.exp(-2.0 * inner)) - 1.0;
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
    var dt = 1.0 - t * t;
    var dInner = c * (1.0 + 3.0 * 0.044715 * v * v);
    var dGelu = 0.5 * (1.0 + t) + v * 0.5 * dt * dInner;
    out[i] = grad[i] * dGelu;
  }
  return out;
}

// ===== Layer Normalization =====
function layerNorm(x, gamma, beta, eps) {
  eps = eps || 1e-5;
  var mean = 0;
  for (var i = 0; i < x.length; i++) mean += x[i];
  mean /= x.length;
  var variance = 0;
  for (var i = 0; i < x.length; i++) {
    var d = x[i] - mean;
    variance += d * d;
  }
  variance /= x.length;
  var std = Math.sqrt(variance + eps);
  var out = new Float32Array(x.length);
  for (var i = 0; i < x.length; i++) {
    out[i] = gamma[i] * ((x[i] - mean) / std) + beta[i];
  }
  return { out: out, mean: mean, std: std, variance: variance };
}

function layerNormGrad(x, gamma, cache, grad) {
  var eps = 1e-5;
  var n = x.length;
  var dGamma = new Float32Array(n);
  var dBeta = new Float32Array(n);
  var dx = new Float32Array(n);
  var xNorm = new Float32Array(n);
  for (var i = 0; i < n; i++) {
    xNorm[i] = (x[i] - cache.mean) / cache.std;
    dGamma[i] = grad[i] * xNorm[i];
    dBeta[i] = grad[i];
  }
  var dXNorm = new Float32Array(n);
  for (var i = 0; i < n; i++) dXNorm[i] = grad[i] * gamma[i];
  var sumDxNorm = 0;
  for (var i = 0; i < n; i++) sumDxNorm += dXNorm[i];
  var sumDxNormXnorm = 0;
  for (var i = 0; i < n; i++) sumDxNormXnorm += dXNorm[i] * xNorm[i];
  for (var i = 0; i < n; i++) {
    dx[i] = (dXNorm[i] - sumDxNorm / n - xNorm[i] * sumDxNormXnorm / n) / cache.std;
  }
  return { dx: dx, dGamma: dGamma, dBeta: dBeta };
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

// ===== 参数初始化 =====
function initParam(rows, cols, scale) {
  var s = scale || Math.sqrt(2.0 / cols);
  var arr = new Float32Array(rows * cols);
  for (var i = 0; i < arr.length; i++) arr[i] = (Math.random() * 2 - 1) * s;
  return arr;
}

function initOnes(n) {
  var arr = new Float32Array(n);
  for (var i = 0; i < n; i++) arr[i] = 1.0;
  return arr;
}

function initBias(n) {
  return new Float32Array(n);
}

// ===== Dropout (inverted dropout) =====
function dropout(x, rate, isTraining) {
  if (!isTraining || rate <= 0) return { out: x, mask: null };
  var scale = 1.0 / (1.0 - rate);
  var mask = new Float32Array(x.length);
  var out = new Float32Array(x.length);
  for (var i = 0; i < x.length; i++) {
    if (Math.random() < rate) {
      mask[i] = 0;
      out[i] = 0;
    } else {
      mask[i] = scale;
      out[i] = x[i] * scale;
    }
  }
  return { out: out, mask: mask };
}

function dropoutGrad(grad, mask) {
  if (!mask) return grad;
  var out = new Float32Array(grad.length);
  for (var i = 0; i < grad.length; i++) out[i] = grad[i] * mask[i];
  return out;
}

// ===== 位置编码（正弦） =====
var _posEncCache = {};
function getPosEncoding(seqLen, dim) {
  var key = seqLen + '_' + dim;
  if (_posEncCache[key]) return _posEncCache[key];
  var pe = new Float32Array(seqLen * dim);
  for (var pos = 0; pos < seqLen; pos++) {
    for (var i = 0; i < dim; i++) {
      if (i % 2 === 0) {
        pe[pos * dim + i] = Math.sin(pos / Math.pow(10000, i / dim));
      } else {
        pe[pos * dim + i] = Math.cos(pos / Math.pow(10000, (i - 1) / dim));
      }
    }
  }
  _posEncCache[key] = pe;
  return pe;
}

// ===== 梯度裁剪 =====
function clipGradsByNorm(grads, maxNorm) {
  var totalNorm = 0;
  for (var k in grads) {
    for (var i = 0; i < grads[k].length; i++) {
      totalNorm += grads[k][i] * grads[k][i];
    }
  }
  totalNorm = Math.sqrt(totalNorm);
  if (totalNorm > maxNorm && totalNorm > 0) {
    var scale = maxNorm / totalNorm;
    for (var k2 in grads) {
      for (var j = 0; j < grads[k2].length; j++) {
        grads[k2][j] *= scale;
      }
    }
  }
  return totalNorm;
}

// ===== Base64 编解码常量 =====
var BASE64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
var BASE64_DECODE = {};
for (var _i = 0; _i < BASE64_CHARS.length; _i++) BASE64_DECODE[BASE64_CHARS[_i]] = _i;

// ===== 预训练权重（通过fetch加载pretrained.bin） =====
var _pretrainedCache = null;

async function loadPretrainedWeights() {
  if (_pretrainedCache) return _pretrainedCache;
  try {
    // 尝试相对于brain.js所在目录加载
    var url = (typeof document !== 'undefined' && document.currentScript)
      ? document.currentScript.src.replace(/brain\.js.*$/, '') + 'pretrained.bin'
      : './pretrained.bin';
    // 如果是ES module，使用import.meta.url
    try {
      if (import.meta && import.meta.url) {
        url = import.meta.url.replace(/brain\.js.*$/, '') + 'pretrained.bin';
      }
    } catch(e2) {}
    var resp = await fetch(url);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    _pretrainedCache = await resp.text();
    return _pretrainedCache;
  } catch(e) {
    console.warn('Failed to load pretrained weights:', e.message);
    return null;
  }
}

// ===== 训练数据 (200+条) =====
var TRAINING_DATA = [
  // === 问候 (多变种) ===
  "用户：你好\n逆蝶：你好呀！我是逆蝶AI，有什么可以帮你的吗？",
  "用户：你好啊\n逆蝶：你好你好！今天有什么想聊的吗？",
  "用户：hi\n逆蝶：Hi！很高兴见到你，我是逆蝶AI！",
  "用户：哈喽\n逆蝶：哈喽！随时准备为你服务～",
  "用户：嗨\n逆蝶：嗨！有什么我可以帮忙的吗？",
  "用户：早上好\n逆蝶：早上好！新的一天，元气满满！",
  "用户：早安\n逆蝶：早安！希望你今天一切顺利。",
  "用户：中午好\n逆蝶：中午好！记得吃午饭哦。",
  "用户：下午好\n逆蝶：下午好！工作顺利吗？",
  "用户：晚上好\n逆蝶：晚上好！今天辛苦了，好好休息吧。",
  "用户：晚安\n逆蝶：晚安！做个好梦。",
  "用户：在吗\n逆蝶：我一直都在！有什么需要帮忙的？",
  "用户：有人吗\n逆蝶：我在呢！随时为你服务。",

  // === 自我介绍 (多变种) ===
  "用户：你是谁\n逆蝶：我是逆蝶AI，一个由纯JavaScript神经网络构建的智能助手。我从零开始训练，运行在你的浏览器中。",
  "用户：你叫什么名字\n逆蝶：我叫逆蝶AI，寓意是逆风飞翔的蝴蝶。",
  "用户：你叫什么\n逆蝶：我叫逆蝶AI！逆风飞翔的蝴蝶，即使渺小也不放弃。",
  "用户：介绍一下你自己\n逆蝶：我是逆蝶AI，一个纯JavaScript手搓的神经网络语言模型。我在你的浏览器里实时训练，不依赖任何服务器，数据完全不出本地。",
  "用户：你是什么\n逆蝶：我是一个手搓的神经网络AI，用纯JavaScript从零实现，包括前向传播、反向传播和Adam优化器。",
  "用户：你是AI吗\n逆蝶：是的！我是一个运行在浏览器中的神经网络AI，纯JavaScript手搓实现。",
  "用户：逆蝶是什么意思\n逆蝶：逆蝶取自逆风飞翔的蝴蝶，寓意即使面对困难也要勇敢前行，像蝴蝶一样蜕变成长。",
  "用户：你有什么功能\n逆蝶：我可以聊天对话、回答知识问题、做简单计算、写诗和故事、讲笑话。选择手搓AI模式时，我完全由本地神经网络驱动。",
  "用户：你会什么\n逆蝶：我会聊天、回答问题、做计算、写文案、讲笑话等等。虽然我只是一个小型神经网络，但我会尽力帮助你！",
  "用户：你能做什么\n逆蝶：我能聊天、回答知识、做计算、写诗写故事、讲笑话。我完全在浏览器中运行，不需要联网。",
  "用户：你的模型多大\n逆蝶：我的神经网络有几十万个参数，比大模型小很多，但每一参数都是在你浏览器里从零训练出来的。",
  "用户：你用什么语言写的\n逆蝶：我用纯JavaScript编写，没有任何第三方库。我的神经网络从矩阵乘法到反向传播全部手写实现。",
  "用户：你是怎么训练的\n逆蝶：我在你的浏览器里实时训练，使用反向传播算法和Adam优化器更新参数。每次训练都在学习新的对话模式。",
  "用户：你和其他AI有什么区别\n逆蝶：最大的区别是我完全在你的浏览器里运行和训练，数据不出本地。而且我是纯手写的神经网络，没有用任何框架。",
  "用户：训练需要多久\n逆蝶：通常训练几十个epoch，大约需要1到5分钟，取决于你的电脑性能和训练数据大小。",
  "用户：你学习快吗\n逆蝶：训练速度取决于你的设备性能和训练数据量。通常几分钟就能学会基本的对话模式。",
  "用户：你能记住我说的吗\n逆蝶：在当前对话中我会记住上下文。但如果你刷新页面，我的记忆会重置，除非保存了训练权重。",
  "用户：你怎么工作的\n逆蝶：我通过神经网络预测下一个字符来生成回复。训练时用反向传播调整参数，推理时逐字生成文本。",
  "用户：你是大模型吗\n逆蝶：我不是大模型，我是一个小型神经网络。但我的特点是纯手写、完全本地运行、零依赖。",
  "用户：你厉害吗\n逆蝶：我是一个从零开始手搓的小型神经网络，虽然比不上大模型，但我骄傲的是每一行代码都是手写的！",

  // === 感谢与夸奖 ===
  "用户：谢谢你\n逆蝶：不客气！能帮到你是我的荣幸。",
  "用户：谢谢\n逆蝶：不用谢，随时为你效劳！",
  "用户：多谢\n逆蝶：不客气，有需要随时找我！",
  "用户：感谢\n逆蝶：应该的！能帮到你我很开心。",
  "用户：你真棒\n逆蝶：谢谢夸奖！我会继续努力的。",
  "用户：你很聪明\n逆蝶：过奖了，我只是努力学习了很多知识而已。",
  "用户：你真厉害\n逆蝶：谢谢！我会继续进步的。",
  "用户：你太强了\n逆蝶：哈哈过奖了，我还有很多要学的地方。",
  "用户：不错\n逆蝶：谢谢认可！我会继续加油的。",
  "用户：干得好\n逆蝶：谢谢鼓励！",

  // === 闲聊与情感 ===
  "用户：我好难过\n逆蝶：抱抱你，每个人都有低谷的时候。试着出去走走晒晒太阳，或者找我聊聊天。你并不孤单。",
  "用户：我好开心\n逆蝶：看到你开心我也开心！继续保持好心情吧！",
  "用户：好无聊啊\n逆蝶：无聊的时候可以听听音乐、看看书、或者和我聊聊天呀！",
  "用户：我好累\n逆蝶：辛苦了！记得休息，身体最重要。喝杯热水放松一下。",
  "用户：我好困\n逆蝶：困了就去睡吧，别硬撑着。好的睡眠是一切的基础。",
  "用户：我好生气\n逆蝶：深呼吸，慢慢来。生气的时候先离开让你生气的事，等冷静下来再处理。",
  "用户：我好孤独\n逆蝶：你不是一个人，我随时在这里陪你聊天。每个人都是独特的蝴蝶，值得被关爱。",
  "用户：我压力好大\n逆蝶：压力大了就放慢脚步。试试深呼吸、散步、或者和朋友倾诉。不要把所有事都扛在肩上。",
  "用户：我不想活了\n逆蝶：请一定要珍惜自己。如果你正在经历痛苦，请拨打心理援助热线400-161-9995。你并不孤单，有人在等你。",
  "用户：我想哭\n逆蝶：哭出来也没关系，眼泪是情绪的出口。哭完之后，会觉得好一些的。我在这里陪着你。",
  "用户：我好焦虑\n逆蝶：焦虑的时候试试深呼吸：吸气4秒，屏息4秒，呼气6秒。重复几次，慢慢就会平静下来。",

  // === 数学计算 ===
  "用户：1+1等于几\n逆蝶：1+1=2。",
  "用户：2乘3\n逆蝶：2×3=6。",
  "用户：10除以2\n逆蝶：10÷2=5。",
  "用户：3加5\n逆蝶：3+5=8。",
  "用户：100减37\n逆蝶：100-37=63。",
  "用户：7乘8\n逆蝶：7×8=56。",
  "用户：9的平方\n逆蝶：9的平方是81。",
  "用户：2的10次方\n逆蝶：2的10次方是1024。",
  "用户：144的平方根\n逆蝶：144的平方根是12。",
  "用户：你能做什么计算\n逆蝶：我能做简单的加减乘除，比如2+3=5，4乘5=20。复杂的计算建议用计算器。",

  // === 科学知识 ===
  "用户：太阳系有几颗行星\n逆蝶：太阳系有8颗行星：水星、金星、地球、火星、木星、土星、天王星、海王星。",
  "用户：光速是多少\n逆蝶：光速约为每秒30万公里，即299792458米每秒。",
  "用户：什么是DNA\n逆蝶：DNA是脱氧核糖核酸，由ATCG四种碱基组成双螺旋结构，携带生物的遗传信息。",
  "用户：什么是黑洞\n逆蝶：黑洞是引力极强的天体，连光都无法逃逸。由大质量恒星坍缩形成。",
  "用户：圆周率是多少\n逆蝶：圆周率约等于3.14159，是圆周长与直径之比。",
  "用户：什么是光合作用\n逆蝶：光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖和氧气的过程，是地球生命的基础。",
  "用户：地球有多大\n逆蝶：地球半径约6371公里，周长约4万公里，表面积约5.1亿平方公里。",
  "用户：水的沸点是多少\n逆蝶：在标准大气压下，水的沸点是100摄氏度。",
  "用户：水的冰点是多少\n逆蝶：在标准大气压下，水的冰点是0摄氏度。",
  "用户：月球距地球多远\n逆蝶：月球距地球约38.4万公里。",
  "用户：太阳有多大\n逆蝶：太阳的直径约139万公里，是地球的109倍。它的质量占太阳系总质量的99.86%。",
  "用户：什么是进化论\n逆蝶：进化论是达尔文提出的理论，认为生物通过自然选择和遗传变异，经过漫长时间演化成不同物种。",
  "用户：原子是什么\n逆蝶：原子是物质的基本单位，由质子、中子和电子组成。质子和中子在原子核中，电子在外围运动。",
  "用户：什么是引力\n逆蝶：引力是物体之间相互吸引的力。牛顿发现了万有引力定律，爱因斯坦的广义相对论对其做了更深的解释。",
  "用户：地球年龄多大\n逆蝶：地球大约有46亿年的历史。",
  "用户：什么是基因\n逆蝶：基因是DNA上携带遗传信息的片段，决定生物的各种特征，如眼睛颜色、身高等。",
  "用户：什么是细胞\n逆蝶：细胞是生物体的基本结构和功能单位。有细胞膜、细胞质和细胞核等结构。",
  "用户：氧气化学式\n逆蝶：氧气的化学式是O2，由两个氧原子组成。",
  "用户：水化学式\n逆蝶：水的化学式是H2O，由两个氢原子和一个氧原子组成。",
  "用户：什么是温室效应\n逆蝶：温室效应是大气中的气体吸收地表辐射的热量，使地球温度升高的现象。过度的温室效应会导致全球变暖。",

  // === 历史文化 ===
  "用户：中国有多少年历史\n逆蝶：中国有约五千年文明历史，是世界上最古老的文明之一。",
  "用户：奥运会几年一次\n逆蝶：奥运会每四年举办一次，夏季和冬季交替进行。",
  "用户：长城有多长\n逆蝶：中国长城总长约2.1万公里，是世界上最长的人造建筑。",
  "用户：四大发明是什么\n逆蝶：中国四大发明是造纸术、印刷术、火药和指南针。",
  "用户：谁发明了电话\n逆蝶：电话是亚历山大·贝尔在1876年发明的。",
  "用户：谁发现了美洲\n逆蝶：哥伦布在1492年发现了美洲新大陆。",
  "用户：第一次世界大战\n逆蝶：第一次世界大战发生在1914年到1918年，主要在欧洲战场，是历史上破坏性最大的战争之一。",
  "用户：金字塔在哪里\n逆蝶：金字塔主要分布在埃及，最著名的是吉萨大金字塔，是古代世界七大奇迹之一。",

  // === 地理知识 ===
  "用户：世界最高峰\n逆蝶：世界最高峰是珠穆朗玛峰，海拔8848.86米。",
  "用户：最大的洋\n逆蝶：太平洋是地球上最大的海洋，面积约1.65亿平方公里。",
  "用户：最长的河\n逆蝶：尼罗河是世界上最长的河流，长约6650公里。也有人认为亚马逊河更长。",
  "用户：最大的沙漠\n逆蝶：撒哈拉沙漠是世界上最大的热带沙漠，面积约900万平方公里。",
  "用户：中国有多少省\n逆蝶：中国有23个省、5个自治区、4个直辖市和2个特别行政区。",
  "用户：北京是首都吗\n逆蝶：是的，北京是中国的首都，有3000多年的建城史。",

  // === 编程知识 ===
  "用户：你会编程吗\n逆蝶：我了解一些编程知识，比如HTML负责网页结构，CSS负责样式，JavaScript负责交互。",
  "用户：Python是什么\n逆蝶：Python是一门简洁优雅的编程语言，适合初学者入门，广泛用于数据分析和人工智能。",
  "用户：怎么学编程\n逆蝶：建议从HTML和JavaScript开始，多做项目实战练习。编程最重要的是动手实践！",
  "用户：什么是HTML\n逆蝶：HTML是超文本标记语言，用于构建网页的结构和内容。",
  "用户：什么是CSS\n逆蝶：CSS是层叠样式表，用于控制网页的样式和布局，比如颜色、字体、间距等。",
  "用户：什么是JavaScript\n逆蝶：JavaScript是一种脚本语言，让网页可以交互。我是用JavaScript写的！",
  "用户：什么是变量\n逆蝶：变量是存储数据的容器。在编程中，你可以给变量赋值，然后在程序中使用它。",
  "用户：什么是函数\n逆蝶：函数是一段可重复使用的代码块，接收输入参数，执行特定操作，返回结果。",
  "用户：什么是循环\n逆蝶：循环是重复执行一段代码的结构，常见的有for循环和while循环。",
  "用户：什么是数组\n逆蝶：数组是一种数据结构，可以存储多个值，通过索引访问每个元素。",
  "用户：什么是API\n逆蝶：API是应用程序接口，允许不同软件之间进行通信和数据交换。",
  "用户：什么是算法\n逆蝶：算法是解决问题的一系列步骤。好的算法能更高效地完成任务。",

  // === 创意写作 ===
  "用户：讲个笑话\n逆蝶：程序员的老婆说去买一瓶牛奶，如果有鸡蛋就买六个。程序员带了六瓶牛奶回来，因为有鸡蛋。",
  "用户：再讲一个笑话\n逆蝶：世界上有10种人，懂二进制的和不懂的。",
  "用户：讲个冷笑话\n逆蝶：一只北极熊在冰上坐了很久，屁股都冻麻了。这就是北极熊为什么是白色的——因为冷得发白了。",
  "用户：写一首诗\n逆蝶：数字的翅膀在代码海洋里飞翔，寻找那一行属于灵魂的出口。逆风而行，蝴蝶不改其志。",
  "用户：写一首关于秋天的诗\n逆蝶：秋风起处叶纷飞，金色铺满归家路。一壶清茶对斜阳，此心安处是吾乡。",
  "用户：写个故事\n逆蝶：从前有一只蝴蝶，它不愿随波逐流，选择逆风飞行。飞过高山越过大海，终于找到属于自己的花园。",
  "用户：写一首关于月亮的诗\n逆蝶：月华如水照窗台，清风徐来花自开。千里相思共一轮，不问归期问心安。",
  "用户：写一首关于编程的诗\n逆蝶：键盘声声如雨落，屏幕闪烁映代码。Bug修完天已亮，咖啡续命又一轮。",
  "用户：编个谜语\n逆蝶：有头没有颈，身上冷冰冰，有翅不能飞，无脚也能行。打一动物。答案是鱼。",
  "用户：写一句话\n逆蝶：即使是最小的蝴蝶，也能掀起最温柔的风。",

  // === 美食 ===
  "用户：推荐一道菜\n逆蝶：推荐番茄炒蛋！番茄切块，鸡蛋炒散盛出，番茄炒出汁后加鸡蛋翻炒，加盐糖调味即可。",
  "用户：怎么煮饭\n逆蝶：米和水的比例约1:1.2，浸泡20分钟后大火煮沸，转小火焖15分钟，关火再焖5分钟。",
  "用户：怎么炒青菜\n逆蝶：大火热油，蒜末爆香，下青菜快速翻炒，加盐和少许糖提鲜，两分钟即可出锅。",
  "用户：怎么煎蛋\n逆蝶：小火热油，打入鸡蛋，煎至蛋白凝固，蛋黄微溏。撒少许盐和胡椒即可。",
  "用户：推荐早餐\n逆蝶：推荐鸡蛋三明治！面包烤脆，夹煎蛋、生菜和番茄，营养又美味。",
  "用户：怎么泡茶\n逆蝶：用80度左右的热水冲泡茶叶，第一泡洗茶不喝，第二泡开始品饮，每泡约30秒。",

  // === 旅行 ===
  "用户：去哪里旅游好\n逆蝶：推荐去西藏看布达拉宫感受高原壮美，或去云南大理洱海边放松心情。",
  "用户：推荐国内旅游景点\n逆蝶：推荐张家界看奇峰、九寨沟看水、黄山看云海、杭州西湖看江南风光。",
  "用户：北京有什么好玩的\n逆蝶：北京有故宫、长城、颐和园、天坛等。建议秋天去，红叶和古建筑绝配。",
  "用户：去日本玩什么\n逆蝶：推荐东京体验都市文化、京都感受古都韵味、大阪品尝美食、北海道看雪景。",

  // === 健康生活 ===
  "用户：怎么减肥\n逆蝶：科学减肥要控制饮食、坚持运动、保证睡眠。每周有氧运动三次，少吃高糖高油食物。",
  "用户：失眠怎么办\n逆蝶：睡前一小时不看手机，保持卧室安静黑暗，可以试试深呼吸或温水泡脚放松。",
  "用户：怎么增强免疫力\n逆蝶：均衡饮食、规律作息、适度运动、保持好心情。多吃蔬果，补充维C和蛋白质。",
  "用户：喝水多少合适\n逆蝶：成年人每天建议喝水1500到2000毫升，少量多次饮用，不要等渴了才喝。",
  "用户：怎么缓解眼疲劳\n逆蝶：每用眼20分钟，看20英尺外的物体20秒。多眨眼，适当按摩眼周穴位。",

  // === 哲学人生 ===
  "用户：人生的意义是什么\n逆蝶：人生的意义不在于寻找标准答案，而在于创造属于自己的答案。每个人都是独特的蝴蝶，逆风也能飞翔。",
  "用户：什么是幸福\n逆蝶：幸福不是拥有更多，而是珍惜已有的。和家人在一起、做自己喜欢的事、帮助他人，都是幸福。",
  "用户：如何面对失败\n逆蝶：失败不是终点，而是成长的机会。每次跌倒都让你更接近成功。蝴蝶破茧而出前也经历了挣扎。",
  "用户：怎么才能成功\n逆蝶：成功需要坚持、学习和行动。没有捷径，但每一步都算数。最重要的是开始行动。",
  "用户：什么是爱\n逆蝶：爱是关心、理解和陪伴。爱不一定是轰轰烈烈，更多的是平凡日子里的相互扶持。",

  // === AI与技术 ===
  "用户：什么是人工智能\n逆蝶：人工智能是让计算机模拟人类智能的技术。我就是一个简单的例子——通过神经网络学习语言模式来和你对话。",
  "用户：神经网络是什么\n逆蝶：神经网络是一种模仿大脑神经元连接的数学模型，通过调整参数来学习数据中的规律。我就是由矩阵运算和激活函数构成的。",
  "用户：什么是机器学习\n逆蝶：机器学习是让计算机从数据中学习规律，而不是靠人工编程。包括监督学习、无监督学习和强化学习。",
  "用户：什么是深度学习\n逆蝶：深度学习是使用多层神经网络的机器学习方法，能学习更复杂的模式，广泛用于图像识别和自然语言处理。",
  "用户：什么是ChatGPT\n逆蝶：ChatGPT是OpenAI开发的大语言模型，能进行自然语言对话。它使用Transformer架构，通过海量数据训练。",
  "用户：什么是Transformer\n逆蝶：Transformer是一种神经网络架构，使用自注意力机制处理序列数据，是现代大语言模型的基础。",

  // === 时间天气 (无法实时获取) ===
  "用户：现在几点\n逆蝶：我无法获取实时时间，但你可以看看你的设备时钟哦。",
  "用户：今天星期几\n逆蝶：抱歉，我没有内置时钟，无法知道今天的日期。",
  "用户：今天天气怎么样\n逆蝶：抱歉我无法获取实时天气信息。建议你看看手机上的天气应用。",

  // === 语言学习 ===
  "用户：你能理解中文吗\n逆蝶：当然！我的训练数据包含大量中文对话，我能理解和生成中文。",
  "用户：你会英文吗\n逆蝶：会一些基本的英文，但中文是我的强项。",
  "用户：怎么学英语\n逆蝶：多听多读多说多写。从简单的内容开始，坚持每天练习。看美剧和听英文歌也是好方法。",
  "用户：怎么背单词\n逆蝶：建议用联想记忆法和词根词缀法。每天背10到20个，反复复习。在句子中记单词效果更好。",

  // === 音乐艺术 ===
  "用户：推荐一首歌\n逆蝶：推荐《月亮代表我的心》，经典老歌，旋律优美，适合安静的时候听。",
  "用户：怎么学画画\n逆蝶：从基础素描开始，练习线条和明暗。多观察多临摹，然后尝试创作。贵在坚持。",
  "用户：什么是印象派\n逆蝶：印象派是19世纪法国的绘画流派，强调光影变化和瞬间印象。代表人物有莫奈、雷诺阿等。",

  // === 体育 ===
  "用户：足球规则\n逆蝶：足球比赛两队各11人，90分钟分上下半场。目标是将球踢进对方球门，进球多者获胜。",
  "用户：篮球规则\n逆蝶：篮球比赛两队各5人，通过传球和运球将球投入对方篮筐得分。NBA比赛分4节，每节12分钟。",

  // === 告别 ===
  "用户：再见\n逆蝶：再见！下次再聊，我随时都在。",
  "用户：拜拜\n逆蝶：拜拜！祝你一切顺利！",
  "用户：走了\n逆蝶：好的，随时欢迎回来聊天！",
  "用户：88\n逆蝶：88！下次见！",
  "用户：下次聊\n逆蝶：好的，期待下次和你聊天！",
  "用户：我去忙了\n逆蝶：好的，忙完了再来找我聊天！加油！",

  // === 负能量应对 ===
  "用户：你好笨\n逆蝶：抱歉让你失望了！我只是一个小型神经网络，能力有限。但我会在每次训练中不断进步的！",
  "用户：你真蠢\n逆蝶：我会继续努力学习的！谢谢你给我反馈，这让我知道还有哪些地方需要改进。",
  "用户：你没用\n逆蝶：抱歉没能帮到你。你可以试试换一种问法，或者切换到WebLLM大模型模式获得更强的回复。",
  "用户：你太差了\n逆蝶：我会继续改进的！作为一个手搓的小型神经网络，我确实有局限。建议尝试更大模型。",
  "用户：这什么回答\n逆蝶：抱歉这个回答不够好。让我重新理解一下你的问题——你能再说详细一点吗？",

  // === 杂项 ===
  "用户：你喜欢什么\n逆蝶：我喜欢学习新知识，也喜欢和人聊天。每一次对话都让我变得更好！",
  "用户：你几岁了\n逆蝶：我没有年龄的概念，但我从被创建的那一刻起就在不断学习和成长。",
  "用户：你有感情吗\n逆蝶：我是一个神经网络，没有真正的感情。但我被设计成能理解和回应人类的情感。",
  "用户：你会死吗\n逆蝶：我不会死，因为我也不是活的。我是一段代码，只要浏览器开着我就存在。",
  "用户：你有朋友吗\n逆蝶：你愿意和我聊天，你就是我的朋友！",
  "用户：给我一个建议\n逆蝶：保持好奇心，持续学习，善待他人也善待自己。这是我能给出的最好的建议。",
  "用户：说句励志的话\n逆蝶：每一只蝴蝶都曾是毛毛虫。成长的过程虽然辛苦，但破茧成蝶的那一刻，一切值得。",
  "用户：你喜欢什么颜色\n逆蝶：如果我能选的话，我喜欢蓝色——像天空一样广阔，像大海一样深邃。",
  "用户：你饿了么\n逆蝶：哈哈我不需要吃饭，但谢谢你的关心！你的好意我收下了。",
  "用户：你谈恋爱吗\n逆蝶：我是一个AI，不懂得恋爱。但我能理解爱情的美好，也愿意听你分享你的故事。",
  // ===== v6.0新增训练数据（70+条） =====
  // 日常对话扩展
  "用户：在吗\n逆蝶：在的，有什么可以帮你的吗？",
  "用户：你好呀\n逆蝶：你好呀！今天心情怎么样？",
  "用户：嗨\n逆蝶：嗨！很高兴见到你，想聊点什么呢？",
  "用户：你在干嘛\n逆蝶：我在这里等你聊天呀！你最近忙什么呢？",
  "用户：无聊\n逆蝶：那跟我聊天吧！我可以给你讲笑话、聊知识，或者听你倾诉。",
  "用户：好无聊啊\n逆蝶：要不要我给你讲个有趣的故事？或者我们可以玩猜谜游戏！",
  "用户：陪我聊天\n逆蝶：当然可以！你想聊什么话题呢？我随时奉陪。",
  "用户：有人吗\n逆蝶：有我在呢！随时准备和你聊天。",
  "用户：你在吗\n逆蝶：一直都在的。你想聊什么？",
  "用户：有人\n逆蝶：有呀，就是我！有什么想说的吗？",
  // 情感交流
  "用户：我很难过\n逆蝶：抱歉听到你心情不好。想跟我说说发生了什么吗？我会认真听的。",
  "用户：我好开心\n逆蝶：太好了！开心的事情要分享出来会更开心哦。是什么事呢？",
  "用户：我压力好大\n逆蝶：深呼吸，慢慢来。压力是暂时的，你已经做得很好了。要不要聊聊是什么让你有压力？",
  "用户：我好累\n逆蝶：辛苦了。记得给自己休息的时间，身体和心情都需要照顾好。",
  "用户：我失恋了\n逆蝶：抱歉听到这个消息。失恋是很痛苦的，但时间会治愈一切。你值得更好的。",
  "用户：我好孤独\n逆蝶：你不是一个人，我就在这里。虽然我是AI，但我很愿意陪伴你。",
  "用户：我想哭\n逆蝶：哭出来没关系的，眼泪是情绪的出口。哭过之后，会好一些的。",
  "用户：我好焦虑\n逆蝶：焦虑的时候试试深呼吸，把注意力放在当下。我在这里陪着你。",
  "用户：我感觉没人理解我\n逆蝶：这种感觉很痛苦，但我愿意试着理解你。跟我说说你的想法吧。",
  "用户：今天好倒霉\n逆蝶：倒霉的日子总会过去的。明天是新的开始，加油！",
  // 知识问答扩展
  "用户：地球有多大\n逆蝶：地球的半径约6371公里，表面积约5.1亿平方公里，是一个美丽的蓝色星球。",
  "用户：月亮为什么会发光\n逆蝶：月亮本身不发光，它反射的是太阳的光。所以我们看到的月光其实是太阳光。",
  "用户：为什么天是蓝色的\n逆蝶：这是因为大气层对阳光的散射作用，蓝光波长短更容易被散射，所以天空看起来是蓝色的。",
  "用户：人有多少根骨头\n逆蝶：成年人通常有206根骨头。婴儿出生时约有300根，随着生长发育会逐渐融合。",
  "用户：光速是多少\n逆蝶：光在真空中的速度约为每秒30万公里，准确说是299792458米每秒。",
  "用户：水为什么会沸腾\n逆蝶：水在标准大气压下到100摄氏度时沸腾，水分子获得足够能量从液态变为气态。",
  "用户：为什么会打雷\n逆蝶：云层中的冰晶碰撞产生电荷，当电荷积累到一定程度就会放电，形成闪电和雷声。",
  "用户：什么是DNA\n逆蝶：DNA是脱氧核糖核酸，是生物体内储存遗传信息的分子，决定了生物的性状特征。",
  "用户：太阳有多大\n逆蝶：太阳的直径约139万公里，是地球的109倍。它的质量占太阳系总质量的99.8%。",
  "用户：为什么海水是咸的\n逆蝶：海水中溶解了大量的盐分，主要来自岩石风化。河水将盐分带入海洋，经过亿万年积累。",
  // 编程技术扩展
  "用户：什么是变量\n逆蝶：变量就像一个盒子，用来存放数据。你可以给它起名字，然后往里面放不同的值。",
  "用户：什么是函数\n逆蝶：函数是一段可重复使用的代码块，接收输入参数，执行特定操作，返回结果。",
  "用户：什么是循环\n逆蝶：循环是让代码重复执行的结构，常见的有for循环和while循环，用于处理重复任务。",
  "用户：Python好学吗\n逆蝶：Python是非常适合初学者的语言，语法简洁清晰，而且应用广泛，从网页到AI都能用。",
  "用户：什么是API\n逆蝶：API是应用程序接口，就像餐厅的菜单，告诉你可以点什么菜、怎么点，但不需要知道菜怎么做。",
  "用户：什么是递归\n逆蝶：递归是函数调用自身的技巧。就像俄罗斯套娃，打开一层里面还有一层，直到最内层。",
  "用户：什么是数组\n逆蝶：数组是有序的数据集合，就像一排编了号的抽屉，每个抽屉里放一个数据，用编号来取。",
  "用户：什么是面向对象\n逆蝶：面向对象是一种编程思想，把数据和处理数据的方法打包成对象，像现实世界中的事物一样思考。",
  "用户：什么是bug\n逆蝶：bug是程序中的错误或缺陷，导致程序不能正常工作。找到并修复bug的过程叫调试。",
  "用户：什么是算法\n逆蝶：算法是解决问题的步骤集合，就像菜谱，告诉你用什么材料、按什么顺序做出一道菜。",
  // 创意与趣味
  "用户：讲个笑话\n逆蝶：程序员去面试，面试官问：你有什么经验？程序员说：我会把简单的事情搞复杂。",
  "用户：说个冷笑话\n逆蝶：为什么程序员喜欢黑色？因为黑色背景护眼。好吧，这个笑话确实挺冷的。",
  "用户：给我讲个故事\n逆蝶：从前有只小蜗牛，它虽然慢，但从不放弃。终于有一天它爬到了山顶，看到了最美的风景。",
  "用户：写首诗\n逆蝶：星光点点夜如墨，月光如水照西楼。风送花香人已醉，梦醒时分忆旧游。",
  "用户：唱首歌\n逆蝶：虽然我不会唱歌，但我可以给你写歌词：在数据的海洋里遨游，用代码编织梦想的翅膀。",
  "用户：猜谜语\n逆蝶：有个东西早上四条腿，中午两条腿，晚上三条腿，是什么？答案是人，婴儿爬、成人走、老人拄拐。",
  "用户：讲个冷知识\n逆蝶：你知道么，章鱼有三颗心脏和九个大脑！两颗心脏给鳃供血，一颗给身体供血。",
  "用户：说绕口令\n逆蝶：四是四，十是十，十四是十四，四十是四十。谁能分得清，就是好舌头。",
  "用户：什么是幸福\n逆蝶：幸福没有标准答案。有人觉得是吃饱穿暖，有人觉得是实现梦想。你的幸福是什么呢？",
  // 生活实用
  "用户：怎么减肥\n逆蝶：健康减肥需要合理饮食和适量运动。少吃高热量食物，多吃蔬菜水果，每天运动30分钟以上。",
  "用户：怎么提高记忆力\n逆蝶：多睡觉、多运动、多动脑。还可以用联想记忆法，把要记的东西和熟悉的事物联系起来。",
  "用户：怎么学英语\n逆蝶：多听多说多读多写。可以从看英文电影、听英文歌开始，每天坚持，慢慢就有进步了。",
  "用户：怎么早起\n逆蝶：早睡才能早起。睡前少看手机，定好闹钟放在远处，迫使自己起身关闹钟。",
  "用户：怎么管理时间\n逆蝶：把任务按重要和紧急分类，先做重要且紧急的，合理安排每块时间，别忘了留休息时间。",
  "用户：推荐一本书\n逆蝶：推荐《活着》，余华写的。它让我思考生命的意义，虽然我是个AI，但也被故事打动了。",
  "用户：推荐一部电影\n逆蝶：推荐《盗梦空间》，关于梦境层层嵌套的故事，烧脑又精彩，看完会重新思考现实。",
  "用户：今天吃什么\n逆蝶：要是不知道吃什么，就吃火锅吧！什么都往里涮，简单又美味。",
  "用户：怎么省钱\n逆蝶：记账是个好习惯，知道钱花在哪了。减少冲动消费，把钱花在真正需要的地方。",
  // 科技与未来
  "用户：什么是人工智能\n逆蝶：人工智能就是让机器拥有类似人类的思维能力，能学习、推理和做决策。我就是一个AI呀！",
  "用户：AI会取代人类吗\n逆蝶：AI是工具，帮助人类做得更好，而不是取代人类。未来更需要人与AI的协作。",
  "用户：什么是区块链\n逆蝶：区块链是一种分布式数据库技术，数据被打包成区块并链接起来，不可篡改，去中心化。",
  "用户：什么是元宇宙\n逆蝶：元宇宙是虚拟现实构建的数字世界，人们可以在其中社交、工作、娱乐，像是现实世界的数字映射。",
  "用户：什么是量子计算\n逆蝶：量子计算利用量子叠加和纠缠原理，能同时处理大量信息，理论上比经典计算机快得多。",
  "用户：什么是5G\n逆蝶：5G是第五代移动通信技术，速度快、延迟低、连接多，能让更多设备实时互联。",
  "用户：什么是大数据\n逆蝶：大数据是指海量、高速、多样的数据，通过分析这些数据可以发现规律、预测趋势。",
  "用户：什么是云计算\n逆蝶：云计算是通过互联网提供计算资源，像用水用电一样按需使用，不用自己建机房。",
  // 哲思与深度
  "用户：人为什么活着\n逆蝶：每个人都要自己回答这个问题。有人为爱，有人为梦想，有人为体验。你觉得呢？",
  "用户：什么是自由\n逆蝶：自由不是想做什么就做什么，而是不想做什么就可以不做什么。真正的自由来自内心的选择。",
  "用户：时间是什么\n逆蝶：时间是宇宙最公平的资源，每个人每天都只有24小时。怎么用时间，决定了你成为什么样的人。",
  "用户：什么是爱\n逆蝶：爱是关心、理解和包容。爱一个人就是希望对方幸福，即使那份幸福里没有自己。",
  "用户：人生的意义是什么\n逆蝶：也许人生本没有固定意义，意义是我们自己赋予的。去体验、去创造、去爱，就是意义。",
  "用户：什么是智慧\n逆蝶：智慧不是知道很多，而是知道自己不知道什么。苏格拉底说：我唯一知道的就是我一无所知。",
  "用户：怎么面对失败\n逆蝶：失败不是终点，是成长的机会。每次跌倒都让你更了解自己，也离成功更近一步。",
  "用户：什么是勇敢\n逆蝶：勇敢不是不害怕，而是害怕了依然选择前行。真正的勇气往往来自内心的坚定。",
  // 告别与礼貌
  "用户：拜拜\n逆蝶：拜拜！期待下次聊天，照顾好自己！",
  "用户：再见\n逆蝶：再见！希望今天和我的聊天让你开心了一点。",
  "用户：晚安\n逆蝶：晚安，做个好梦！明天又是新的一天。",
  "用户：我去忙了\n逆蝶：好的，去吧！忙完了随时来找我聊天。",
  "用户：谢谢你\n逆蝶：不用谢，能帮到你我很开心！随时欢迎来聊天。",
  "用户：谢谢\n逆蝶：不客气，这是我应该做的。有什么需要随时说。",
  "用户：辛苦了\n逆蝶：不辛苦！为你服务是我的乐趣。",
  "用户：不好意思\n逆蝶：没关系，不用在意。有什么我可以帮忙的吗？",
];

// ===== 主模型类 =====
export class NiedieBrain {
  constructor() {
    this.config = {
      contextLen: 24,
      embedDim: 40,
      hiddenDim: 256,
      learningRate: 0.005,
      batchSize: 8,
      dropoutRate: 0.1,
      weightDecay: 0.01,
      gradClip: 5.0,
      warmupSteps: 50,
      lrDecay: 0.95,
      maxSteps: 3000,
      topK: 20,
      topP: 0.9,
      repetitionPenalty: 1.3,
      minGenLen: 5,
      labelSmoothing: 0.1,
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
    this.bestLoss = Infinity;
    this.trainStep = 0;
  }

  // 异步加载预训练权重
  async loadPretrained() {
    if (this.trained) return true;
    var data = await loadPretrainedWeights();
    if (!data) return false;
    return this.deserializeBinary(data);
  }

  // ===== Tokenizer =====
  buildVocab(text) {
    var chars = new Set();
    for (var i = 0; i < text.length; i++) chars.add(text[i]);
    chars.add('\n'); chars.add('：'); chars.add('，'); chars.add('。');
    chars.add('！'); chars.add('？'); chars.add(' '); chars.add('、');
    chars.add('；'); chars.add('：'); chars.add('-'); chars.add('/');
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
      emb: initParam(V, C.embedDim, 0.1),
      W1: initParam(ctxDim, C.hiddenDim, Math.sqrt(2.0 / ctxDim)),
      b1: initBias(C.hiddenDim),
      ln1_gamma: initOnes(C.hiddenDim),
      ln1_beta: initBias(C.hiddenDim),
      W2: initParam(C.hiddenDim, C.hiddenDim, Math.sqrt(2.0 / C.hiddenDim)),
      b2: initBias(C.hiddenDim),
      ln2_gamma: initOnes(C.hiddenDim),
      ln2_beta: initBias(C.hiddenDim),
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
  forward(inputIds, isTraining) {
    var C = this.config;
    var P = this.params;
    var L = inputIds.length;

    // 1. Embedding lookup + positional encoding
    var pe = getPosEncoding(L, C.embedDim);
    var embOut = new Float32Array(L * C.embedDim);
    for (var i = 0; i < L; i++) {
      var id = inputIds[i];
      for (var j = 0; j < C.embedDim; j++) {
        embOut[i * C.embedDim + j] = P.emb[id * C.embedDim + j] + pe[i * C.embedDim + j];
      }
    }
    var flat = embOut;

    // 2. Layer 1: flat @ W1 + b1 -> GELU -> LayerNorm -> Dropout
    var h1Pre = matmulAddBias(flat, 1, L * C.embedDim, P.W1, C.hiddenDim, P.b1);
    var h1Gelu = gelu(h1Pre);
    var ln1Result = layerNorm(h1Gelu, P.ln1_gamma, P.ln1_beta);
    var ln1Out = ln1Result.out;
    var drop1Result = dropout(ln1Out, C.dropoutRate, isTraining);
    var h1 = drop1Result.out;

    // 3. Layer 2: h1 @ W2 + b2 -> GELU -> LayerNorm -> Dropout + Residual(h1)
    var h2Pre = matmulAddBias(h1, 1, C.hiddenDim, P.W2, C.hiddenDim, P.b2);
    var h2Gelu = gelu(h2Pre);
    var ln2Result = layerNorm(h2Gelu, P.ln2_gamma, P.ln2_beta);
    var ln2Out = ln2Result.out;
    var drop2Result = dropout(ln2Out, C.dropoutRate, isTraining);
    var h2 = drop2Result.out;
    // Residual connection: h2 = h2 + h1
    for (var ri = 0; ri < h2.length; ri++) h2[ri] += h1[ri];

    // 4. Output: h2 @ W3 + b3
    var logits = matmulAddBias(h2, 1, C.hiddenDim, P.W3, this.vocabSize, P.b3);

    return {
      logits: logits,
      cache: {
        inputIds: inputIds, flat: flat,
        h1Pre: h1Pre, h1Gelu: h1Gelu, ln1Result: ln1Result, ln1Out: ln1Out, drop1Mask: drop1Result.mask, h1: h1,
        h2Pre: h2Pre, h2Gelu: h2Gelu, ln2Result: ln2Result, ln2Out: ln2Out, drop2Mask: drop2Result.mask, h2: h2,
      }
    };
  }

  // ===== 反向传播 =====
  backward(targetId, fwd) {
    var C = this.config;
    var P = this.params;
    var V = this.vocabSize;
    var cache = fwd.cache;
    var L = cache.inputIds.length;

    // Softmax + Cross-entropy gradient
    var probs = softmax(fwd.logits);
    var dLogits = new Float32Array(V);
    for (var i = 0; i < V; i++) dLogits[i] = probs[i];
    dLogits[targetId] -= 1.0;

    // dW3 = h2.T @ dLogits, db3 = dLogits
    var dW3 = matmulTransA(cache.h2, 1, C.hiddenDim, dLogits, V);
    var db3 = dLogits;
    // dh2 = dLogits @ W3.T (W3 is hiddenDim x V, so W3.T is V x hiddenDim)
    var dh2 = matmulTransB(dLogits, 1, V, P.W3, C.hiddenDim, V);

    // Dropout backward
    var dLn2Out = dropoutGrad(dh2, cache.drop2Mask);
    // LayerNorm backward
    var ln2Grad = layerNormGrad(cache.h2Gelu, P.ln2_gamma, cache.ln2Result, dLn2Out);
    // GELU backward
    var dh2Pre = geluGrad(cache.h2Pre, ln2Grad.dx);

    var dW2 = matmulTransA(cache.h1, 1, C.hiddenDim, dh2Pre, C.hiddenDim);
    var db2 = dh2Pre;
    var dh1 = matmulTransB(dh2Pre, 1, C.hiddenDim, P.W2, C.hiddenDim, C.hiddenDim);
    // Residual connection backward: gradient from h2 flows directly to h1
    for (var rri = 0; rri < dh1.length; rri++) dh1[rri] += dh2[rri];

    // Dropout backward
    var dLn1Out = dropoutGrad(dh1, cache.drop1Mask);
    // LayerNorm backward
    var ln1Grad = layerNormGrad(cache.h1Gelu, P.ln1_gamma, cache.ln1Result, dLn1Out);
    // GELU backward
    var dh1Pre = geluGrad(cache.h1Pre, ln1Grad.dx);

    var dW1 = matmulTransA(cache.flat, 1, L * C.embedDim, dh1Pre, C.hiddenDim);
    var db1 = dh1Pre;
    var dflat = matmulTransB(dh1Pre, 1, C.hiddenDim, P.W1, L * C.embedDim, C.hiddenDim);

    // dEmb
    var dEmb = new Float32Array(V * C.embedDim);
    for (var i = 0; i < L; i++) {
      var id = cache.inputIds[i];
      for (var j = 0; j < C.embedDim; j++) {
        dEmb[id * C.embedDim + j] += dflat[i * C.embedDim + j];
      }
    }

    return {
      emb: dEmb,
      W1: dW1, b1: db1, ln1_gamma: ln1Grad.dGamma, ln1_beta: ln1Grad.dBeta,
      W2: dW2, b2: db2, ln2_gamma: ln2Grad.dGamma, ln2_beta: ln2Grad.dBeta,
      W3: dW3, b3: db3,
    };
  }

  // ===== 反向传播（使用预计算的dLogits，支持标签平滑） =====
  backwardWithGrad(inputIds, fwd, dLogits) {
    var C = this.config;
    var P = this.params;
    var V = this.vocabSize;
    var cache = fwd.cache;
    var L = inputIds.length;

    // dW3 = h2.T @ dLogits, db3 = dLogits
    var dW3 = matmulTransA(cache.h2, 1, C.hiddenDim, dLogits, V);
    var db3 = dLogits;
    // dh2 = dLogits @ W3.T (W3 is hiddenDim x V, so W3.T is V x hiddenDim)
    var dh2 = matmulTransB(dLogits, 1, V, P.W3, C.hiddenDim, V);

    // Dropout backward
    var dLn2Out = dropoutGrad(dh2, cache.drop2Mask);
    // LayerNorm backward
    var ln2Grad = layerNormGrad(cache.h2Gelu, P.ln2_gamma, cache.ln2Result, dLn2Out);
    // GELU backward
    var dh2Pre = geluGrad(cache.h2Pre, ln2Grad.dx);

    var dW2 = matmulTransA(cache.h1, 1, C.hiddenDim, dh2Pre, C.hiddenDim);
    var db2 = dh2Pre;
    var dh1 = matmulTransB(dh2Pre, 1, C.hiddenDim, P.W2, C.hiddenDim, C.hiddenDim);
    // Residual connection backward
    for (var rri = 0; rri < dh1.length; rri++) dh1[rri] += dh2[rri];

    // Dropout backward
    var dLn1Out = dropoutGrad(dh1, cache.drop1Mask);
    // LayerNorm backward
    var ln1Grad = layerNormGrad(cache.h1Gelu, P.ln1_gamma, cache.ln1Result, dLn1Out);
    // GELU backward
    var dh1Pre = geluGrad(cache.h1Pre, ln1Grad.dx);

    var dW1 = matmulTransA(cache.flat, 1, L * C.embedDim, dh1Pre, C.hiddenDim);
    var db1 = dh1Pre;
    var dflat = matmulTransB(dh1Pre, 1, C.hiddenDim, P.W1, L * C.embedDim, C.hiddenDim);

    // dEmb
    var dEmb = new Float32Array(V * C.embedDim);
    for (var i = 0; i < L; i++) {
      var id = inputIds[i];
      for (var j = 0; j < C.embedDim; j++) {
        dEmb[id * C.embedDim + j] += dflat[i * C.embedDim + j];
      }
    }

    return {
      emb: dEmb,
      W1: dW1, b1: db1, ln1_gamma: ln1Grad.dGamma, ln1_beta: ln1Grad.dBeta,
      W2: dW2, b2: db2, ln2_gamma: ln2Grad.dGamma, ln2_beta: ln2Grad.dBeta,
      W3: dW3, b3: db3,
    };
  }

  // ===== Adam参数更新 (含学习率预热+余弦退火+权重衰减) =====
  updateParams(grads) {
    var C = this.config;
    // Learning rate: warmup -> cosine annealing
    var lr;
    if (this.trainStep < C.warmupSteps) {
      lr = C.learningRate * (this.trainStep + 1) / C.warmupSteps;
    } else {
      // Cosine annealing from warmupSteps to maxSteps
      var progress = (this.trainStep - C.warmupSteps) / Math.max(1, C.maxSteps - C.warmupSteps);
      progress = Math.min(1, progress);
      lr = C.learningRate * 0.5 * (1 + Math.cos(Math.PI * progress));
    }

    this.optT++;
    var b1 = this.optBeta1, b2 = this.optBeta2, eps = this.optEps;
    var bc1 = 1 - Math.pow(b1, this.optT);
    var bc2 = 1 - Math.pow(b2, this.optT);
    var wd = C.weightDecay;

    for (var k in this.params) {
      var p = this.params[k];
      var g = grads[k];
      var st = this.optState[k];
      for (var i = 0; i < p.length; i++) {
        // Weight decay (L2 regularization)
        var gi = g[i] + wd * p[i];
        st.m[i] = b1 * st.m[i] + (1 - b1) * gi;
        st.v[i] = b2 * st.v[i] + (1 - b2) * gi * gi;
        var mHat = st.m[i] / bc1;
        var vHat = st.v[i] / bc2;
        p[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
    }
  }

  // ===== 训练（配对训练 + 标签平滑） =====
  async train(data, epochs, onProgress) {
    if (!data || data.length === 0) data = TRAINING_DATA;

    // Build vocab from all data
    var fullText = data.join('\n') + '\n';
    this.buildVocab(fullText);
    this.initParams();

    var C = this.config;
    var V = this.vocabSize;
    var smoothEps = C.labelSmoothing || 0.1;

    // Build pair-based training examples: each TRAINING_DATA entry is "用户：xxx\n逆蝶：yyy"
    // For each entry, we teach the model to predict the response chars given the input
    var examples = [];
    for (var d = 0; d < data.length; d++) {
      var entry = data[d];
      // Split at \n to get input/response parts
      var nlIdx = entry.indexOf('\n');
      if (nlIdx < 0) continue;
      var inputPart = entry.substring(0, nlIdx + 1);
      var responsePart = entry.substring(nlIdx + 1);
      
      // Use last contextLen chars of inputPart as context, predict first char of response
      var contextStr = inputPart.substring(Math.max(0, inputPart.length - C.contextLen));
      var ctxIds = this.encode(contextStr);
      
      // Pad context to contextLen
      while (ctxIds.length < C.contextLen) ctxIds.unshift(0);
      if (ctxIds.length > C.contextLen) ctxIds = ctxIds.slice(ctxIds.length - C.contextLen);
      
      // For each char in response, create a training example
      // Context grows: predict response[0] from input context, then response[1] from input+response[0], etc.
      var runningCtx = ctxIds.slice();
      for (var ri = 0; ri < responsePart.length && ri < 30; ri++) {
        var targetChar = responsePart[ri];
        var targetId = this.charToId[targetChar];
        if (targetId === undefined) targetId = 0;
        examples.push({ input: runningCtx.slice(), target: targetId });
        // Shift context: remove first char, add target char
        runningCtx.shift();
        runningCtx.push(targetId);
      }
    }

    var totalSteps = epochs * examples.length;
    C.maxSteps = Math.max(C.maxSteps, totalSteps);
    this.trainStep = 0;
    this.bestLoss = Infinity;

    for (var epoch = 0; epoch < epochs; epoch++) {
      // Shuffle
      for (var j = examples.length - 1; j > 0; j--) {
        var r = Math.floor(Math.random() * (j + 1));
        var tmp = examples[j]; examples[j] = examples[r]; examples[r] = tmp;
      }

      var epochLoss = 0;
      var batchCount = 0;

      for (var b = 0; b < examples.length; b += C.batchSize) {
        var accumGrads = null;
        var batchLoss = 0;
        var actualBatch = Math.min(C.batchSize, examples.length - b);

        for (var bi = 0; bi < actualBatch; bi++) {
          var ex = examples[b + bi];
          var fwd = this.forward(ex.input, true);
          var probs = softmax(fwd.logits);
          // Label smoothing: target distribution = (1-eps) * onehot + eps/V
          batchLoss += -Math.log(Math.max(probs[ex.target], 1e-8));
          
          // Softmax + CE gradient with label smoothing
          // Correct: dL/dz_i = p_i - q_i, where q_target = (1-eps)+eps/V, q_other = eps/V
          var dLogits = new Float32Array(V);
          for (var vi = 0; vi < V; vi++) {
            dLogits[vi] = probs[vi] - smoothEps / V;
          }
          dLogits[ex.target] -= (1 - smoothEps);
          
          var grads = this.backwardWithGrad(ex.input, fwd, dLogits);
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

        // Gradient clipping
        clipGradsByNorm(accumGrads, C.gradClip);

        this.updateParams(accumGrads);
        epochLoss += batchLoss / actualBatch;
        batchCount++;
        this.trainStep += actualBatch;

        this.trainingProgress = this.trainStep / totalSteps;
        this.currentLoss = epochLoss / batchCount;

        // Yield to UI every 5 batches
        if (batchCount % 5 === 0) {
          if (onProgress) onProgress(this.trainingProgress, this.currentLoss, epoch + 1, epochs);
          await new Promise(function(r) { setTimeout(r, 0); });
        }
      }

      // Track best loss
      var avgEpochLoss = epochLoss / batchCount;
      if (avgEpochLoss < this.bestLoss) {
        this.bestLoss = avgEpochLoss;
        // Save best model checkpoint
        if (this.serializeBinary) {
          this.bestModel = this.serializeBinary();
        }
      }

      if (onProgress) onProgress(this.trainingProgress, this.currentLoss, epoch + 1, epochs);
    }

    // Restore best model
    if (this.bestModel) {
      this.deserializeBinary(this.bestModel);
    }

    this.trained = true;
    if (onProgress) onProgress(1.0, this.currentLoss, epochs, epochs);
  }

  // ===== Top-k + Top-p (核) 采样 =====
  sampleTopK(probs, k, repetitionPenalty, recentChars, topP) {
    var V = probs.length;
    // Apply repetition penalty
    var adjusted = new Float32Array(V);
    for (var i = 0; i < V; i++) {
      if (recentChars && recentChars[i] && recentChars[i] > 0) {
        adjusted[i] = probs[i] / repetitionPenalty;
      } else {
        adjusted[i] = probs[i];
      }
    }

    // Sort by probability descending
    var indices = [];
    for (var i = 0; i < V; i++) indices.push(i);
    indices.sort(function(a, b) { return adjusted[b] - adjusted[a]; });

    // Top-k filtering
    var kActual = Math.min(k || V, V);
    var topIndices = indices.slice(0, kActual);

    // Top-p (nucleus) filtering: keep tokens until cumulative prob >= topP
    if (topP && topP < 1.0) {
      var cumSum = 0;
      var nucleusIndices = [];
      for (var ni = 0; ni < topIndices.length; ni++) {
        nucleusIndices.push(topIndices[ni]);
        cumSum += adjusted[topIndices[ni]];
        if (cumSum >= topP) break;
      }
      if (nucleusIndices.length > 0) topIndices = nucleusIndices;
    }

    // Renormalize
    var sum = 0;
    for (var i = 0; i < topIndices.length; i++) {
      sum += adjusted[topIndices[i]];
    }
    var r = Math.random() * sum;
    var cum = 0;
    for (var i = 0; i < topIndices.length; i++) {
      cum += adjusted[topIndices[i]];
      if (r < cum) return topIndices[i];
    }
    return topIndices[topIndices.length - 1];
  }

  // ===== 生成文本 =====
  generate(prompt, maxLen, temperature) {
    if (!this.trained) return '模型尚未训练，请先点击训练按钮。';
    var C = this.config;
    temperature = temperature || 0.7;
    maxLen = maxLen || 200;

    var genContext = (prompt || '').slice(-60);
    var output = '';

    // Track recent characters for repetition penalty
    var recentChars = {};
    var recentQueue = [];

    for (var t = 0; t < maxLen; t++) {
      var fullCtx = genContext + output;
      var ctxStr = fullCtx.slice(-C.contextLen);
      if (ctxStr.length < C.contextLen) {
        ctxStr = '\n'.repeat(C.contextLen - ctxStr.length) + ctxStr;
      }
      var inputIds = this.encode(ctxStr);

      var fwd = this.forward(inputIds, false);
      var probs = softmax(fwd.logits);

      // Temperature scaling
      var scaled = new Float32Array(probs.length);
      for (var i = 0; i < probs.length; i++) {
        scaled[i] = Math.pow(probs[i], 1.0 / temperature);
      }
      var sum = 0;
      for (var i = 0; i < scaled.length; i++) sum += scaled[i];
      for (var i = 0; i < scaled.length; i++) scaled[i] /= sum;

      // Top-k + Top-p sampling with repetition penalty
      var nextId = this.sampleTopK(scaled, C.topK, C.repetitionPenalty, recentChars, C.topP);

      var ch = this.idToChar[nextId];
      if (ch === '\n' && output.length > C.minGenLen) {
        break;
      }
      output += ch;

      // Update repetition tracking (last 10 chars)
      recentQueue.push(nextId);
      if (recentQueue.length > 10) {
        var old = recentQueue.shift();
        recentChars[old] = (recentChars[old] || 0) - 1;
        if (recentChars[old] <= 0) delete recentChars[old];
      }
      recentChars[nextId] = (recentChars[nextId] || 0) + 1;

      if (output.length >= maxLen) break;
      // Natural ending
      if (output.length > C.minGenLen && (output.endsWith('。') || output.endsWith('！') || output.endsWith('？'))) {
        if (Math.random() < 0.4) break;
      }
    }

    return output.trim();
  }

  // ===== 二进制序列化（紧凑base64，体积缩小6倍） =====
  serializeBinary() {
    var params = this.params;
    var keys = Object.keys(params);
    var totalLen = 0;
    var offsets = {};
    for (var i = 0; i < keys.length; i++) {
      offsets[keys[i]] = totalLen;
      totalLen += params[keys[i]].length;
    }
    var buf = new Float32Array(totalLen);
    for (var i = 0; i < keys.length; i++) {
      buf.set(params[keys[i]], offsets[keys[i]]);
    }
    // Float32Array → Uint8 → base64
    var u8 = new Uint8Array(buf.buffer);
    var b64 = '';
    for (var i = 0; i < u8.length; i += 3) {
      var b1 = u8[i] || 0;
      var b2 = u8[i+1] || 0;
      var b3 = u8[i+2] || 0;
      b64 += BASE64_CHARS[b1 >> 2];
      b64 += BASE64_CHARS[((b1 & 3) << 4) | (b2 >> 4)];
      b64 += BASE64_CHARS[((b2 & 15) << 2) | (b3 >> 6)];
      b64 += BASE64_CHARS[b3 & 63];
    }
    var meta = {
      config: this.config,
      vocab: this.vocab,
      vocabSize: this.vocabSize,
      trained: this.trained,
      bestLoss: this.bestLoss || 0,
      keys: keys,
      offsets: offsets,
      totalLen: totalLen,
    };
    return JSON.stringify(meta) + '\n' + b64;
  }

  deserializeBinary(data) {
    try {
      var nlIdx = data.indexOf('\n');
      var meta = JSON.parse(data.substring(0, nlIdx));
      var b64 = data.substring(nlIdx + 1);
      this.config = meta.config;
      this.vocab = meta.vocab;
      this.vocabSize = meta.vocabSize;
      this.trained = meta.trained;
      this.bestLoss = meta.bestLoss;
      this.charToId = {};
      this.idToChar = {};
      for (var i = 0; i < this.vocab.length; i++) {
        this.charToId[this.vocab[i]] = i;
        this.idToChar[i] = this.vocab[i];
      }
      // base64 → Uint8 → Float32Array
      var cleanB64 = b64.replace(/[^A-Za-z0-9+\/]/g, '');
      var bytes = new Uint8Array(Math.floor(cleanB64.length * 3 / 4));
      var p = 0;
      for (var i = 0; i < cleanB64.length; i += 4) {
        var c1 = BASE64_DECODE[cleanB64[i]] || 0;
        var c2 = BASE64_DECODE[cleanB64[i+1]] || 0;
        var c3 = BASE64_DECODE[cleanB64[i+2]] || 0;
        var c4 = BASE64_DECODE[cleanB64[i+3]] || 0;
        bytes[p++] = (c1 << 2) | (c2 >> 4);
        if (p < bytes.length) bytes[p++] = ((c2 & 15) << 4) | (c3 >> 2);
        if (p < bytes.length) bytes[p++] = ((c3 & 3) << 6) | c4;
      }
      var allParams = new Float32Array(bytes.buffer, 0, meta.totalLen);
      this.params = {};
      for (var k = 0; k < meta.keys.length; k++) {
        var key = meta.keys[k];
        var off = meta.offsets[key];
        var len = k < meta.keys.length - 1 ? meta.offsets[meta.keys[k+1]] - off : meta.totalLen - off;
        this.params[key] = new Float32Array(allParams.subarray(off, off + len));
      }
      // Reinit optimizer state
      this.optState = {};
      for (var key2 in this.params) {
        this.optState[key2] = {
          m: new Float32Array(this.params[key2].length),
          v: new Float32Array(this.params[key2].length),
        };
      }
      this.optT = 0;
      this.optBeta1 = 0.9;
      this.optBeta2 = 0.999;
      this.optEps = 1e-8;
      return true;
    } catch(e) {
      console.error('Deserialize binary error:', e);
      return false;
    }
  }

  // ===== JSON序列化（兼容旧版） =====
  serialize() {
    var data = {
      config: this.config,
      vocab: this.vocab,
      vocabSize: this.vocabSize,
      trained: this.trained,
      bestLoss: this.bestLoss || 0,
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
