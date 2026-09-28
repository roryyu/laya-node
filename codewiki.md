# laya-node CodeWiki

> 面向开发者的代码Wiki · 覆盖 `src/`、`bin/`、`tools/`、`test/`、`models/` 全部关键路径
> 版本 `0.1.0` · Node.js >= 22 · Apache-2.0 · 对齐上游 Laya `0.3.5` (`573e5b6`)

---

## 目录

1. [项目定位](#1-项目定位)
2. [快速开始](#2-快速开始)
3. [架构总览](#3-架构总览)
4. [核心概念](#4-核心概念)
5. [模块详解](#5-模块详解)
6. [端到端数据流](#6-端到端数据流)
7. [模型制品契约](#7-模型制品契约)
8. [Python 工具链](#8-python-工具链)
9. [CLI 与 MCP 服务](#9-cli-与-mcp-服务)
10. [测试与验收](#10-测试与验收)
11. [设计决策与不变量](#11-设计决策与不变量)
12. [扩展指南](#12-扩展指南)
13. [故障排查手册](#13-故障排查手册)
14. [API 速查表](#14-api-速查表)

---

## 1. 项目定位

`laya-node` 把上游 Python 的 **Laya**（RL 训练的「类型化决策推理」模型）包装成 **Node.js + Transformers.js** 的本地推理 SDK，并额外提供 **模型路由** 与 **MCP 服务**。

### 它做什么 / 不做什么

| 能力 | 说明 |
| --- | --- |
| ✅ 标签分类 | `choice` — 在候选集中输出概率分布 + 最优标签 |
| ✅ 有序等级评分 | `score` — 输出等级概率分布 + 从 0 开始的期望等级 |
| ✅ 命题概率 | `noul` — 输出「命题成立」的概率 |
| ✅ 动作概率 | 每题附带 `act_probability`（`n_act` 个动作头的 softmax 首项） |
| ✅ 大候选集预筛 | `shortlist` — 用编码器 embedding 做 Top-K 余弦召回 |
| ✅ 自动模型路由 | 按语言 / 工作流 / 显式指定在 3 个 checkpoint 间选择并做 LRU |
| ✅ MCP 工具服务 | 5 个只读工具，stdio / Streamable HTTP 两种传输 |
| ❌ 自由文本生成 | 模型只输出 logits，不解码文本 |
| ❌ 联网检索 | `local_files_only` 语义，绝不静默下载模型 |
| ❌ 自动执行业务动作 | 所有工具 `readOnlyHint: true`，Skill 明确禁止把结果当授权 |

### 与上游的关系

```
上游 convaiinnovations/laya (safetensors)
        │  tools/export_onnx.py  ← 唯一使用 Python 训练栈的环节
        ▼
ONNX 制品 (models/<checkpoint>/)  ← 纯推理，Node 侧零 Python
        │
        ▼
laya-node (JS) / laya (Py) 数值对照  ← golden.json, atol=1e-4, rtol=1e-3
```

**上游仓库是 safetensors，不能直接运行**（`runtime.js:43` 显式拒绝 `convaiinnovations/laya` 前缀）。

---

## 2. 快速开始

```bash
# 1) 一次性导出完整模型（需要 Python，仅此一步）
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/requirements.txt
.venv/bin/python tools/export_onnx.py --checkpoint english --output models/english
# 另外两个 checkpoint：multilingual / typed-decisions，各自单独导出

# 2) 跑起来
node examples/predict.js
# 或
node bin/laya.js predict --model ./models/english --input examples/request.json

# 3) 测试
npm test            # node --test test/*.test.js
npm run test:types  # tsc --strict 校验 index.d.ts
```

只想验证运行链路（无语义能力）：

```bash
.venv/bin/python tools/export_onnx.py --tiny --output artifacts/tiny-english-v2
```

---

## 3. 架构总览

### 3.1 分层

```
┌──────────────────────────────────────────────────────────────────┐
│  入口层    bin/laya.js (CLI)      bin/laya-mcp.js (MCP 服务)     │
├──────────────────────────────────────────────────────────────────┤
│  服务层    mcp.js (工具/队列)      mcp-http.js (HTTP 传输)        │
├──────────────────────────────────────────────────────────────────┤
│  编排层    router.js (路由+LRU)   agent.js (Agent 门面)           │
│            shortlist.js (Top-K 预筛)                              │
├──────────────────────────────────────────────────────────────────┤
│  领域层    questions.js (DSL)      presets.js (业务模板)          │
│            lang.js (语言检测)      email.js (邮件清洗)             │
│            math.js (概率/奖励/校准)                                │
├──────────────────────────────────────────────────────────────────┤
│  编码层    sequence.js (tokenize/collate)                         │
├──────────────────────────────────────────────────────────────────┤
│  执行层    runtime.js (ONNX/MLX)  mlx.js (Python sidecar)        │
├──────────────────────────────────────────────────────────────────┤
│  制品层    models/ artifacts/  config.json + onnx/model.onnx      │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 模块依赖图

```
index.js ──► agent.js ──► runtime.js ──► questions.js / sequence.js / math.js
    │           │  │            │
    │           │  │            └──► mlx.js ──spawn──► tools/mlx_runtime.py
    │           │  └──► sequence.js (buildSequence, collateItems)
    │           └──► shortlist.js ──► questions.js
    ├──► router.js ──► agent.js (默认 loader)
    ├──► presets.js → index.js 导出为 API
    ├──► lang.js, email.js, math.js  （无状态叶子模块）
    └──► mcp.js ──► index.js, questions.js
            └──► mcp-http.js ──► mcp.js
```

**依赖规则**：`math.js` / `questions.js` / `lang.js` / `email.js` / `sequence.js` 是叶子，不 import 任何 laya 模块，可在无模型环境下单测。所有 `loadRuntime` 路径都是**惰性 `await import('@huggingface/transformers')`**（`runtime.js:55`），因此 `route` / 数学工具 / 语言检测不初始化任何原生推理库。

### 3.3 三种推理后端

| device | dtype | 平台 | 加速路径 | 备注 |
| --- | --- | --- | --- | --- |
| `cpu` | `fp32` / `q8` / `q4` | 全平台 | onnxruntime-node | 默认 |
| `coreml` | 仅 `fp32` | macOS | CoreML EP + CPU 兜底 | 需 `coreml_patched: true` |
| `mlx` | `fp32` / `fp16` | Apple Silicon | Python sidecar + Metal | 直读 safetensors，最快（~60ms） |

组合约束（`runtime.js:34-42`）：
- `coreml` + `q8/q4` → 拒绝（`MatMulNBits` 不被 CoreML 支持）
- `mlx` + `q8/q4` → 拒绝（量化只适用于 ONNX CPU）
- `coreml` 在非 darwin → 拒绝

---

## 4. 核心概念

### 4.1 三种决策原语

`QTYPES` 是模型协议的硬约定，**顺序不可改**（`questions.js:1`）：

```js
QTYPES = { choice: 0, score: 1, noul: 2 }
```

| 类型 | 输入 | 模型输出 | SDK 归约 |
| --- | --- | --- | --- |
| `choice` | 候选标签（字符串数组 or 标签→描述对象） | `[k]` logits | `choice` = argmax 标签；`probabilities` 逐标签 |
| `score` | 有序等级数组 | `[k]` logits | `score` = `Σ i·p[i]`（**0 起算的期望等级**）；`legend` 保留原描述 |
| `noul` | 可选 `{true, false}` 描述 | `[2]` logits | `noul` = `p[1]`（真命题概率）；`confidence` = `max(p1, 1-p1)` |

**DSL 形态**（用户侧）：

```js
{
  intent:   { type: 'choice', instructions: 'What does the customer want?', criteria: { refund: 'money back', bug: 'a bug' } },
  urgency:  { type: 'score',  instructions: 'How urgent?', criteria: ['none', 'soon', 'blocking'] },
  refund:   { type: 'noul',   instructions: 'Does the customer ask for money back?' }
}
```

**归一化后的内部形态**（`normalizeQuestion` 输出）：

```js
{ t: 'choice'|'score'|'noul', ins: '<字符串>', crit: <归一化 criteria> }
```

### 4.2 State 序列化

`serializeState(state)` 把 JS 值变成模型输入文本，**逐字节对齐 Python `json.dumps`**：

- 接受 `string` / 纯 JSON 对象 / 数组；拒绝 `null`、数字、布尔顶层
- 分隔符固定为 `, ` 和 `: `（**保留 Python 的空格**，`questions.js:6` 注释）
- 拒绝 `undefined`、`BigInt`、`NaN`/`Infinity`、`Date` 等类实例、循环引用
- 允许 `Object.create(null)` 原型的裸对象

`jsonText(value, ascii=true)` 用于 `instructions` 非字符串时的转义，保证非 ASCII 以 `\uXXXX` 输出。

### 4.3 Token 预算

序列由三段组成，预算由两个配置控制：

```
[CLS] {type} question: {instructions} [SEP]
  [MASK] option_0   ← markers[0] 指向这个 MASK
  [MASK] option_1   ← markers[1]
  ...
[SEP] {body} [SEP]
```

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `headMaxLength` | 192 | 问题头 + 全部候选的上限 |
| `maxLength` | 512 | 整条序列上限 |
| `max_position_embeddings` | 8192 | 编码器绝对上限（`prepare` 会校验） |

分配策略（`sequence.js:30-43`）：

1. 每个候选先 `slice(0, 48)` token
2. `budget = headMaxLength - ΣoptionLengths`
3. 若 `budget < 16`，按 `per = max(4, floor((headMaxLength-16)/n))` 二次均分
4. `head = head.slice(0, max(8, budget))`
5. **仍超预算 → 抛 `RangeError`**，提示改用 `shortlist`。绝不静默丢弃尾部候选

正文截断：`room = maxLength - ids.length - 1`，`truncateLeft` 保留**末尾**（客服工单常见），默认保留**开头**。

### 4.4 温度、置信度与校准

温度是模型的**输出后校准参数**，不是采样温度：

```js
// agent.js:10 — 优先级：分桶 > 按类型 > 1
const scale = clampTemperature(
  cfg.temperature_by_options?.[tempBucket(q.t, k)] ?? cfg.temperature?.[QTYPES[q.t]] ?? 1
);
```

`tempBucket(qtype, k)` 生成 `"{type}:{bucket}"`，bucket 分档：`k≤2 → '2'`、`≤5 → '3-5'`、`≤10 → '6-10'`、否则 `'11+'`。

`confidence` 是**归一化熵的补**，不是正确率：

```
confidence = 1 - H(p) / log(k)      (k ≥ 2；k < 2 时直接返回 1)
```

`clampTemperature` 把值夹到 `[TEMP_MIN=0.5, TEMP_MAX=5]`，**非法值一律回退到 1**。若 checkpoint 里的温度被限幅，构造函数发 `process.emitWarning('LAYA_TEMPERATURE_CLAMPED')`，MCP 的 `diagnostics.temperature_clamped` 也会置位——因为限幅后的概率**不再对应已校准的语义**。

`math.js` 另提供离线校准工具：
- `eceScore(confidence, correct, bins=15)` — 期望校准误差
- `properReward(q, target, qtype, mask, {wSph, wRps})` — strictly proper scoring rule；`score` 类型额外带 RPS（秩概率得分）项
- `tdLambdaTargets(pTrue, batch, lam)` — 按 `ep_group` 分组、`ep_step` 排序的倒序 TD(λ) 递推，不修改入参

---

## 5. 模块详解

### 5.1 `src/questions.js` — 问题 DSL 与状态序列化

| 导出 | 职责 |
| --- | --- |
| `QTYPES` / `QTYPE_NAMES` | 冻结的类型↔索引双向表 |
| `isRecord(v)` | 纯对象判定：原型必须是 `Object.prototype` 或 `null` |
| `jsonText(value, ascii)` | Python `json.dumps` 兼容序列化，带循环引用检测 |
| `serializeState(state)` | 顶层 `state` → 字符串 |
| `criteriaEntries(criteria)` | 统一数组/对象形式为 `[label, description]` 对 |
| `normalizeQuestion(def)` | 校验 + 归一化为 `{t, ins, crit}` |
| `normalizeQuestions(obj)` | 批量归一化，错误信息带上问题 ID |
| `renderOptions(q)` | 渲染成模型可见的候选文本 |

**`renderOptions` 的输出直接进入 tokenize**（`sequence.js:31`），格式固定：

```js
choice → ['refund: money returned or a duplicate charge reversed', 'other']
score  → ['level 0: calm and neutral', 'level 1: clearly annoyed']
noul   → ['false: no, the statement does not hold', 'true: yes, the statement holds']
```

细节：
- `choice` 的 value 为 `null`/`''` 时退化为纯标签
- `noul` 固定**先 false 后 true**（对应 `p[0]=假`、`p[1]=真`）
- `choice` 以**数组**给出时，保留调用者的候选顺序——因为数字形字符串作为对象键会被 JS 重排（`questions.js:63` 注释）

**校验拒绝**：`choice` 空 criteria / 重复标签 / 空标签；`score` 非数组或空；`noul` 含 `true`/`false` 以外的键；`instructions` 缺失。

### 5.2 `src/sequence.js` — 编码与批处理

```js
specialTokens(tok)  // 解析 cls/sep/mask/pad 的 id；缺失即抛错
encode(tok, text)   // add_special_tokens: false
buildSequence(tok, state, q, opts)  // → { ids, markers, qtype }
collateItems(items, padId)          // → Batch
```

- `specialTokens` 先读 `tokenizer.<name>_token_id`，缺失则回退到把 `config.<name>_token` 文本编码成单 token
- `buildSequence` 用 `tokenizer.mask_token` 作为**候选分隔符**，并把它从指令/正文中替换为空格（防止用户文本伪造 mask 边界）
- `optionOrder` 支持候选置换（必须是对 `0..n-1` 的完整排列）
- `collateItems` 是**纯数组协议**，不创建 Tensor——Tensor 只在 `runtime.js` 的适配层构造，这让它可以脱离模型单测

`Batch` 结构：

```js
{ input_ids: n×L, attention_mask: n×L, marker_pos: n×K, marker_mask: n×K, qtype: n }
```

`K = max(2, ...各题候选数)`，不足用 `0`/`false` 填充。

### 5.3 `src/math.js` — 概率与校准

| 函数 | 关键不变量 |
| --- | --- |
| `positiveInteger(n, label)` | 所有长度/预算参数的统一入口校验 |
| `clampTemperature(v, lo, hi)` | 非有限值 → **1**（不是 `lo`）；字符串会先 trim 再 `Number()` |
| `tempBucket(qtype, k)` | 接受数字或字符串 qtype；未知类型抛 `TypeError` |
| `softmax(logits, T)` | 先减 max 再除 T（数值稳定）；非空有限数组 |
| `confidenceFromProbs(p, k)` | 概率须在 `[0,1]`；`log` 取 `max(x, 1e-12)` 兜底 |
| `eceScore(conf, correct, bins)` | 分桶用左开右闭 `(b/bins, (b+1)/bins]`；空输入返回 `NaN` |
| `properReward(q, target, qtype, mask, opts)` | `log` 项 `logFloor=-9.21`；mask 外的候选置 0 不参与 log |
| `tdLambdaTargets(pTrue, batch, lam)` | `lam ∈ [0,1]`；`target[row][0]=1-value` |

### 5.4 `src/agent.js` — Agent 门面

```js
const agent = await load(modelPath, { device, dtype });  // 或 Agent.load
const r = await agent.predict(state, questions, { maxLength, headMaxLength, truncateLeft });
await agent.dispose();
```

**构造契约**：`new Agent(runtime)` 要求 `runtime.run` 和 `runtime.tokenizer` 都存在，否则提示「请使用 `await load()` 创建」——防止手工构造半成品对象。

**并发模型**：`#use(operation)` 把每个推理包成 Promise 加入 `#pending` 集合，`dispose()` 等所有在途任务结算后再释放 runtime。这是一个**协作式优雅关闭**：没有中断正在进行的 ONNX 调用。

**`formatAnswers(entries, output, cfg)`** 是纯函数，核心转换表：

```js
choice → choice = labels[argmax(p)];  probabilities = {label: p}
score  → score   = Σ i·p[i];           legend = {"0": crit[0], "1": crit[1], ...}
noul   → noul    = p[1];               confidence = max(p[1], 1-p[1])
所有类型 → action.act_probability = softmax(act_logits[row])[0]
```

答案容器用 `Object.create(null)`，**问题 ID 无法污染原型**（`core.test.js` 有专门断言）。

**`embed(texts, {maxLength=512, batchSize=32})`**：
- `maxLength ≥ 2`（保留起止 token）
- 截断时 `ids.splice(maxLength-1, ..., ids.at(-1))` —— ModernBERT 的起止 token 不参与正文截断，与 Python fast tokenizer 对齐
- 返回 `output.embeddings`，marker 传 `[0]` 占位

**API 别名**：`RLAgent`、`load`、`systemOne` / `system_one` 全部映射到 `Agent` / `predict`，保持与 Python 导出一致。

### 5.5 `src/runtime.js` — 制品加载与推理执行

```js
const INPUTS  = ['input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype'];
const OUTPUTS = ['logits', 'act_logits', 'embeddings'];
```

`loadRuntime(modelPath, options)` 的守卫顺序（每一步都有独立错误信息）：

1. `modelPath` 非空字符串
2. `device ∈ {cpu, coreml, mlx}`，平台/dtype 组合校验
3. 拒绝 `convaiinnovations/laya*`（safetensors 格式）
4. 判定本地路径 vs Hub 仓库 id；本地缺 `config.json` → 报出该跑哪条 `export_onnx.py` 命令
5. `AutoConfig.from_pretrained` → `validateConfig`：
   - `model_type === 'custom'` && `laya.format_version === 1`
   - `agent_config` 存在，`max_len`/`head_max_len`/`hidden_size`/`n_act` 为正整数
   - `max_len ≤ max_position_embeddings`
   - `temperature` 必须是**长度 3** 的数组
6. `device === 'coreml'` 且 `coreml_patched !== true` → 拒绝（4D 注意力广播会让 CoreML EP 崩溃）
7. `AutoTokenizer` + `specialTokens()` 预检
8. 构建 session，校验 ONNX 图的 **inputNames / outputNames 与协议双向完全一致**
9. 任一步失败都 `await model.dispose()` 后重抛

`run(batch)` 的 shape 契约：

| 张量 | dims | dtype |
| --- | --- | --- |
| `input_ids` / `attention_mask` | `[n, L]` | `int64` |
| `marker_pos` | `[n, K]` | `int64` |
| `marker_mask` | `[n, K]` | `bool` |
| `qtype` | `[n]` | `int64` |
| ← `logits` | `[n, K]` | |
| ← `act_logits` | `[n, n_act]` | |
| ← `embeddings` | `[n, hidden_size]` | |

输出**同时校验 shape 和有限性**（`t.data.every(Number.isFinite)`），非有限值直接报错而不是静默传播 NaN。

**CoreML 特例**（`runtime.js:95-99`）：Transformers.js 对 coreml 只注册单个 EP，图中 `int64`/`bool`/`Shape` 算子无人接管会直接失败。因此显式 `executionProviders ??= ['coreml', 'cpu']`，让 CoreML 只接管它支持的 float 子图。

### 5.6 `src/mlx.js` — Python Sidecar 桥

Apple Silicon 上的最快路径（~60ms/次）。**不走 ONNX，直接读上游 safetensors**。

```
Node                                Python (tools/mlx_runtime.py serve)
 │ spawn(python, [mlx_runtime.py, serve, --fp16?])                        │
 │◄── NDJSON {"ready":true,"dtype":...,"model_dir":...}  (30s 超时)      │
 │── {"id":1,"inputs":{...}}                                             │
 │◄── {"id":1,"outputs":{...}}  或  {"id":1,"error":"..."}                │
```

- Python 解释器：`process.env.LAYA_MLX_PYTHON` → 否则项目 `.venv/bin/python`
- **请求严格串行**：Node 侧 `queue` + `pending` 单飞，`flush()` 保证同一时刻只有一条在途，因为 Python 端按 stdin 顺序处理
- `dispose()` 关 readline → kill → 等 `exit`
- 错误处理完备：启动失败、进程提前退出（`exit` 事件 → reject）都有路径

### 5.7 `src/router.js` — 路由与模型生命周期

**路由优先级**（严格从上到下短路）：

```
1. options.model            显式 checkpoint
2. options.task             显式任务（同一套别名）
3. autoTaskDetection + 工作流匹配 → typed-decisions
4. options.lang             'en'/'eng'/'english' 前缀 → english，否则 multilingual
5. detectLanguage(state)
   ├─ script === 'unknown'        → default
   ├─ script !== 'latin'          → multilingual
   ├─ !is_english                 → multilingual
   └─ 否则                        → english
```

`RouteDecision` 携带 `model / repo(路径) / reason(人类可读理由) / detection / workflow`——**每个决策都可解释**。

**工作流匹配**（`matchTypedDecisionsWorkflow`）要求问题 ID 集合**精确相等**（长度相等 + 全部命中），避免前缀误匹配：

| workflow | 必需问题 ID |
| --- | --- |
| `agent_trace_observability` | action, needs_review, outcome, risk, urgency |
| `customer_service` | action, category, churn_risk, needs_human, urgency |
| `invoice_processing` | discrepancy_severity, disposition, duplicate, matches_order, urgency |
| `security_incidents` | credential_compromise, disposition, severity, true_positive, urgency |

**模型别名**：`en`/`laya`/`default`→`english`；`multi`/`ml`/`laya-multilingual`→`multilingual`；`typed`/`typed_decisions`/`decisions`→`typed-decisions`。全部经 `Object.hasOwn` 白名单校验（`__proto__` 会抛错）。

**LRU 槽位机制**：

```js
slot = { key, users: 0, agent: null, ready, waiters: [], retired: false }
```

- `#slot(name)` 惰性建槽，`ready` 缓存加载 Promise → **并发加载自动去重**（`router.load('english')` 与 `load('en')` 只加载一次）
- 访问即 `#slots.delete(key); #slots.set(key, slot)` → Map 插入序 = LRU 序
- `#trim(protect)`：超过 `maxLoaded` 时释放最旧的、`users===0` 且已加载的槽
- `users` 引用计数 + `waiters`：正在被使用的模型不会被 retire，`#release` 归零时唤醒等待者
- `#retire` 同时注册 `.then(ok, err)`——**避免未观察的 `finally` 派生 Promise 造成 unhandled rejection**
- `attach(name, agent)` 拒绝把同一 agent 挂到两个模型名（防止重复释放）

默认 `maxLoaded = 1`；三个完整 FP32 模型各占数 GB，**不建议默认全预加载**。

### 5.8 `src/shortlist.js` — 大候选集预筛

```
state + criteria ──► renderOptions ──► [query, opt0, opt1, ...]
                                            │
                                       embedFn(encoder)
                                            ▼
                                     余弦相似度排序 ──► Top-K
```

- `k >= 候选数` → **passthrough**，不调 embedding，返回 `scores: null`、`probability_scope: 'all'`
- 否则 `probability_scope: 'shortlisted'` —— **缩减后的概率只在入围候选内归一化，可能漏选**，SDK 与 MCP 描述都显式声明
- 数值稳定性：先按**最大绝对值缩放**再归一化，避免超大向量溢出（余弦不变）
- 排序 `b.score - a.score || a.i - b.i` —— 分数相同时**保持原始候选顺序**（确定性）
- `predictShortlist` 只缩减 `choice` 题，其他题原样透传
- `embedFnFromAgent(agent, opts)` 把 `Agent.embed` 适配成 `EmbeddingFunction`（支持同步/异步返回、TypedArray）

### 5.9 `src/lang.js` — 语言检测

两级启发式：

1. **文字系统**（`RANGES`，22 个 Unicode 区段）：先按码点判定 greek/cyrillic/han/kana/hangul/arabic…；`0x250` 以下与 `0x1e00-0x1eff` 计入 `latin`
2. **拉丁语系**（仅 8 种：en/fr/de/es/pt/it/nl/ro）：变音符号率 `≥ 0.02` 判为非英语，再用停用词表打分

返回结构：

```js
{ script, script_profile, language, is_english, language_undecided, diacritic_rate, non_latin_fraction }
```

**`language_undecided: true` 表示证据不足**，调用方应显式传 `lang`。`stateText` 递归展开 state（深度上限 6）拼成待检测文本。

### 5.10 `src/email.js` — 邮件正文清洗

三段式清理，全部在本地正则完成：

1. **引用截断**：遇到 `On ... wrote:` / `-----Original Message-----` / `________` / `From:` 头即停
2. **签名截断**：扫描位置限制在 `max(1, min(0.6·lines, lines-8))` 之后，只匹配 ≤40 字符的签名行（`--` / `Best regards` / `Sent from my iPhone` …）
3. **免责声明句剔除**：命中 `confidential` / `intended solely for...` / `if you received this email in error` 的段落，按句切分后逐句过滤

`emailState(subject, body, opts)` 组装 `{ subject, body, from?, ...extra }`；`extra` 里的 `null`/`undefined` 会被剔除。

### 5.11 `src/presets.js` — 业务模板

五个工厂函数，**提示词刻意保留上游英文原文**（`presets.js:1` 注释：迁移语言会导致决策行为漂移）：

| preset | 用途 | 问题 |
| --- | --- | --- |
| `triageQuestions()` | 客服工单 | intent(choice), is_urgent, frustration(score), refund_requested, churn_risk |
| `emailQuestions(categories?)` | 邮件分派 | category(choice), is_spam, is_phishing, urgency(score), needs_reply |
| `guardQuestions()` | 提示词风险 | jailbreak, prompt_injection, sensitive_data, harm_severity(score), topic(choice) |
| `moderationQuestions()` | 内容审核 | toxic, harassment, threat, spam, severity(score) |
| `routerQuestions()` | 请求分类 | difficulty(score), domain(choice), needs_tools, is_sensitive |

注意 `guardQuestions` 的 `is_phishing` 提供了 `{true, false}` 描述，触发 `renderOptions` 的自定义文案路径。

### 5.12 `src/index.js` — 统一出口

所有公共 API 都带 **snake_case 别名**（`detect_language` / `render_options` / `predict_shortlist` / `td_lambda_targets` …），与 Python 导出对齐。`__version__ = '0.1.0'` 是版本唯一来源（`mcp.js` 从这里 import）。

---

## 6. 端到端数据流

### 6.1 `Agent.predict` 全链路

```
predict(state, questions, opts)
  │
  ├─ prepare()
  │   ├─ serializeState(state)                     → 校验 + 字符串
  │   ├─ normalizeQuestions(questions)              → [{id, {t,ins,crit}}]
  │   ├─ positiveInteger(maxLength/headMaxLength)   → 预算校验
  │   ├─ maxLength ≤ metadata.max_position_embeddings
  │   ├─ buildSequence(tok, state, q) × N           → {ids, markers, qtype}
  │   └─ collateItems(items, pad)                   → Batch
  │
  └─ #use(async () => {
        runtime.run(batch)      ← 惰性 transformers → Tensor → ONNX/MLX
          ├─ shape 校验 [n,K]/[n,n_act]/[n,hidden]
          └─ 有限性校验
        formatAnswers(entries, output, cfg)
          ├─ 温度解析：by_options[bucket] ?? temperature[type] ?? 1
          ├─ softmax(logits.slice(0,k), T)         → p
          ├─ confidenceFromProbs(p)                 → 归一化熵补
          └─ act_probability = softmax(act_logits)[0]
      })
  → { model: 'laya-rl-agent', answers, usage: { input_tokens, output_tokens: 0 } }
```

**`output_tokens` 恒为 0 是刻意的**——模型不生成文本。

### 6.2 Shortlist 链路

```
predictShortlist(agent, state, questions, {embedFn, k=20})
  │
  ├─ normalizeQuestions
  ├─ 对每个 choice 题:
  │     renderOptions → [query, ...candidates]
  │     embedFn(...) → [[q],[c0],[c1],...]        ← 必须 rows === candidates+1
  │     max-abs 归一化 → 余弦 → 排序 → Top-K
  │     reduced[id] = 缩减后的 criteria
  │     shortlist[id] = { labels, scores, passthrough, original_count, probability_scope }
  │
  └─ predict.call(agent, state, reduced, opts)     ← 复用普通 predict
```

### 6.3 Router.predict 链路

```
router.predict(state, questions, options)
  │
  ├─ route(state, questions, options)  → RouteDecision（含 reason）
  ├─ #slot(decision.model)  → slot.users++
  ├─ await slot.ready                   → Agent（并发去重）
  ├─ #trim(slot)                        → 释放最旧的空闲模型
  ├─ agent.predict(...)
  └─ { ...result, routing: { ...decision } }
  finally: #release(slot); #trim()
```

### 6.4 MCP 工具链路

```
JSON-RPC tools/call
  └─ createServer() 的统一 wrapper
       ├─ 序列化 args → >256KiB ? throw
       ├─ handler(args, extra)
       │    └─ enqueue(op, signal)
       │         ├─ pending ≥ maxPending ? throw
       │         ├─ tail.then(op)   ← 全局串行
       │         ├─ signal.aborted 前置检查
       │         ├─ resolveQuestions()  1–32 题 / choice ≤1000 / serializeState
       │         ├─ router.route()
       │         ├─ inspectModel()   文件存在 + 元数据
       │         │    ├─ 不可用 → throw（绝不自动下载/替换）
       │         │    └─ tiny && !allowTiny → throw
       │         ├─ router.load()
       │         ├─ agent.predict() | predictShortlist()
       │         └─ { ...prediction, routing, diagnostics }
       │         └─ signal.aborted 后置检查（已完成的 ONNX 推理结果被丢弃）
       └─ result(): { content:[text], structuredContent }
```

**串行化的理由**（`mcp.js:78` 注释）：让 shortlist 的 embedding 与随后的 choice 推理命中**同一个已加载模型**，避免 `maxLoaded=1` 时反复换入换出。

`diagnostics` 字段：

```js
{ checkpoint, tiny, temperature_clamped, state_truncation, max_length, head_max_length }
```

---

## 7. 模型制品契约

### 7.1 目录布局

```
models/english/
├── config.json            # transformers 配置 + laya 元数据（必须有）
├── rl_agent_config.json   # 上游 agent 配置（必须有，inspect 校验）
├── tokenizer.json         # fast tokenizer
├── tokenizer_config.json
├── manifest.json          # 校验清单 + 来源 revision
├── golden.json            # Python 参考数值（验收基线）
└── onnx/
    ├── model.onnx              # 1.6 GiB (fp32)
    └── model_quantized.onnx    # 564 MiB (q8，量化后生成)
```

### 7.2 `config.json` 的 `laya` 段

```jsonc
{
  "model_type": "custom",
  "laya": {
    "format_version": 1,
    "checkpoint": "english",
    "tiny": false,
    "source": "convaiinnovations/laya",
    "revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
    "hidden_size": 1024,
    "max_position_embeddings": 8192,
    "n_act": 2,
    "coreml_patched": true,        // CoreML EP 必需
    "agent_config": {              // → Agent.cfg
      "encoder": "answerdotai/ModernBERT-large",
      "head_layers": 2,
      "max_len": 512,
      "head_max_len": 192,
      "max_prefixes": 6,
      "act_costs": { "escalate": 0.5 },
      "cost_wrong_act": 3.0,
      "amp_dtype": "bf16",
      "model_name": "rl-agent",
      "temperature": [1.637, 1.251, 1.983],        // [choice, score, noul]
      "temperature_by_options": { "choice:3-5": 1.760, "choice:11+": 0.1006, ... },
      "training": { "updates": 7313, "epochs_completed": 1, "hours": 1.96, "world_size": 1 }
    }
  }
}
```

**契约点**：
- `model_type` 必须是 `custom`（不是任何 HF 架构名）
- `format_version` 必须为 1
- `temperature` 长度必须为 3
- `max_len ≤ max_position_embeddings`
- `coreml_patched: true` 才能用 `device: 'coreml'`

### 7.3 ONNX 图签名

```
inputs : input_ids[n,L]:int64  attention_mask[n,L]:int64
         marker_pos[n,K]:int64   marker_mask[n,K]:bool    qtype[n]:int64
outputs: logits[n,K]            act_logits[n,2]          embeddings[n,1024]
```

`runtime.js:105-108` 要求 inputNames/outputNames 与协议**双向完全一致**（多一个少一个都报错）——防止把不兼容的图当成 Laya 模型跑。

### 7.4 `manifest.json`

记录每个文件的 `sha256` + `bytes`，外加 `source_revision`（必须等于 `config.laya.revision`）、`reference_laya_version`、`runtime`（torch / onnxruntime 版本）、`verification`（7 个用例，atol 1e-4 / rtol 1e-3）。

`--verify` 逐文件流式重算 SHA-256，并做**路径越界检查**（`full.startsWith(root + path.sep)`）。

### 7.5 三个 checkpoint

| 名称 | 别名 | 用途 |
| --- | --- | --- |
| `english` | `en`, `laya`, `default` | 英语文本 |
| `multilingual` | `multi`, `ml` | 中文及其他非英语 |
| `typed-decisions` | `typed`, `decisions` | 四个结构化工作流（默认**不自动启用**，需 `autoTaskDetection`） |

### 7.6 golden.json 验收

`tools/export_onnx.py` 在导出时用 **Python laya 0.3.5 跑一遍 7 个用例**并把 logits/act_logits/embeddings 落盘为 `golden.json`。`integration.test.js` 和 `verify_models.js` 用 `atol=1e-4, rtol=1e-3` 对照——这是 JS 侧数值正确性的**唯一权威来源**。

---

## 8. Python 工具链

> Python 只在**导出 / 验证 / 加速**环节出现，Node 运行时不需要 Python（除 `device: 'mlx'`）。

| 脚本 | 作用 | 关键点 |
| --- | --- | --- |
| `tools/export_onnx.py` | safetensors → ONNX + golden + manifest | `--checkpoint` / `--output`（**禁止覆盖已有目录**）/ `--tiny` / `--verify-only` / `--revision` |
| `tools/quantize_onnx.py` | ONNX → int8 | 产出 `model_quantized.onnx`（1608 MiB → 564 MiB，加载 1386ms → 537ms） |
| `tools/prepare_coreml.py` | CoreML 兼容改写 | 把 4D 注意力广播改成 CoreML 可处理形式；无损，CPU 同样可用；写 `coreml_patched: true` |
| `tools/mlx_runtime.py` | 纯 MLX 推理（直读 safetensors） | `validate` / `bench` / `serve` 三个子命令 |
| `tools/verify_models.js` | **Node** 验收脚本 | 每个模型独立进程运行，避免累计会话内存影响测量 |
| `tools/bench_threads.mjs` | onnxruntime 线程数基准 | Apple Silicon 上 `intraOpNumThreads = P-core 数` 比默认快 ~20% |

**MLX 数学语义**（`mlx_runtime.py` 头注释，与 ONNX 导出一致）：

```
encoder(ModernBERT-large, 28 层, hidden 1024, local_attention 窗口 ±64, RoPE θ=160000/10000)
  → mean pool                                  → embeddings
  → +type_emb → 2 × TransformerEncoder         → h
  → gather(marker_pos)                         → h0
  → act_head(cat([h0, top1, top1-top2, ent, k/255]))
  → scorer(h0)                                 → logits
```

`act_logits` 的 5 维输入是理解动作头语义的关键：`top1 - top2`（分类间隔）与 `ent`（熵）让模型能表达「该不该 escalate」。

**依赖**（`tools/requirements.txt`）：`torch==2.8.0`、`transformers==5.0.0`、`laya==0.3.5`、`onnxruntime==1.23.2`、`onnx==1.19.0`、`numpy==2.3.3`。

---

## 9. CLI 与 MCP 服务

### 9.1 `bin/laya.js` — 命令行

```
laya-node predict --model <path> --input <file|-> [--preset ...] [--offline] [--revision ...]
                  [--max-length N] [--head-max-length N] [--truncate-left]
laya-node route   --input <file|-> [--lang zh] [--model english] [--task ...]
laya-node inspect --model <path> [--verify]
```

设计要点：
- `--input -` 从 stdin 读，32 MiB 上限
- `inspect --verify` 逐文件 SHA-256 + 字节数校验，含路径越界防护
- 错误只写 **stderr**，`process.exitCode = 1`；stdout 保持纯净 JSON
- `predict` 用 `try/finally` 保证 `agent.dispose()`
- `route` **不加载模型**，纯语言判断

### 9.2 `bin/laya-mcp.js` — MCP 服务入口

```
--transport stdio|http   --host 127.0.0.1   --port 7777
--models-dir /path       --english PATH  --multilingual PATH  --typed-decisions PATH
--max-loaded N (默认 1)  --max-pending N (默认 8，最大 64)  --allow-tiny
```

环境变量：`LAYA_MODELS_DIR`、`LAYA_MODEL_ENGLISH`、`LAYA_MODEL_MULTILINGUAL`、`LAYA_MODEL_TYPED_DECISIONS`（命令行覆盖）。

**stdio 模式的关键防护**（`laya-mcp.js:46`）：

```js
console.log = console.info = console.debug = (...v) => console.error(...v);
```

依赖库的普通日志会破坏 stdout 上的 JSON-RPC 帧，因此全部重定向到 stderr。stdio 模式下监听 `SIGINT`/`SIGTERM`/`stdin end` 三种退出信号。

### 9.3 五个 MCP 工具

| 工具 | 是否推理 | 说明 |
| --- | --- | --- |
| `laya_status` | 否 | 检查模型文件可用性、已加载列表、队列深度、上限；**不加载权重**；`checksums_verified` **恒为 `false`** |
| `laya_presets` | 否 | 返回五个内置问题模板的完整定义 |
| `laya_route` | 否 | 纯预览会选哪个 checkpoint 及理由 |
| `laya_predict` | 是 | choice / score / noul |
| `laya_shortlist` | 是 | 先 embedding 预筛 Top-K 再决策 |

**限制**（全部在 `laya_status.limits` 中自报）：

```js
{ argument_bytes: 262144, questions: 32, choice_candidates: 1000 }
```

**工具 instructions**（发给 MCP 客户端）关键约束：
> `confidence 不是正确率，noul 是概率，score 是从 0 开始的等级期望。安全判断只作辅助；不得把模型结果当作执行危险操作的授权。`

所有工具 `annotations = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false }`。

### 9.4 `src/mcp-http.js` — Streamable HTTP

**无状态设计**：每个 HTTP 请求创建**独立的 MCP server 实例**（`service.createServer('streamable-http')`），避免多客户端使用相同 JSON-RPC id 相互干扰；而**模型缓存与推理队列是共享的有状态**的。

安全与协议约束：

| 约束 | 实现 |
| --- | --- |
| 仅回环 | `host ∈ {127.0.0.1, localhost, ::1}`，否则拒绝 |
| DNS rebinding 防护 | 严格校验 `Host` 白名单 `{127.0.0.1:port, localhost:port, [::1]:port}`，不匹配 → 403 |
| Origin 校验 | 存在 `Origin` 头时必须在白名单内 |
| 仅 POST | `GET /mcp` → 405（无状态 MCP 不提供 GET SSE / DELETE 会话） |
| 健康检查 | `GET /health` → 200；**只表示进程存活** |
| 正文上限 | 1 MiB（`Content-Length` 预检 + 流式累计双重检查） |
| 接收超时 | 10s → 408 |
| Content-Type | 必须 `application/json` → 415 |
| 并发 | `connections ≥ 64` → 503；`maxConnections = 128` |
| 超时 | `requestTimeout 30s` / `headersTimeout 10s` / `keepAliveTimeout 5s` |

**关键工程细节**（`mcp-http.js:80`）：

```js
// SDK 的 JSON 响应 Promise 在断开时可能不结算，不能让它阻塞清理。
await Promise.race([transport.handleRequest(req, res, body), disconnected]);
```

`close()` 的顺序也是刻意的：先 `service.close()` 等在途推理结束，**再** `closeAllConnections()` 关掉还在读正文的连接，最后等所有 handler task 结算。

### 9.5 `skills/laya-decision/SKILL.md`

随包分发的 Agent Skill（109 行），教 Claude/Codex 如何正确使用这五个工具。核心纪律：

1. 先 `laya_status` 确认模型 `available`
2. `tiny: true` **不得用于真实语义结论**；`available: true` 只代表文件存在，不代表哈希完整或准确率合格
3. 中文/非英语**显式传 `lang`**，不依赖自动检测
4. 模型缺失时报告具体 checkpoint 和路径，**不下载大模型、不覆盖模型目录、不静默回退**
5. `choice` 若要允许拒绝分类，**必须显式设置 `other`/`unknown` 候选**，否则总会从候选中选一个
6. HTTP 模式下只关闭自己的连接，**不停止共享服务**

---

## 10. 测试与验收

```
test/
├── core.test.js          问题 DSL、JSON 序列化、原型污染、formatAnswers、shortlist 数学
├── runtime.test.js       制品校验、device/dtype 矩阵、损坏配置、dispose
├── router-shortlist.test.js  路由优先级、并发去重、LRU 释放、attach 冲突
├── integration.test.js   真实 Transformers.js/ONNX ↔ Python golden 对照
├── mcp.test.js           五个工具、限制、InMemory + 真实 stdio 子进程
├── mcp-http.test.js      Host/Origin/状态码/超时/取消/关闭顺序
├── cli.test.js           help、stderr-only、退出码
└── types.ts              tsc --strict 类型契约
```

**运行矩阵**：

```bash
npm test                      # 全部
npm run test:integration      # 真实 ONNX 对照
npm run test:mcp              # MCP 套件
npm run test:types            # tsc --strict
npm run test:models           # 三模型验收，落 artifacts/model-verification.json
```

**测试模型选择**：默认 `artifacts/tiny-english-v2`；设 `LAYA_TEST_MODEL` 指向完整模型时**该模型不可用就直接失败**（不静默降级为 skip）。tiny 随机模型必须显式 `--allow-tiny`。

**值得注意的测试断言**（都是历史 bug 的固化）：

- `jsonText` 拒绝循环引用 / `undefined` / `BigInt` / `Infinity` / `Date`
- 问题 ID `__proto__` 不能污染 `answers` 原型
- `model: '__proto__'` 必须抛错
- CLI 错误时 `stderr` 有输出、`stdout` 为空、退出码非零
- MCP `content[0].text` 与 `structuredContent` 必须一致
- 工作流匹配不能被「带分隔符的伪 ID」误命中

---

## 11. 设计决策与不变量

### 11.1 数值一致性优先于便利

| 决策 | 原因 |
| --- | --- |
| `jsonText` 保留 Python 的 `", "` / `": "` | 一个空格变化就会改变 token 序列，进而改变 logits |
| `instructions` 非 ASCII 转成 `\uXXXX` | 与 Python `json.dumps` 的 `ensure_ascii=True` 对齐 |
| embedding 截断保留起止 token | ModernBERT 的 CLS/SEP 不参与正文截断 |
| `optionOrder` 排序有 `a.i - b.i` 平局兜底 | 保证结果确定性 |
| 温度优先级 `by_options > type > 1` | 分桶温度是训练时校准的，必须优先命中 |

### 11.2 显式失败优于静默降级

- 候选超预算 → 抛错并提示用 `shortlist`，**不静默丢候选**
- 模型缺失 → 报出该跑哪条导出命令，**不自动下载**
- tiny 模型 → 除非 `--allow-tiny`，**一律拒绝**
- shape/非有限输出 → **抛错**，不让 NaN 传播进业务判断
- `clampTemperature` 非法值 → 回退 1 **并 emit warning**，让用户知道置信度不再可信

### 11.3 安全边界

- **原型污染**：`Object.create(null)` + `Object.hasOwn` 白名单 + `isRecord` 拒绝类实例
- **stdio 完整性**：`console.log/info/debug` 全部重定向 stderr
- **HTTP 本地暴露**：回环 + Host/Origin 白名单 + 仅 POST + 大小/并发上限
- **不记录用户输入**：MCP 只在 stderr 打印错误消息
- **不执行动作**：所有工具 `readOnlyHint: true`；Skill 明确禁止把结果当授权

### 11.4 资源生命周期

```
Agent   : #pending Set → dispose() 等所有在途任务结算 → runtime.dispose()
Router  : slot.users 引用计数 + waiters + #retire（已注册错误处理）
MCP     : close() = 等 tail（队列排空）→ router.dispose()
HTTP    : close() = service.close() → closeAllConnections() → 等 tasks
```

所有这些 `close()`/`dispose()` 都是**幂等**的（`??=` 缓存 Promise）。

### 11.5 中文命名导出

所有公共 API 同时提供 camelCase 与 snake_case 别名，因为这是从 Python 迁移的 SDK——用户可能按 Python 习惯调用。这个约束贯穿 `index.js` 和 `index.d.ts`。

---

## 12. 扩展指南

### 12.1 加一个新的问题类型

不建议——`QTYPES` 是 ONNX 图的硬协议。`qtype` 会作为 int64 送进图里的类型 embedding，改枚举会让已训练的模型失效。

要加新的**问题模板**（而非新类型），只需在 `presets.js` 加工厂函数并在 `index.js` 导出。

### 12.2 加一个新的 checkpoint

```bash
# 1. 上游加新 checkpoint 后，在 tools/export_onnx.py 的 --checkpoint choices 里加一项
.venv/bin/python tools/export_onnx.py --checkpoint <name> --output models/<name>

# 2. runtime.js 的 DEFAULT_MODELS 数组加一项
export const DEFAULT_MODELS = Object.freeze(Object.fromEntries(
  ['english', 'multilingual', 'typed-decisions', '<name>'].map((n) => [n, path.join(MODEL_ROOT, n)]),
));

# 3. router.js 的 ALIASES 加别名
```

### 12.3 加一个新的 MCP 工具

在 `mcp.js` 的 `createMcpService` 里 `register(name, description, schema, handler)`，schema 用 zod，wrapper 会自动做 256 KiB 检查和错误转 `isError`。若需要模型，在 handler 里用 `router.load(decision.model)`。

### 12.4 加一个新的工作流路由

在 `router.js` 的 `WORKFLOWS` 加一行「问题 ID 数组」。要求：ID 集合**精确匹配**（长度相等 + 全命中）。要让它自动触发，还需实例化时设 `autoTaskDetection: true`。

### 12.5 换推理后端

在 `runtime.js` 加分支，返回的对象必须满足接口：

```js
{ tokenizer, config, metadata, source, run(batch), dispose() }
```

`run(batch)` 接收 `Batch`（纯数组），返回 `{ logits, act_logits, embeddings }`，三者都要做 shape + 有限性校验。Tensor 构造只在 ONNX 路径做，保持 `collateItems` 的纯数组性质以便单测。

---

## 13. 故障排查手册

| 错误信息 | 原因 | 处理 |
| --- | --- | --- |
| `不是 laya-node v1 ONNX 制品` | `model_type ≠ 'custom'` 或 `format_version ≠ 1` | 重新 `export_onnx.py` |
| `上游仓库是 safetensors 格式，不能直接运行` | 传了 `convaiinnovations/laya` | 用导出后的本地目录 |
| `未找到导出模型 ... 先运行 python tools/export_onnx.py` | 目录缺 `config.json` | 按提示导出 |
| `该模型未做 CoreML 兼容改写` | `coreml_patched ≠ true` | 跑 `tools/prepare_coreml.py <dir>` |
| `CoreML EP 仅在 macOS` | 平台不符 | 换 `device: 'cpu'` |
| `CoreML EP 仅验证过 fp32` | coreml + q8/q4 | 改 `device: 'cpu'` + `dtype: 'q8'` |
| `ONNX 图的输入/输出与 Laya 协议不一致` | 图签名不匹配 | 重新导出 |
| `ONNX logits shape 不匹配` | 图被改写或 marker 数不符 | 检查 `marker_pos` 的 K |
| `候选超过 token 预算` | 候选太多 | 降低 `headMaxLength` 不行——应改用 `predictShortlist` |
| `maxLength 超过编码器最大位置数` | 超过 `max_position_embeddings`(8192) | 调小 `maxLength` |
| `LAYA_TEMPERATURE_CLAMPED` | checkpoint 温度超出 [0.5,5] | 置信度不可视为已校准；重新导出 |
| `tiny 随机模型不具备语义能力` | 未加 `--allow-tiny` | 用真实模型 |
| `推理队列已满` | 超过 `maxPending` | 等待或提高 `--max-pending`（≤64） |
| `模型 … 不可用`（MCP） | 文件缺失 | 报告路径，**不自动下载** |
| `工具参数超过 256 KiB` | 状态过大 | 按语义分段多次调用 |
| `不允许的 Host` / `不允许的 Origin` | 非本机来源 | 只从 127.0.0.1 访问 |
| `HTTP 无状态 MCP 不提供 GET SSE` | 浏览器打开 `/mcp` | 正常行为，用 POST |
| `MLX sidecar 启动超时（30s）` | 缺 mlx / 权重未下载 | `.venv/bin/pip install mlx`；设 `LAYA_MLX_PYTHON` |
| `MLX logits 含非有限数值` | 数值溢出 | 降 `maxLength` 或用 fp32 |

---

## 14. API 速查表

### 14.1 核心

```js
import { Agent, load, Router, DEFAULT_MODELS } from 'laya-node';

const agent = await load(DEFAULT_MODELS.english, { device: 'cpu', dtype: 'fp32' });
const r = await agent.predict(state, {
  intent:  { type: 'choice', instructions: '...', criteria: ['a', 'b'] },
  urgency: { type: 'score',  instructions: '...', criteria: ['low', 'high'] },
  refund:  { type: 'noul',   instructions: '...' },
}, { maxLength: 512, headMaxLength: 192, truncateLeft: false });

// r = { model, answers: {...}, usage: { input_tokens, output_tokens: 0 } }
await agent.dispose();
```

### 14.2 完整导出面

| 分组 | 导出 |
| --- | --- |
| **Agent** | `Agent`, `RLAgent`, `load` |
| **Router** | `Router`, `RouteDecision`, `DEFAULT_MODELS` |
| **问题 DSL** | `QTYPES`, `QTYPE_NAMES`, `serializeState`, `renderOptions` / `render_options` |
| **业务预设** | `triageQuestions`, `emailQuestions`, `guardQuestions`, `moderationQuestions`, `routerQuestions`（均带 `_questions` 别名） |
| **语言** | `detectLanguage` / `detect_language`, `detectScript` / `detect_script`, `isEnglish` / `is_english` |
| **邮件** | `cleanEmailBody` / `clean_email_body`, `emailState` / `email_state` |
| **预筛** | `shortlistChoice` / `shortlist_choice`, `predictShortlist` / `predict_shortlist`, `embedFnFromAgent` / `embed_fn_from_agent` |
| **数学** | `properReward` / `proper_reward`, `tdLambdaTargets` / `td_lambda_targets`, `eceScore` / `ece_score`, `confidenceFromProbs` / `confidence_from_probs`, `clampTemperature`, `tempBucket`, `softmax`, `TEMP_MIN`, `TEMP_MAX` |
| **MCP**（子路径） | `laya-node/mcp` → `createMcpService`, `createMcpServer`<br>`laya-node/mcp/http` → `startMcpHttpServer` |
| **版本** | `__version__` |

### 14.3 环境变量

| 变量 | 作用 |
| --- | --- |
| `LAYA_TEST_MODEL` | 测试/验收使用的模型目录（指定后不可用即失败） |
| `LAYA_MLX_PYTHON` | MLX sidecar 的 Python 解释器路径 |
| `LAYA_MODELS_DIR` | MCP 模型根目录 |
| `LAYA_MODEL_ENGLISH` / `_MULTILINGUAL` / `_TYPED_DECISIONS` | 单独指定 checkpoint |
| `LAYA_DTYPE` | 示例脚本的 dtype（`q8` 等） |
| `LAYA_DEVICE` | 示例脚本的 device（`coreml` / `mlx`） |
| `LAYA_THREADS` | `intraOpNumThreads`（P-core 数，Apple Silicon 提速 ~20%） |

### 14.4 npm scripts

| script | 作用 |
| --- | --- |
| `test` | `node --test test/*.test.js` |
| `test:integration` | 真实 ONNX ↔ golden 对照 |
| `test:mcp` | MCP + HTTP 套件 |
| `test:models` | 三模型验收，产出 JSON 报告 |
| `test:types` | `tsc --noEmit --strict`（TS 5.9.3） |
| `mcp` / `mcp:http` | 启动 MCP 服务 |
| `example` | `node examples/predict.js` |

---

## 附：源码索引

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `src/index.js` | 15 | 公共出口 + snake_case 别名 |
| `src/questions.js` | 95 | 问题 DSL、JSON 序列化、选项渲染 |
| `src/math.js` | 83 | 温度、softmax、confidence、ECE、properReward、TD(λ) |
| `src/sequence.js` | 65 | 特殊 token、序列构建、批处理 |
| `src/agent.js` | 109 | Agent 门面、答案格式化、embedding、优雅关闭 |
| `src/runtime.js` | 136 | 制品加载、ONNX/MLX 执行、shape 契约 |
| `src/mlx.js` | 89 | Python sidecar NDJSON 桥 |
| `src/router.js` | 143 | 路由决策、别名、LRU 槽位 |
| `src/shortlist.js` | 50 | Top-K 余弦预筛 |
| `src/lang.js` | 83 | 文字系统 + 拉丁语系检测 |
| `src/email.js` | 27 | 邮件引用/签名/免责声明清洗 |
| `src/presets.js` | 68 | 五个业务问题模板 |
| `src/mcp.js` | 177 | MCP 服务、五个工具、串行队列 |
| `src/mcp-http.js` | 121 | Streamable HTTP、回环安全、生命周期 |
| `src/index.d.ts` | 132 | 完整 TS 类型契约 |
| `bin/laya.js` | 102 | CLI |
| `bin/laya-mcp.js` | 83 | MCP 进程入口 |
| `tools/export_onnx.py` | 283 | 模型导出 + golden 生成 |
| `tools/mlx_runtime.py` | 270 | 纯 MLX 推理 / 校验 / 基准 |
| `tools/verify_models.js` | 123 | 验收与性能报告 |
| `tools/quantize_onnx.py` | 90 | int8 量化 |
| `tools/prepare_coreml.py` | 116 | CoreML 兼容改写 |
| `skills/laya-decision/SKILL.md` | 109 | Agent 使用纪律 |
