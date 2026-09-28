# laya-node

Laya 的独立 Node.js 类型化决策 SDK，基于 Transformers.js 加载完整 ONNX 模型，将文本或 JSON 状态转换为分类、等级评分、真假概率等结构化结果。

本项目不是聊天文本生成器，也不是上游官方发行版。它将 Laya 的推理与辅助能力迁移到 JavaScript；Python 仅用于模型导出、数值对照和可选的 MLX 后端 sidecar（Apple Silicon），默认推理不依赖 Python 服务或远程 LLM API。与上游的关系及不直接复用上游代码的原因见[「与上游 Laya 的关系」](#与上游-laya-的关系)。

## 功能概览

- **三类决策**：`choice` 标签选择、`score` 有序等级评分、`noul` 命题为真的概率；一次请求可包含多个问题。
- **本地推理**：完整编码器、决策头和动作头在同一 ONNX 图中执行；支持 CPU FP32、int8 量化、CoreML EP 与 Apple Silicon MLX 后端（见「推理加速」）。
- **多模型路由**：在 `english`、`multilingual`、`typed-decisions` 之间选择，支持显式指定、语言检测、可选工作流匹配。
- **候选筛选**：利用 embedding 与余弦相似度选出 Top-K，再执行 `choice` 决策。
- **业务预设**：客服分流、邮件处理、提示词安全、内容审核、请求路由，以及邮件正文清理。
- **工程支持**：异步 SDK、ESM、TypeScript 类型声明、CLI、按需加载、LRU 淘汰和资源释放。
- **Agent 接入**：stdio 与 Streamable HTTP 两种 MCP 方式，配套 Codex / Claude Code 共用的 `laya-decision` Skill。
- **数值验收**：对照 Python Laya 检查 token、marker、原始张量、最终答案和 embedding，校验模型文件哈希。

## 环境与安装

- Node.js **>= 22**，使用 ESM `import`。
- 推理依赖：`@huggingface/transformers@4.3.0`；MCP 适配依赖：`@modelcontextprotocol/sdk@1.30.0`、`zod@4.6.5`。SDK 主入口不导入 MCP 模块。
- Python 及 [tools/requirements.txt](tools/requirements.txt) 在导出、量化、CoreML 改写和重新生成数值对照时需要；MLX 后端还需额外 `.venv/bin/pip install mlx`。
- 预训练权重不包含在 npm 发布包中；完整模型可能占用数 GB 内存，请预留磁盘和内存空间。

以下命令均在 `laya-node` 项目根目录执行：

```bash
npm ci
```

在另一个 Node.js 项目中，可安装此本地目录：

```bash
npm install /absolute/path/to/laya-node
```

库入口为 `laya-node`，类型声明见 [src/index.d.ts](src/index.d.ts)。安装为依赖时建议显式传入模型绝对路径；`DEFAULT_MODELS` 指向 SDK 自身目录下的 `models/`，不是调用方工作目录。

## 模型准备

### 使用已有模型

如果已有导出制品，无需再次安装 Python 依赖或重新导出：

```bash
node bin/laya.js inspect --model ./models/english --verify
npm run example -- ./models/english
```

三个 checkpoint 的默认位置与用途：

| checkpoint | 默认目录 | 用途 |
| --- | --- | --- |
| `english` | `models/english` | 英语通用决策，也是默认模型 |
| `multilingual` | `models/multilingual` | 中文及其他非英语文本 |
| `typed-decisions` | `models/typed-decisions` | 特定工作流决策，默认不自动选用 |

### 首次导出完整模型

仅导出需要的 checkpoint。下列输出目录必须尚不存在；脚本会拒绝覆盖已有目录。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/requirements.txt

.venv/bin/python tools/export_onnx.py --checkpoint english --output models/english
.venv/bin/python tools/export_onnx.py --checkpoint multilingual --output models/multilingual
.venv/bin/python tools/export_onnx.py --checkpoint typed-decisions --output models/typed-decisions
```

导出脚本默认从 `convaiinnovations/laya` 获取原始权重，固定模型 revision 为 `1c5edc17a7acd8701df6fc341c0d179f1c62c982`。可通过 `--revision` 指定版本、`--cache-dir` 指定缓存，或通过 `--source` 使用直接包含 `rl_agent_config.json` 的本地原始 checkpoint 目录。

导出制品包含：

```text
models/english/
├── config.json             # ONNX 协议版本、模型元数据与推理配置
├── rl_agent_config.json    # 原始 Agent 配置
├── tokenizer.json
├── tokenizer_config.json
├── onnx/model.onnx         # 编码器、决策头、动作头及 embedding 输出
├── golden.json             # Python 数值对照样本
└── manifest.json           # 来源版本、文件大小、SHA-256、验收信息
```

已有图可重新生成 Python 对照数据，不覆盖 ONNX 权重：

```bash
.venv/bin/python tools/export_onnx.py --checkpoint english --output models/english --verify-only
```

`--verify-only` 会更新 `golden.json` 和 `manifest.json`，仍需要 Python 及原始 checkpoint；`checkpoint`、`tiny`、`revision` 必须与现有制品一致。只检查文件完整性时使用 `inspect --verify` 即可。

### tiny 技术验证模型

无需下载完整权重时，可在安装好 Python 依赖后生成随机小模型：

```bash
.venv/bin/python tools/export_onnx.py --tiny --output artifacts/tiny-english-v2
npm run example -- ./artifacts/tiny-english-v2
```

该目录若已存在，应直接复用。**tiny 仅验证运行协议、动态图和数值一致性，不具备真实语义决策能力，也不能作为完整模型验收结果。**

### 推理加速：q8 量化 / CoreML / MLX

默认 CPU FP32 在 Apple Silicon（M2 Max）上热推理约 710ms。下列选项不改变问题定义与答案语义：

| 路径 | 准备命令 | 加载配置 | 热推理实测* | 收益 / 代价 |
| --- | --- | --- | --- | --- |
| CPU 线程调优 | 无 | `sessionOptions: { intraOpNumThreads: 8 }` | ~552ms | 无损；设为 P-core 数约快 22% |
| int8 量化（q8） | `tools/quantize_onnx.py` | `dtype: 'q8'` | 约慢 15% | 模型 -65%、加载 -61%、内存大降 |
| CoreML EP | `tools/prepare_coreml.py` | `device: 'coreml'` | ~845ms | 本机型为负优化，先自测再启用 |
| MLX 后端（推荐） | `.venv/bin/pip install mlx` | `device: 'mlx'` | **~58ms（约 12×）** | 数值与 CPU 一致；冷启动数秒，需 Python sidecar |

\* 3 个问题 / 约 300 token 的示例请求热路径均值，绝对值因机器而异，建议用自己的负载复核。

#### int8 量化（q8）

```bash
.venv/bin/python tools/quantize_onnx.py models/english
# 产出 onnx/model_quantized.onnx：1608 MiB → 564 MiB，加载 1386ms → 537ms
```

int8 weight-only（MatMulNBits，块 128，激活保持 fp32），无需校准数据；golden 验收 logits 误差 <0.2、embedding cos >0.999，决策与 fp32 一致，代价是推理延迟约增加 15%。使用 `load(path, { dtype: 'q8' })`。`--mode q4` 与 `--mode dynamic` 在该模型上数值超差，仅作对照；若先跑过 CoreML 改写，量化产物同样兼容。

#### CoreML EP

```bash
.venv/bin/python tools/prepare_coreml.py models/english   # 就地无损改写，幂等
```

ModernBERT 注意力的 4D 广播掩码会让 CoreML 运行时崩溃，脚本在每个注意力相加前插入 Shape+Expand 使输入同形：数值与原模型逐位一致，CPU 性能不变，并在 config 中标记 `coreml_patched`（未改写的模型会拒绝以 CoreML 加载）。使用 `load(path, { device: 'coreml' })`，仅 macOS、仅 fp32（量化模型不支持 CoreML）。注意：该模型会被切成约 273 个子图，M2 Max 实测 ~845ms 慢于 CPU，启用前先跑自己的基准。

#### MLX 后端（Apple Silicon 最快）

```bash
.venv/bin/pip install mlx    # 复用现有 .venv
```

```js
const agent = await load('./models/english', { device: 'mlx' });   // 默认 fp32，与 CPU 结果一致
// 可选：{ device: 'mlx', dtype: 'fp16' }
// 可选：{ device: 'mlx', mlxModelDir: '/absolute/path/to/safetensors 目录' }
```

首次使用自动下载上游 safetensors（约 840MB）到 `.cache/huggingface`；也可用 `mlxModelDir` 指定本地目录。加载时启动 [tools/mlx_runtime.py](tools/mlx_runtime.py) 常驻 sidecar（NDJSON stdio 协议），随 `agent.dispose()` 关闭；解释器默认取 `.venv/bin/python`，可用环境变量 `LAYA_MLX_PYTHON` 覆盖。fp32 对照 golden.json 七个样本 logits 偏差 ≤4e-5、embedding cos=1.0；fp16 logits ≤4.5e-3、cos ≥0.999998。冷启动需数秒（权重加载与 Metal 编译），适合常驻 SDK 或服务，不适合单次命令。

示例脚本支持对应环境变量：

```bash
LAYA_DEVICE=mlx node examples/predict.js            # MLX fp32
LAYA_DEVICE=mlx LAYA_DTYPE=fp16 node examples/predict.js
LAYA_DTYPE=q8 node examples/predict.js              # 量化 ONNX
LAYA_DEVICE=coreml node examples/predict.js         # CoreML（需先改写模型）
LAYA_THREADS=8 node examples/predict.js             # ONNX 算子内线程数
```

## SDK 快速开始

下面的 ESM 示例使用已导出的中文模型。模型加载、推理和释放都需要 `await`：

```js
import { load } from 'laya-node';

const agent = await load('./models/multilingual', { localFilesOnly: true });
try {
  const result = await agent.predict(
    { message: '今天重复扣款了，请尽快帮我退款。' },
    {
      department: {
        type: 'choice',
        instructions: 'Which team should handle this request?',
        criteria: {
          billing: 'invoices, payments and refunds',
          technical: 'bugs and outages',
          other: 'other requests',
        },
      },
      urgency: {
        type: 'score',
        instructions: 'How urgent is this request?',
        criteria: ['no time pressure', 'soon', 'hard deadline'],
      },
      refund: {
        type: 'noul',
        instructions: 'Does the customer request a refund?',
      },
    },
  );

  console.log(result.answers.department.choice);
  console.log(result.answers.urgency.score);
  console.log(result.answers.refund.noul);
  console.log(result.usage);
} finally {
  await agent.dispose();
}
```

### 输入与输出语义

`state` 支持字符串、JSON 对象或数组；不接受顶层数字、布尔值、`null`，也不能包含循环引用、`undefined`、`BigInt`、非有限数值或类实例。`questions` 是问题 ID 到定义的对象，各问题需提供 `type` 和 `instructions`。

| 类型 | `criteria` | 主要结果 |
| --- | --- | --- |
| `choice` | 非空且不重复的字符串数组，或标签到 JSON 描述的对象 | `choice` 为最高概率标签；`probabilities` 为标签概率 |
| `score` | 非空有序等级数组 | `score` 为等级索引的概率加权期望；`legend` 为索引说明；`probabilities` 为各等级概率 |
| `noul` | 可省略，或仅含 `true` / `false` 描述的对象 | `noul` 为命题为真的概率，范围 `[0, 1]`，不是布尔值 |

注意：

- `score` 的等级索引从 `0` 开始，结果可能为小数，不是最高概率等级，也不是自动转换的百分制。
- `choice` / `score` 的 `confidence` 为 `1 - H(p) / log(k)`，单候选为 `1`；`noul` 的 `confidence` 为 `max(p, 1-p)`。它们不是同一口径的正确率估计。
- 每个答案还包含 `action.act_probability`，即动作头第一个输出的 softmax 概率；SDK 不会据此自动执行业务动作。
- 答案中的概率、评分和置信度保留四位小数，概率之和可能有舍入误差。
- `usage.input_tokens` 是各问题实际序列长度之和；模型不生成文本，`usage.output_tokens` 固定为 `0`。
- `result.model` 固定为 `laya-rl-agent`；实际 checkpoint 查看 `agent.metadata.checkpoint` 或 Router 返回的 `routing.model`。

### 常用 API 与选项

| API | 作用 |
| --- | --- |
| `load(path, options)` / `Agent.load(path, options)` | 异步加载模型，省略路径时使用默认英语模型 |
| `agent.predict(state, questions, options)` | 返回结构化决策结果 |
| `agent.systemOne(...)` / `agent.system_one(...)` | `predict` 的别名 |
| `agent.prepare(...)` | 检查输入并构造 token、marker 和 batch，不执行推理 |
| `agent.predictRaw(...)` | 返回构造数据及 `logits`、`act_logits`、`embeddings` |
| `agent.embed(texts, options)` | 提取编码器均值池化 embedding；支持 `maxLength`、`batchSize` |
| `agent.dispose()` | 拒绝新推理，等待在途操作后释放模型；重复调用复用释放过程 |

加载选项包括 `localFilesOnly`、`cacheDir`、`revision`、`progressCallback`、`sessionOptions`、`device: 'cpu' | 'coreml' | 'mlx'` 和 `dtype: 'fp32' | 'q8' | 'q4' | 'fp16'`（`fp16` 仅 MLX，量化仅 ONNX，CoreML 需先改写模型，详见「推理加速」）；MLX 还可用 `mlxModelDir` 指定 safetensors 目录。本地路径只读取本地文件；远程仓库必须是符合本项目协议的 ONNX 制品，不能直接传入上游 safetensors 仓库。

推理选项：

- `maxLength`：总 token 预算，默认读取模型的 `max_len`，不能超过编码器位置上限。
- `headMaxLength`：问题和候选部分的预算，默认读取模型的 `head_max_len`。
- `truncateLeft`：默认 `false`，正文超长时保留开头；设为 `true` 时保留尾部。

候选部分超预算会明确报错，不会静默丢弃候选；可调整预算或使用 shortlist。`RLAgent` 是 `Agent` 的兼容别名，并不提供训练接口。部分工具同时导出 camelCase 与 snake_case 名称，但参数仍采用 JavaScript options 对象。

## 自动路由与模型生命周期

```js
import { Router, triageQuestions } from 'laya-node';

const router = new Router({
  models: {
    english: './models/english',
    multilingual: './models/multilingual',
  },
  maxLoaded: 1,
  localFilesOnly: true,
});

try {
  const state = { message: '发票重复扣款，请退款。' };
  const questions = triageQuestions();
  console.log(router.route(state, questions)); // 仅选择模型，不加载权重
  const result = await router.predict(state, questions, { lang: 'zh' });
  console.log(result.routing, result.answers);
} finally {
  await router.dispose();
}
```

路由优先级从高到低为：显式 `model` → 显式 `task` → 已启用的工作流匹配 → 显式 `lang` → 语言启发式 → 无可识别字母时使用 `default`。

- `model` / `task` 接受模型名或别名，如 `english` / `en`、`multilingual` / `multi`、`typed-decisions` / `typed`；它们不是任意任务描述或模型文件路径。路径通过构造器的 `models` 配置。
- `autoTaskDetection` 默认关闭。开启后，仅按问题 ID 集合精确匹配 `agent_trace_observability`、`customer_service`、`invoice_processing`、`security_incidents`，不做通用语义任务识别；具体 ID 见 [src/router.js](src/router.js)。
- 语言检测为文字与词汇启发式，短文本或混合语言可能误判；已知语言时显式提供 `lang`。
- 默认 `maxLoaded: 1`，复用相同模型的并发加载，对空闲模型做 LRU 淘汰。在途加载或推理可能暂时超过该数量，因此它不是硬性内存上限。
- `preload(names)` 会预加载并提高 `maxLoaded` 以保留模型；无参数会预加载全部模型，需注意内存。
- `attach(name, agent)` 可交给 Router 管理现有实例；`unload(name)` 卸载指定模型，省略名称卸载全部；`dispose()` 关闭 Router 并等待资源释放。

仅观察路由无需模型文件：

```bash
node examples/router.js
# 已准备 english 和 multilingual 后，可同时执行推理
node examples/router.js --predict
```

## 候选筛选与业务预设

`predictShortlist` 仅缩减 `choice` 问题：对问题、状态与候选计算 embedding 余弦相似度，保留 Top-K，再调用模型。其他问题类型保持不变。

```js
import { load, predictShortlist, embedFnFromAgent } from 'laya-node';

const agent = await load('./models/english');
try {
  const result = await predictShortlist(agent, 'Please refund my duplicate payment.', {
    intent: {
      type: 'choice',
      instructions: 'Which intent best describes the request?',
      criteria: ['refund', 'outage', 'sales', 'cancel', 'password', 'invoice'],
    },
  }, { k: 3, embedFn: embedFnFromAgent(agent) });
  console.log(result.answers.intent, result.shortlist.intent);
} finally {
  await agent.dispose();
}
```

`k` 默认 `20`。候选数不超过 `k` 时直接透传，不调用 embedding；需要缩减时必须提供 `embedFn`，也可使用外部 embedding 函数。`shortlistChoice` 仅返回入围标签。

**缩减后的概率只在入围标签之间归一化，不代表完整标签集。** 查看 `shortlist.<问题 ID>.probability_scope`：`shortlisted` 表示已筛选，`all` 表示未缩减。Embedding 筛选可能遗漏正确候选，应单独评估召回率。

| 预设 | 推荐 `state` 字段 | 主要问题 |
| --- | --- | --- |
| `triageQuestions()` | `message` | 意图、紧急程度、挫败程度、退款、流失风险 |
| `emailQuestions(categories?)` | `body`，可加 `subject` | 分类、垃圾邮件、钓鱼、紧急程度、是否需回复 |
| `guardQuestions()` | `prompt` | 越狱、提示词注入、敏感数据、危害、主题 |
| `moderationQuestions()` | `post` | 不文明内容、骚扰、威胁、垃圾信息、违规程度 |
| `routerQuestions()` | `request` | 难度、领域、工具需求、敏感性 |

`routerQuestions()` 是请求分类问题集，与选择 checkpoint 的 `Router` 不同。预设保留上游英文提示词，每次调用返回独立定义。

邮件辅助函数 `cleanEmailBody(body, maxChars = 3000)` 启发式清理引用、签名和免责声明；`emailState(subject, body, { sender, clean, ...extra })` 生成邮件状态，默认清理正文。这不是 HTML 安全过滤器或完整邮件解析器。

还导出 `softmax`、`confidenceFromProbs`、`clampTemperature`、`tempBucket`、`eceScore`、`properReward`、`tdLambdaTargets` 等数学工具；完整签名见 [类型声明](src/index.d.ts)。

## CLI

源码目录使用 `node bin/laya.js`；安装后的命令名为 `laya-node`。

```bash
node bin/laya.js --help
node bin/laya.js predict --model ./models/english --input examples/request.json --offline
node bin/laya.js route --input examples/request.json --lang zh
node bin/laya.js inspect --model ./models/english --verify
```

通过 stdin 与业务预设推理：

```bash
printf '%s\n' '{"state":{"message":"重复扣款，请退款。"}}' | node bin/laya.js predict --model ./models/multilingual --preset triage --input - --offline
```

- 输入结构为 `{ "state": ..., "questions": { ... } }`；`--input` / `-i` 指定文件，`-` 或省略参数时读取 stdin，stdin 上限为 32 MiB。
- `--preset` 支持 `triage`、`email`、`guard`、`moderation`、`router`；输入自带 `questions` 时优先使用输入，不与预设合并。
- `predict` 使用 `--model` 指定的路径或 ONNX 仓库，可设置 `--offline`、`--revision`、`--max-length`、`--head-max-length`、`--truncate-left`。
- `route` **只输出路由决定，不执行推理**；此时 `--model` 是模型名或别名，支持 `--task` 和 `--lang`。`predict` 不通过这些参数自动路由；自动选择并推理请使用 SDK 的 `Router.predict()`。
- `inspect` 读取本地制品元数据和 manifest；加 `--verify` 校验全部清单文件的大小与 SHA-256，不做语义质量评估。
- 结果 JSON 写入 stdout，错误写入 stderr 并以非零状态退出。

## Codex / Claude Code：MCP + Skill

MCP 提供可调用的本地推理工具；Skill 指导 Agent 何时调用、如何构造问题及解读概率。两者需要分别接入。支持以下两种传输方式，无需配置 API Key 或 Python 常驻服务。

| 方式 | 服务如何启动 | 客户端配置 | 模型生命周期 |
| --- | --- | --- | --- |
| stdio（默认） | 客户端启动子进程 | Node.js / 脚本路径及模型目录 | 每个客户端进程独立缓存 |
| Streamable HTTP | 用户预先启动独立常驻服务 | `http://127.0.0.1:7777/mcp` | 多客户端共享缓存与队列 |

每个服务进程选择一种传输方式；原有 stdio 配置不变。HTTP 客户端不需要知道项目或模型在哪里，但服务启动端仍需配置本地模型。

### 启动 MCP 服务

stdio 通常由客户端自动启动，也可以直接运行并等待 stdin JSON-RPC：

```bash
node bin/laya-mcp.js --help
node bin/laya-mcp.js --models-dir /absolute/path/to/laya-node/models
```

安装为依赖后命令名为 `laya-node-mcp`。stdio 客户端应直接运行 `node` + 脚本绝对路径或此可执行文件，不使用会向 stdout 输出启动横幅的 `npm run mcp`。脚本和模型路径含空格时仍需作为独立参数传入。

Streamable HTTP 在 `laya-node` 项目根目录启动，保持此进程运行：

```bash
npm run mcp:http -- --port 7777
# 等价命令；模型不在默认目录时可指定 --models-dir
node bin/laya-mcp.js --transport http --host 127.0.0.1 --port 7777 --models-dir /absolute/path/to/models
```

上面两条命令二选一。默认使用 SDK 自身的 `models/`，启动时不加载权重；第一次推理才加载。HTTP 模式不通过 stdout 传协议，因此可以使用 `npm run mcp:http`。关闭 stdin 不会退出，`Ctrl+C` / SIGTERM 会停止服务并释放模型。

- MCP 地址：`http://127.0.0.1:7777/mcp`，客户端应使用 Streamable HTTP，而不是旧 SSE transport。
- 健康检查：`GET http://127.0.0.1:7777/health`；只表示服务存活，不表示模型可用、已加载或通过验收。
- 采用无状态 JSON 响应，每个 POST 独立处理；不提供 GET SSE、会话恢复或 DELETE 会话操作。浏览器直接打开 `/mcp` 返回 405 是正常行为。
- 仅支持回环地址 `127.0.0.1`、`localhost`（绑定 IPv4）、`::1`。Host / Origin 校验限制本机地址和实际端口，不开放跨源 CORS；这是无鉴权本机服务，不应通过代理公开到局域网或公网。
- HTTP 正文上限 1 MiB，解码后的工具参数仍限 256 KiB；最多同时处理 64 个 MCP HTTP 请求，推理另受 `--max-pending` 限制。

### 共用工具与模型配置

| 工具 | 功能 |
| --- | --- |
| `laya_status` | 查看模型文件可用性、已加载模型和队列，不加载权重 |
| `laya_presets` | 查看全部或指定业务问题模板，不执行推理 |
| `laya_route` | 预览 checkpoint 路由，不执行推理 |
| `laya_predict` | `choice` / `score` / `noul` 推理；`questions` 与 `preset` 二选一 |
| `laya_shortlist` | embedding Top-K 筛选后推理；默认 `k: 20`，标注概率范围 |

推理参数为 `{ state, questions? , preset?, model?, lang?, maxLength?, headMaxLength?, truncateLeft? }`；shortlist 额外接受 `k`。MCP 的 `instructions` 限定为非空字符串，每次最多 32 个问题、每个 choice 最多 1000 个候选、score 最多 100 个等级；问题 ID 和 choice 标签长度为 1–200 字符。参数 JSON 上限 256 KiB，超出时返回错误，由 Skill 指导分段处理。现有 SDK 的输入契约不变。

成功响应同时返回 `structuredContent` 和相同内容的 JSON 文本 `content`；失败返回 `isError: true`。推理结果增加 `routing` 和 `diagnostics`（真实 checkpoint、tiny 标记、温度限幅、token 预算和正文截断策略）。策略不是实际截断检测；正文超预算仍沿用 SDK 的截断行为。

模型路径配置优先级：单模型命令行参数 > 单模型环境变量 > `--models-dir` / `LAYA_MODELS_DIR` > SDK 自身的 `models/`。三个单模型选项为 `--english`、`--multilingual`、`--typed-decisions`，对应 `LAYA_MODEL_ENGLISH`、`LAYA_MODEL_MULTILINGUAL`、`LAYA_MODEL_TYPED_DECISIONS`。建议始终使用绝对路径。

- 始终仅本地 CPU FP32；不自动下载、不回退模型；工具参数不能指定文件路径或远程仓库。
- 按需加载、默认 `--max-loaded 1`；模型切换时可能短暂同时存在两个模型，不是硬性内存限额。stdio 每个客户端独立启动进程；同一 HTTP 服务的多个客户端共享模型，某个客户端关闭不释放共享缓存。
- 推理串行排队，`--max-pending 8` 为服务内在途加排队上限，HTTP 多客户端共用此队列。队列满时显式报错；状态和模板查询无需等待推理。
- stdio 取消通知或 HTTP 请求连接断开会阻止尚未执行的排队推理；已启动的 ONNX 调用不能强行中断，完成后丢弃结果。无状态 HTTP 不维护跨独立 POST 的会话级取消通知映射。
- SIGINT / SIGTERM 会停止接收工作、跳过排队任务并等待在途推理释放资源；stdin EOF 仅关闭 stdio 服务。
- tiny 默认禁止推理，只有协议测试才使用 `--allow-tiny`；它不具备真实语义能力。
- 所有工具只读，不执行业务动作。stdio stdout 仅传 JSON-RPC，诊断写入 stderr；服务不主动记录请求正文。客户端如何保留会话由客户端配置决定。
- `laya_status.available` 不是完整性或准确率验收；需要哈希校验时运行原有 `inspect --verify`。

### 接入 Claude Code

在希望启用的项目目录中选择一种方式运行：

```bash
# stdio：替换为真实路径
claude mcp add --transport stdio --scope project laya -- node /absolute/path/to/laya-node/bin/laya-mcp.js --models-dir /absolute/path/to/laya-node/models

# HTTP：先启动上面的独立 HTTP 服务
claude mcp add --transport http --scope project laya http://127.0.0.1:7777/mcp
```

或者选择 [stdio 模板](examples/claude-mcp.json) / [HTTP 模板](examples/claude-mcp-http.json)，将 `mcpServers.laya` 合并到该项目的 `.mcp.json`。同名 `laya` 只保留一种方式，不覆盖其他服务器。项目配置可能需要用户批准；在 Claude Code 中用 `/mcp` 检查连接。仅 stdio 模式需要替换脚本路径；若 GUI 找不到 nvm 的 `node`，将 `command` 改为本机 Node.js >=22 的可执行文件绝对路径。

### 接入 Codex

选择 [stdio 模板](examples/codex-mcp.toml) / [HTTP 模板](examples/codex-mcp-http.toml)，将 `[mcp_servers.laya]` 段合并到 `~/.codex/config.toml` 或受信任项目的 `.codex/config.toml`。stdio 需替换占位路径；HTTP 只需以下配置：

```toml
[mcp_servers.laya]
url = "http://127.0.0.1:7777/mcp"
startup_timeout_sec = 20
tool_timeout_sec = 180
```

同名配置选择一种方式，HTTP 不同时设置 stdio 的 `command` / `args`，不要覆盖其他配置。示例的 180 秒工具超时为首次完整模型加载预留时间。重新加载客户端后确认能发现上述五个工具。

### 安装共用 Skill

源文件：[skills/laya-decision/SKILL.md](skills/laya-decision/SKILL.md)。stdio 与 HTTP 共用同一个 Skill，无需分别安装。复制整个 `laya-decision` 文件夹到目标项目对应目录：

| 客户端 | 项目级位置 | 用户级位置 |
| --- | --- | --- |
| Codex | `.agents/skills/laya-decision/` | `~/.agents/skills/laya-decision/` |
| Claude Code | `.claude/skills/laya-decision/` | `~/.claude/skills/laya-decision/` |

例如在目标项目执行以下命令；目录已存在时先比较内容，不覆盖已有 Skill：

```bash
mkdir -p .agents/skills .claude/skills
cp -Rn /absolute/path/to/laya-node/skills/laya-decision .agents/skills/
cp -Rn /absolute/path/to/laya-node/skills/laya-decision .claude/skills/
```

Codex 显式调用 `$laya-decision`，Claude Code 显式调用 `/laya-decision`；也可用自然语言要求「用本地 Laya 判断这条工单的类别与退款意图」。Skill 不会自行修改客户端配置或安装模型。

### 不改客户端配置的验证

```bash
# 官方 MCP Client 启动子进程：握手、工具发现、模型状态（不加载权重）
node examples/mcp-client.js
# 实际用本地 multilingual 模型完成中文推理（stdio）
node examples/mcp-client.js --predict
# 先在另一终端 npm run mcp:http；以下只连接服务，不启动子进程
node examples/mcp-client.js --url http://127.0.0.1:7777/mcp
node examples/mcp-client.js --url http://127.0.0.1:7777/mcp --predict
npm run test:mcp
# 指定完整模型复验 stdio / HTTP 推理与 Python golden 样本
LAYA_TEST_MODEL=./models/english npm run test:mcp
```

HTTP 示例的 `--url` 与 `--models-dir` 不能同时使用，模型路径由服务启动端配置。示例退出仅关闭客户端连接，不停止共享 HTTP 服务。

可嵌入入口（主 SDK 入口保持不变）：

- `import { createMcpServer } from 'laya-node/mcp'` 返回 `{ server, close }`，用于单个 stdio 连接；将 `server` 接入 transport，结束时 `await close()`。
- `import { startMcpHttpServer } from 'laya-node/mcp/http'`，`await startMcpHttpServer({ host: '127.0.0.1', port: 7777, models: { ... } })` 返回 `{ url, close }`；`await close()` 等待在途推理并释放监听端口、模型与连接。
- 高级嵌入可使用 `createMcpService(options)` 创建共享缓存，调用 `service.createServer('streamable-http')` 生成独立协议实例；关闭单个实例不关闭缓存，最终需分别关闭协议实例并 `await service.close()`。

接入格式依据：[Codex MCP](https://developers.openai.com/codex/mcp)、[Codex Skills](https://developers.openai.com/codex/skills)、[Claude Code MCP](https://code.claude.com/docs/en/mcp)、[Claude Code Skills](https://code.claude.com/docs/en/skills)。

## 测试与验收

```bash
npm test
npm run test:types
npm run test:integration

# 指定完整模型执行集成测试
LAYA_TEST_MODEL=./models/english npm run test:integration

# 逐个验收三个完整模型，并保存数值和性能报告
npm run test:models -- --output artifacts/model-verification.json

# 仅验收一个完整模型
npm run test:models -- --model ./models/english --iterations 5

# tiny 仅作为技术回归，必须显式允许
npm run test:models -- --model ./artifacts/tiny-english-v2 --allow-tiny
```

- `npm test` 覆盖输入校验、序列预算、数学函数、语言与邮件工具、路由生命周期、shortlist、CLI、运行时错误及真实 ONNX 对照。
- `npm run test:mcp` 覆盖 stdio / HTTP 官方客户端握手与推理、多客户端共享缓存与队列、Host / Origin 校验、请求大小限制、取消及关闭生命周期。
- 模型相关测试默认使用 `artifacts/tiny-english-v2`，缺少默认制品时部分测试会跳过；显式指定不存在的 `LAYA_TEST_MODEL` 会失败。不要把跳过测试当成模型已验收。
- `test:types` 通过 `npm exec` 使用固定的 TypeScript `5.9.3`，首次可能需要联网获取编译器。
- `test:models` 默认要求三个完整模型全部存在，不会静默跳过，也不会默认接受 tiny；每个模型在独立进程中验收。
- 验收检查文件哈希、token / marker 精确一致、原始张量与答案及 embedding；数值容差为 `atol=1e-4`、`rtol=1e-3`。
- 性能报告包含新进程加载时间、热推理均值 / P50 / P95 和 RSS。冷加载未清空操作系统文件缓存，热推理包含样例内全部问题及后处理，不能直接当作单问题延迟。

本地已有报告见 `artifacts/model-verification.json`。数值对照通过表示迁移实现与参考样本一致，不等于业务准确率、概率校准或安全效果已达标。

## 实现结构

```text
state + questions
  → 输入校验与 JSON 序列化
  → 每个问题构造一条序列：问题 + 候选 marker + 状态正文
  → 多问题 padding 成 batch
  → Transformers.js / ONNX（或 MLX sidecar）：编码器 + 类型嵌入 + 决策头 + 动作头
  → 温度处理、softmax、结果格式化
  → answers + usage
```

| 路径 | 职责 |
| --- | --- |
| `src/index.js` / `src/index.d.ts` | 公共导出与 TypeScript API |
| `src/agent.js` | 推理编排、答案格式化、embedding、资源生命周期 |
| `src/runtime.js` | 模型加载、配置与张量协议校验，CPU / CoreML / MLX 执行分发 |
| `src/mlx.js` | MLX sidecar 进程管理：NDJSON stdio、就绪握手、请求队列与释放 |
| `src/questions.js` / `src/sequence.js` | 问题校验、选项渲染、token 预算与 batch 构建 |
| `src/router.js` / `src/lang.js` | checkpoint 路由、模型缓存与语言启发式 |
| `src/shortlist.js` | embedding 候选筛选 |
| `src/presets.js` / `src/email.js` / `src/math.js` | 业务预设、文本清理与数学工具 |
| `bin/laya.js` | CLI 入口 |
| `bin/laya-mcp.js` / `src/mcp.js` / `src/mcp.d.ts` | 双传输 CLI、共用 MCP 工具、模型队列与类型声明 |
| `src/mcp-http.js` / `src/mcp-http.d.ts` | Streamable HTTP 监听、安全校验与连接生命周期 |
| `skills/laya-decision/` | Codex / Claude Code 共用决策 Skill |
| `tools/` | Python 导出、q8 量化、CoreML 改写、MLX 运行时与数值验收 |
| `examples/` / `test/` | 用法示例与测试 |
| `models/` / `artifacts/` / `.cache/` | 本地模型、验证产物和缓存，不随 npm 包发布 |

ONNX 输入为 `input_ids`、`attention_mask`、`marker_pos`、`marker_mask`、`qtype`；输出为 `logits`、`act_logits`、`embeddings`。普通文本分类 ONNX 或仅编码器模型不能直接替代该完整图。

## 与上游 Laya 的关系

引入 MLX sidecar 后常见疑问：既然已经依赖 Python 进程，为什么不直接使用上游 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)？因为两者定位不同——sidecar 只借用 Python 的数值计算（约 50MB 的 `mlx` 包，无 PyTorch），上游则把整个决策系统放在 Python/PyTorch 生态中：

| 维度 | 本项目的 MLX sidecar | 上游 laya |
| --- | --- | --- |
| Python 依赖 | 仅 `mlx`（约 50MB，无 PyTorch） | PyTorch + transformers 完整科学栈（2–4GB） |
| Python 代码范围 | [tools/mlx_runtime.py](tools/mlx_runtime.py) 约 260 行纯前向，私有 NDJSON 协议 | 训练、渲染、推理一体的研究代码 |
| 序列构造 | JS 侧实现，`golden.json` 逐 token 锁定一致性 | 全在 Python 侧，JS 生态不可复用 |
| MLX 加速 | 已实现（约 58ms） | 无 MLX 后端；同等速度仍需自行实现前向 |
| 工程层 | 输入校验、路由、生命周期、MCP、CLI、类型与测试 | 需自行构建或跨语言桥接 |
| API 稳定性 | golden 数值验收与测试锁定的私有协议 | 研究仓库，无 API 承诺 |

- **上游省不掉这份工作**：上游没有 MLX 路径，切换过去后想要同样的推理速度，仍需手写 ModernBERT + 决策头的 MLX 前向，即 `mlx_runtime.py` 的实现一行都少不掉。
- **依赖重量决定产品边界**：本项目默认路径零 Python 依赖（`npm install` 即用），MLX 是可选增强；上游强制完整科学栈。
- **架构主权**：本项目由 JS 拥有全部输入构造（问题渲染、token 预算、截断、marker 位置），Python 只做纯函数矩阵乘，数值一致性因此可被 golden 样本逐 token 验收；换用上游会把序列构造交回 Python，Router / shortlist / MCP / CLI 全部退化为跨语言桥接。
- **何时该用上游**：训练、微调、修改奖励函数或复现实验是上游的领域。纯推理部署场景，上游不提供本项目的任何工程能力。

## 使用边界与许可

- 默认本地 CPU FP32；可选 int8 量化、CoreML EP 与 Apple Silicon MLX 后端（见「推理加速」）。不提供 CUDA、WebGPU 或浏览器运行支持；MLX 依赖本机 Python sidecar 进程。
- checkpoint 温度会限幅到 `[0.5, 5]`；超界或无效时发出 `LAYA_TEMPERATURE_CLAMPED` 警告，受影响置信度不能视为已校准。
- 这是推理 SDK，提供本机 stdio / Streamable HTTP MCP 适配，不包含模型训练、自动执行业务操作、公开网络服务或通用 Agent 工具执行循环。
- 安全、审核、退款等高风险场景应结合业务规则、人工复核和真实数据评估，不能只依赖单次模型概率。
- 代码采用 Apache-2.0，详见 [LICENSE](LICENSE)。上游来源、参考版本和迁移说明见 [NOTICE](NOTICE)；模型权重与第三方依赖遵循各自许可证。
