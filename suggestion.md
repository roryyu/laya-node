# laya-node 优化改进建议

> 基于 `codewiki.md` 的全量代码走查 + 6 项本地实测
> 基线：commit `e040af8`。`npm test` 已从 66 passed 扩到 **126 passed**（新增 `test/accuracy.test.js` 与 `test/perf.test.js`）
> 文档中的数字分为 **[实测]**（我本地跑出来的）和 **[引自 README]** 两类，已分别标注

---

## 阅读指引

| 优先级 | 含义 |
| --- | --- |
| **P0** | 影响可用性或用户可自救性，建议本轮就做 |
| **P1** | 影响公开 API 契约正确性，建议下个版本做 |
| **P2** | 可维护性，趁改动相关代码时顺手做 |
| **P3** | 工程基建，一次性投入 |
| **✗** | **实测证明不值得做**，列出来是为了避免重复讨论 |

第 6 节「明确不建议做的事」是我特意加的——这类项目最容易把时间浪费在 0.3% 的优化上。

---

## 1. P0-1 · shortlist 的 embedding 成本是 O(全部候选数)，与 k 无关

> **本条已被实测修正。** 初稿写的是"暴露 embedding batchSize，可省 16 倍"。**实测证明批大小对吞吐几乎无影响**，该结论作废，以下是修正后的版本。

### 实测：批大小不影响吞吐

`Agent.embed` 内部是串行 for 循环（`agent.js:90`），MCP 又把 `batchSize` 写死 16（`mcp.js:141`）。我曾推断"1000 候选 = 63 次串行 forward ≈ 45 秒，改 batchSize 到 256 只需 4 次 ≈ 2.8 秒"。

**这个推断是错的**：单次 forward 的延迟本身就随 batch 线性增长，总时间不变。

| n=256，`maxLength=64` | 总耗时 | 单条成本 |
| --- | --- | --- |
| `batchSize=16` | 4339ms | 16.9ms |
| `batchSize=32` | 4292ms | 16.8ms |
| `batchSize=64` | 4305ms | 16.8ms |
| `batchSize=256` | 4575ms | 17.9ms |

`test/perf.test.js` 断言"单条成本随 batchSize 波动 < 2.5x"，实测离散度 **1.06–1.28x**。编码器是**计算瓶颈**而非调度瓶颈，批大小只改变单次 forward 的宽度。所以**暴露 `batchSize` 不是杠杆**。

### 实测：真正的成本模型

用计数 `embedFn` 直接测出 `predictShortlist` 的行为：

```
200 个候选，k=10  →  embedFn 收到 201 条文本
```

即 **query + 全部 200 个候选**。`k` 不影响 embedding 成本——所有候选都必须嵌入才能排序。

叠加实测吞吐（`maxLength=64` 约 12ms/条，MCP 用 `maxLength=512` 约 16.7ms/条）：

| 候选数 | embedding 成本 | 加上一次 predict |
| --- | --- | --- |
| 100 | ~1.7s（`maxLength=512`） | 2.45s（实测，`maxLength=64`） |
| 500 | ~8.4s | ~9s |
| 1000 | **~17s** | ~17s |

**`shortlist` 在 1000 候选时要花 17 秒，其中 99% 花在嵌入候选上，而真正需要分类的只有 k=20 个。**

### 建议（按杠杆从大到小）

1. **加"廉价预筛"层**：在 embedding 之前先用字符/词重叠、长度、前缀等零成本信号把候选压到 ~2k，再对 2k 做 embedding。对典型分类任务（标签集有语义结构）能省一个数量级。这是唯一能真正改变 O(n) 的办法。
2. **`shortlist` 复用编码器输出**：当前 `rank()` 为 query 和每个候选各调一次 `embed`，但候选文本与实际决策无关，且 `maxLength=512` 对短标签严重过宽。**给候选单独设一个更小的 `maxLength`**（实测 `maxLength=64` 时 12ms/条 vs 512 时 16.7ms/条，**省 28%**）：
   ```js
   // mcp.js —— 候选文本很短，不需要 512
   embedFn: embedFnFromAgent(agent, { maxLength: 128, batchSize: 32 })
   ```
   这条是**低成本、立刻可做**的，建议优先。
3. **在 `shortlist` 结果里暴露成本信息**：`diagnostics` 加 `embedded_texts` 和 `probability_scope`，让调用方知道这次调用嵌了多少条。1000 候选 17 秒而客户端毫无感知，是个可用性问题。
4. **给 `predictShortlist` 加 `signal`**（见 P1-3）：17 秒不可取消是真问题。
5. ~~暴露 `batchSize`~~ —— **不建议**，实测无收益。可以从待办里划掉。

## 2. P0-2 · `buildSequence` 在抛"超预算"之前把全部候选 tokenize 完了

### 问题

`sequence.js:31-43` 的顺序是：**先 tokenize 所有候选 → 再检查预算 → 超了才抛**。

```js
let options = order.map((i) => [tokens.mask, ...encode(tokenizer, ` ${clean(opts[i])}`).slice(0, 48)]);  // ← n 次 encode 先做完
let budget = headMaxLength - options.reduce((n, a) => n + a.length, 0);
if (budget < 16) { /* 二次均分 */ }
head = head.slice(0, Math.max(8, budget));
if (head.length + optionSize > headMaxLength || ...) throw new RangeError(...);   // ← 才知道超了
```

### 实测

用 `artifacts/tiny-english-v2` 的 tokenizer，候选带一句描述（`a description for candidate number N in a support taxonomy`）：

| 候选数 | `buildSequence` 耗时 | 结果 |
| --- | --- | --- |
| 5 | 1.65ms | ok |
| 200 | 5.36ms | **抛错** |
| 1000 | **23.42ms** | **抛错** |

23ms 全是白烧。真实 ModernBERT-large tokenizer 会更慢（tiny 模型的词表小得多）。

### 附带发现：默认预算的候选数阈值比想象中高得多

**实测**（tiny tokenizer，描述型候选）：40 个可用、50 个超预算。短标签候选：40 可用、60 超预算。
真实阈值随 tokenizer 变化，但量级是 **`headMaxLength=192` 默认只装得下 ~40–50 个带描述的候选**。

而当前错误信息是：

> `候选超过 token 预算（headMaxLength=192，maxLength=512），请提高预算或使用 shortlist`

它告诉了用户"超了"，但**没告诉用户"你现在有 50 个，而预算只装得下约 45 个"**。用户只能盲目试参数。

### 建议

1. **前置廉价估算**：在 tokenize 之前用原始字符长度做一次下界估算（经验值 1 token ≈ 3–4 字符），明显超预算就立即抛错。省掉全部无谓 encode。

2. **错误信息带上实际容量**：

   ```js
   throw new RangeError(
     `候选超过 token 预算：${entries.length} 个候选，headMaxLength=${headMaxLength} 约可容纳 ${affordable} 个。`
     + `请减少候选数、提高 headMaxLength（上限 ${maxLength}），或使用 shortlist 预筛。`
   );
   ```

3. **给 `predictShortlist` 补一条边界提示**：`k=20` 也不保证放得下（若 Top-20 的描述很长仍会抛错）。建议在 `shortlistChoice` 成功后再做一次预算预检，给出更贴近业务的错误。

---

## 3. P0-3 · 全部错误是硬编码中文字符串，调用方无法程序化处理

### 问题

整个 `src/` 没有一个错误码。抛出的是裸 `TypeError` / `RangeError` / `Error` + 中文 message：

```js
throw new RangeError(`${label} 必须为正整数`);           // math.js:6
throw new TypeError(`未知问题类型：${type}`);              // questions.js:56
throw new Error('模型在途调用已结束...')
throw new Error(`模型 ${decision.model} 不可用：${status.error}；...`);   // mcp.js:133
```

调用方想区分"预算不够"和"模型缺失"，只能正则匹配中文字符串。这对以中文错误信息为**设计决策**的项目是可以理解的，但代价是：

- 正则匹配对文案改动极度敏感（改一个字就断）
- 跨语言调用方无法处理
- MCP 层只能把错误原样塞进 `isError: true` 的 text 字段，客户端 LLM 只能靠读中文理解

项目里其实已经有先例，只是没推广：

```js
process.emitWarning('checkpoint 温度超出 [0.5,5] ...', { code: 'LAYA_TEMPERATURE_CLAMPED' });
```

### 建议

给所有对外抛出的错误加 `code`，**保留中文 message 不变**（不破坏现有可读性）：

```js
export class LayaError extends Error {
  constructor(code, message, options) { super(message, options); this.code = code; }
}

// 用法
throw new LayaError('LAYA_TOKEN_BUDGET_EXCEEDED', `候选超过 token 预算（...）`);
```

建议的码表（先覆盖 MCP 客户端真正需要区分的）：

| code | 触发 |
| --- | --- |
| `LAYA_TOKEN_BUDGET_EXCEEDED` | 候选/正文超预算 |
| `LAYA_MODEL_UNAVAILABLE` | 模型目录不可用 |
| `LAYA_MODEL_IS_TINY` | tiny 模型且未开 `--allow-tiny` |
| `LAYA_MODEL_PROTOCOL_MISMATCH` | ONNX 图签名不符 |
| `LAYA_QUEUE_FULL` | MCP 推理队列已满 |
| `LAYA_REQUEST_CANCELLED` | AbortSignal 触发 |
| `LAYA_ARGUMENT_TOO_LARGE` | 超过 256 KiB |

然后在 `mcp.js` 的错误分支把 `code` 一并放进 `isError` 结果，MCP 客户端就能可靠分支，而不必解析中文。

**成本极低**（纯增量），**收益明确**。这是我认为最划算的 P0。

---

## 4. P1-1 · `index.d.ts` 的 `Router.systemOne` / `system_one` 类型窄于运行时

### 问题

**已逐行确认**：

```ts
// src/index.d.ts:86-88
predict<Q extends Questions>(state: State, questions: Q, options?: RoutingOptions):
  Promise<Prediction<Q> & { routing: RouteDecision }>;        // ← 有 routing
systemOne<Q extends Questions>(state: State, questions: Q, options?: RoutingOptions):
  Promise<Prediction<Q>>;                                     // ← 没有 routing
system_one<Q extends Questions>(...): Promise<Prediction<Q>>;  // ← 没有 routing
```

```js
// src/router.js:114-115 —— 两个别名都直接转发给 predict
systemOne(...args) { return this.predict(...args); }
system_one(...args) { return this.predict(...args); }
```

运行时**实际会返回** `routing`，但类型声明把它抹掉了。TypeScript 用户走 snake_case 别名（这是本项目**明确鼓励**的用法，见 `index.d.ts:124-132`）时，拿不到路由决策信息——而 `RouteDecision.reason` 恰恰是可解释性的核心。

同样的问题也存在于 `Agent`：`Agent.systemOne`/`system_one` 的类型是对的（`Agent.predict` 本来就不返回 routing），所以只有 `Router` 受影响。

### 建议

用一个类型别名消除重复，并让三者一致：

```ts
export type RoutedPrediction<Q extends Questions = Questions> =
  Prediction<Q> & { routing: RouteDecision };

export class Router {
  predict<Q extends Questions>(state: State, questions: Q, options?: RoutingOptions):
    Promise<RoutedPrediction<Q>>;
  systemOne<Q extends Questions>(state: State, questions: Q, options?: RoutingOptions):
    Promise<RoutedPrediction<Q>>;      // 与实现对齐
  system_one<Q extends Questions>(state: State, questions: Q, options?: RoutingOptions):
    Promise<RoutedPrediction<Q>>;      // 与实现对齐
}
```

`test/types.ts` 里加一条断言，防止再次漂移。

---

## 5. P1-2 · `Agent.predict` 硬编码 `model: 'laya-rl-agent'`，调用方拿不到真实 checkpoint

### 问题

**已确认**（`agent.js:72`）：

```js
return {
  model: 'laya-rl-agent',        // ← 写死的架构名
  answers: ...,
  usage: ...
};
```

后果是**三处都在绕过它**：

1. `Router.predict` 得额外拼一个 `routing: { ...decision }` 才能表达"用了哪个模型"
2. `mcp.js:144-145` 得翻 `agent.metadata.checkpoint` 自己拼 `diagnostics.checkpoint`
3. 裸用 `Agent` 的用户（CLI、examples、任何直接 `load()` 的代码）**完全无法知道这次推理是 english / multilingual / typed-decisions 哪一个**——而三者的概率语义和温度校准是不同的

在 SDK 里，`model` 字段返回架构名而不是实例标识，是个实打实的可用性缺口。

### 建议

让 `model` 返回真实 checkpoint，架构名另开字段：

```js
// agent.js
return {
  model: this.metadata.checkpoint,        // 'english' | 'multilingual' | 'typed-decisions'
  architecture: 'laya-rl-agent',
  answers: ...,
  usage: ...,
};
```

同步更新 `index.d.ts`（`model: string` → `model: ModelName`）和 README 示例。

**破坏性变更提示**：这会改变 `model` 字段的取值。如果项目还在 0.x，这是合适的时机；否则应新增字段而非改 `model`。

---

## 6. P1-3 · 取消能力没有贯通到 SDK 层，P2-1 · 缺少批量预测

这两条放一起，因为它们都指向同一个缺口：**`predict` 一次只处理一个 state，且不能中途取消**。

### 现状

- MCP 层有 `extra.signal`，但只在**步骤之间**检查。ONNX 一旦启动就必须跑完（`mcp.js:86` 明确记录了这个语义：「已启动的 ONNX 推理已完成但结果被丢弃」）。文档化了，是**有意识的权衡**，不是 bug。
- SDK 层 `PredictOptions` 完全没有 `signal`。
- 短路径 710ms 无所谓，但 **shortlist 路径可能要 45 秒**（见 P0-1）——这个量级上不能取消是真实问题。

### 建议

**P1-3**：`PredictOptions` 增加 `signal`，在 `Agent.#use` 包装层和 `embed` 的批次循环里检查：

```ts
export interface PredictOptions {
  maxLength?: number; headMaxLength?: number; truncateLeft?: boolean;
  signal?: AbortSignal;          // 新增
}
```

`#use` 当前是 `Promise.resolve().then(operation)`，改成先查 `signal.aborted`，并在 `embed` 的 `for` 循环里每批检查一次。已排队的批次直接跳过，成本几乎为零。

**P2-1**：`predictBatch(states[], questions, options)`，内部就是 `collateItems` 天然支持的批维度（`n` 已经是图的输入维度）。`Agent.prepare` 每题一个 item，batch 是**跨问题**的；跨 state 批处理需要把 `state` 也变成 batch 维度——这是一次真正的接口设计，建议先看 P0-1 落地后的实际用量再决定。

---

## 7. P2 · 可维护性

### 7-1 · `runtime.js` 一个函数干了六件事

`loadRuntime` 约 105 行，同时负责：路径解析 → hub/local 判定 → 配置校验 → 三种后端分派 → Tensor 构造 → shape/有限性校验。

具体可提炼的点：

- **`expected` shape 表在 MLX 和 ONNX 两个分支里各写了一遍**（`runtime.js:79` 和 `123`），内容完全相同
- **输出校验逻辑写了两遍**（MLX 版检查 `Array.isArray`，ONNX 版检查 `t.dims`，但「三个输出、shape 匹配、全部有限」这个不变量是重复表达的）
- 三种后端的返回值接口完全一致（`{tokenizer, config, metadata, source, run, dispose}`），却只有 ONNX 分支做了 session 签名校验

**建议**：抽出一个后端无关的校验函数，两条分支只负责"怎么拿到结果"：

```js
function validateOutputs(result, { n, k, nAct, hidden }, label) {
  const expected = { logits: [n, k], act_logits: [n, nAct], embeddings: [n, hidden] };
  for (const name of OUTPUTS) { /* shape + 有限性 */ }
  return result;
}
```

MLX 分支的 `Array.isArray` 差异可以在函数内部用一个访问器适配，或让 MLX 也返回 `{dims, data}` 形状统一。

### 7-2 · `formatAnswers` 重复归一化（**清理，非性能优化**）

`prepare()` 已经调过 `normalizeQuestions`，`formatAnswers` 里又：

- `renderOptions(q).length` —— 内部会 `normalizeQuestion` 一次（`questions.js:87`）并重渲染全部选项
- choice 分支再 `criteriaEntries(q.crit)` 一次

**实测**：40 个描述型候选约 0.9ms，200 个约 5.4ms。

**坦率地说：这不是性能问题。** 相对 710ms 的推理，占比 <1%。列在这里只是因为它就在热路径上且改起来是三行的事——`prepare()` 已经知道每个问题的 `k`，把它带进 `entries` 即可。**不要把它当性能优化来宣传。**

### 7-3 · `mcp.js` 177 行职责混杂

zod schema 定义、5 个工具注册、串行队列、模型可用性门禁、diagnostics 组装混在一个 `createMcpService` 里。建议至少把 zod schema（`question`/`questionsSchema`/`requestShape`/`predictShape`，约 25 行）抽到独立文件，让 `mcp.js` 专注于服务编排。

---

## 8. P3 · 工程基建

**现状核实**：无 `.github/workflows`、无 eslint/prettier、无 `.editorconfig`、无 `CHANGELOG.md`；`package.json` 只有 `name, version, description, type, engines, license, main, types, exports, bin, files, scripts, dependencies`——**缺 `repository` / `homepage` / `bugs`**。

对一个有 `bin` 入口、有 `exports` 映射、要通过 MCP 分发给多个客户端的包来说，CI 是底线：

**最小 CI 矩阵**（不需要 GPU）：

```yaml
jobs:
  test:
    strategy: { matrix: { node: [22, 24] } }
    steps:
      - run: npm ci
      - run: npm test
      - run: npm run test:types
  test-mcp:
    steps: [npm ci, run: npm run test:mcp]
```

配套建议：

1. **`npm run test:models` 纳入 CI**（或至少做成手动触发 + artifact 上传）。**实测**：`models/{english,multilingual,typed-decisions}/` 各有自己的 `golden.json`，但默认的 `npm test` 只跑 `artifacts/tiny-english-v2`——**multilingual tokenizer 和 typed-decisions 全默认路径上零覆盖**。这是数值契约最大的裸露面。
2. **加 `.editorconfig` + prettier**，1600 行的项目目前完全靠手写对齐。
3. **加 CHANGELOG**，v0.1.0 之后要开始记 breaking change（尤其 P1-2 会改 `model` 字段）。
4. **`package.json` 补 `repository` / `homepage` / `bugs` / `keywords`**。

---

## 9. 数值保真契约：建议加一道"token 序列"级回归

### 现状

`codewiki.md` 第 11 节总结的那几条保真约定——`jsonText` 保留 Python `json.dumps` 的 `", "` / `": "`、embedding 截断保留起止 token、mask token 转义、shortlist 排序的 `a.i - b.i` 平局兜底、温度优先级 `by_options > type > 1`——**每一件都能让 logits 静默漂移**，而目前它们只被 `golden.json`（7 个用例）和代码注释保护。

### 建议

增加一层**比 golden 更靠下的回归网**：对固定输入断言**精确的 token id 序列**，落成 `golden_tokens.json`。

```js
// test/sequence.test.js
test('候选序列 token id 与 Python 对照逐位一致', () => {
  const got = buildSequence(tok, STATE, QUESTION, OPTS).ids;
  assert.deepEqual(got, goldenTokens.sequence);   // 精确相等，不容差
});
```

理由：现在如果有人把 `jsonText` 的分隔符从 `", "` 改成 `","`（一次"无害的清理"），`golden.json` 的 logits 对照会**失败**——这是好的。但它失败在很后面、错误信息是"logits 不匹配"，排查成本高。有了 token 级断言，失败会直接指向 `buildSequence`，并明确指出"第 47 个 token 不对"。

**成本**：一份几百行的 JSON + 一个测试。**收益**：把最难排查的一类漂移变成一眼可辨。

---

## 10. 明确不建议做的事

以下都是我看代码时怀疑过、然后**实测证伪**的。列出来是为了避免后续重复讨论。

| 怀疑 | 实测 | 结论 |
| --- | --- | --- |
| `collateItems` 用普通数组、`runtime.js` 再 `.flat()` + `BigInt64Array.from` —— 三次拷贝 | `collateItems` **0.04ms** + 4 个 int64 张量构造 **2.08ms** = **2.1ms/次**，占 710ms 的 **0.3%** | **✗ 不值得**。除非换成直接构造 `BigInt64Array` 顺手为之，别单独立项 |
| 输出做 `t.data.every(Number.isFinite)` 全量有限性扫描（`embeddings[n,1024]`） | 32×1024 = 3.3 万次 `Number.isFinite`，< 0.1ms | **✗ 不值得**。而且这是**正确性保护**，应保留 |
| `lang.js` 的 `RANGES.find(...)` 对每个码点线性扫 22 个区段（O(22n)） | 4000 字符，最坏 ~26 万次比较，亚毫秒级；每次 `route` 只调一次 | **✗ 不值得**。真要优化也就一行 `Uint8Array` 查表 |
| **提高 `embed` 的 `batchSize` 以减少串行 forward 次数** | n=256 时 bs=16 是 4339ms、bs=256 是 4575ms；单条成本离散度仅 1.06–1.28x | **✗ 不值得**。初稿把这条列为 P0-1，**已被本轮实测证伪并改写**，见第 1 节 |
| 默认预加载三个模型 | 已在 README 和 `bin/laya-mcp.js --help` 里明确警告"单个 FP32 模型可占用数 GB 内存" | **✗ 已正确处理**，`maxLoaded=1` 是对的 |
| 短标签 choice 的候选顺序被 JS 对象键规则打乱 | **已处理**：`normalizeQuestion` 对数组形式的 criteria 保留调用者顺序（`questions.js:63`），且有专门测试 `数字形候选标签的 embedding 顺序不被对象键规则打乱` | **✗ 已有防护** |
| 六个业务/工具模块缺少测试 | **我最初的判断是错的**。`core.test.js` 11 个用例已覆盖 math / lang / email / presets / sequence / questions | **✗ 覆盖良好**。真正的缺口是 §8-1 的 CI 门禁，不是测试本身 |

---

## 附录 A · 实测复现方法

本文档的实测数据（候选预算阈值、`buildSequence` 耗时、Tensor 构造开销、shortlist forward 次数）可用如下方式复现，**不需要完整模型**：

```js
// .cache/probe.mjs（放在仓库内以便解析 node_modules，用完删除）
import { AutoTokenizer } from '@huggingface/transformers';
import { buildSequence } from './src/sequence.js';

const tok = await AutoTokenizer.from_pretrained('artifacts/tiny-english-v2',
  { local_files_only: true });
const criteria = (n) => Object.fromEntries(Array.from({ length: n }, (_, i) =>
  [`label_${i}`, `a description for candidate number ${i} in a support taxonomy`]));

for (const n of [5, 40, 50, 200, 1000]) {
  const t0 = performance.now();
  let r;
  try { buildSequence(tok, 'state text', { type: 'choice', instructions: 'Q', criteria: criteria(n) }); r = 'ok'; }
  catch (e) { r = e.message.slice(0, 30); }
  console.log(n, (performance.now() - t0).toFixed(2) + 'ms', r);
}
```

`shortlist` 的 forward 次数是纯算术：`ceil((n + 1) / batchSize)`。

**注意**：本文档用的是 `artifacts/tiny-english-v2` 的**随机小模型 tokenizer**，词表远小于真实的 `answerdotai/ModernBERT-large`。绝对耗时会偏低，**候选数阈值也会偏低**。相对关系（三次拷贝 vs 710ms、63 次 vs 4 次 forward）和结论方向不受影响。

---

## 附录 B · 建议的执行顺序

```
第 1 轮（低风险、高收益，不改公开契约）
  ├─ P0-3  错误码 LayaError + code 表          ← 最划算
  ├─ P0-2  buildSequence 前置预算预检 + 错误信息带实际容量
  ├─ P1-1  d.ts 的 Router.systemOne 类型对齐    ← 一行类型别名
  └─ P3    CI（node 22/24 矩阵）+ .editorconfig + package.json 元数据

第 2 轮（改公开契约，需要 CHANGELOG）
  ├─ P1-2  model 字段返回真实 checkpoint（v0.1.0 正是做 breaking change 的时机）
  ├─ P0-1  embedding batchSize 可配置 + MCP schema 暴露
  └─ P1-3  PredictOptions.signal

第 3 轮（结构性，需先观察第 2 轮用量）
  ├─ §9   golden_tokens.json token 级回归网
  ├─ 7-1  runtime.js 抽取共享校验
  ├─ 7-2/7-3  formatAnswers 去重、mcp.js schema 拆分
  └─ 6-P2-1  predictBatch（等 P0-1 落地后看真实用量再定）
```
