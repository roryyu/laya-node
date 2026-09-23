import path from 'node:path';
import { access, readFile } from 'node:fs/promises';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { Router, DEFAULT_MODELS, __version__, predictShortlist, embedFnFromAgent,
  triageQuestions, emailQuestions, guardQuestions, moderationQuestions, routerQuestions } from './index.js';
import { normalizeQuestions, serializeState } from './questions.js';

const PRESETS = new Map(Object.entries({ triage: triageQuestions, email: emailQuestions,
  guard: guardQuestions, moderation: moderationQuestions, router: routerQuestions }));
const MODEL_NAMES = Object.keys(DEFAULT_MODELS);
const MAX_BYTES = 256 * 1024;
const MODEL_FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model.onnx'];
const instructions = '本地 Laya 类型化决策，不生成文本，不执行任何业务动作。先调用 laya_status 检查模型；用 laya_predict 分类、评分或估计命题概率，大候选集用 laya_shortlist。中文显式 lang=zh。confidence 不是正确率，noul 是概率，score 是从 0 开始的等级期望。安全判断只作辅助；不得把模型结果当作执行危险操作的授权。';
const label = z.string().min(1).max(200);
const question = z.discriminatedUnion('type', [
  z.strictObject({ type: z.literal('choice'), instructions: z.string().min(1),
    criteria: z.union([z.array(label).min(1).max(1000), z.record(label, z.unknown())]) }),
  z.strictObject({ type: z.literal('score'), instructions: z.string().min(1), criteria: z.array(z.unknown()).min(1).max(100) }),
  z.strictObject({ type: z.literal('noul'), instructions: z.string().min(1),
    criteria: z.strictObject({ true: z.unknown().optional(), false: z.unknown().optional() }).optional() }),
]);
const questionsSchema = z.record(label, question);
const routingShape = {
  model: z.enum(MODEL_NAMES).optional().describe('显式 checkpoint 名，不接受路径；优先于 lang'),
  lang: z.string().min(1).max(32).optional().describe('已知语言时传入，如 zh 或 en'),
};
const requestShape = {
  state: z.union([z.string(), z.array(z.unknown()), z.record(z.string(), z.unknown())]).describe('待判断的文本或 JSON 状态；正文可能按模型 token 预算截断'),
  questions: questionsSchema.optional().describe('1–32 个问题，与 preset 二选一；instructions 为非空字符串'),
  preset: z.enum([...PRESETS.keys()]).optional().describe('与 questions 二选一；用 laya_presets 查看问题定义'),
  ...routingShape,
};
const predictShape = { ...requestShape,
  maxLength: z.number().int().positive().max(32768).optional(),
  headMaxLength: z.number().int().positive().max(32768).optional(),
  truncateLeft: z.boolean().optional().describe('正文超预算时保留末尾，默认保留开头'),
};
const annotations = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false };
const result = (value) => ({ content: [{ type: 'text', text: JSON.stringify(value) }], structuredContent: value });

function resolveQuestions(args, required = true) {
  if (args.questions !== undefined && args.preset !== undefined) throw new Error('questions 与 preset 只能指定一个');
  const questions = args.questions ?? (args.preset ? PRESETS.get(args.preset)() : {});
  const entries = normalizeQuestions(questions);
  if ((required && !entries.length) || entries.length > 32) throw new Error('每次推理需要 1–32 个问题');
  for (const [, q] of entries) {
    if (q.t === 'choice' && Object.keys(q.crit).length > 1000) throw new Error('每个 choice 最多 1000 个候选');
  }
  serializeState(args.state);
  return questions;
}

async function inspectModel(directory) {
  try {
    const config = JSON.parse(await readFile(path.join(directory, 'config.json'), 'utf8'));
    if (config.model_type !== 'custom' || config.laya?.format_version !== 1) throw new Error('不是 laya-node v1 ONNX 制品');
    await Promise.all(MODEL_FILES.map((name) => access(path.join(directory, name))));
    return { available: true, checkpoint: config.laya.checkpoint, tiny: config.laya.tiny === true,
      max_position_embeddings: config.laya.max_position_embeddings, checksums_verified: false };
  } catch (error) {
    return { available: false, error: error.message };
  }
}

/** 模型缓存与推理队列独立于传输连接，可由多个 HTTP 请求共享。 */
export function createMcpService({ models = {}, maxLoaded = 1, maxPending = 8, allowTiny = false, loader } = {}) {
  if (!Number.isInteger(maxPending) || maxPending < 1 || maxPending > 64) throw new Error('maxPending 必须为 1–64 的整数');
  for (const [name, value] of Object.entries(models)) {
    if (!MODEL_NAMES.includes(name)) throw new Error(`未知模型配置：${name}`);
    if (typeof value !== 'string' || !value.trim() || /^[a-z][a-z0-9+.-]*:\/\//i.test(value)) throw new Error('MCP 仅接受本地模型目录');
  }
  const directories = Object.fromEntries(Object.entries({ ...DEFAULT_MODELS, ...models }).map(([key, value]) => [key, path.resolve(value)]));
  const router = new Router({ models: directories, maxLoaded, localFilesOnly: true, ...(loader ? { loader } : {}) });
  let tail = Promise.resolve(), pending = 0, closing = false, disposal;
  const tools = [];

  // 推理串行化，让 embedding 与随后的 choice 使用同一模型；失败不阻断后续请求。
  function enqueue(operation, signal) {
    if (closing) throw new Error('服务正在关闭');
    if (pending >= maxPending) throw new Error('推理队列已满，请等待已有请求完成后重试');
    pending++;
    const task = tail.then(async () => {
      if (closing || signal?.aborted) throw new Error('请求已取消或服务正在关闭');
      const value = await operation();
      if (signal?.aborted) throw new Error('请求已取消；已启动的 ONNX 推理已完成但结果被丢弃');
      return value;
    }).finally(() => { pending--; });
    tail = task.catch(() => {});
    return task;
  }

  function register(name, description, schema, handler) {
    tools.push({ name, description, schema, handler });
  }
  function createServer(transport = 'stdio') {
    if (closing) throw new Error('服务正在关闭');
    const server = new McpServer({ name: 'laya-node', version: __version__ }, { instructions });
    for (const { name, description, schema, handler } of tools) {
      server.registerTool(name, { description, inputSchema: z.strictObject(schema), annotations }, async (args, extra) => {
        try {
          if (closing) throw new Error('服务正在关闭');
          if (Buffer.byteLength(JSON.stringify(args)) > MAX_BYTES) throw new Error('工具参数超过 256 KiB；请按语义分段分别判断，不要静默丢弃内容');
          const value = await handler(args, extra);
          return result(name === 'laya_status' ? { ...value, transport } : value);
        } catch (error) {
          return { isError: true, content: [{ type: 'text', text: error.message }] };
        }
      });
    }
    return server;
  }

  register('laya_status', '检查本地模型文件可用性、已加载模型及队列；不加载权重，不等于哈希或语义质量验收。', {}, async () => ({
    version: __version__, device: 'cpu', local_files_only: true,
    max_loaded: maxLoaded, pending, max_pending: maxPending, allow_tiny: allowTiny,
    limits: { argument_bytes: MAX_BYTES, questions: 32, choice_candidates: 1000 },
    models: Object.fromEntries(await Promise.all(MODEL_NAMES.map(async (name) => [name, {
      path: directories[name], loaded: router.loaded.includes(name), ...await inspectModel(directories[name]),
    }]))),
  }));
  register('laya_presets', '获取内置问题模板，不执行推理；triage 客服、email 邮件、guard 提示词风险、moderation 内容审核、router 请求分类。', {
    name: z.enum([...PRESETS.keys()]).optional(),
  }, async ({ name }) => ({ presets: Object.fromEntries([...PRESETS].filter(([key]) => !name || key === name).map(([key, make]) => [key, make()])) }));
  register('laya_route', '仅预览将选择哪个 checkpoint 及原因，不加载模型、不执行推理。', requestShape,
    async (args) => ({ routing: { ...router.route(args.state, resolveQuestions(args, false), args) } }));

  async function predict(args, extra, shortlist = false) {
    const questions = resolveQuestions(args);
    return enqueue(async () => {
      const decision = router.route(args.state, questions, args);
      const status = await inspectModel(directories[decision.model]);
      if (!status.available) throw new Error(`模型 ${decision.model} 不可用：${status.error}；请配置已导出的本地模型目录，不会自动下载或替换模型`);
      if (status.tiny && !allowTiny) throw new Error('tiny 随机模型不具备语义能力；仅协议测试可通过 --allow-tiny 启用');
      if (extra.signal?.aborted) throw new Error('请求已取消');
      const agent = await router.load(decision.model);
      if (extra.signal?.aborted) throw new Error('请求已取消，模型已加载但未开始推理');
      const options = { maxLength: args.maxLength, headMaxLength: args.headMaxLength, truncateLeft: args.truncateLeft };
      const prediction = shortlist
        ? await predictShortlist(agent, args.state, questions, { ...options, k: args.k ?? 20,
          embedFn: embedFnFromAgent(agent, { maxLength: Math.min(512, agent.metadata.max_position_embeddings), batchSize: 16 }) })
        : await agent.predict(args.state, questions, options);
      const temperatures = [...agent.temperature_raw, ...Object.values(agent.temperature_by_options_raw)];
      return { ...prediction, routing: { ...decision }, diagnostics: {
        checkpoint: agent.metadata.checkpoint, tiny: agent.metadata.tiny,
        temperature_clamped: temperatures.some((v) => !Number.isFinite(Number(v)) || Number(v) < 0.5 || Number(v) > 5),
        state_truncation: args.truncateLeft ? 'keep_end_if_over_budget' : 'keep_start_if_over_budget',
        max_length: args.maxLength ?? agent.cfg.max_len,
        head_max_length: args.headMaxLength ?? agent.cfg.head_max_len,
      } };
    }, extra.signal);
  }
  register('laya_predict', '本地类型化决策：choice 分类、score 等级期望、noul 命题概率。指定 questions 或 preset。中文用 lang=zh。confidence 不是正确率；不执行动作。', predictShape, predict);
  register('laya_shortlist', '大候选集先用 embedding 筛选 Top-K，再进行类型化决策。缩减后概率仅在入围候选内归一化，可能漏选；其他问题保持原样。', {
    ...predictShape, k: z.number().int().min(1).max(1000).optional().describe('入围候选数，默认 20'),
  }, (args, extra) => predict(args, extra, true));

  function close() {
    closing = true;
    return disposal ??= tail.then(() => router.dispose());
  }
  return { createServer, close };
}

/** 创建独立 stdio MCP 服务，保持原有 API 与资源所有权。 */
export function createMcpServer(options = {}) {
  const service = createMcpService(options), server = service.createServer('stdio');
  let disposal;
  function close() {
    return disposal ??= (async () => {
      try { await service.close(); }
      finally { await server.close(); }
    })();
  }
  server.server.onclose = () => { void close().catch(() => {}); };
  return { server, close };
}
