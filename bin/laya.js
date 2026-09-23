#!/usr/bin/env node
import { parseArgs } from 'node:util';
import { readFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { load, Router, DEFAULT_MODELS, triageQuestions, emailQuestions, guardQuestions, moderationQuestions, routerQuestions } from '../src/index.js';

const HELP = `laya-node：Node.js 类型化决策 SDK

用法：
  laya-node predict --model ./models/english --input examples/request.json
  laya-node predict --model ./models/multilingual --preset triage --input -
  laya-node route --input examples/request.json [--lang zh] [--model english]
  laya-node inspect --model ./models/english [--verify]

输入 JSON：{ "state": "文本或 JSON 状态", "questions": { ... } }
可用 preset：triage、email、guard、moderation、router。
predict 默认 CPU FP32；--offline 禁止远程加载，--revision 固定 Hub 版本。
--max-length / --head-max-length 调整预算，--truncate-left 保留正文尾部。

首次运行前导出完整模型（Python 仅在这一步使用）：
  python3 -m venv .venv
  .venv/bin/python -m pip install -r tools/requirements.txt
  .venv/bin/python tools/export_onnx.py --checkpoint english --output models/english
另两个 checkpoint 为 multilingual 和 typed-decisions，需分别导出。
只验证运行链路可使用 --tiny；随机小模型不具备语义决策能力。
  .venv/bin/python tools/export_onnx.py --tiny --output artifacts/tiny-english-v2
  npm test
  npm run test:types
  npm run test:models -- --output artifacts/model-verification.json
验收入口逐个检查三个完整模型；--model 可只检查一个，缺失模型会失败。
已有图可用 --verify-only 重跑 Python 对照，更新 golden 和 manifest。
测试默认使用 artifacts/tiny-english-v2；完整模型也可设置 LAYA_TEST_MODEL。

模型加载和 predict 均需 await；结束后调用 dispose()。
snake_case 为导出名称别名，JS 参数采用 options 对象，不是 Python 位置参数。
概率取决于 checkpoint；限幅过的温度不代表经过重新校准。
单个 FP32 模型可占用数 GB 内存；不要默认同时预加载三个模型。
`;
const PRESETS = new Map(Object.entries({ triage: triageQuestions, email: emailQuestions, guard: guardQuestions, moderation: moderationQuestions, router: routerQuestions }));

async function inputJSON(file) {
  if (file && file !== '-') return JSON.parse(await readFile(file, 'utf8'));
  if (process.stdin.isTTY) throw new Error('请使用 --input 文件或向 stdin 写入 JSON');
  const chunks = []; let size = 0;
  for await (const chunk of process.stdin) {
    size += chunk.length;
    if (size > 32 * 1024 * 1024) throw new Error('stdin JSON 超过 32 MiB');
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}
async function inspect(directory, verify) {
  const root = path.resolve(directory), config = JSON.parse(await readFile(path.join(root, 'config.json'), 'utf8'));
  if (config.laya?.format_version !== 1) throw new Error('不是 laya-node v1 制品');
  const manifest = JSON.parse(await readFile(path.join(root, 'manifest.json'), 'utf8'));
  if (manifest.format_version !== 1 || manifest.source_revision !== config.laya.revision) throw new Error('manifest 版本或来源 revision 不匹配');
  for (const required of ['config.json', 'rl_agent_config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model.onnx', 'golden.json']) {
    if (!manifest.files || !Object.hasOwn(manifest.files, required)) throw new Error(`manifest 缺少 ${required}`);
  }
  if (verify) for (const [name, expected] of Object.entries(manifest.files)) {
    const full = path.resolve(root, name);
    if (!full.startsWith(root + path.sep)) throw new Error('manifest 含越界路径');
    const hash = createHash('sha256'); let bytes = 0;
    for await (const chunk of createReadStream(full)) { hash.update(chunk); bytes += chunk.length; }
    if (hash.digest('hex') !== expected.sha256 || bytes !== expected.bytes) throw new Error(`制品校验失败：${name}`);
  }
  return { ...config.laya, manifest, checksums_verified: verify };
}
async function main() {
  const { values: v, positionals } = parseArgs({ allowPositionals: true, options: {
    help: { type: 'boolean', short: 'h' }, model: { type: 'string' }, input: { type: 'string', short: 'i' },
    preset: { type: 'string' }, lang: { type: 'string' }, task: { type: 'string' },
    offline: { type: 'boolean' }, revision: { type: 'string' }, verify: { type: 'boolean' },
    'max-length': { type: 'string' }, 'head-max-length': { type: 'string' }, 'truncate-left': { type: 'boolean' },
  } });
  if (v.help || !positionals.length) { process.stdout.write(HELP); return; }
  const [command] = positionals;
  if (positionals.length !== 1 || !['predict', 'route', 'inspect'].includes(command)) throw new Error('未知命令，请运行 --help');
  if (command === 'inspect') return inspect(v.model ?? DEFAULT_MODELS.english, v.verify === true);
  const request = await inputJSON(v.input);
  if (!request || !Object.hasOwn(request, 'state')) throw new Error('输入缺少 state');
  if (v.preset && !PRESETS.has(v.preset)) throw new Error('未知 preset');
  const questions = request.questions ?? (v.preset ? PRESETS.get(v.preset)() : {});
  if (command === 'route') return new Router().route(request.state, questions, { lang: v.lang, model: v.model, task: v.task });
  const agent = await load(v.model, { localFilesOnly: v.offline, revision: v.revision });
  try {
    return await agent.predict(request.state, questions, {
      maxLength: v['max-length'] == null ? undefined : Number(v['max-length']),
      headMaxLength: v['head-max-length'] == null ? undefined : Number(v['head-max-length']),
      truncateLeft: v['truncate-left'],
    });
  } finally { await agent.dispose(); }
}
try {
  const output = await main();
  if (output !== undefined) process.stdout.write(JSON.stringify(output, null, 2) + '\n');
} catch (error) {
  process.stderr.write(`laya-node: ${error.message}\n`);
  process.exitCode = 1;
}
