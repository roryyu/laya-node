import test from 'node:test';
import assert from 'node:assert/strict';
import { access, readFile } from 'node:fs/promises';
import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { createMcpServer } from 'laya-node/mcp';

const root = fileURLToPath(new URL('../', import.meta.url));
const bin = path.join(root, 'bin/laya-mcp.js');
const tiny = path.join(root, 'artifacts/tiny-english-v2');
const model = path.resolve(process.env.LAYA_TEST_MODEL ?? tiny);
const names = ['english', 'multilingual', 'typed-decisions'];
const modelMap = (directory) => Object.fromEntries(names.map((name) => [name, directory]));
const questions = { intent: { type: 'choice', instructions: 'Choose a team', criteria: ['billing', 'tech', 'sales'] },
  refund: { type: 'noul', instructions: 'Refund?' } };
let available = true;
try { await access(path.join(model, 'golden.json')); } catch (error) {
  if (process.env.LAYA_TEST_MODEL) throw error;
  available = false;
}
let tinyAvailable = true;
try { await access(path.join(tiny, 'config.json')); } catch { tinyAvailable = false; }

async function connect(t, options = {}) {
  const app = createMcpServer(options);
  const [a, b] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: 'laya-test', version: '1.0.0' });
  await app.server.connect(b);
  await client.connect(a);
  t.after(async () => { await client.close(); await app.close(); });
  return { app, client, call: (name, args = {}) => client.callTool({ name, arguments: args }) };
}
function data(value) {
  assert.notEqual(value.isError, true, JSON.stringify(value));
  assert.deepEqual(JSON.parse(value.content[0].text), value.structuredContent);
  return value.structuredContent;
}

test('MCP 初始化、工具发现、schema 与只读无模型调用', async (t) => {
  const { client, call } = await connect(t, { models: modelMap(path.join(root, 'artifacts/missing')) });
  assert.match(client.getInstructions(), /confidence/);
  const { tools } = await client.listTools();
  assert.deepEqual(tools.map((tool) => tool.name).sort(), ['laya_status', 'laya_presets', 'laya_route', 'laya_predict', 'laya_shortlist'].sort());
  for (const tool of tools) {
    assert.equal(tool.inputSchema.type, 'object');
    assert.equal(tool.inputSchema.additionalProperties, false);
    assert.equal(tool.annotations.readOnlyHint, true);
    assert.equal(tool.annotations.openWorldHint, false);
  }
  const status = data(await call('laya_status'));
  assert.equal(status.local_files_only, true);
  assert.ok(Object.values(status.models).every((item) => !item.loaded && !item.available));
  assert.equal(data(await call('laya_route', { state: '中文退款', lang: 'zh' })).routing.model, 'multilingual');
  assert.equal(data(await call('laya_route', { state: '中文', lang: 'zh', model: 'english' })).routing.model, 'english');
  const presets = data(await call('laya_presets')).presets;
  assert.equal(Object.keys(presets).length, 5);
  assert.deepEqual(Object.keys(data(await call('laya_presets', { name: 'guard' })).presets), ['guard']);
  const missing = await call('laya_predict', { state: 'hello', questions });
  assert.equal(missing.isError, true);
  assert.match(missing.content[0].text, /不可用/);
  assert.equal(data(await call('laya_status')).pending, 0);
});

test('MCP 非法输入在加载前拒绝，错误后连接继续可用', async (t) => {
  let loads = 0;
  const { call } = await connect(t, { loader: () => { loads++; throw new Error('不应加载'); } });
  const bad = [
    {}, { state: null, questions }, { state: true, questions }, { state: 'x' },
    { state: 'x', questions: {} }, { state: 'x', questions, preset: 'triage' },
    { state: 'x', preset: 'unknown' }, { state: 'x', questions, model: '/tmp/model' },
    { state: 'x', questions, maxLength: 0 }, { state: 'x', questions, extra: true },
    { state: 'x', questions: { q: { type: 'choice', instructions: 'x', criteria: ['a', 'a'] } } },
    { state: 'x', questions: { q: { type: 'score', instructions: 'x', criteria: [] } } },
    { state: 'x', questions: { q: { type: 'noul', instructions: 'x', criteria: { maybe: true } } } },
    { state: 'x', questions: Object.fromEntries(Array.from({ length: 33 }, (_, i) => [String(i), questions.refund])) },
    { state: 'x'.repeat(256 * 1024), questions },
  ];
  for (const args of bad) assert.equal((await call('laya_predict', args)).isError, true);
  assert.equal((await call('laya_shortlist', { state: 'x', questions, k: 0 })).isError, true);
  assert.equal(loads, 0);
  data(await call('laya_status'));
  assert.throws(() => createMcpServer({ models: { english: 'https://example.com/model' } }), /本地/);
  assert.throws(() => createMcpServer({ models: { unknown: '/tmp' } }), /未知模型/);
  assert.throws(() => createMcpServer({ maxPending: 0 }), /maxPending/);
});

test('tiny 默认拒绝用于业务判断', { skip: !tinyAvailable }, async (t) => {
  const { call } = await connect(t, { models: modelMap(tiny) });
  const prediction = await call('laya_predict', { state: 'hello', questions });
  assert.equal(prediction.isError, true);
  assert.match(prediction.content[0].text, /tiny/);
  assert.equal(data(await call('laya_status')).models.english.loaded, false);
});

test('串行队列、背压、失败恢复、缓存复用与关闭释放', { skip: !tinyAvailable }, async (t) => {
  let release, started, loads = 0, disposed = 0, calls = 0;
  const gate = new Promise((resolve) => { release = resolve; });
  const entered = new Promise((resolve) => { started = resolve; });
  const agent = {
    cfg: { max_len: 64, head_max_len: 32 }, metadata: { checkpoint: 'english', tiny: true },
    temperature_raw: [1, 6, 1], temperature_by_options_raw: {},
    async predict() { calls++; if (calls === 1) { started(); await gate; throw new Error('测试推理失败'); }
      return { answers: {}, usage: { input_tokens: 1, output_tokens: 0 } }; },
    async dispose() { disposed++; },
  };
  const { app, call } = await connect(t, { models: modelMap(tiny), allowTiny: true, maxPending: 1,
    loader: async (_, options) => { loads++; assert.equal(options.localFilesOnly, true); return agent; } });
  const first = call('laya_predict', { state: 'hello', questions });
  await entered;
  assert.equal(data(await call('laya_status')).pending, 1);
  assert.match((await call('laya_predict', { state: 'hello', questions })).content[0].text, /队列已满/);
  release();
  assert.equal((await first).isError, true);
  const second = data(await call('laya_predict', { state: 'hello', questions }));
  assert.equal(second.diagnostics.temperature_clamped, true);
  assert.equal(loads, 1);
  await app.close();
  await app.close();
  assert.equal(disposed, 1);
});

test('取消排队请求不执行推理，关闭等待在途推理结束', { skip: !tinyAvailable }, async (t) => {
  let release, started, calls = 0, disposed = false;
  const gate = new Promise((resolve) => { release = resolve; });
  const entered = new Promise((resolve) => { started = resolve; });
  const agent = {
    cfg: { max_len: 64, head_max_len: 32 }, metadata: { checkpoint: 'english', tiny: true },
    temperature_raw: [1, 1, 1], temperature_by_options_raw: {},
    async predict() { calls++; started(); await gate; return { answers: {} }; },
    async dispose() { disposed = true; },
  };
  const { client, call, app } = await connect(t, { models: modelMap(tiny), allowTiny: true, loader: async () => agent });
  const first = call('laya_predict', { state: 'hello', questions });
  await entered;
  const controller = new AbortController();
  const second = client.callTool({ name: 'laya_predict', arguments: { state: 'hello', questions } }, undefined, { signal: controller.signal });
  const rejected = assert.rejects(second, /cancelled|abort/i);
  assert.equal(data(await call('laya_status')).pending, 2);
  controller.abort(new Error('cancelled'));
  await rejected;
  const closing = app.close();
  assert.equal(disposed, false);
  release();
  data(await first);
  await closing;
  assert.equal(calls, 1);
  assert.equal(disposed, true);
});

test('官方 MCP Client 通过 stdio 调用真实 ONNX，校验三类答案及 shortlist', { skip: !available, timeout: 180000 }, async (t) => {
  const transport = new StdioClientTransport({ command: process.execPath,
    args: [bin, ...names.flatMap((name) => [`--${name}`, model]), '--allow-tiny'], cwd: path.dirname(root), stderr: 'pipe' });
  let stderr = '';
  transport.stderr.on('data', (chunk) => { stderr += chunk; });
  const client = new Client({ name: 'laya-stdio-test', version: '1.0.0' });
  const errors = [];
  client.onerror = (error) => errors.push(error);
  t.after(() => client.close());
  await client.connect(transport);
  await client.listTools();
  const call = (name, args) => client.callTool({ name, arguments: args }, undefined, { timeout: 180000 });
  assert.equal(data(await call('laya_status', {})).models.english.loaded, false);
  const golden = JSON.parse(await readFile(path.join(model, 'golden.json'), 'utf8'));
  const entry = golden.cases.find((item) => item.name === 'mixed');
  const prediction = data(await call('laya_predict', { state: entry.state, questions: entry.questions, model: 'english' }));
  assert.equal(prediction.routing.model, 'english');
  for (const [id, expected] of Object.entries(entry.result.answers)) {
    const actual = prediction.answers[id];
    assert.equal(actual.type, expected.type);
    for (const field of ['confidence', 'noul', 'score']) {
      if (field in expected) assert.ok(Math.abs(actual[field] - expected[field]) < 0.001);
    }
    if ('choice' in expected) assert.equal(actual.choice, expected.choice);
  }
  assert.equal(prediction.usage.output_tokens, 0);
  const preset = data(await call('laya_predict', { state: { message: 'Please refund my payment.' }, preset: 'triage', model: 'english' }));
  assert.ok(Object.keys(preset.answers).length > 1);
  const selected = data(await call('laya_shortlist', { state: 'refund', questions, model: 'english', k: 2 }));
  assert.equal(selected.shortlist.intent.probability_scope, 'shortlisted');
  assert.equal(selected.shortlist.intent.labels.length, 2);
  assert.ok(selected.shortlist.intent.labels.includes(selected.answers.intent.choice));
  const passthrough = data(await call('laya_shortlist', { state: 'refund', questions, model: 'english', k: 3 }));
  assert.equal(passthrough.shortlist.intent.probability_scope, 'all');
  const chinese = data(await call('laya_predict', { state: '退款', questions: { refund: questions.refund }, lang: 'zh' }));
  assert.equal(chinese.routing.model, 'multilingual');
  const status = data(await call('laya_status', {}));
  assert.equal(status.models.multilingual.loaded, true);
  assert.equal(status.models.english.loaded, false);
  assert.equal(status.pending, 0);
  assert.deepEqual(errors, [], stderr);
});

test('stdio 环境变量、参数优先级及 SIGTERM 退出', { timeout: 10000 }, async (t) => {
  const transport = new StdioClientTransport({ command: process.execPath,
    args: [bin, '--models-dir', '/configured/models', '--english', '/configured/english'],
    env: { LAYA_MODELS_DIR: '/env/models', LAYA_MODEL_ENGLISH: '/env/english', LAYA_MODEL_MULTILINGUAL: '/env/multilingual' },
    stderr: 'pipe' });
  const client = new Client({ name: 'laya-config-test', version: '1.0.0' });
  t.after(() => client.close());
  await client.connect(transport);
  const status = data(await client.callTool({ name: 'laya_status', arguments: {} }));
  assert.equal(status.models.english.path, '/configured/english');
  assert.equal(status.models.multilingual.path, '/env/multilingual');
  assert.equal(status.models['typed-decisions'].path, '/configured/models/typed-decisions');
  const closed = new Promise((resolve) => { client.onclose = resolve; });
  process.kill(transport.pid, 'SIGTERM');
  await closed;
});

test('Skill 与接入模板可随包复用', async () => {
  const skill = await readFile(path.join(root, 'skills/laya-decision/SKILL.md'), 'utf8');
  assert.match(skill, /^---\nname: laya-decision\ndescription: .+\n---/);
  assert.ok(skill.split('\n').length < 500);
  for (const name of ['laya_status', 'laya_presets', 'laya_route', 'laya_predict', 'laya_shortlist']) assert.ok(skill.includes(name));
  const claude = JSON.parse(await readFile(path.join(root, 'examples/claude-mcp.json'), 'utf8'));
  assert.equal(claude.mcpServers.laya.type, 'stdio');
  assert.ok(claude.mcpServers.laya.args[0].endsWith('/bin/laya-mcp.js'));
  const codex = await readFile(path.join(root, 'examples/codex-mcp.toml'), 'utf8');
  assert.match(codex, /\[mcp_servers\.laya\]/);
  assert.match(codex, /tool_timeout_sec = 180/);
});

test('stdio 帮助、非法参数与 stdin EOF 正常退出', { timeout: 10000 }, async () => {
  const help = spawnSync(process.execPath, [bin, '--help'], { encoding: 'utf8' });
  assert.equal(help.status, 0);
  assert.match(help.stdout, /stdio MCP/);
  for (const args of [['--unknown'], ['--max-loaded', '0'], ['--max-pending', '999'], ['--models-dir', 'https://example.com']]) {
    const output = spawnSync(process.execPath, [bin, ...args], { encoding: 'utf8', timeout: 5000 });
    assert.equal(output.status, 1);
    assert.equal(output.stdout, '');
    assert.match(output.stderr, /laya-node-mcp/);
  }
  const child = spawn(process.execPath, [bin], { stdio: ['pipe', 'pipe', 'pipe'] });
  const exit = once(child, 'exit');
  child.stdin.end();
  assert.deepEqual(await exit, [0, null]);
});
