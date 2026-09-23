import test from 'node:test';
import assert from 'node:assert/strict';
import { access, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { execFile, spawn, spawnSync } from 'node:child_process';
import { promisify, stripVTControlCharacters } from 'node:util';
import { once } from 'node:events';
import { request } from 'node:http';
import path from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { LATEST_PROTOCOL_VERSION } from '@modelcontextprotocol/sdk/types.js';
import { startMcpHttpServer } from 'laya-node/mcp/http';

const root = fileURLToPath(new URL('../', import.meta.url));
const tiny = path.join(root, 'artifacts/tiny-english-v2');
const model = path.resolve(process.env.LAYA_TEST_MODEL ?? tiny);
const models = (directory) => ({ english: directory, multilingual: directory, 'typed-decisions': directory });
const questions = { refund: { type: 'noul', instructions: 'Refund?' } };
const headers = { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream', 'MCP-Protocol-Version': LATEST_PROTOCOL_VERSION };
const callBody = (state) => ({ jsonrpc: '2.0', id: 7, method: 'tools/call', params: { name: 'laya_predict', arguments: { state, questions, model: 'english' } } });
let tinyAvailable = true, available = true;
try { await access(path.join(tiny, 'config.json')); } catch { tinyAvailable = false; }
try { await access(path.join(model, 'golden.json')); } catch (error) {
  if (process.env.LAYA_TEST_MODEL) throw error;
  available = false;
}
async function start(t, options = {}) {
  const app = await startMcpHttpServer({ port: 0, models: models(path.join(root, 'artifacts/missing')), ...options });
  t.after(() => app.close());
  return app;
}
async function connect(t, url) {
  const client = new Client({ name: 'laya-http-test', version: '1.0.0' });
  const transport = new StreamableHTTPClientTransport(new URL(url));
  t.after(() => client.close());
  await client.connect(transport);
  return { client, transport, call: (name, args = {}) => client.callTool({ name, arguments: args }, undefined, { timeout: 180000 }) };
}
function data(value) {
  assert.notEqual(value.isError, true, JSON.stringify(value));
  assert.deepEqual(value.structuredContent, JSON.parse(value.content[0].text));
  return value.structuredContent;
}
function fakeAgent(predict, dispose = async () => {}) {
  return { cfg: { max_len: 64, head_max_len: 32 }, metadata: { checkpoint: 'english', tiny: true },
    temperature_raw: [1, 1, 1], temperature_by_options_raw: {}, predict, dispose };
}

test('HTTP 官方客户端握手、工具发现、重复请求及客户端独立关闭', async (t) => {
  const app = await start(t);
  const a = await connect(t, app.url), b = await connect(t, app.url);
  assert.equal(a.transport.sessionId, undefined);
  assert.match(a.client.getInstructions(), /本地 Laya/);
  assert.equal((await a.client.listTools()).tools.length, 5);
  assert.equal(data(await a.call('laya_status')).transport, 'streamable-http');
  await a.client.close();
  assert.equal(data(await b.call('laya_route', { state: '中文', lang: 'zh' })).routing.model, 'multilingual');
  assert.ok(data(await b.call('laya_presets', { name: 'triage' })).presets.triage);
  assert.equal((await b.call('laya_predict', { state: 'x', questions })).isError, true);
  assert.equal((await b.call('laya_predict', { state: 'x' })).isError, true);
  assert.equal(data(await b.call('laya_status')).pending, 0);
});

test('HTTP Host、Origin、请求大小与协议校验', async (t) => {
  const { url } = await start(t);
  assert.equal((await fetch(new URL('/health', url))).status, 200);
  assert.equal((await fetch(new URL('/unknown', url))).status, 404);
  for (const method of ['GET', 'DELETE', 'OPTIONS']) {
    const response = await fetch(url, { method });
    assert.equal(response.status, 405);
    assert.equal(response.headers.get('allow'), 'POST');
    assert.equal(response.headers.get('access-control-allow-origin'), null);
  }
  for (const extra of [{ Host: 'attacker.example' }, { Origin: 'https://attacker.example' }, { Origin: 'null' }]) {
    // fetch 可能重写 Host，使用原始 HTTP 请求确保测试头确实发出。
    const status = await new Promise((resolve, reject) => {
      const req = request(url, { method: 'POST', headers: { ...headers, ...extra } }, (res) => { res.resume(); resolve(res.statusCode); });
      req.on('error', reject); req.end('{}');
    });
    assert.equal(status, 403, JSON.stringify(extra));
  }
  assert.equal((await fetch(url, { method: 'POST', body: '{}' })).status, 415);
  assert.equal((await fetch(url, { method: 'POST', headers, body: 'invalid' })).status, 400);
  assert.equal((await fetch(url, { method: 'POST', headers, body: 'x'.repeat(1024 * 1024 + 1) })).status, 413);
  const stream = new ReadableStream({ start(controller) {
    controller.enqueue(new Uint8Array(600000)); controller.enqueue(new Uint8Array(600000)); controller.close();
  } });
  assert.equal((await fetch(url, { method: 'POST', headers, body: stream, duplex: 'half' })).status, 413);
  const badAccept = await fetch(url, { method: 'POST', headers: { ...headers, Accept: 'text/plain' }, body: '{}' });
  assert.equal(badAccept.status, 406);
  const badVersion = await fetch(url, { method: 'POST', headers: { ...headers, 'MCP-Protocol-Version': '1900-01-01' }, body: JSON.stringify(callBody('x')) });
  assert.equal(badVersion.status, 400);
  const { call } = await connect(t, url);
  data(await call('laya_status'));
  await assert.rejects(startMcpHttpServer({ host: '0.0.0.0' }), /回环/);
  await assert.rejects(startMcpHttpServer({ port: 65536 }), /port/);
});

test('HTTP 共享缓存、全局背压、相同 RPC id 不串结果且失败后可恢复', { skip: !tinyAvailable }, async (t) => {
  let release, entered, loads = 0, disposed = 0, active = 0, maxActive = 0;
  const gate = new Promise((resolve) => { release = resolve; });
  const started = new Promise((resolve) => { entered = resolve; });
  const agent = fakeAgent(async (state) => {
    active++; maxActive = Math.max(maxActive, active);
    try {
      if (state === 'first') { entered(); await gate; }
      if (state === 'fail') throw new Error('模拟推理失败');
      return { answers: { state } };
    } finally { active--; }
  }, async () => { disposed++; });
  const app = await start(t, { models: models(tiny), allowTiny: true, maxPending: 2, loader: async () => { loads++; return agent; } });
  const { call } = await connect(t, app.url);
  const post = async (state) => (await fetch(app.url, { method: 'POST', headers, body: JSON.stringify(callBody(state)) })).json();
  const first = post('first');
  await started;
  const second = post('second');
  for (let i = 0; i < 100; i++) {
    if (data(await call('laya_status')).pending === 2) break;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.equal(data(await call('laya_status')).pending, 2);
  assert.match((await post('overflow')).result.content[0].text, /队列已满/);
  release();
  assert.equal(data((await first).result).answers.state, 'first');
  assert.equal(data((await second).result).answers.state, 'second');
  assert.equal((await post('fail')).result.isError, true);
  assert.equal(data((await post('recovered')).result).answers.state, 'recovered');
  assert.equal(data(await call('laya_status')).models.english.loaded, true);
  assert.equal(loads, 1);
  assert.equal(maxActive, 1);
  assert.equal(disposed, 0);
  await app.close(); await app.close();
  assert.equal(disposed, 1);
});

test('HTTP 客户端断开后跳过排队推理，不关闭共享服务', { skip: !tinyAvailable, timeout: 10000 }, async (t) => {
  let release, entered, calls = 0;
  const gate = new Promise((resolve) => { release = resolve; });
  const started = new Promise((resolve) => { entered = resolve; });
  const agent = fakeAgent(async () => { calls++; entered(); await gate; return { answers: {} }; });
  const { url } = await start(t, { models: models(tiny), allowTiny: true, loader: async () => agent });
  const { call } = await connect(t, url);
  const first = call('laya_predict', { state: 'first', questions });
  await started;
  const controller = new AbortController();
  const second = fetch(url, { method: 'POST', headers, body: JSON.stringify(callBody('second')), signal: controller.signal });
  const rejection = assert.rejects(second, /abort/i);
  for (let i = 0; i < 100; i++) {
    if (data(await call('laya_status')).pending === 2) break;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.equal(data(await call('laya_status')).pending, 2);
  controller.abort(); await rejection;
  await new Promise((resolve) => setTimeout(resolve, 30));
  release(); data(await first);
  data(await call('laya_predict', { state: 'after cancellation', questions }));
  assert.equal(calls, 2);
});

test('HTTP 关闭等待在途推理，跳过排队任务并仅释放一次', { skip: !tinyAvailable, timeout: 10000 }, async (t) => {
  let release, entered, calls = 0, disposed = 0, active = false, closed = false;
  const gate = new Promise((resolve) => { release = resolve; });
  const started = new Promise((resolve) => { entered = resolve; });
  t.after(() => release());
  const agent = fakeAgent(async () => {
    calls++; active = true; entered(); await gate; active = false;
    return { answers: {} };
  }, async () => { assert.equal(active, false); disposed++; });
  const app = await start(t, { models: models(tiny), allowTiny: true, loader: async () => agent });
  const { call } = await connect(t, app.url);
  const first = call('laya_predict', { state: 'first', questions });
  await started;
  const second = call('laya_predict', { state: 'second', questions });
  const settled = Promise.allSettled([first, second]);
  for (let i = 0; i < 100; i++) {
    if (data(await call('laya_status')).pending === 2) break;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.equal(data(await call('laya_status')).pending, 2);
  const closing = app.close();
  assert.equal(app.close(), closing);
  void closing.then(() => { closed = true; });
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(closed, false);
  assert.equal(disposed, 0);
  release();
  await closing;
  await settled;
  assert.equal(calls, 1);
  assert.equal(disposed, 1);
});

test('HTTP 配置模板与共用 Skill，客户端示例保留两种传输', { timeout: 20000 }, async (t) => {
  const url = 'http://127.0.0.1:7777/mcp';
  const claude = JSON.parse(await readFile(path.join(root, 'examples/claude-mcp-http.json'), 'utf8'));
  assert.deepEqual(claude.mcpServers.laya, { type: 'http', url });
  const codex = await readFile(path.join(root, 'examples/codex-mcp-http.toml'), 'utf8');
  assert.match(codex, /\[mcp_servers\.laya\]/);
  assert.ok(codex.includes(`url = "${url}"`));
  assert.match(codex, /tool_timeout_sec = 180/);
  assert.doesNotMatch(codex, /^(command|args)\s*=/m);
  const skill = await readFile(path.join(root, 'skills/laya-decision/SKILL.md'), 'utf8');
  for (const text of ['stdio', 'Streamable HTTP', 'npm run mcp:http', url, '/health']) assert.ok(skill.includes(text));
  const example = path.join(root, 'examples/mcp-client.js');
  const invalid = spawnSync(process.execPath, [example, '--url', url, '--models-dir', '/unused'], { encoding: 'utf8' });
  assert.equal(invalid.status, 1); assert.match(invalid.stderr, /不能同时使用/);
  const run = promisify(execFile);
  const stdio = await run(process.execPath, [example, '--models-dir', path.join(root, 'artifacts/missing')], { timeout: 10000 });
  assert.match(stripVTControlCharacters(stdio.stdout), /transport: 'stdio'/);
  const app = await start(t);
  const http = await run(process.execPath, [example, '--url', app.url], { timeout: 10000 });
  assert.match(stripVTControlCharacters(http.stdout), /transport: 'streamable-http'/);
  assert.match(http.stdout, /laya_predict/);
  assert.equal((await fetch(new URL('/health', app.url))).status, 200);
});

test('HTTP 真实 ONNX golden 与 shortlist，另一个客户端复用模型', { skip: !available, timeout: 180000 }, async (t) => {
  const app = await start(t, { models: models(model), allowTiny: true });
  const a = await connect(t, app.url), b = await connect(t, app.url);
  const golden = JSON.parse(await readFile(path.join(model, 'golden.json'), 'utf8'));
  const entry = golden.cases.find((item) => item.name === 'mixed');
  const output = data(await a.call('laya_predict', { state: entry.state, questions: entry.questions, model: 'english' }));
  assert.equal(output.routing.model, 'english');
  for (const [id, expected] of Object.entries(entry.result.answers)) {
    assert.equal(output.answers[id].type, expected.type);
    for (const field of ['confidence', 'noul', 'score']) if (field in expected) assert.ok(Math.abs(output.answers[id][field] - expected[field]) < 0.001);
    if ('choice' in expected) assert.equal(output.answers[id].choice, expected.choice);
  }
  await a.client.close();
  assert.equal(data(await b.call('laya_status')).models.english.loaded, true);
  const shortlist = data(await b.call('laya_shortlist', { state: 'refund', model: 'english', k: 2,
    questions: { intent: { type: 'choice', instructions: 'Choose a team', criteria: ['billing', 'tech', 'sales'] } } }));
  assert.equal(shortlist.shortlist.intent.probability_scope, 'shortlisted');
  assert.ok(shortlist.shortlist.intent.labels.includes(shortlist.answers.intent.choice));
});

test('HTTP CLI 不依赖 stdin，端口冲突报错，SIGTERM 释放端口', { timeout: 20000 }, async (t) => {
  const bin = path.join(root, 'bin/laya-mcp.js');
  for (const args of [['--transport', 'invalid'], ['--port', '7777'], ['--transport', 'http', '--host', '0.0.0.0'], ['--transport', 'http', '--port', 'NaN']]) {
    const output = spawnSync(process.execPath, [bin, ...args], { encoding: 'utf8', timeout: 5000 });
    assert.equal(output.status, 1); assert.equal(output.stdout, '');
  }
  const child = spawn(process.execPath, [bin, '--transport', 'http', '--port', '0'], { stdio: ['pipe', 'pipe', 'pipe'] });
  const exited = once(child, 'exit');
  t.after(() => { if (child.exitCode === null) child.kill('SIGTERM'); });
  let stderr = '';
  const ready = new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', () => reject(new Error(`服务提前退出：${stderr}`)));
    child.stderr.on('data', (chunk) => { stderr += chunk; const url = stderr.match(/http:\/\/127\.0\.0\.1:\d+\/mcp/); if (url) resolve(url[0]); });
  });
  child.stdin.end();
  const url = await ready;
  assert.equal((await fetch(new URL('/health', url))).status, 200);
  const port = Number(new URL(url).port);
  const conflict = spawnSync(process.execPath, [bin, '--transport', 'http', '--port', String(port)], { encoding: 'utf8', timeout: 5000 });
  assert.equal(conflict.status, 1); assert.match(conflict.stderr, /EADDRINUSE/);
  child.kill('SIGTERM');
  assert.deepEqual(await exited, [0, null]);
  const replacement = await startMcpHttpServer({ port });
  await replacement.close();
});
