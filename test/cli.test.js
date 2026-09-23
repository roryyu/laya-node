import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { access, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const model = path.join(root, 'artifacts/tiny-english-v2');
let available = true;
try { await access(path.join(model, 'manifest.json')); } catch { available = false; }
function run(args, input = '') {
  const result = spawnSync(process.execPath, [path.join(root, 'bin/laya.js'), ...args], {
    input, encoding: 'utf8', cwd: root, timeout: 30000, maxBuffer: 4 * 1024 * 1024,
  });
  assert.equal(result.error, undefined);
  return result;
}
const request = { state: '中文退款请求', questions: { refund: { type: 'noul', instructions: 'Refund?' } } };
test('CLI 帮助和无模型路由', () => {
  assert.match(run(['--help']).stdout, /Python 仅在这一步/);
  const route = run(['route', '--input', '-'], JSON.stringify(request));
  assert.equal(route.status, 0, route.stderr);
  assert.equal(JSON.parse(route.stdout).model, 'multilingual');
  assert.equal(route.stderr, '');
  const explicit = run(['route', '--lang', 'zh', '--model', 'en', '-i', 'examples/request.json']);
  assert.equal(JSON.parse(explicit.stdout).model, 'english');
});
test('CLI 错误只输出 stderr 且状态非零', () => {
  for (const [args, input] of [
    [['unknown'], ''], [['route'], '{bad json'], [['route'], '{}'],
    [['route', '--model', 'unknown'], JSON.stringify(request)],
    [['predict', '--preset', 'unknown'], JSON.stringify(request)],
    [['inspect', '--model', './artifacts/nonexistent'], ''],
  ]) {
    const result = run(args, input);
    assert.equal(result.status, 1);
    assert.equal(result.stdout, '');
    assert.match(result.stderr, /laya-node:/);
  }
});
test('CLI predict 与 inspect 通过真实 ONNX', { skip: !available && '需要 tiny 制品' }, () => {
  const prediction = run(['predict', '--model', model, '--offline', '-i', '-'], JSON.stringify(request));
  assert.equal(prediction.status, 0, prediction.stderr);
  const output = JSON.parse(prediction.stdout);
  assert.equal(output.answers.refund.type, 'noul');
  assert.equal(output.usage.output_tokens, 0);
  const inspection = run(['inspect', '--model', model, '--verify']);
  assert.equal(inspection.status, 0, inspection.stderr);
  assert.equal(JSON.parse(inspection.stdout).checksums_verified, true);
  const invalid = run(['predict', '--model', model, '--max-length', '0'], JSON.stringify(request));
  assert.equal(invalid.status, 1);
  assert.equal(invalid.stdout, '');
});
test('显式错误模型路径不会被集成测试跳过', () => {
  const result = spawnSync(process.execPath, ['--test', 'test/integration.test.js'], {
    cwd: root, encoding: 'utf8', timeout: 30000,
    env: { ...Object.fromEntries(Object.entries(process.env).filter(([key]) => key !== 'NODE_TEST_CONTEXT')),
      LAYA_TEST_MODEL: path.join(root, 'artifacts/nonexistent') },
  });
  assert.equal(result.status, 1);
  assert.match(result.stdout + result.stderr, /显式指定的验收模型不可用/);
});
test('验收入口不忽略缺失模型', () => {
  const result = spawnSync(process.execPath, ['tools/verify_models.js', '--model', './artifacts/nonexistent'], {
    cwd: root, encoding: 'utf8', timeout: 30000,
  });
  assert.equal(result.status, 1);
  assert.equal(JSON.parse(result.stdout).results[0].status, 'failed');
});
test('推理与 MCP 直接依赖固定，推理不启动外部进程', async () => {
  const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'));
  assert.deepEqual(pkg.dependencies, { '@huggingface/transformers': '4.3.0', '@modelcontextprotocol/sdk': '1.30.0', zod: '4.6.5' });
  for (const file of ['agent.js', 'runtime.js', 'router.js']) {
    const source = await readFile(path.join(root, 'src', file), 'utf8');
    assert.doesNotMatch(source, /child_process|\bspawn\(|\bexec\(/);
  }
});
