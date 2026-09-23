#!/usr/bin/env node
// 显式验收完整权重；每个模型在独立进程中运行，避免累计会话内存影响测量。
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { load, DEFAULT_MODELS } from '../src/index.js';

const ATOL = 1e-4, RTOL = 1e-3;
const script = fileURLToPath(import.meta.url);
function close(actual, expected, location = '', stats = { max_absolute_error: 0 }) {
  if (typeof expected === 'number') {
    const delta = Math.abs(actual - expected);
    assert.ok(Number.isFinite(actual) && delta <= ATOL + RTOL * Math.abs(expected), `${location}: ${actual} != ${expected}`);
    stats.max_absolute_error = Math.max(stats.max_absolute_error, delta);
  } else if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual), location);
    assert.equal(actual.length, expected.length, location);
    expected.forEach((v, i) => close(actual[i], v, `${location}[${i}]`, stats));
  } else if (expected && typeof expected === 'object') {
    assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), location);
    for (const k of Object.keys(expected)) close(actual[k], expected[k], `${location}.${k}`, stats);
  } else assert.equal(actual, expected, location);
  return stats;
}
async function checksums(directory, manifest) {
  assert.equal(manifest.format_version, 1);
  for (const required of ['config.json', 'rl_agent_config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model.onnx', 'golden.json']) {
    assert.ok(Object.hasOwn(manifest.files, required), `manifest 缺少 ${required}`);
  }
  for (const [name, expected] of Object.entries(manifest.files)) {
    const full = path.resolve(directory, name);
    assert.ok(full.startsWith(directory + path.sep), 'manifest 含越界路径');
    const hash = createHash('sha256'); let bytes = 0;
    for await (const chunk of createReadStream(full)) { hash.update(chunk); bytes += chunk.length; }
    assert.equal(hash.digest('hex'), expected.sha256, name);
    assert.equal(bytes, expected.bytes, name);
  }
}
async function verify(directory, allowTiny, iterations) {
  directory = path.resolve(directory);
  const manifest = JSON.parse(await readFile(path.join(directory, 'manifest.json'), 'utf8'));
  await checksums(directory, manifest);
  const golden = JSON.parse(await readFile(path.join(directory, 'golden.json'), 'utf8'));
  assert.ok(golden.cases.length >= 6, 'golden 缺少动态图场景');
  const rssBefore = process.memoryUsage().rss, start = performance.now();
  const agent = await load(directory, { localFilesOnly: true });
  const cold = performance.now() - start;
  try {
    assert.ok(allowTiny || !agent.metadata.tiny, '随机小模型不能作为完整权重验收；仅技术回归时传 --allow-tiny');
    assert.equal(agent.metadata.revision, manifest.source_revision);
    const rawStats = { max_absolute_error: 0 }, resultStats = { max_absolute_error: 0 };
    for (const entry of golden.cases) {
      const raw = await agent.predictRaw(entry.state, entry.questions, entry.options);
      assert.deepEqual(raw.batch, entry.inputs, `${entry.name}: token ids、markers、mask 不一致`);
      close(raw.output, entry.output, entry.name, rawStats);
      if (entry.result) close(await agent.predict(entry.state, entry.questions), entry.result, `${entry.name}.result`, resultStats);
    }
    close(await agent.embed(golden.embedding.texts, { maxLength: 64, batchSize: 3 }), golden.embedding.values, 'embedding', rawStats);
    const request = golden.cases[0];
    await agent.predict(request.state, request.questions);
    const samples = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await agent.predict(request.state, request.questions);
      samples.push(performance.now() - start);
    }
    samples.sort((a, b) => a - b);
    return {
      model: directory, checkpoint: agent.metadata.checkpoint, tiny: agent.metadata.tiny,
      revision: agent.metadata.revision, status: 'passed', checksums_verified: true,
      cases: golden.cases.length, embedding_texts: golden.embedding.texts.length,
      token_ids_and_markers_exact: true, atol: ATOL, rtol: RTOL,
      raw_max_absolute_error: rawStats.max_absolute_error, answer_max_absolute_error: resultStats.max_absolute_error,
      cold_load_ms: cold, warm_iterations: iterations,
      warm_mean_ms: samples.reduce((sum, v) => sum + v, 0) / samples.length,
      warm_p50_ms: samples[Math.floor(samples.length / 2)], warm_p95_ms: samples[Math.ceil(samples.length * 0.95) - 1],
      benchmark_question_count: Object.keys(request.questions).length,
      rss_before_bytes: rssBefore, rss_loaded_bytes: process.memoryUsage().rss,
      peak_rss_bytes: process.resourceUsage().maxRSS * 1024,
      environment: { node: process.version, platform: process.platform, arch: process.arch, device: 'cpu', dtype: 'fp32' },
    };
  } finally { await agent.dispose(); }
}
async function main() {
  const { values } = parseArgs({ options: {
    model: { type: 'string', multiple: true }, output: { type: 'string' },
    iterations: { type: 'string', default: '5' }, 'allow-tiny': { type: 'boolean' },
    worker: { type: 'boolean' }, help: { type: 'boolean', short: 'h' },
  } });
  if (values.help) {
    console.log('用法：npm run test:models -- [--model 导出目录，可重复] [--iterations 5] [--output 报告.json]\n默认逐个验收三个完整 checkpoint；缺少模型或数值不匹配会失败。--allow-tiny 仅用于随机小模型。\n冷启动指新进程加载，不清空操作系统文件缓存；热推理包含全部问题和后处理。');
    return;
  }
  const iterations = Number(values.iterations);
  assert.ok(Number.isSafeInteger(iterations) && iterations > 0, 'iterations 必须为正整数');
  if (values.worker) {
    console.log(JSON.stringify(await verify(values.model[0], values['allow-tiny'], iterations)));
    return;
  }
  const results = [];
  for (const model of values.model ?? Object.values(DEFAULT_MODELS)) {
    const args = [script, '--worker', '--model', model, '--iterations', String(iterations)];
    if (values['allow-tiny']) args.push('--allow-tiny');
    const child = spawnSync(process.execPath, args, { encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 });
    if (child.stderr) process.stderr.write(child.stderr);
    if (child.status === 0) results.push(JSON.parse(child.stdout));
    else results.push({ model, status: 'failed', error: child.error?.message ?? child.stderr.trim() ?? String(child.signal) });
  }
  const report = { timestamp: new Date().toISOString(), results };
  const json = JSON.stringify(report, null, 2) + '\n';
  if (values.output) await writeFile(values.output, json);
  process.stdout.write(json);
  if (results.some((r) => r.status !== 'passed')) process.exitCode = 1;
}
try { await main(); } catch (error) {
  process.stderr.write(`模型验收失败：${error.message}\n`);
  process.exitCode = 1;
}
