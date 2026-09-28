#!/usr/bin/env node
// 扫描 onnxruntime intraOpNumThreads 对推理延迟的影响，找出本机最优线程数。
// 用法：node tools/bench_threads.mjs [模型目录] [线程数列表，如 0,4,6,8]
import { readFile } from 'node:fs/promises';
import { load } from '../src/index.js';

const modelPath = process.argv[2] || undefined;
const threadList = (process.argv[3] ?? '0,4,6,8,10,12').split(',').map(Number);
const request = JSON.parse(await readFile(new URL('../examples/request.json', import.meta.url), 'utf8'));
const WARMUP = 2, ROUNDS = 5;

console.log(`线程数扫描：${Object.keys(request.questions).length} 个问题，预热 ${WARMUP} 次，计时 ${ROUNDS} 次取中位数`);
const results = [];
for (const threads of threadList) {
  const agent = await load(modelPath, { sessionOptions: { intraOpNumThreads: threads } });
  for (let i = 0; i < WARMUP; i++) await agent.predict(request.state, request.questions);
  const samples = [];
  for (let i = 0; i < ROUNDS; i++) {
    const started = performance.now();
    await agent.predict(request.state, request.questions);
    samples.push(performance.now() - started);
  }
  await agent.dispose();
  samples.sort((a, b) => a - b);
  const median = samples[Math.floor(samples.length / 2)];
  results.push({ threads, median, min: samples[0] });
  console.log(`intraOpNumThreads=${String(threads).padStart(2)}  中位 ${median.toFixed(0)}ms  最快 ${samples[0].toFixed(0)}ms`);
}
const best = results.reduce((a, b) => (b.median < a.median ? b : a));
console.log(`\n最优：intraOpNumThreads=${best.threads}（中位 ${best.median.toFixed(0)}ms）`);
