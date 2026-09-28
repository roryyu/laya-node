// 性能基准：测量并报告，断言只覆盖跨机器稳定的性质。
//
// 设计原则：绝对延迟依赖硬件（README 记录 M2 Max CPU fp32 约 710ms，本机不同），
// 因此默认不对绝对时间设断言，否则 CI 必然抖动。改为断言与硬件无关的不变量：
//   - 批处理不会让 embedding 吞吐显著变差（防止有人把批处理改坏）
//   - 序列长度是延迟的主要驱动因素（记录，供调预算参考）
//   - shortlist 的 passthrough 路径确实零成本
// 绝对预算只在 LAYA_PERF_BUDGET=1 时启用，供本地/CI 专用 runner 使用。
//
// 产物：artifacts/performance.json，LAYA_PERF=sweep 时跑完整扫描。
import test from 'node:test';
import assert from 'node:assert/strict';
import { access, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { load, predictShortlist, embedFnFromAgent } from '../src/index.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const SWEEP = process.env.LAYA_PERF === 'sweep';
const BUDGET = process.env.LAYA_PERF_BUDGET === '1';
const gib = (n) => (n / 2 ** 30).toFixed(2);

const CANDIDATES = [
  process.env.LAYA_TEST_MODEL && path.resolve(process.env.LAYA_TEST_MODEL),
  path.join(root, 'models/english'),
  path.join(root, 'artifacts/tiny-english-v2'),
].filter(Boolean);
let directory = null;
for (const dir of CANDIDATES) {
  try { await access(path.join(dir, 'config.json')); directory = dir; break; } catch {}
}

const QUESTIONS = {
  intent: { type: 'choice', instructions: 'What does the customer want in `message`?', criteria: ['refund', 'bug', 'billing', 'other'] },
  urgency: { type: 'score', instructions: 'How urgent is the request?', criteria: ['none', 'soon', 'blocking'] },
  refund: { type: 'noul', instructions: 'Is a refund requested?' },
};
const SHORT = 'I was charged twice and need a refund today.';
const LONG = 'the customer reports a duplicate charge. '.repeat(60);

async function measure(label, reps, fn) {
  await fn();
  const t = [];
  for (let i = 0; i < reps; i++) { const s = performance.now(); await fn(); t.push(performance.now() - s); }
  t.sort((a, b) => a - b);
  return { label, reps, p50: +t[Math.floor(t.length / 2)].toFixed(1), min: +t[0].toFixed(1), max: +t.at(-1).toFixed(1) };
}

test('性能基准', { skip: !directory && '需要已导出的模型；运行 tools/export_onnx.py --tiny' }, async (t) => {
  const loadStart = performance.now();
  const agent = await load(directory, { localFilesOnly: true });
  t.after(() => agent.dispose());
  const rows = [];
  const rssLoad = process.memoryUsage().rss;
  rows.push({ label: 'cold_load', reps: 1, p50: +(performance.now() - loadStart).toFixed(1), min: 0, max: 0 });
  t.diagnostic(`模型 ${path.basename(directory)} tiny=${agent.metadata.tiny} 冷加载 ${(performance.now() - loadStart).toFixed(0)}ms  rss=${gib(rssLoad)}GiB`);

  await t.test('推理延迟由序列长度主导', async () => {
    const one = await measure('predict 1题 noul', 3, () => agent.predict(SHORT, { refund: QUESTIONS.refund }));
    const three = await measure('predict 3题 (choice+score+noul)', 3, () => agent.predict(SHORT, QUESTIONS));
    const long = await measure(`predict 3题 长正文(~${Math.min(agent.cfg.max_len, 512)} tok)`, 1, () => agent.predict(LONG, QUESTIONS));
    rows.push(one, three, long);
    t.diagnostic(`短单题 ${one.p50}ms → 短三题 ${three.p50}ms → 长正文 ${long.p50}ms（正文放大约 ${(long.p50 / three.p50).toFixed(1)} 倍）`);

    assert.ok(Number.isFinite(long.p50) && long.p50 > 0, '长正文延迟必须是有限正数');
    if (BUDGET) {
      const cap = agent.metadata.tiny ? 500 : 6000;
      assert.ok(three.p50 < cap, `predict 预算超标：${three.p50}ms > ${cap}ms`);
    }
    assert.ok(three.p50 < one.p50 * 6,
      `3 题耗时 ${three.p50}ms 相对 1 题 ${one.p50}ms 超过 6 倍，疑似非线性`);
  });

  await t.test('embedding 吞吐与 batchSize 基本无关（批处理不改变单条成本）', async () => {
    const n = SWEEP ? 256 : 64;
    const texts = Array.from({ length: n }, (_, i) => `support taxonomy candidate number ${i}`);
    const sizes = [...new Set([16, 32, 64, n])].sort((a, b) => a - b);
    const perText = {};
    for (const bs of sizes) {
      const r = await measure(`embed n=${n} batchSize=${bs}`, SWEEP ? 3 : 1, () => agent.embed(texts, { maxLength: 64, batchSize: bs }));
      perText[bs] = r.p50 / n;
      rows.push(r);
    }
    const values = Object.values(perText);
    const spread = Math.max(...values) / Math.min(...values);
    for (const [bs, v] of Object.entries(perText)) t.diagnostic(`  batchSize=${bs.padStart(3)}  ${v.toFixed(2)} ms/条`);
    t.diagnostic(`  单条成本离散度 ${spread.toFixed(2)}x —— 编码器是计算瓶颈，批大小只改变单次 forward 的宽度`);

    // 关键：加大批大小不会显著改变单条吞吐。若这条失败，说明批处理被改坏
    // （例如退化成逐条 forward，或 padding 到 maxLength 而非批内最长）
    assert.ok(spread < 2.5, `单条成本随 batchSize 波动 ${spread.toFixed(2)}x，超过 2.5x 说明批处理失效`);
  });

  await t.test('embed 返回结构正确且全部有限', async () => {
    const out = await agent.embed(['alpha', 'beta', 'gamma'], { maxLength: 32, batchSize: 2 });
    assert.equal(out.length, 3);
    const dim = agent.metadata.hidden_size;
    for (const row of out) {
      assert.equal(row.length, dim, `embedding 维度应为 ${dim}`);
      assert.ok(row.every(Number.isFinite), 'embedding 含非有限数值');
    }
    const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    assert.ok(norm(out[0]) > 0, '非空文本的 embedding 范数不应为 0');
  });

  await t.test('shortlist 端到端随候选数线性增长', async () => {
    const sizes = SWEEP ? [100, 500, 1000] : [100];
    const measured = [];
    for (const n of sizes) {
      const criteria = Object.fromEntries(Array.from({ length: n }, (_, i) => [`label_${i}`, `support taxonomy candidate number ${i}`]));
      const r = await measure(`shortlist ${n} 候选 → k=20`, 1, () => predictShortlist(agent, SHORT,
        { pick: { type: 'choice', instructions: 'Which label fits?', criteria } },
        { k: 20, embedFn: embedFnFromAgent(agent, { maxLength: 64, batchSize: 32 }) }));
      rows.push(r); measured.push({ n, ms: r.p50 });
    }
    for (const { n, ms } of measured) t.diagnostic(`  ${String(n).padStart(4)} 候选 → k=20  ${(ms / 1000).toFixed(2)}s  (${(ms / n).toFixed(1)}ms/候选)`);
    for (let i = 1; i < measured.length; i++) {
      assert.ok(measured[i].ms < measured[i - 1].ms * 3,
        `${measured[i - 1].n}→${measured[i].n} 候选耗时增长超过 3 倍，疑似非线性`);
    }
  });

  await t.test('shortlist passthrough 不调用 embed（零 embedding 成本）', async () => {
    let called = 0;
    const embedFn = () => { called++; return []; };
    const out = await predictShortlist(agent, SHORT,
      { pick: { type: 'choice', instructions: 'Q', criteria: ['a', 'b', 'c'] } }, { k: 10, embedFn });
    assert.equal(called, 0, 'k >= 候选数时不应调用 embedFn');
    assert.equal(out.shortlist.pick.probability_scope, 'all');
    assert.equal(out.shortlist.pick.passthrough, true);
    assert.equal(out.shortlist.pick.scores, null);
    assert.equal(out.shortlist.pick.original_count, 3);
  });

  await t.test('shortlist 嵌入全部候选，成本是 O(n) 而非 O(k)', async () => {
    // 候选必须全部嵌入才能排序，所以 embedding 成本与候选总数成正比，与 k 无关。
    // 这条断言把该行为钉住：k 变小不会降低 embedding 成本，任何"优化"若改变了
    // 嵌入条数，这里会失败，从而让成本变化可见而不是静默发生。
    const criteria = Object.fromEntries(Array.from({ length: 200 }, (_, i) => [`l${i}`, `candidate ${i}`]));
    let texts = 0;
    const embedFn = (t) => { texts += t.length; return t.map(() => new Array(agent.metadata.hidden_size).fill(0.1)); };
    const out = await predictShortlist(agent, SHORT,
      { pick: { type: 'choice', instructions: 'Q', criteria } }, { k: 10, embedFn });
    assert.equal(texts, 201, '必须嵌入 query + 全部 200 个候选，与 k=10 无关');
    assert.equal(out.shortlist.pick.labels.length, 10);
    assert.equal(out.shortlist.pick.probability_scope, 'shortlisted');
    assert.equal(out.shortlist.pick.original_count, 200);
    t.diagnostic(`  200 候选 → k=10 仍嵌入 ${texts} 条；k 不影响 embedding 成本`);
  });

  const rssPeak = process.memoryUsage().rss;
  const report = {
    generated_at: new Date().toISOString(),
    model: path.basename(directory),
    tiny: agent.metadata.tiny,
    encoder: agent.cfg.encoder,
    hidden_size: agent.metadata.hidden_size,
    node: process.version,
    platform: `${process.platform}-${process.arch}`,
    cpu_count: (await import('node:os')).cpus().length,
    sweep: SWEEP,
    rss_load_gib: +gib(rssLoad),
    rss_peak_gib: +gib(rssPeak),
    measurements: rows,
  };
  await mkdir(path.join(root, 'artifacts'), { recursive: true });
  await writeFile(path.join(root, 'artifacts/performance.json'), JSON.stringify(report, null, 2) + '\n');
  t.diagnostic(`报告已写入 artifacts/performance.json  rss 峰值=${gib(rssPeak)}GiB`);
});
