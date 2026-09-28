// 数值准确性与不变性：多 checkpoint 覆盖、确定性、批不变性、答案语义契约。
// 现有 integration.test.js 已覆盖 token 级 parity 与 golden 容差比对；本文件补的是
//   1) multilingual(mmBERT-base) / typed-decisions 在默认 npm test 路径上的覆盖
//   2) 同一输入的确定性、批顺序不变性、批组成不变性
//   3) 不需要模型的答案语义契约（纯函数层）
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { load } from '../src/index.js';
import { formatAnswers } from '../src/agent.js';
import { normalizeQuestions } from '../src/questions.js';
import { confidenceFromProbs, softmax, tempBucket } from '../src/math.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const ATOL = 1e-4, RTOL = 1e-3;

function close(actual, expected, where = '', atol = ATOL, rtol = RTOL) {
  if (typeof expected === 'number') {
    assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= atol + rtol * Math.abs(expected),
      `${where}: ${actual} != ${expected}`);
  } else if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length, where);
    expected.forEach((v, i) => close(actual[i], v, `${where}[${i}]`, atol, rtol));
  } else if (expected && typeof expected === 'object') {
    assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), where);
    for (const k of Object.keys(expected)) close(actual[k], expected[k], `${where}.${k}`, atol, rtol);
  } else assert.equal(actual, expected, where);
}
const rowFor = (logits, id, order) => logits[order.indexOf(id)];
// 只比前 min(k) 列：同一问题在不同批里 K 可能不同，多出的列是 padding marker
const delta = (a, b) => { let m = 0; for (let i = 0; i < Math.min(a.length, b.length); i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

const CHECKPOINTS = ['english', 'multilingual', 'typed-decisions'];
const CANDIDATES = [
  process.env.LAYA_TEST_MODEL && path.resolve(process.env.LAYA_TEST_MODEL),
  ...CHECKPOINTS.map((n) => path.join(root, 'models', n)),
  path.join(root, 'artifacts/tiny-english-v2'),
].filter(Boolean);

const available = [];
for (const dir of CANDIDATES) {
  try { await access(path.join(dir, 'golden.json')); if (!available.includes(dir)) available.push(dir); } catch {}
}
const MISSING = '需要已导出的模型；运行 tools/export_onnx.py --tiny 或导出完整 checkpoint';

// ─────────────────────────────────────────────────────────────
// 1. 多 checkpoint：token 级 parity + 输出容差比对
// ─────────────────────────────────────────────────────────────

for (const directory of available) {
  const name = path.basename(directory);
  test(`数值对照 [${name}]`, { skip: !available.length && MISSING }, async (t) => {
    const golden = JSON.parse(await readFile(path.join(directory, 'golden.json'), 'utf8'));
    const start = performance.now();
    const agent = await load(directory, { localFilesOnly: true });
    t.after(() => agent.dispose());
    t.diagnostic(`${name}: cold_load_ms=${(performance.now() - start).toFixed(0)} tiny=${agent.metadata.tiny} encoder=${agent.cfg.encoder}`);

    for (const entry of golden.cases) {
      await t.test(`${name}/${entry.name} token 逐位一致`, async () => {
        const raw = await agent.predictRaw(entry.state, entry.questions, entry.options);
        assert.deepEqual(raw.batch, entry.inputs, 'token ids / markers / mask / qtype 必须与 Python 逐位相同');
        close(raw.output, entry.output, `${name}/${entry.name} output`);
      });
    }

    await t.test(`${name}/答案语义与 Python 一致`, async () => {
      for (const entry of golden.cases) {
        if (!entry.result) continue;
        close(await agent.predict(entry.state, entry.questions, entry.options), entry.result, `${name}/${entry.name} result`);
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────
// 2. 不变性：确定性 / 批顺序 / 批组成
// ─────────────────────────────────────────────────────────────

const probe = available[0];
test('推理不变性', { skip: !probe && MISSING }, async (t) => {
  const golden = JSON.parse(await readFile(path.join(probe, 'golden.json'), 'utf8'));
  const mixed = golden.cases.find((c) => c.name === 'mixed');
  const agent = await load(probe, { localFilesOnly: true });
  t.after(() => agent.dispose());
  const name = path.basename(probe);
  const Q = mixed.questions, ids = Object.keys(Q);
  const logitsOf = (questions) => agent.predictRaw(mixed.state, questions, mixed.options).then((r) => r.output.logits);

  const batched = await logitsOf(Q);

  await t.test('同一输入重复推理逐位相同（确定性）', async () => {
    for (let i = 0; i < 3; i++) {
      const again = await logitsOf(Q);
      for (const id of ids) {
        assert.equal(delta(rowFor(batched, id, ids), rowFor(again, id, ids)), 0,
          `${name}: ${id} 第 ${i + 1} 次重跑出现非零差异`);
      }
    }
  });

  await t.test('批内问题顺序不影响结果', async () => {
    const reversed = [...ids].reverse();
    const out = await logitsOf(Object.fromEntries(reversed.map((id) => [id, Q[id]])));
    for (const id of ids) {
      assert.equal(delta(rowFor(batched, id, ids), rowFor(out, id, reversed)), 0,
        `${name}: ${id} 换序后结果改变，说明结果依赖批内位置`);
    }
  });

  await t.test('批组成（padding 长度）影响在 1e-5 内', async () => {
    // 单独跑时该题的 K 就是它自己的候选数；混批后 K 被 padding 到批内最大候选数。
    // 被 padding 的 marker 槽不是完全惰性的，因此 k<K 的题会漂移约 1e-6~1e-5。
    // 把上界钉在这里：漂移明显变大时说明 padding 槽开始参与归约。
    const solo = {};
    for (const id of ids) solo[id] = (await logitsOf({ [id]: Q[id] }))[0];
    const other = golden.cases.find((c) => c.name === 'many-options');
    const mixQ = { ...Q, ...other.questions }, mixIds = Object.keys(mixQ);
    const out = await logitsOf(mixQ);
    for (const id of ids) {
      const diff = delta(rowFor(out, id, mixIds), solo[id]);
      assert.ok(diff <= 1e-5, `${name}: ${id} 混批漂移 ${diff.toExponential(3)} 超过 1e-5（padding marker 槽疑似参与计算）`);
      t.diagnostic(`${id}: k=${solo[id].length} 混批漂移=${diff.toExponential(3)}`);
    }
    // 候选数等于批内 K 的题必须完全一致（无 padding 可漂移）
    const full = ids.find((id) => solo[id].length === Math.max(...Object.values(solo).map((v) => v.length)));
    if (full) {
      assert.equal(delta(rowFor(out, full, mixIds), solo[full]), 0,
        `${name}: ${full} 的 k 已等于批内 K，不应有任何漂移`);
    }
  });
});

// ─────────────────────────────────────────────────────────────
// 3. 答案语义契约（纯函数，不需要模型）
// ─────────────────────────────────────────────────────────────

const SPEC = normalizeQuestions({
  choice: { type: 'choice', instructions: 'Q', criteria: { a: 'first', b: 'second', c: 'third' } },
  score: { type: 'score', instructions: 'Q', criteria: ['low', 'mid', 'high', 'critical'] },
  noul: { type: 'noul', instructions: 'Q' },
});
const LOGITS = [1.2, 0.4, -0.7];           // choice/score 用（3 列，按 K 截断）
const NOUL = [0.3, 1.4];                   // noul 用
const ACT = [0.5, 1.5];                    // act_logits 单行

test('答案归约语义', async (t) => {
  const out = formatAnswers(SPEC, {
    logits: [LOGITS, LOGITS, NOUL],
    act_logits: [ACT, ACT, ACT],
  }, {});
  const p = softmax(LOGITS);

  await t.test('choice 取 argmax 标签且概率和为 1', () => {
    assert.equal(out.choice.choice, 'a');
    assert.equal(out.choice.type, 'choice');
    assert.deepEqual(Object.keys(out.choice.probabilities), ['a', 'b', 'c']);
    close(Object.values(out.choice.probabilities).reduce((x, y) => x + y, 0), 1, 'probabilities 和');
    p.forEach((v, i) => close(out.choice.probabilities[['a', 'b', 'c'][i]], v, `p[${i}]`));
  });

  await t.test('score 是从 0 起算的等级期望', () => {
    assert.equal(out.score.type, 'score');
    assert.deepEqual(out.score.legend, { 0: 'low', 1: 'mid', 2: 'high', 3: 'critical' });
    // legend 有 4 级但只有 3 列 logits：期望值必须只按实际概率的等级累加
    const expect = p.reduce((s, v, i) => s + i * v, 0);
    close(out.score.score, expect, 'score 期望');
    assert.ok(out.score.score >= 0 && out.score.score < 3, 'score 不得越界到不存在的等级');
  });

  await t.test('noul 取 p[1] 为命题概率', () => {
    assert.equal(out.noul.type, 'noul');
    close(out.noul.noul, Math.exp(1.4) / (Math.exp(0.3) + Math.exp(1.4)), 'noul');
    close(out.noul.confidence, Math.max(out.noul.noul, 1 - out.noul.noul), 'noul confidence');
  });

  await t.test('confidence 等于归一化熵的补', () => {
    close(out.choice.confidence, confidenceFromProbs(p, 3), 'choice confidence');
    assert.ok(out.choice.confidence >= 0 && out.choice.confidence <= 1);
    // 单候选必然完全确定
    assert.equal(confidenceFromProbs([1], 1), 1);
    // 完全均匀分布的 k 元分布 confidence 为 0
    close(confidenceFromProbs([1 / 4, 1 / 4, 1 / 4, 1 / 4], 4), 0, '均匀分布');
  });

  await t.test('action.act_probability 是 act_logits 的 softmax 首项', () => {
    const expected = Math.exp(0.5) / (Math.exp(0.5) + Math.exp(1.5));
    for (const id of ['choice', 'score', 'noul']) close(out[id].action.act_probability, expected, `${id}.act_probability`);
  });

  await t.test('对称 logits 在任何温度下都保持均匀', () => {
    const q = normalizeQuestions({ q: { type: 'choice', instructions: 'Q', criteria: ['a', 'b'] } });
    for (const T of [0.5, 1, 5]) {
      const r = formatAnswers(q, { logits: [[0, 0]], act_logits: [ACT] }, { temperature: [T, T, T] });
      close(r.q.probabilities.a, 0.5, `T=${T}`);
    }
  });

  await t.test('四位小数舍入不破坏概率和的一致性', () => {
    const sum = Object.values(out.choice.probabilities).reduce((x, y) => x + y, 0);
    assert.ok(Math.abs(sum - 1) < 1e-3, `舍入后概率和 ${sum} 偏离 1 超过 1e-3`);
  });
});

test('温度分桶解析', async (t) => {
  await t.test('分桶边界', () => {
    assert.equal(tempBucket('choice', 1), 'choice:2');
    assert.equal(tempBucket('choice', 2), 'choice:2');
    assert.equal(tempBucket('choice', 3), 'choice:3-5');
    assert.equal(tempBucket('choice', 5), 'choice:3-5');
    assert.equal(tempBucket('choice', 6), 'choice:6-10');
    assert.equal(tempBucket('choice', 10), 'choice:6-10');
    assert.equal(tempBucket('choice', 11), 'choice:11+');
    assert.equal(tempBucket(1, 4), 'score:3-5');       // 数字 qtype 也接受
    assert.equal(tempBucket('noul', 2), 'noul:2');
  });

  await t.test('分桶温度优先于按类型温度', () => {
    const cfg = { temperature: [5, 5, 5], temperature_by_options: { 'choice:2': 0.5 } };
    const out = formatAnswers(normalizeQuestions({ q: { type: 'choice', instructions: 'Q', criteria: ['a', 'b'] } }),
      { logits: [[1, 0]], act_logits: [[0, 0]] }, cfg);
    // 分桶 T=0.5 而非按类型 T=5：前者更锐，p(a) 必须更大
    close(out.q.probabilities.a, 1 / (1 + Math.exp(-2)), 'p(a)');
    assert.ok(out.q.probabilities.a > 0.5, `分桶温度未生效：p(a)=${out.q.probabilities.a}`);
  });

  await t.test('无分桶时回退到按类型温度', () => {
    const q = normalizeQuestions({ q: { type: 'noul', instructions: 'Q' } });
    const sharp = formatAnswers(q, { logits: [[0, 1]], act_logits: [[0, 0]] }, { temperature: [1, 1, 0.5] });
    const plain = formatAnswers(q, { logits: [[0, 1]], act_logits: [[0, 0]] }, {});
    close(sharp.q.noul, 1 / (1 + Math.exp(-2)), 'T=0.5 的 noul');
    close(plain.q.noul, 1 / (1 + Math.exp(-1)), 'T=1 的 noul');
    assert.ok(sharp.q.noul > plain.q.noul, 'T=0.5 应让 true 概率更接近 1');
  });

  await t.test('超出 [0.5,5] 的温度被限幅到边界而非报错', () => {
    const q = normalizeQuestions({ q: { type: 'noul', instructions: 'Q' } });
    const hot = formatAnswers(q, { logits: [[0, 1]], act_logits: [[0, 0]] }, { temperature: [1, 1, 1e9] });
    close(hot.q.noul, 1 / (1 + Math.exp(-(1 - 0) / 5)), 'T=1e9 被限幅为 T=5');
  });
});
