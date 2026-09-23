import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { load, predictShortlist, embedFnFromAgent, Router } from '../src/index.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const directory = process.env.LAYA_TEST_MODEL ?? path.join(root, 'artifacts/tiny-english-v2');
let available = true;
try { await access(path.join(directory, 'manifest.json')); } catch (error) {
  if (process.env.LAYA_TEST_MODEL) throw new Error(`显式指定的验收模型不可用：${directory}`, { cause: error });
  available = false;
}

function close(actual, expected, location = '', atol = 1e-4, rtol = 1e-3) {
  if (typeof expected === 'number') {
    assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= atol + rtol * Math.abs(expected), `${location}: ${actual} != ${expected}`);
  } else if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length, location);
    expected.forEach((v, i) => close(actual[i], v, `${location}[${i}]`, atol, rtol));
  } else if (expected && typeof expected === 'object') {
    assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), location);
    for (const k of Object.keys(expected)) close(actual[k], expected[k], `${location}.${k}`, atol, rtol);
  } else assert.equal(actual, expected, location);
}

test('真实 Transformers.js / ONNX：Python golden 对照', { skip: !available && '先运行 tools/export_onnx.py --tiny；完整权重可设置 LAYA_TEST_MODEL' }, async (t) => {
  const golden = JSON.parse(await readFile(path.join(directory, 'golden.json'), 'utf8'));
  const start = performance.now(), agent = await load(directory, { localFilesOnly: true });
  t.after(() => agent.dispose());
  t.diagnostic(`模型 ${directory}；cold_load_ms=${(performance.now() - start).toFixed(1)}`);
  for (const entry of golden.cases) {
    await t.test(entry.name, async () => {
      const raw = await agent.predictRaw(entry.state, entry.questions, entry.options);
      assert.deepEqual(raw.batch, entry.inputs, 'token ids / markers / mask 必须与 Python 一致');
      close(raw.output, entry.output, 'output');
      if (entry.result) close(await agent.predict(entry.state, entry.questions), entry.result, 'result');
    });
  }
  await t.test('编码器 embedding 和 shortlist', async () => {
    close(await agent.embed(golden.embedding.texts, { maxLength: 64, batchSize: 3 }), golden.embedding.values, 'embeddings');
    if (golden.embedding.inputs) {
      const tokens = golden.embedding.inputs;
      const expected = tokens.input_ids.map((ids, i) => ids.filter((_, j) => tokens.attention_mask[i][j]));
      const actual = golden.embedding.texts.map((text) => {
        const ids = agent.tok.encode(text, { add_special_tokens: true });
        return ids.length > 64 ? [...ids.slice(0, 63), ids.at(-1)] : ids;
      });
      assert.deepEqual(actual, expected, '长 embedding 文本截断后保留特殊 token');
    }
    await assert.rejects(agent.embed(['hello'], { maxLength: 1 }), /至少为 2/);
    const questions = { intent: { type: 'choice', instructions: 'Choose a team', criteria: ['billing', 'tech', 'sales'] } };
    const result = await predictShortlist(agent, 'refund my invoice', questions, { k: 2, embedFn: embedFnFromAgent(agent, { maxLength: 64 }) });
    assert.equal(result.shortlist.intent.labels.length, 2);
    assert.equal(result.shortlist.intent.probability_scope, 'shortlisted');
    assert.ok(result.shortlist.intent.labels.includes(result.answers.intent.choice));
  });
  await t.test('空问题和释放等待在途推理', async () => {
    assert.deepEqual((await agent.predict('', {})).usage, { input_tokens: 0, output_tokens: 0 });
    const another = await load(directory);
    const prediction = another.predict('hello', { a: { type: 'noul', instructions: 'Is it true?' } });
    const disposing = another.dispose();
    assert.ok((await prediction).answers.a);
    await disposing;
    await assert.rejects(another.predict('hello', {}), /释放/);
  });
  await t.test('Router 通过真实模型推理', async () => {
    const router = new Router({ models: { english: directory, multilingual: directory } });
    try {
      const result = await router.predict('你好，退款', { a: { type: 'noul', instructions: 'Refund?' } });
      assert.equal(result.routing.model, 'multilingual');
      assert.ok(Number.isFinite(result.answers.a.noul));
    } finally { await router.dispose(); }
  });
});
