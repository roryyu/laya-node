import test from 'node:test';
import assert from 'node:assert/strict';
import { Router, shortlistChoice, predictShortlist } from '../src/index.js';
const deferred = () => { let resolve; const promise = new Promise((r) => { resolve = r; }); return { promise, resolve }; };
const questions = { intent: { type: 'choice', instructions: 'Choose', criteria: ['a', 'b', 'c'] } };
function stub(label, dispose = () => {}) { return { predict: async () => ({ model: label, answers: {}, usage: { input_tokens: 1, output_tokens: 0 } }), dispose }; }

test('路由优先级与非默认 typed-decisions', () => {
  const r = new Router();
  assert.equal(r.route('中文').model, 'multilingual');
  assert.equal(r.route('Please refund').model, 'english');
  assert.equal(r.route('中文', {}, { model: 'en', task: 'typed', lang: 'zh' }).model, 'english');
  assert.equal(r.route('中文', {}, { task: 'typed_decisions', lang: 'en' }).model, 'typed-decisions');
  const workflow = Object.fromEntries(['action', 'category', 'churn_risk', 'needs_human', 'urgency'].map((k) => [k, {}]));
  assert.equal(r.route('English', workflow).model, 'english');
  assert.equal(new Router({ autoTaskDetection: true }).route('English', workflow).model, 'typed-decisions');
  const collision = { 'action|category|churn_risk|needs_human|urgency': {} };
  assert.equal(new Router({ autoTaskDetection: true }).route('English', collision).model, 'english');
  assert.equal(r.route('', {}, { lang: 'zh-CN' }).model, 'multilingual');
  assert.throws(() => r.route('', {}, { model: '__proto__' }));
  assert.throws(() => new Router({ maxLoaded: 0 }));
});
test('并发加载去重和 LRU 释放', async () => {
  const gate = deferred(); let loads = 0; const disposed = [];
  const router = new Router({ loader: async (name) => { loads++; await gate.promise; return stub(name, () => disposed.push(name)); } });
  const first = router.load('english'), second = router.load('en');
  gate.resolve();
  assert.equal(await first, await second); assert.equal(loads, 1);
  await router.load('multilingual');
  assert.deepEqual(router.loaded, ['multilingual']); assert.equal(disposed.length, 1);
  await router.dispose(); await router.dispose(); assert.equal(disposed.length, 2);
  await assert.rejects(router.load('english'), /释放/);
});
test('LRU 和 unload 不释放在途推理', async () => {
  const entered = deferred(), gate = deferred(); let disposed = 0;
  const busy = { predict: async () => { entered.resolve(); await gate.promise; assert.equal(disposed, 0); return { answers: {} }; }, dispose: () => disposed++ };
  const router = new Router({ loader: async () => busy });
  const prediction = router.predict('English', questions);
  await entered.promise;
  const unloading = router.unload('english');
  assert.equal(disposed, 0);
  gate.resolve(); await prediction; await unloading;
  assert.equal(disposed, 1); assert.deepEqual(router.loaded, []);
});
test('加载期间 unload 等待使用者完成', async () => {
  const gate = deferred(); let disposed = false;
  const router = new Router({ loader: async () => { await gate.promise; return stub('x', () => { disposed = true; }); } });
  const prediction = router.predict('English', questions);
  const unload = router.unload();
  gate.resolve();
  assert.equal((await prediction).routing.model, 'english');
  await unload; assert.equal(disposed, true);
});
test('失败加载可重试，preload 保留全部模型', async () => {
  let calls = 0;
  const router = new Router({ loader: async () => { if (++calls === 1) throw new Error('network'); return stub('ok'); } });
  await assert.rejects(router.load('english'), /network/);
  await router.preload(['english', 'multilingual', 'typed-decisions']);
  assert.equal(router.loaded.length, 3); assert.equal(router.maxLoaded, 3);
  await router.dispose();
});
test('attach 避免重复加载和双重释放', async () => {
  let calls = 0, disposed = 0;
  const router = new Router({ loader: () => { calls++; throw new Error('不得调用'); } });
  const agent = stub('attached', () => disposed++);
  await router.attach('english', agent);
  assert.equal(await router.load('english'), agent);
  assert.equal((await router.predict('hello', {})).model, 'attached');
  await assert.rejects(router.attach('multilingual', agent), /两个模型名/);
  assert.equal(calls, 0);
  await router.dispose(); assert.equal(disposed, 1);
});
test('shortlist 排序稳定、零向量和非有限值', async () => {
  const embed = async () => [[1, 0], [1, 0], [1, 0], [NaN, Infinity]];
  assert.deepEqual(await shortlistChoice('x', ['a', 'b', 'c'], embed, 2), ['a', 'b']);
  assert.deepEqual(await shortlistChoice('x', ['a', 'b'], async () => [[0, 0], [1, 0], [2, 0]], 1), ['a']);
  assert.deepEqual(await shortlistChoice('x', ['a', 'b'], async () => [[1e308, 1e308], [1e308, 1e308], [0, 1]], 1), ['a']);
});
test('数字形候选标签的 embedding 顺序不被对象键规则打乱', async () => {
  const selected = await shortlistChoice('x', ['10', '2', '0'], async (texts) => {
    assert.deepEqual(texts.slice(1), ['10', '2', '0']);
    return [[1, 0], [1, 0], [0, 1], [-1, 0]];
  }, 1);
  assert.deepEqual(selected, ['10']);
});
test('shortlist passthrough 不调用 embed，非法维度拒绝', async () => {
  assert.deepEqual(await shortlistChoice('x', ['a', 'b'], () => { throw new Error('不得调用'); }, 2), ['a', 'b']);
  await assert.rejects(shortlistChoice('x', ['a', 'b'], async () => [[1], [2, 3], [4]], 1), /二维/);
  await assert.rejects(shortlistChoice('x', ['a', 'a'], () => [], 1), /重复/);
  await assert.rejects(shortlistChoice('x', ['a'], () => [], 0), /正整数/);
});
test('shortlist 不修改调用者 schema 且标注概率范围', async () => {
  let captured;
  const agent = { predict: async (_state, q) => { captured = q; return { answers: {} }; } };
  const result = await predictShortlist(agent, 'x', questions, { k: 1, embedFn: async () => [[1, 0], [0, 1], [1, 0], [-1, 0]] });
  assert.deepEqual(captured.intent.criteria, ['b']);
  assert.deepEqual(questions.intent.criteria, ['a', 'b', 'c']);
  assert.equal(result.shortlist.intent.probability_scope, 'shortlisted');
});
