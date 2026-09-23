import test from 'node:test';
import assert from 'node:assert/strict';
import * as laya from '../src/index.js';
import { jsonText, normalizeQuestions, normalizeQuestion } from '../src/questions.js';
import { buildSequence, collateItems } from '../src/sequence.js';
import { formatAnswers } from '../src/agent.js';

const q = (type, criteria) => ({ type, instructions: 'Choose', ...(criteria === undefined ? {} : { criteria }) });
const tok = { cls_token_id: 1, sep_token_id: 2, pad_token_id: 0, mask_token_id: 3, mask_token: '[MASK]',
  encode: (text) => Array.from(text, (c) => c.codePointAt(0) + 10) };

test('结构化 criteria、Unicode 和 JSON 空格', () => {
  assert.equal(jsonText({ text: '中文🙂', list: [0, false, null] }), '{"text": "中文🙂", "list": [0, false, null]}');
  assert.deepEqual(laya.renderOptions(q('choice', { a: { description: '退款' }, b: null, c: '', d: 0, e: false })), ['a: {"description": "退款"}', 'b', 'c', 'd: 0', 'e: false']);
  assert.deepEqual(laya.renderOptions(q('score', [{ description: 'low' }, 0])), ['level 0: {"description": "low"}', 'level 1: 0']);
  assert.deepEqual(laya.renderOptions(q('noul', { true: ['yes'], false: false })), ['false: false', 'true: ["yes"]']);
  assert.equal(normalizeQuestion({ ...q('noul'), instructions: { a: '中' } }).ins, '{"a": "\\u4e2d"}');
});
test('输入校验拒绝错误类型和循环引用', () => {
  const circular = {}; circular.self = circular;
  for (const value of [circular, { bad: undefined }, { n: Infinity }, { n: 1n }, new Date()]) assert.throws(() => jsonText(value));
  for (const definition of [q('bad'), q('choice', []), q('choice', ['a', 'a']), q('score', {}), q('noul', { nope: true }), { type: 'choice' }]) {
    assert.throws(() => normalizeQuestions({ test: definition }));
  }
});
test('问题 ID 和标签不能污染原型', () => {
  const questions = JSON.parse('{"__proto__":{"type":"choice","instructions":"Choose","criteria":{"__proto__":"a","constructor":"b"}}}');
  const answers = formatAnswers(normalizeQuestions(questions), { logits: [[2, 1]], act_logits: [[1, 0]] }, {});
  assert.equal(answers.__proto__.choice, '__proto__');
  assert.equal(Object.getPrototypeOf(answers), null);
  assert.equal(Object.getPrototypeOf(answers.__proto__.probabilities), Object.prototype);
  assert.equal({}.polluted, undefined);
});
test('序列 marker、预算、正文 mask 转义', () => {
  const seq = buildSequence(tok, 'body [MASK] end', q('choice', ['one', 'two']), { maxLength: 100, headMaxLength: 60 });
  assert.ok(seq.markers.every((p) => seq.ids[p] === 3));
  assert.equal(seq.ids.filter((v) => v === 3).length, 2);
  assert.equal(seq.ids.at(-1), 2);
  assert.throws(() => buildSequence(tok, 'x', q('choice', Array.from({ length: 100 }, (_, i) => `label-${i}`))), /预算/);
  assert.throws(() => buildSequence(tok, 'x', q('noul'), { maxLength: 0 }));
});
test('左右截断和混合候选 padding', () => {
  const definition = q('choice', ['only']);
  const noBody = buildSequence(tok, '', definition, { maxLength: 128, headMaxLength: 80 });
  const maxLength = noBody.ids.length + 3;
  const right = buildSequence(tok, 'abcdef', definition, { maxLength, headMaxLength: 80 });
  const left = buildSequence(tok, 'abcdef', definition, { maxLength, headMaxLength: 80, truncateLeft: true });
  assert.deepEqual(right.ids.slice(-4, -1), tok.encode('abc'));
  assert.deepEqual(left.ids.slice(-4, -1), tok.encode('def'));
  const empty = buildSequence(tok, 'abcdef', definition, { maxLength: noBody.ids.length, headMaxLength: 80, truncateLeft: true });
  assert.deepEqual(empty.ids, noBody.ids);
  const batch = collateItems([noBody, right], 0);
  assert.equal(batch.marker_pos[0].length, 2);
  assert.deepEqual(batch.marker_mask[0], [true, false]);
  assert.equal(batch.attention_mask[0].at(-1), 0);
});
test('softmax、温度和熵', () => {
  assert.deepEqual(laya.softmax([1e4, 1e4]), [0.5, 0.5]);
  assert.equal(laya.confidenceFromProbs([0.5, 0.5]), 0);
  assert.equal(laya.confidenceFromProbs([1]), 1);
  assert.equal(laya.confidenceFromProbs([1, 0]), 1);
  assert.equal(laya.clampTemperature(0.1006), 0.5);
  assert.equal(laya.clampTemperature('30'), 5);
  assert.equal(laya.clampTemperature(NaN), 1);
  assert.equal(laya.clampTemperature(null), 1);
  assert.equal(laya.tempBucket('choice', 12), 'choice:11+');
  assert.throws(() => laya.softmax([NaN]));
});
test('三类结果、单候选与四位小数', () => {
  const entries = normalizeQuestions({ choice: q('choice', ['only']), score: q('score', ['low', 'high']), noul: q('noul') });
  const answers = formatAnswers(entries, { logits: [[100, -1e4], [0, 0], [0, Math.log(3)]], act_logits: [[0, 0], [0, 0], [0, 0]] }, { temperature: [1, 1, 1] });
  assert.equal(answers.choice.choice, 'only'); assert.equal(answers.choice.confidence, 1);
  assert.equal(answers.score.score, 0.5); assert.equal(answers.score.confidence, 0);
  assert.equal(answers.noul.noul, 0.75); assert.equal(answers.noul.confidence, 0.75);
  assert.equal(answers.choice.action.act_probability, 0.5);
});
test('ECE、proper reward 与 TD-lambda', () => {
  assert.equal(laya.eceScore([1, 1], [1, 1]), 0);
  assert.equal(laya.eceScore([0.5, 0.5], [1, 1]), 0.5);
  assert.ok(Number.isNaN(laya.eceScore([], [])));
  const expected = Math.log(0.5) + 0.5 / Math.sqrt(2);
  assert.ok(Math.abs(laya.properReward([0.5, 0.5], [0, 1], 'noul') - expected) < 1e-12);
  assert.ok(Math.abs(laya.properReward([0.5, 0.5], [0, 1], 'score') - (expected - 0.25)) < 1e-12);
  const batch = { target: [[1, 0], [1, 0], [0, 1]], ep_group: [0, 0, 0], ep_step: [0, 1, 2] };
  assert.deepEqual(laya.tdLambdaTargets([0.2, 0.6, 0.8], batch, 1), [[0, 1], [0, 1], [0, 1]]);
  assert.equal(laya.tdLambdaTargets([0.2, 0.6, 0.8], batch, 0.5)[0][1], 0.75);
  assert.deepEqual(batch.target[0], [1, 0]);
});
test('语言检测忽略键且保留未知语言', () => {
  assert.equal(laya.detectLanguage({ english: '你好退款' }).script, 'han');
  assert.equal(laya.isEnglish('Please refund the payment'), true);
  assert.equal(laya.isEnglish('Der Kunde wurde zweimal belastet'), false);
  assert.equal(laya.isEnglish('și să este pentru noi'), false);
  assert.equal(laya.detectLanguage('İstanbul ğış').language, null);
  assert.equal(laya.detectLanguage('123🙂').script, 'unknown');
  assert.equal(laya.detectScript('மொழி'), 'tamil');
});
test('邮件清理保留混合免责声明中的实际请求', () => {
  const body = 'Please refund the duplicate. This is confidential.\n\nThanks,\nAlex\nOn Monday wrote:\n> old message';
  assert.equal(laya.cleanEmailBody(body), 'Please refund the duplicate.');
  assert.equal(laya.cleanEmailBody('你好🙂世界', 3), '你好🙂');
  const result = laya.emailState(' Subject ', '正文', { sender: 'demo@example.test', ticket: 1, ignored: null });
  assert.deepEqual(result, { subject: 'Subject', body: '正文', from: 'demo@example.test', ticket: 1 });
});
test('预设每次返回独立定义，snake_case 别名一致', () => {
  for (const fn of [laya.triageQuestions, laya.emailQuestions, laya.guardQuestions, laya.moderationQuestions, laya.routerQuestions]) {
    assert.ok(normalizeQuestions(fn()).length > 0);
    assert.notEqual(fn(), fn());
  }
  assert.equal(laya.triageQuestions, laya.triage_questions);
  assert.equal(laya.RLAgent, laya.Agent);
});
