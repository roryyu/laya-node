import { QTYPE_NAMES, QTYPES } from './questions.js';
export const TEMP_MIN = 0.5;
export const TEMP_MAX = 5;
export const round4 = (n) => Math.round(n * 1e4) / 1e4;
export function positiveInteger(n, label) {
  if (!Number.isSafeInteger(n) || n < 1) throw new RangeError(`${label} 必须为正整数`);
  return n;
}
export function clampTemperature(value, lo = TEMP_MIN, hi = TEMP_MAX) {
  const n = typeof value === 'number' || (typeof value === 'string' && value.trim()) ? Number(value) : NaN;
  return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : 1;
}
export function tempBucket(qtype, k) {
  const type = typeof qtype === 'number' ? QTYPE_NAMES[qtype] : qtype;
  if (!Object.hasOwn(QTYPES, type)) throw new TypeError('未知问题类型');
  positiveInteger(k, '候选数');
  return `${type}:${k <= 2 ? '2' : k <= 5 ? '3-5' : k <= 10 ? '6-10' : '11+'}`;
}
export function softmax(logits, temperature = 1) {
  if (!logits.length || !Array.from(logits).every(Number.isFinite)) throw new RangeError('logits 必须为非空有限数值数组');
  if (!Number.isFinite(temperature) || temperature <= 0) throw new RangeError('温度必须为正数');
  let max = -Infinity;
  for (const x of logits) max = Math.max(max, x);
  const exp = Array.from(logits, (x) => Math.exp((x - max) / temperature));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((x) => x / sum);
}
export function confidenceFromProbs(p, k = p.length) {
  positiveInteger(k, '候选数');
  if (p.length < k || Array.from(p).some((x) => !Number.isFinite(x) || x < 0 || x > 1)) throw new RangeError('非法概率');
  if (k < 2) return 1;
  const entropy = Array.from(p).slice(0, k).reduce((h, x) => h - x * Math.log(Math.max(x, 1e-12)), 0);
  return Math.max(0, Math.min(1, 1 - entropy / Math.log(k)));
}
export function eceScore(confidence, correct, bins = 15) {
  positiveInteger(bins, 'bins');
  if (confidence.length !== correct.length) throw new RangeError('置信度与标签长度不一致');
  if (!confidence.length) return NaN;
  if (confidence.some((x) => !Number.isFinite(x) || x < 0 || x > 1) || correct.some((x) => ![0, 1, false, true].includes(x))) throw new RangeError('非法置信度或正确性标签');
  let ece = 0;
  for (let b = 0; b < bins; b++) {
    let count = 0, sum = 0, hits = 0;
    for (let i = 0; i < confidence.length; i++) if (confidence[i] > b / bins && confidence[i] <= (b + 1) / bins) {
      count++; sum += confidence[i]; hits += Number(correct[i]);
    }
    if (count) ece += Math.abs(sum - hits) / confidence.length;
  }
  return ece;
}
/** 单条分布的 strictly proper scoring rule；批量由调用者 map。 */
export function properReward(q, target, qtype, mask = q.map(() => true), { wSph = 0.5, wRps = 1, logFloor = -9.21 } = {}) {
  if (!q.length || target.length !== q.length || mask.length !== q.length) throw new RangeError('分布、目标和 mask 长度必须相同且非空');
  if ([...q, ...target].some((x) => !Number.isFinite(x) || x < 0 || x > 1)) throw new RangeError('概率必须位于 [0,1]');
  let log = 0, dot = 0, norm = 0, cq = 0, ct = 0, rps = 0;
  for (let i = 0; i < q.length; i++) {
    const v = mask[i] ? q[i] : 0;
    log += target[i] * Math.max(logFloor, Math.log(Math.max(v, 1e-12)));
    dot += target[i] * v; norm += v * v;
    cq += v; ct += target[i];
    if (mask[i]) rps += (cq - ct) ** 2;
  }
  const ordinal = qtype === 'score' || qtype === QTYPES.score;
  return log + wSph * dot / Math.max(Math.sqrt(norm), 1e-9) -
    (ordinal ? wRps * rps / (Math.max(2, mask.filter(Boolean).length) - 1) : 0);
}
/** 与上游相同的按 episode 分组、倒序递推；不修改输入。 */
export function tdLambdaTargets(pTrue, batch, lam = 1) {
  if (!Number.isFinite(lam) || lam < 0 || lam > 1) throw new RangeError('lambda 必须位于 [0,1]');
  const target = batch.target.map((row) => [...row]);
  if (!batch.ep_group) return target;
  if ([pTrue, batch.ep_group, batch.ep_step].some((a) => !a || a.length !== target.length)) throw new RangeError('trajectory 长度不一致');
  const groups = new Map();
  batch.ep_group.forEach((g, i) => { if (g >= 0) { if (!groups.has(g)) groups.set(g, []); groups.get(g).push(i); } });
  for (const indices of groups.values()) {
    indices.sort((a, b) => batch.ep_step[a] - batch.ep_step[b]);
    let value = target[indices.at(-1)][1];
    for (let j = indices.length - 1; j >= 0; j--) {
      if (j < indices.length - 1) value = (1 - lam) * pTrue[indices[j + 1]] + lam * value;
      target[indices[j]][0] = 1 - value; target[indices[j]][1] = value;
    }
  }
  return target;
}
