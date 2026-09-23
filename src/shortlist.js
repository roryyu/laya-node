import { criteriaEntries, renderOptions, serializeState, normalizeQuestions, jsonText } from './questions.js';
import { positiveInteger } from './math.js';

async function rank(state, criteria, embedFn, k, instructions) {
  positiveInteger(k, 'k');
  const entries = criteriaEntries(criteria), labels = entries.map(([key]) => key);
  if (k >= entries.length) return { labels, scores: null, passthrough: true, original_count: entries.length };
  if (typeof embedFn !== 'function') throw new TypeError('embedFn 必须为函数');
  const body = serializeState(state);
  const query = instructions == null || instructions === '' ? body : `${typeof instructions === 'string' ? instructions : jsonText(instructions)}\n${body}`;
  const texts = renderOptions({ type: 'choice', instructions: '', criteria });
  const raw = await embedFn([query, ...texts]);
  if (!Array.isArray(raw) || raw.length !== entries.length + 1) throw new RangeError('embedding 行数必须等于候选数 + 1');
  const dim = raw[0]?.length;
  if (!Number.isInteger(dim) || dim < 1 || raw.some((row) => (!Array.isArray(row) && !ArrayBuffer.isView(row)) || row.length !== dim)) throw new RangeError('embedding 必须为规则二维数组');
  // 按最大绝对值缩放避免超大向量溢出，余弦相似度保持不变。
  const normalize = (row) => {
    const values = Array.from(row, (v) => typeof v === 'number' && Number.isFinite(v) ? v : 0);
    const scale = values.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    if (!scale) return values;
    const scaled = values.map((v) => v / scale), norm = Math.sqrt(scaled.reduce((sum, v) => sum + v * v, 0));
    return scaled.map((v) => v / norm);
  };
  const matrix = raw.map(normalize), q = matrix[0];
  const ranked = labels.map((label, i) => ({ label, i, score: matrix[i + 1].reduce((sum, v, j) => sum + v * q[j], 0) }));
  ranked.sort((a, b) => b.score - a.score || a.i - b.i);
  const selected = ranked.slice(0, k);
  return { labels: selected.map((r) => r.label), scores: selected.map((r) => r.score), passthrough: false, original_count: labels.length };
}
export async function shortlistChoice(state, criteria, embedFn, k = 20, { instructions } = {}) {
  return (await rank(state, criteria, embedFn, k, instructions)).labels;
}
export async function predictShortlist(agent, state, questions, { embedFn, k = 20, ...predictOptions } = {}) {
  positiveInteger(k, 'k');
  const entries = normalizeQuestions(questions), reduced = Object.create(null), shortlist = Object.create(null);
  for (const [id, q] of entries) {
    if (q.t !== 'choice') { reduced[id] = questions[id]; continue; }
    const ranking = await rank(state, q.crit, embedFn, k, q.ins);
    reduced[id] = { type: 'choice', instructions: q.ins, criteria: Array.isArray(q.crit)
      ? [...ranking.labels] : Object.fromEntries(ranking.labels.map((label) => [label, q.crit[label]])) };
    shortlist[id] = { ...ranking, probability_scope: ranking.passthrough ? 'all' : 'shortlisted' };
  }
  const predict = agent.predict ?? agent.systemOne ?? agent.system_one;
  if (typeof predict !== 'function') throw new TypeError('agent 必须提供 predict 或 system_one');
  return { ...await predict.call(agent, state, reduced, predictOptions), shortlist };
}
export function embedFnFromAgent(agent, options = {}) {
  if (typeof agent.embed !== 'function') throw new TypeError('agent 必须支持编码器 embedding');
  return (texts) => agent.embed(texts, options);
}
