export const QTYPES = Object.freeze({ choice: 0, score: 1, noul: 2 });
export const QTYPE_NAMES = Object.freeze({ 0: 'choice', 1: 'score', 2: 'noul' });
export const isRecord = (v) => v !== null && typeof v === 'object' &&
  (Object.getPrototypeOf(v) === Object.prototype || Object.getPrototypeOf(v) === null);

/** 保留 Python json.dumps 的分隔空格，避免无意改变模型输入。 */
export function jsonText(value, ascii = false) {
  const seen = new Set();
  const quote = (text) => {
    const result = JSON.stringify(text);
    return ascii ? result.replace(/[\u007f-\uffff]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`) : result;
  };
  function visit(v) {
    if (v === null) return 'null';
    if (typeof v === 'string') return quote(v);
    if (typeof v === 'boolean') return String(v);
    if (typeof v === 'number' && Number.isFinite(v)) return JSON.stringify(v);
    if (!Array.isArray(v) && !isRecord(v)) throw new TypeError('输入必须为有限数值组成的 JSON，不能包含 undefined、BigInt 或类实例');
    if (seen.has(v)) throw new TypeError('输入不能包含循环引用');
    seen.add(v);
    const result = Array.isArray(v)
      ? `[${Array.from(v, visit).join(', ')}]`
      : `{${Object.entries(v).map(([k, val]) => `${quote(k)}: ${visit(val)}`).join(', ')}}`;
    seen.delete(v);
    return result;
  }
  return visit(value);
}

export function serializeState(state) {
  if (typeof state !== 'string' && !isRecord(state) && !Array.isArray(state)) {
    throw new TypeError('state 必须为字符串、JSON 对象或数组');
  }
  return typeof state === 'string' ? state : jsonText(state);
}
export const renderCriterion = (value) => typeof value === 'string' ? value : jsonText(value);

export function criteriaEntries(criteria) {
  let entries;
  if (Array.isArray(criteria)) {
    if (!criteria.every((c) => typeof c === 'string' && c.length > 0)) throw new TypeError('choice 列表必须由非空字符串组成');
    if (new Set(criteria).size !== criteria.length) throw new TypeError('choice 标签不能重复');
    entries = criteria.map((key) => [key, null]);
  } else if (isRecord(criteria)) {
    entries = Object.entries(criteria);
    if (entries.some(([key]) => !key)) throw new TypeError('choice 标签不能为空');
  } else throw new TypeError('choice criteria 必须为对象或字符串数组');
  if (!entries.length) throw new TypeError('choice criteria 不能为空');
  for (const [, val] of entries) renderCriterion(val);
  return entries;
}

export function normalizeQuestion(definition) {
  if (!isRecord(definition)) throw new TypeError('问题必须为对象');
  const type = definition.type ?? definition.t;
  if (!Object.hasOwn(QTYPES, type)) throw new TypeError(`不支持的问题类型：${type}`);
  const instructions = definition.instructions ?? definition.ins;
  if (instructions === undefined || instructions === null) throw new TypeError('问题缺少 instructions');
  const ins = typeof instructions === 'string' ? instructions : jsonText(instructions, true);
  let crit = definition.criteria ?? definition.crit;
  if (type === 'choice') {
    const entries = criteriaEntries(crit);
    // 数字形字符串作为对象键会被 JS 重排；列表必须保留调用者给定的候选顺序。
    crit = Array.isArray(crit) ? entries.map(([label]) => label) : Object.fromEntries(entries);
  }
  if (type === 'score') {
    if (!Array.isArray(crit) || !crit.length) throw new TypeError('score criteria 必须为非空等级数组');
    crit.forEach(renderCriterion);
  }
  if (type === 'noul') {
    crit ??= {};
    if (!isRecord(crit) || Object.keys(crit).some((k) => k !== 'true' && k !== 'false')) throw new TypeError('noul criteria 只能包含 true/false');
    Object.values(crit).forEach(renderCriterion);
  }
  return { t: type, ins, crit };
}

export function normalizeQuestions(questions) {
  if (!isRecord(questions)) throw new TypeError('questions 必须为问题 ID 到问题定义的对象');
  return Object.entries(questions).map(([id, def]) => {
    try { return [id, normalizeQuestion(def)]; }
    catch (error) { throw new TypeError(`问题 ${JSON.stringify(id)}：${error.message}`, { cause: error }); }
  });
}

export function renderOptions(definition) {
  const q = normalizeQuestion(definition);
  const text = (value, fallback) => value == null || value === '' ? fallback : renderCriterion(value);
  if (q.t === 'choice') return criteriaEntries(q.crit).map(([key, val]) => val == null || val === '' ? key : `${key}: ${renderCriterion(val)}`);
  if (q.t === 'score') return q.crit.map((c, i) => `level ${i}: ${renderCriterion(c)}`);
  return [
    `false: ${text(q.crit.false, 'no, the statement does not hold')}`,
    `true: ${text(q.crit.true, 'yes, the statement holds')}`,
  ];
}
