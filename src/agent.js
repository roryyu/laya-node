import { loadRuntime } from './runtime.js';
import { normalizeQuestions, serializeState, renderOptions, criteriaEntries, QTYPES } from './questions.js';
import { buildSequence, collateItems, specialTokens } from './sequence.js';
import { clampTemperature, confidenceFromProbs, tempBucket, softmax, round4, positiveInteger } from './math.js';

export function formatAnswers(entries, output, cfg) {
  const answers = Object.create(null);
  entries.forEach(([id, q], row) => {
    const k = renderOptions(q).length;
    const scale = clampTemperature(cfg.temperature_by_options?.[tempBucket(q.t, k)] ?? cfg.temperature?.[QTYPES[q.t]] ?? 1);
    const p = softmax(output.logits[row].slice(0, k), scale);
    const confidence = round4(confidenceFromProbs(p));
    const action = { act_probability: round4(softmax(output.act_logits[row])[0]) };
    if (q.t === 'choice') {
      const labels = criteriaEntries(q.crit).map(([label]) => label);
      const best = p.indexOf(Math.max(...p));
      answers[id] = { type: 'choice', choice: labels[best], probabilities: Object.fromEntries(labels.map((label, i) => [label, round4(p[i])])), confidence, action };
    } else if (q.t === 'score') {
      answers[id] = { type: 'score', score: round4(p.reduce((sum, v, i) => sum + i * v, 0)),
        legend: Object.fromEntries(q.crit.map((c, i) => [String(i), c])),
        probabilities: Object.fromEntries(p.map((v, i) => [String(i), round4(v)])), confidence, action };
    } else {
      answers[id] = { type: 'noul', noul: round4(p[1]), confidence: round4(Math.max(p[1], 1 - p[1])), action };
    }
  });
  return answers;
}

export class Agent {
  #runtime;
  #pending = new Set();
  #closing = false;
  #disposal;
  constructor(runtime) {
    if (!runtime?.run || !runtime?.tokenizer) throw new TypeError('请使用 await load(modelPath) 或 Agent.load() 创建 Agent');
    this.#runtime = runtime;
    this.cfg = structuredClone(runtime.config);
    this.metadata = structuredClone(runtime.metadata);
    this.source = runtime.source;
    this.tok = runtime.tokenizer;
    this.temperature_raw = [...(this.cfg.temperature ?? [1, 1, 1])];
    this.temperature_by_options_raw = { ...this.cfg.temperature_by_options };
    this.temperature = this.temperature_raw.map((x) => clampTemperature(x));
    this.temperature_by_options = Object.fromEntries(Object.entries(this.temperature_by_options_raw).map(([k, v]) => [k, clampTemperature(v)]));
    const values = [...this.temperature_raw, ...Object.values(this.temperature_by_options_raw)];
    if (values.some((v) => !Number.isFinite(Number(v)) || clampTemperature(v) !== Number(v))) {
      process.emitWarning('checkpoint 温度超出 [0.5,5] 或无效，已限幅；受影响置信度不能视为已校准', { code: 'LAYA_TEMPERATURE_CLAMPED' });
    }
  }
  static async load(modelPath, options) { return new Agent(await loadRuntime(modelPath, options)); }
  prepare(state, questions, options = {}) {
    if (this.#closing) throw new Error('Agent 已释放或正在释放');
    serializeState(state);
    const entries = normalizeQuestions(questions);
    const maxLength = options.maxLength ?? this.cfg.max_len;
    const headMaxLength = options.headMaxLength ?? this.cfg.head_max_len;
    positiveInteger(maxLength, 'maxLength');
    if (maxLength > this.metadata.max_position_embeddings) throw new RangeError('maxLength 超过编码器最大位置数');
    const items = entries.map(([, q]) => buildSequence(this.tok, state, q, { maxLength, headMaxLength, truncateLeft: options.truncateLeft }));
    return { entries, items, batch: items.length ? collateItems(items, specialTokens(this.tok).pad) : null };
  }
  #use(operation) {
    if (this.#closing) return Promise.reject(new Error('Agent 已释放或正在释放'));
    const task = Promise.resolve().then(operation);
    this.#pending.add(task);
    return task.finally(() => this.#pending.delete(task));
  }
  async predict(state, questions, options = {}) {
    const prepared = this.prepare(state, questions, options);
    return this.#use(async () => {
      const output = prepared.batch ? await this.#runtime.run(prepared.batch) : null;
      return { model: 'laya-rl-agent', answers: output ? formatAnswers(prepared.entries, output, this.cfg) : {},
        usage: { input_tokens: prepared.items.reduce((n, item) => n + item.ids.length, 0), output_tokens: 0 } };
    });
  }
  async predictRaw(state, questions, options = {}) {
    const prepared = this.prepare(state, questions, options);
    return this.#use(async () => ({ ...prepared, output: prepared.batch ? await this.#runtime.run(prepared.batch) : null }));
  }
  systemOne(...args) { return this.predict(...args); }
  system_one(...args) { return this.predict(...args); }

  async embed(texts, { maxLength = 512, batchSize = 32 } = {}) {
    positiveInteger(maxLength, 'maxLength'); positiveInteger(batchSize, 'batchSize');
    if (maxLength < 2) throw new RangeError('embedding maxLength 至少为 2，以保留起止特殊 token');
    if (maxLength > this.metadata.max_position_embeddings) throw new RangeError('embedding maxLength 超过编码器上限');
    if (!Array.isArray(texts) || texts.some((x) => typeof x !== 'string')) throw new TypeError('texts 必须为字符串数组');
    return this.#use(async () => {
      const result = [];
      for (let start = 0; start < texts.length; start += batchSize) {
        const items = texts.slice(start, start + batchSize).map((text) => {
          const ids = Array.from(this.tok.encode(text, { add_special_tokens: true }));
          // ModernBERT/mmBERT 的起止 token 不参与正文截断，与 Python fast tokenizer 一致。
          if (ids.length > maxLength) ids.splice(maxLength - 1, ids.length - maxLength + 1, ids.at(-1));
          return { ids, markers: [0], qtype: 0 };
        });
        const output = await this.#runtime.run(collateItems(items, specialTokens(this.tok).pad));
        result.push(...output.embeddings);
      }
      return result;
    });
  }
  dispose() {
    this.#closing = true;
    return this.#disposal ??= Promise.allSettled([...this.#pending]).then(() => this.#runtime.dispose());
  }
}
export const RLAgent = Agent;
export const load = (modelPath, options) => Agent.load(modelPath, options);
