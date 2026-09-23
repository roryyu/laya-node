import { load } from './agent.js';
import { DEFAULT_MODELS } from './runtime.js';
import { detectLanguage } from './lang.js';
import { positiveInteger } from './math.js';

const ALIASES = new Map(Object.entries({ en: 'english', laya: 'english', default: 'english', multi: 'multilingual', ml: 'multilingual',
  'laya-multilingual': 'multilingual', typed: 'typed-decisions', typed_decisions: 'typed-decisions',
  'laya-typed-decisions': 'typed-decisions', decisions: 'typed-decisions' }));
const WORKFLOWS = {
  agent_trace_observability: ['action', 'needs_review', 'outcome', 'risk', 'urgency'],
  customer_service: ['action', 'category', 'churn_risk', 'needs_human', 'urgency'],
  invoice_processing: ['discrepancy_severity', 'disposition', 'duplicate', 'matches_order', 'urgency'],
  security_incidents: ['credential_compromise', 'disposition', 'severity', 'true_positive', 'urgency'],
};
export function normalizeModelName(name) {
  const normalized = String(name).trim().toLowerCase(), key = ALIASES.get(normalized) ?? normalized;
  if (!Object.hasOwn(DEFAULT_MODELS, key)) throw new RangeError(`未知模型 ${name}；可选 english、multilingual、typed-decisions`);
  return key;
}
export function matchTypedDecisionsWorkflow(questions) {
  const ids = new Set(Object.keys(questions ?? {}));
  return Object.entries(WORKFLOWS).find(([, keys]) => keys.length === ids.size && keys.every((key) => ids.has(key)))?.[0] ?? null;
}
export class RouteDecision {
  constructor(model, repo, reason, detection = null, workflow = null) { Object.assign(this, { model, repo, reason, detection, workflow }); }
}

export class Router {
  #slots = new Map();
  #loader;
  #closed = false;
  #disposal;
  #retiring = new Set();
  constructor({ models = {}, maxLoaded = 1, default: defaultModel = 'english', autoTaskDetection = false, loader = load, ...loadOptions } = {}) {
    this.models = { ...DEFAULT_MODELS, ...Object.fromEntries(Object.entries(models).map(([k, v]) => [normalizeModelName(k), v])) };
    this.maxLoaded = positiveInteger(maxLoaded, 'maxLoaded');
    this.default = normalizeModelName(defaultModel);
    this.autoTaskDetection = autoTaskDetection;
    this.loadOptions = loadOptions;
    this.#loader = loader;
  }
  get loaded() { return [...this.#slots].filter(([, s]) => s.agent).map(([name]) => name); }
  route(state, questions = {}, { model, task, lang } = {}) {
    const decision = (key, reason, detection, workflow) => new RouteDecision(key, this.models[key], reason, detection, workflow);
    if (model != null) return decision(normalizeModelName(model), `显式 model=${model}`);
    if (task != null) return decision(normalizeModelName(task), `显式 task=${task}`);
    const workflow = matchTypedDecisionsWorkflow(questions);
    if (workflow && this.autoTaskDetection) return decision('typed-decisions', `匹配已启用的工作流 ${workflow}`, null, workflow);
    if (lang != null) {
      const key = ['en', 'eng', 'english'].includes(String(lang).toLowerCase().split('-')[0]) ? 'english' : 'multilingual';
      return decision(key, `显式 lang=${lang}`, null, workflow);
    }
    const det = detectLanguage(state);
    if (det.script === 'unknown') return decision(this.default, '没有检测到字母，使用默认模型', det, workflow);
    if (det.script !== 'latin') return decision('multilingual', `非拉丁文字 ${det.script}`, det, workflow);
    if (!det.is_english) return decision('multilingual', `非英语拉丁文本，语言=${det.language ?? '未确定'}`, det, workflow);
    return decision('english', '英语或没有非英语证据的短拉丁文本', det, workflow);
  }
  #assertOpen() { if (this.#closed) throw new Error('Router 已释放'); }
  #slot(name) {
    this.#assertOpen();
    const key = normalizeModelName(name);
    let slot = this.#slots.get(key);
    if (!slot) {
      slot = { key, users: 0, agent: null, waiters: [], retired: false };
      slot.ready = Promise.resolve().then(() => this.#loader(this.models[key], this.loadOptions)).then((agent) => {
        slot.agent = agent; return agent;
      }).catch((error) => { if (this.#slots.get(key) === slot) this.#slots.delete(key); throw error; });
    }
    this.#slots.delete(key); this.#slots.set(key, slot);
    return slot;
  }
  #retire(slot) {
    if (slot.retired) return slot.disposal;
    slot.retired = true;
    if (this.#slots.get(slot.key) === slot) this.#slots.delete(slot.key);
    slot.disposal = (async () => {
      try { await slot.ready; } catch { return; }
      if (slot.users) await new Promise((resolve) => slot.waiters.push(resolve));
      await slot.agent.dispose?.();
    })();
    this.#retiring.add(slot.disposal);
    // 同时注册成功和失败处理，避免未观察到的 finally 派生 Promise。
    slot.disposal.then(() => this.#retiring.delete(slot.disposal), () => this.#retiring.delete(slot.disposal));
    return slot.disposal;
  }
  async #trim(protect) {
    const disposals = [];
    for (const slot of this.#slots.values()) {
      if (this.#slots.size <= this.maxLoaded) break;
      if (slot !== protect && slot.users === 0 && slot.agent) disposals.push(this.#retire(slot));
    }
    await Promise.all(disposals);
  }
  #release(slot) {
    slot.users--;
    if (slot.users === 0) for (const resolve of slot.waiters.splice(0)) resolve();
  }
  async load(name) {
    const slot = this.#slot(name); slot.users++;
    try { return await slot.ready; }
    finally { this.#release(slot); await this.#trim(slot); }
  }
  async predict(state, questions, options = {}) {
    const decision = this.route(state, questions, options), slot = this.#slot(decision.model);
    slot.users++;
    try {
      const agent = await slot.ready;
      await this.#trim(slot);
      const result = await agent.predict(state, questions, options);
      return { ...result, routing: { ...decision } };
    } finally { this.#release(slot); await this.#trim(); }
  }
  systemOne(...args) { return this.predict(...args); }
  system_one(...args) { return this.predict(...args); }
  async preload(names = Object.keys(this.models)) {
    this.#assertOpen();
    const keys = [...new Set(names.map(normalizeModelName))];
    this.maxLoaded = Math.max(this.maxLoaded, new Set([...this.#slots.keys(), ...keys]).size);
    await Promise.all(keys.map((key) => this.load(key)));
    return this;
  }
  async attach(name, agent) {
    this.#assertOpen();
    if (typeof agent?.predict !== 'function') throw new TypeError('agent 必须提供 predict');
    const key = normalizeModelName(name), old = this.#slots.get(key);
    if (old?.agent === agent) return agent;
    if ([...this.#slots.values()].some((s) => s.agent === agent)) throw new Error('同一 agent 不能挂载到两个模型名，避免重复释放');
    const retirement = old ? this.#retire(old) : Promise.resolve();
    this.#slots.set(key, { key, users: 0, agent, ready: Promise.resolve(agent), waiters: [], retired: false });
    this.maxLoaded = Math.max(this.maxLoaded, this.#slots.size);
    await retirement;
    return agent;
  }
  async unload(name) {
    const slots = name == null ? [...this.#slots.values()] : [this.#slots.get(normalizeModelName(name))].filter(Boolean);
    await Promise.all(slots.map((s) => this.#retire(s)));
  }
  dispose() {
    this.#closed = true;
    return this.#disposal ??= this.unload().then(() => Promise.all([...this.#retiring]));
  }
}
