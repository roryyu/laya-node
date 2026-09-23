import { normalizeQuestion, renderOptions, serializeState, QTYPES } from './questions.js';
import { positiveInteger } from './math.js';

export function encode(tokenizer, text) {
  return Array.from(tokenizer.encode(text, { add_special_tokens: false }), Number);
}
export function specialTokens(tokenizer) {
  const values = {};
  for (const name of ['cls', 'sep', 'mask', 'pad']) {
    let id = tokenizer[`${name}_token_id`];
    if (id == null) {
      const configured = tokenizer.config?.[`${name}_token`];
      const text = typeof configured === 'string' ? configured : configured?.content;
      const encoded = text ? encode(tokenizer, text) : [];
      if (encoded.length === 1) id = encoded[0];
    }
    if (!Number.isInteger(id) || id < 0) throw new Error(`分词器缺少 ${name}_token_id`);
    values[name] = id;
  }
  if (!tokenizer.mask_token) throw new Error('分词器缺少 mask_token');
  return values;
}

export function buildSequence(tokenizer, state, definition, { maxLength = 512, headMaxLength = 192, truncateLeft = false, optionOrder } = {}) {
  positiveInteger(maxLength, 'maxLength'); positiveInteger(headMaxLength, 'headMaxLength');
  const q = normalizeQuestion(definition), tokens = specialTokens(tokenizer), opts = renderOptions(q);
  const clean = (s) => s.split(tokenizer.mask_token).join(' ');
  const order = optionOrder ?? opts.map((_, i) => i);
  if (order.length !== opts.length || new Set(order).size !== order.length || order.some((i) => !Number.isInteger(i) || i < 0 || i >= opts.length)) throw new RangeError('optionOrder 必须是候选索引的完整排列');
  let head = encode(tokenizer, `${q.t} question: ${clean(q.ins)}`);
  let options = order.map((i) => [tokens.mask, ...encode(tokenizer, ` ${clean(opts[i])}`).slice(0, 48)]);
  let budget = headMaxLength - options.reduce((n, a) => n + a.length, 0);
  if (budget < 16) {
    const per = Math.max(4, Math.floor((headMaxLength - 16) / Math.max(1, options.length)));
    options = options.map((a) => a.slice(0, per));
    budget = headMaxLength - options.reduce((n, a) => n + a.length, 0);
  }
  head = head.slice(0, Math.max(8, budget));
  // 超预算时明确拒绝，不能静默丢掉后面的候选或正文。
  const optionSize = options.reduce((n, a) => n + a.length, 0);
  if (head.length + optionSize > headMaxLength || head.length + optionSize + 4 > maxLength) {
    throw new RangeError(`候选超过 token 预算（headMaxLength=${headMaxLength}，maxLength=${maxLength}），请提高预算或使用 shortlist`);
  }
  const ids = [tokens.cls, ...head, tokens.sep], markers = [];
  for (const option of options) { markers.push(ids.length); ids.push(...option); }
  ids.push(tokens.sep);
  const room = Math.max(0, maxLength - ids.length - 1);
  const body = encode(tokenizer, clean(serializeState(state)));
  ids.push(...(room === 0 ? [] : truncateLeft ? body.slice(-room) : body.slice(0, room)), tokens.sep);
  return { ids, markers, qtype: QTYPES[q.t] };
}

/** 纯数组协议方便单测；Tensor 的创建只在运行时适配层进行。 */
export function collateItems(items, padId) {
  if (!items.length) throw new RangeError('不能整理空 batch');
  const length = Math.max(...items.map((i) => i.ids.length));
  const k = Math.max(2, ...items.map((i) => i.markers.length));
  return {
    input_ids: items.map((i) => [...i.ids, ...Array(length - i.ids.length).fill(padId)]),
    attention_mask: items.map((i) => [...Array(i.ids.length).fill(1), ...Array(length - i.ids.length).fill(0)]),
    marker_pos: items.map((i) => [...i.markers, ...Array(k - i.markers.length).fill(0)]),
    marker_mask: items.map((i) => [...Array(i.markers.length).fill(true), ...Array(k - i.markers.length).fill(false)]),
    qtype: items.map((i) => i.qtype),
  };
}
