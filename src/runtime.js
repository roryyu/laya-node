import { access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isRecord } from './questions.js';
import { positiveInteger } from './math.js';
import { specialTokens } from './sequence.js';

export const MODEL_ROOT = fileURLToPath(new URL('../models/', import.meta.url));
export const DEFAULT_MODELS = Object.freeze(Object.fromEntries(
  ['english', 'multilingual', 'typed-decisions'].map((name) => [name, path.join(MODEL_ROOT, name)]),
));
export const INPUTS = ['input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype'];
export const OUTPUTS = ['logits', 'act_logits', 'embeddings'];

function validateConfig(config) {
  const meta = config.laya;
  if (config.model_type !== 'custom' || !isRecord(meta) || meta.format_version !== 1) {
    throw new Error('不是 laya-node v1 ONNX 制品；请先执行 tools/export_onnx.py 转换完整 Laya 模型');
  }
  const cfg = meta.agent_config;
  if (!isRecord(cfg)) throw new Error('模型缺少 agent_config');
  positiveInteger(cfg.max_len, '模型 max_len'); positiveInteger(cfg.head_max_len, '模型 head_max_len');
  positiveInteger(meta.hidden_size, '模型 hidden_size'); positiveInteger(meta.max_position_embeddings, '模型最大位置数');
  positiveInteger(meta.n_act, '模型 n_act');
  if (cfg.max_len > meta.max_position_embeddings) throw new Error('模型 token 预算超过位置上限');
  if (!Array.isArray(cfg.temperature) || cfg.temperature.length !== 3) throw new Error('模型必须包含三类温度');
  if (cfg.temperature_by_options != null && !isRecord(cfg.temperature_by_options)) throw new Error('模型温度分桶必须为对象');
  return meta;
}

export async function loadRuntime(modelPath = DEFAULT_MODELS.english, options = {}) {
  if (typeof modelPath !== 'string' || !modelPath) throw new TypeError('模型路径不能为空');
  if (options.device != null && options.device !== 'cpu') throw new RangeError('当前已验证的运行设备仅支持 CPU FP32');
  if (/^convaiinnovations\/laya(?:$|-|\/)/.test(modelPath)) throw new Error('上游仓库是 safetensors 格式，不能直接运行；请使用导出后的本地目录或自己的 ONNX Hub 仓库');
  let local = path.isAbsolute(modelPath) || modelPath.startsWith('.');
  if (!local) { try { await access(modelPath); local = true; } catch {} }
  const source = local ? path.resolve(modelPath) : modelPath;
  if (local) {
    try { await access(path.join(source, 'config.json')); }
    catch (cause) {
      const name = path.basename(source), checkpoint = Object.hasOwn(DEFAULT_MODELS, name) ? name : 'english';
      throw new Error(`未找到导出模型 ${source}；先运行 python tools/export_onnx.py --checkpoint ${checkpoint} --output "${source}"`, { cause });
    }
  }
  // 惰性加载，语言路由/数学工具无需初始化任何原生推理库。
  const { AutoConfig, AutoTokenizer, PreTrainedModel, Tensor } = await import('@huggingface/transformers');
  const common = {
    revision: options.revision ?? 'main', cache_dir: options.cacheDir,
    local_files_only: local || options.localFilesOnly === true,
    progress_callback: options.progressCallback,
  };
  const config = await AutoConfig.from_pretrained(source, common);
  const meta = validateConfig(config);
  const tokenizer = await AutoTokenizer.from_pretrained(source, common);
  specialTokens(tokenizer);
  let model;
  try {
    model = await PreTrainedModel.from_pretrained(source, {
      ...common, config, device: options.device ?? 'cpu', dtype: 'fp32',
      session_options: options.sessionOptions ?? {},
    });
    const session = model.sessions.model;
    if (!session || INPUTS.some((name) => !session.inputNames.includes(name)) ||
        session.inputNames.some((name) => !INPUTS.includes(name)) || OUTPUTS.some((name) => !session.outputNames.includes(name))) {
      throw new Error('ONNX 图的输入/输出与 Laya 协议不一致');
    }
    return {
      tokenizer, config: structuredClone(meta.agent_config), metadata: meta, source,
      async run(batch) {
        const n = batch.input_ids.length, length = batch.input_ids[0].length, k = batch.marker_pos[0].length;
        if (length > meta.max_position_embeddings) throw new RangeError('序列超过编码器最大位置数');
        const tensors = Object.fromEntries(INPUTS.map((name) => {
          const values = batch[name];
          const dims = name === 'qtype' ? [n] : [n, name.startsWith('marker') ? k : length];
          const flat = values.flat();
          return [name, name === 'marker_mask'
            ? new Tensor('bool', Uint8Array.from(flat, Number), dims)
            : new Tensor('int64', BigInt64Array.from(flat, BigInt), dims)];
        }));
        const output = await model(tensors);
        const expected = { logits: [n, k], act_logits: [n, meta.n_act], embeddings: [n, meta.hidden_size] };
        const result = {};
        for (const name of OUTPUTS) {
          const t = output[name];
          if (!t || t.dims.length !== 2 || t.dims.some((v, i) => v !== expected[name][i])) throw new Error(`ONNX ${name} shape 不匹配`);
          if (!t.data.every(Number.isFinite)) throw new Error(`ONNX ${name} 含非有限数值`);
          result[name] = Array.from({ length: n }, (_, row) => Array.from(t.data.slice(row * t.dims[1], (row + 1) * t.dims[1])));
        }
        return result;
      },
      dispose: () => model.dispose(),
    };
  } catch (error) { if (model) await model.dispose(); throw error; }
}
