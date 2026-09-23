import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, writeFile, cp, access, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { load } from '../src/index.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const tiny = path.join(root, 'artifacts/tiny-english-v2');
const cache = path.join(root, '.cache/tests');
async function temporary(t) {
  await mkdir(cache, { recursive: true });
  const directory = await mkdtemp(path.join(cache, 'runtime-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}
test('缺失路径、原始仓库和不支持的设备明确拒绝', async () => {
  await assert.rejects(load(path.join(root, 'artifacts/nonexistent')), /先运行.*export_onnx/);
  await assert.rejects(load(path.join(root, 'artifacts/absent/multilingual')), /--checkpoint multilingual/);
  await assert.rejects(load('convaiinnovations/laya'), /safetensors/);
  await assert.rejects(load(tiny, { device: 'cuda' }), /仅支持 CPU/);
});
test('损坏制品配置在初始化会话前拒绝', async (t) => {
  const directory = await temporary(t);
  await writeFile(path.join(directory, 'config.json'), JSON.stringify({ model_type: 'custom', laya: { format_version: 99 } }));
  await assert.rejects(load(directory), /不是 laya-node v1/);
});
let available = true;
try { await access(path.join(tiny, 'manifest.json')); } catch { available = false; }
test('损坏 ONNX 权重不能被当作可用模型', { skip: !available && '需要 tiny 制品' }, async (t) => {
  const directory = await temporary(t);
  for (const name of ['config.json', 'tokenizer.json', 'tokenizer_config.json']) await cp(path.join(tiny, name), path.join(directory, name));
  await mkdir(path.join(directory, 'onnx'));
  await writeFile(path.join(directory, 'onnx/model.onnx'), 'not an ONNX model');
  await assert.rejects(load(directory, { localFilesOnly: true }));
});
