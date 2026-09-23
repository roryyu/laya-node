import { readFile } from 'node:fs/promises';
import { load, DEFAULT_MODELS } from '../src/index.js';

// 首次导出：.venv/bin/python tools/export_onnx.py --checkpoint english --output models/english
// 运行：node examples/predict.js [已导出模型目录]
const request = JSON.parse(await readFile(new URL('./request.json', import.meta.url), 'utf8'));
const agent = await load(process.argv[2] ?? DEFAULT_MODELS.english);
try {
  if (agent.metadata.tiny) console.error('当前是随机小模型，只验证技术链路，不具备真实语义能力。');
  console.log(JSON.stringify(await agent.predict(request.state, request.questions), null, 2));
} finally { await agent.dispose(); }
