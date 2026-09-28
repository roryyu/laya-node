import { readFile } from 'node:fs/promises';
import { load, DEFAULT_MODELS } from '../src/index.js';

// 首次导出：.venv/bin/python tools/export_onnx.py --checkpoint english --output models/english
// int8 量化：.venv/bin/python tools/quantize_onnx.py models/english
// CoreML：.venv/bin/python tools/prepare_coreml.py models/english（无损改写，CPU 同样可用）
// 运行：node examples/predict.js [已导出模型目录]
//   LAYA_DTYPE=q8 使用量化模型；LAYA_DEVICE=coreml 使用 CoreML EP（仅 macOS，且不与 q8 组合）
//   LAYA_THREADS=8 指定 onnxruntime 算子内线程数；Apple Silicon 上设为 P-core 数（M2 Max 为 8）比默认快 ~20%
const request = JSON.parse(await readFile(new URL('./request.json', import.meta.url), 'utf8'));
const agent = await load(process.argv[2] ?? DEFAULT_MODELS.english, {
  dtype: process.env.LAYA_DTYPE,
  device: process.env.LAYA_DEVICE,
  sessionOptions: process.env.LAYA_THREADS ? { intraOpNumThreads: Number(process.env.LAYA_THREADS) } : undefined,
});
try {
  if (agent.metadata.tiny) console.error('当前是随机小模型，只验证技术链路，不具备真实语义能力。');
  console.time('predict');
  const result = await agent.predict(request.state, request.questions);
  console.timeEnd('predict');
  console.log(JSON.stringify(result, null, 2));
  
  // === 返回值详细解释 ===
  console.log('\n=== 详细解释 ===');
  console.log('model: 使用的模型名称');
  console.log('answers: 对每个问题的预测答案');
  
  // 部门选择解释
  console.log('\n1. department (部门选择):');
  console.log(`   - type: "${result.answers.department.type}" - 选择题类型`);
  console.log(`   - choice: "${result.answers.department.choice}" - 预测应由${result.answers.department.choice === 'billing' ? '账单' : result.answers.department.choice === 'technical' ? '技术' : '其他'}团队处理`);
  console.log('   - probabilities: 各选项概率分布');
  Object.entries(result.answers.department.probabilities).forEach(([key, value]) => {
    console.log(`     * ${key}: ${value} (${(value * 100).toFixed(1)}% 概率)`);
  });
  console.log(`   - confidence: ${result.answers.department.confidence} - 置信度(${(result.answers.department.confidence * 100).toFixed(1)}%)`);
  
  // 紧急程度解释
  console.log('\n2. urgency (紧急程度):');
  console.log(`   - type: "${result.answers.urgency.type}" - 评分题类型`);
  console.log(`   - score: ${result.answers.urgency.score} - 预测紧急程度分数(0-2之间)`);
  console.log('   - legend: 分数对应含义');
  Object.entries(result.answers.urgency.legend).forEach(([key, value]) => {
    console.log(`     * ${key}: "${value}"`);
  });
  console.log('   - probabilities: 各分数概率分布');
  Object.entries(result.answers.urgency.probabilities).forEach(([key, value]) => {
    console.log(`     * ${key}: ${value} (${(value * 100).toFixed(1)}% 概率)`);
  });
  console.log(`   - confidence: ${result.answers.urgency.confidence} - 置信度(${(result.answers.urgency.confidence * 100).toFixed(1)}%)`);
  
  // 退款问题解释
  console.log('\n3. refund (退款问题):');
  console.log(`   - type: "${result.answers.refund.type}" - 是非题类型`);
  console.log(`   - noul: ${result.answers.refund.noul} - 预测客户要求退款的概率(${(result.answers.refund.noul * 100).toFixed(1)}%)`);
  console.log(`   - confidence: ${result.answers.refund.confidence} - 置信度(${(result.answers.refund.confidence * 100).toFixed(1)}%)`);
  
  // 使用统计
  console.log('\n4. usage (使用统计):');
  console.log(`   - input_tokens: ${result.usage.input_tokens} - 输入token数量`);
  console.log(`   - output_tokens: ${result.usage.output_tokens} - 输出token数量`);
} finally { await agent.dispose(); }