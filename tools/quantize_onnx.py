#!/usr/bin/env python3
"""对导出目录中的 onnx/model.onnx 做权重量化，仅构建阶段使用。
默认 q8：int8 weight-only（MatMulNBits，块大小 128、对称、fp32 计算精度），产出 onnx/model_quantized.onnx；
激活保持 fp32，不受 ModernBERT GeGLU 激活离群值影响，无需校准；该模型上数值达标（logits 误差 <0.2，embedding cos>0.999）。
--mode q4：int4 weight-only，产出 onnx/model_q4.onnx；该模型上误差超差（embedding cos≈0.96），仅作对照。
--mode dynamic：int8 动态量化（权重+激活）；该模型上决策误差大，仅作对照。
产出后用 golden.json 的完整输入对比 FP32 与量化会话输出，超差即失败。
注意：若先跑过 prepare_coreml.py（model.onnx 含 Expand 改写），量化产物同样兼容 CoreML 广播改写。"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig

# 决策 logits 的量级在 ±20 内，softmax 前允许 0.25 的绝对漂移；embedding 用余弦相似度约束。
LOGIT_ATOL = 0.25
EMBED_COS_MIN = 0.999
DTYPE_MAP = {'input_ids': np.int64, 'attention_mask': np.int64, 'marker_pos': np.int64,
             'marker_mask': np.bool_, 'qtype': np.int64}


def quantize(model_dir, mode):
    source = model_dir / 'onnx/model.onnx'
    if not source.exists():
        raise SystemExit(f'未找到 {source}；先运行 tools/export_onnx.py 导出完整模型')
    started = time.perf_counter()
    if mode == 'q4':
        target = model_dir / 'onnx/model_q4.onnx'
        import onnx
        quantizer = MatMulNBitsQuantizer(onnx.load(str(source)), block_size=32, is_symmetric=True,
                                         accuracy_level=4,
                                         algo_config=DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=4))
        quantizer.process()
        quantizer.model.save_model_to_file(str(target), use_external_data_format=False)
    elif mode == 'q8':
        target = model_dir / 'onnx/model_quantized.onnx'
        import onnx
        quantizer = MatMulNBitsQuantizer(onnx.load(str(source)), block_size=128, is_symmetric=True,
                                         accuracy_level=4,
                                         algo_config=DefaultWeightOnlyQuantConfig(block_size=128, is_symmetric=True, bits=8))
        quantizer.process()
        quantizer.model.save_model_to_file(str(target), use_external_data_format=False)
    else:
        target = model_dir / 'onnx/model_quantized.onnx'
        quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)
    return target, source.stat().st_size, target.stat().st_size, time.perf_counter() - started


def verify(model_dir, target):
    golden = json.loads((model_dir / 'golden.json').read_text(encoding='utf8'))
    reference = ort.InferenceSession(str(model_dir / 'onnx/model.onnx'), providers=['CPUExecutionProvider'])
    quantized = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
    outputs = ['logits', 'act_logits', 'embeddings']
    max_logit_err, max_act_rel, min_cos = 0.0, 0.0, 1.0
    for case in golden['cases']:
        feeds = {k: np.asarray(v, dtype=DTYPE_MAP[k]) for k, v in case['inputs'].items()}
        expected = reference.run(outputs, feeds)
        actual = quantized.run(outputs, feeds)
        max_logit_err = max(max_logit_err, float(np.abs(expected[0] - actual[0]).max()))
        # act_logits 的绝对量级在 10^3，用相对误差约束
        rel = np.abs(expected[1] - actual[1]) / np.maximum(np.abs(expected[1]), 1.0)
        max_act_rel = max(max_act_rel, float(rel.max()))
        dot = (expected[2] * actual[2]).sum(-1)
        cos = dot / (np.linalg.norm(expected[2], axis=-1) * np.linalg.norm(actual[2], axis=-1) + 1e-12)
        min_cos = min(min_cos, float(cos.min()))
    print(json.dumps({'quantized': str(target), 'cases': len(golden['cases']),
                      'max_logit_abs_error': round(max_logit_err, 6),
                      'max_act_logit_rel_error': round(max_act_rel, 6),
                      'min_embedding_cosine': round(min_cos, 6)}, ensure_ascii=False))
    if max_logit_err > LOGIT_ATOL or max_act_rel > 0.05 or min_cos < EMBED_COS_MIN:
        raise SystemExit(f'量化误差超差：logits atol<={LOGIT_ATOL}，act rel<=5%，embedding cos>={EMBED_COS_MIN}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='导出目录，如 models/english')
    parser.add_argument('--mode', choices=['q4', 'q8', 'dynamic'], default='q8')
    args = parser.parse_args()
    model_dir = Path(args.model).resolve()
    target, before, after, elapsed = quantize(model_dir, args.mode)
    print(f'{model_dir.name} ({args.mode}): {before / 2**20:.0f} MiB -> {after / 2**20:.0f} MiB，量化耗时 {elapsed:.1f}s')
    verify(model_dir, target)


if __name__ == '__main__':
    main()
