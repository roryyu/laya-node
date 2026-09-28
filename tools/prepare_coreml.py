#!/usr/bin/env python3
"""把导出模型改写为 CoreML EP 可用版本：就地改写 onnx/model.onnx（无损）。

背景：ModernBERT 注意力的掩码相加是 4D 广播（scores [n,h,L,L] + mask [n,1,?,?]）。
ONNX Runtime 的 CoreML EP 静态检查会把该子图认领走，但 CoreML NeuralNetwork/MLProgram
运行时不支持这种 4D 广播，推理直接报 error code -1（已用 CoreML 原生 API 确认）。
改写方法：在每个 layers.X/attn/Add_2 之前插入 Shape(scores) + Expand(mask, shape)，
让 Add 两个输入同形。Shape/Expand 因 int64 shape 输入留在 CPU 执行，Expand 的 float
输出再进入 CoreML 子图；数值与原模型逐位一致，CPU 推理性能不变。

用法：.venv/bin/python tools/prepare_coreml.py models/english
幂等：已改写的模型重复执行会直接跳过。
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, shape_inference
import onnxruntime as ort

DTYPE_MAP = {'input_ids': np.int64, 'attention_mask': np.int64, 'marker_pos': np.int64,
             'marker_mask': np.bool_, 'qtype': np.int64}
OUTPUTS = ['logits', 'act_logits', 'embeddings']
PATCH_MARK = 'CoreMLMaskExpand'
CONFIG_MARK = 'coreml_patched'


def rewrite_graph(model):
    graph = model.graph
    producer = {o: n for n in graph.node for o in n.output}
    patched = 0
    new_nodes = []
    # 自上而下遍历，新插入的 Shape/Expand 紧跟在对应 Add_2 前；拓扑上 scores/mask 此时均已就绪。
    for node in graph.node:
        if node.op_type == 'Add' and node.name.endswith('/attn/Add_2'):
            scores, mask = node.input[0], node.input[1]
            prod = producer.get(mask)
            if prod is None or prod.op_type != 'Where':
                raise SystemExit(f'{node.name} 的掩码输入不是 Where 输出，改写规则需要更新：{mask}')
            prefix = node.name.replace('/attn/Add_2', '/attn')
            shape_name = f'{prefix}/{PATCH_MARK}Shape'
            expand_name = f'{prefix}/{PATCH_MARK}'
            shape_node = helper.make_node('Shape', inputs=[scores], outputs=[shape_name], name=shape_name)
            expand_node = helper.make_node('Expand', inputs=[mask, shape_name],
                                           outputs=[f'{expand_name}_output_0'], name=expand_name)
            node.input[1] = f'{expand_name}_output_0'
            new_nodes.extend([shape_node, expand_node])
            patched += 1
        new_nodes.append(node)
    if patched == 0:
        raise SystemExit('未找到任何 layers.X/attn/Add_2 节点；模型结构可能已变化，拒绝改写')
    del graph.node[:]
    graph.node.extend(new_nodes)


def verify(model_path, rewritten_bytes):
    golden_path = model_path.parent.parent / 'golden.json'
    golden = json.loads(golden_path.read_text(encoding='utf8'))
    ref = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    new = ort.InferenceSession(rewritten_bytes, providers=['CPUExecutionProvider'])
    max_err = 0.0
    for case in golden['cases']:
        feeds = {k: np.asarray(v, dtype=DTYPE_MAP[k]) for k, v in case['inputs'].items()}
        for x, y in zip(ref.run(OUTPUTS, feeds), new.run(OUTPUTS, feeds)):
            max_err = max(max_err, float(np.abs(x - y).max()))
    print(json.dumps({'cases': len(golden['cases']), 'max_abs_error_vs_original': max_err}, ensure_ascii=False))
    if max_err != 0.0:
        raise SystemExit('改写模型与原 FP32 输出不完全一致，Expand 改写应当无损，已放弃覆盖')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help='导出目录，如 models/english')
    args = parser.parse_args()
    model_dir = Path(args.model).resolve()
    model_path = model_dir / 'onnx/model.onnx'
    config_path = model_dir / 'config.json'
    if not model_path.exists():
        raise SystemExit(f'未找到 {model_path}；先运行 tools/export_onnx.py 导出完整模型')

    model = onnx.load(str(model_path))
    already = any(n.op_type == 'Expand' and PATCH_MARK in n.name for n in model.graph.node)
    config = json.loads(config_path.read_text(encoding='utf8'))
    if already and config.get('laya', {}).get(CONFIG_MARK):
        print(f'{model_dir.name} 已是 CoreML 改写版本，无需重复执行')
        return
    if already != config.get('laya', {}).get(CONFIG_MARK, False):
        raise SystemExit('模型图与 config.json 的 coreml 标记不一致，请先用 tools/export_onnx.py 重新导出')

    rewrite_graph(model)
    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    payload = model.SerializeToString()
    print(f'已改写 {sum(1 for n in model.graph.node if PATCH_MARK in n.name) // 2} 个注意力 Add_2'
          '（插入 Shape+Expand 消除 4D 广播），开始数值验证…')
    verify(model_path, payload)

    tmp = model_path.with_suffix('.onnx.tmp')
    tmp.write_bytes(payload)
    os.replace(tmp, model_path)
    config.setdefault('laya', {})[CONFIG_MARK] = True
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # 清理早期实验产出的独立改写文件；现在改写直接内嵌在 model.onnx 中。
    legacy = model_dir / 'onnx/model_coreml.onnx'
    if legacy.exists():
        legacy.unlink()
    print(f'数值验证通过（逐位一致），已就地更新 {model_path} 并在 config.json 标记 {CONFIG_MARK}=true')


if __name__ == '__main__':
    main()
