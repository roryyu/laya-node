#!/usr/bin/env python3
"""Laya 决策模型的纯 MLX 推理运行时（Apple Silicon Metal/AMX 加速）。

直接加载上游 safetensors 权重，数学语义与 tools/export_onnx.py 的 ExportModel 一致：
encoder(ModernBERT-large) -> mean pool(embeddings) / +type_emb -> 2x TransformerEncoder
-> gather marker -> scorer -> logits; act_head(cat([h0, top1, top1-top2, ent, k/255]))。

用法：
  .venv/bin/python tools/mlx_runtime.py validate   # 对照 models/english/golden.json
  .venv/bin/python tools/mlx_runtime.py bench      # 延迟基准（对比 ONNX CPU ~520ms）
  .venv/bin/python tools/mlx_runtime.py serve      # Node sidecar：stdin/stdout NDJSON 协议

serve 协议（每行一个 JSON）：
  就绪：<- {"ready": true, "dtype": "fp32", "model_dir": "..."}
  请求：-> {"id": 1, "inputs": {input_ids, attention_mask, marker_pos, marker_mask, qtype}}
  响应：<- {"id": 1, "outputs": {logits, act_logits, embeddings}} 或 {"id": 1, "error": "..."}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = (ROOT / '.cache/huggingface/models--convaiinnovations--laya/snapshots'
            / '1c5edc17a7acd8701df6fc341c0d179f1c62c982')

HEADS = 16
HEAD_DIM = 64
HIDDEN = 1024
N_LAYERS = 28
WINDOW = 64            # local_attention 128 -> 对称 ±64
THETA_FULL = 160000.0  # layer % 3 == 0
THETA_LOCAL = 10000.0
NORM_EPS = 1e-5
REVISION = '1c5edc17a7acd8701df6fc341c0d179f1c62c982'


def resolve_model_dir(model_dir=None):
    """显式目录优先；否则复用项目 .cache（缺失时经 huggingface_hub 下载 804MB 权重）。"""
    if model_dir:
        path = Path(model_dir)
        if not (path / 'model.safetensors').is_file():
            raise FileNotFoundError(f'{path} 下没有 model.safetensors')
        return path
    if (SNAPSHOT / 'model.safetensors').is_file():
        return SNAPSHOT
    from huggingface_hub import snapshot_download
    return Path(snapshot_download('convaiinnovations/laya', revision=REVISION,
                                  cache_dir=str(ROOT / '.cache/huggingface'),
                                  allow_patterns=['model.safetensors']))


def _ln(x, weight, bias=None):
    return mx.fast.layer_norm(x, weight, bias, NORM_EPS)


def _rope_cos_sin(length, theta):
    inv_freq = 1.0 / (theta ** (mx.arange(0, HEAD_DIM, 2).astype(mx.float32) / HEAD_DIM))
    freqs = mx.arange(length).astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)          # [L, 64]
    return mx.cos(emb)[None, None], mx.sin(emb)[None, None]  # [1,1,L,64]


def _rotate_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class LayaMLX:
    """一次加载权重，反复前向；与 ONNX 制品同精度(fp32)或 fp16。"""

    def __init__(self, model_dir=None, dtype=mx.float32):
        model_dir = resolve_model_dir(model_dir)
        raw = mx.load(str(model_dir / 'model.safetensors'))
        self.w = {k: v.astype(dtype) for k, v in raw.items() if k != 'temperature'}
        mx.eval(self.w)
        self.dtype = dtype
        self._rope = {}  # (length, theta) -> (cos, sin)

    def _rope_cache(self, length, theta):
        key = (length, theta)
        if key not in self._rope:
            self._rope[key] = _rope_cos_sin(length, theta)
        return self._rope[key]

    def _attention(self, h, prefix, layer, cos, sin, mask):
        b, length, _ = h.shape
        wqkv = self.w[f'{prefix}.layers.{layer}.attn.Wqkv.weight']
        qkv = h @ wqkv.T                                   # [b, L, 3072]
        qkv = qkv.reshape(b, length, 3, HEADS, HEAD_DIM).transpose(0, 2, 3, 1, 4)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]          # [b, 16, L, 64]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM ** -0.5, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(b, length, HIDDEN)
        return o @ self.w[f'{prefix}.layers.{layer}.attn.Wo.weight'].T

    def _mlp(self, h, prefix, layer):
        x = h @ self.w[f'{prefix}.layers.{layer}.mlp.Wi.weight'].T  # [b, L, 5248]
        inp, gate = mx.split(x, 2, axis=-1)
        return (nn.gelu(inp) * gate) @ self.w[f'{prefix}.layers.{layer}.mlp.Wo.weight'].T

    def encode(self, input_ids, attention_mask):
        w = self.w
        length = input_ids.shape[1]
        h = _ln(mx.take(w['encoder.embeddings.tok_embeddings.weight'], input_ids, axis=0),
                w['encoder.embeddings.norm.weight'])

        # 注意力加性掩码：padding + 滑窗（|i-j| <= 64 保留）
        key_pad = mx.where(attention_mask[:, None, None, :] == 0, -1e9, 0.0).astype(self.dtype)
        pos = mx.arange(length)
        band = (mx.abs(pos[:, None] - pos[None, :]) > WINDOW)
        sliding = key_pad + mx.where(band[None, None], -1e9, 0.0).astype(self.dtype)

        for layer in range(N_LAYERS):
            full = layer % 3 == 0
            cos, sin = self._rope_cache(length, THETA_FULL if full else THETA_LOCAL)
            attn_norm = w.get(f'encoder.layers.{layer}.attn_norm.weight')  # 第 0 层为 Identity
            normed = h if attn_norm is None else _ln(h, attn_norm)
            h = h + self._attention(normed, 'encoder', layer, cos, sin,
                                    key_pad if full else sliding)
            h = h + self._mlp(_ln(h, w[f'encoder.layers.{layer}.mlp_norm.weight']), 'encoder', layer)
        return _ln(h, w['encoder.final_norm.weight'])

    def _head_layer(self, h, layer, key_pad):
        p = f'head.layers.{layer}'
        w = self.w
        b, length, _ = h.shape
        n1 = _ln(h, w[f'{p}.norm1.weight'], w[f'{p}.norm1.bias'])
        qkv = n1 @ w[f'{p}.self_attn.in_proj_weight'].T + w[f'{p}.self_attn.in_proj_bias']
        qkv = qkv.reshape(b, length, 3, HEADS, HEAD_DIM).transpose(0, 2, 3, 1, 4)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM ** -0.5, mask=key_pad)
        o = o.transpose(0, 2, 1, 3).reshape(b, length, HIDDEN)
        h = h + o @ w[f'{p}.self_attn.out_proj.weight'].T + w[f'{p}.self_attn.out_proj.bias']
        n2 = _ln(h, w[f'{p}.norm2.weight'], w[f'{p}.norm2.bias'])
        ff = nn.relu(n2 @ w[f'{p}.linear1.weight'].T + w[f'{p}.linear1.bias'])
        return h + ff @ w[f'{p}.linear2.weight'].T + w[f'{p}.linear2.bias']

    def __call__(self, batch):
        w = self.w
        ids = mx.array(batch['input_ids'], mx.int32)
        mask = mx.array(batch['attention_mask'], mx.int32)
        marker_pos = mx.array(batch['marker_pos'], mx.int32)
        marker_mask = mx.array(batch['marker_mask'], mx.bool_)
        qtype = mx.array(batch['qtype'], mx.int32)

        h = self.encode(ids, mask)
        m = mask[..., None].astype(h.dtype)
        embeddings = (h * m).sum(1) / mx.maximum(m.sum(1), 1.0)
        h = h + mx.take(w['type_emb.weight'], qtype, axis=0)[:, None, :]

        key_pad = mx.where(mask[:, None, None, :] == 0, -1e9, 0.0).astype(self.dtype)
        for layer in range(2):
            h = self._head_layer(h, layer, key_pad)

        b, k = marker_pos.shape
        idx = mx.broadcast_to(mx.maximum(marker_pos, 0)[..., None], (b, k, HIDDEN))
        gathered = mx.take_along_axis(h, idx, axis=1)
        s = _ln(gathered, w['scorer.0.weight'], w['scorer.0.bias'])
        s = nn.gelu(s @ w['scorer.1.weight'].T + w['scorer.1.bias'])
        logits = (s @ w['scorer.3.weight'].T + w['scorer.3.bias']).squeeze(-1).astype(mx.float32)
        logits = mx.where(marker_mask, logits, -1e4)

        p = mx.softmax(logits, axis=-1)
        cnt = mx.maximum(marker_mask.sum(-1).astype(mx.float32), 2.0)
        ent = -(p * mx.log(mx.maximum(p, 1e-9))).sum(-1) / mx.log(cnt)
        top2 = mx.sort(mx.topk(p, 2, axis=-1), axis=-1)[:, ::-1]  # mx.topk 升序 -> 转降序
        feats = mx.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, cnt / 255.0], axis=-1)
        act_in = mx.concatenate([h[:, 0].astype(mx.float32), feats], axis=-1)
        a = nn.gelu(act_in @ w['act_head.0.weight'].T + w['act_head.0.bias'])
        act_logits = a @ w['act_head.2.weight'].T + w['act_head.2.bias']

        mx.eval(logits, act_logits, embeddings)
        return {
            'logits': logits.astype(mx.float32).tolist(),
            'act_logits': act_logits.astype(mx.float32).tolist(),
            'embeddings': embeddings.astype(mx.float32).tolist(),
        }


def _diff(got, ref):
    import numpy as np
    g, r = np.array(got, dtype=np.float64), np.array(ref, dtype=np.float64)
    assert g.shape == r.shape, f'shape {g.shape} != {r.shape}'
    abs_diff = np.abs(g - r)
    denom = np.maximum(np.abs(r), 1e-6)
    return float(abs_diff.max()), float((abs_diff / denom).max())


def cmd_validate(args):
    model = LayaMLX(args.model_dir, dtype=mx.float16 if args.fp16 else mx.float32)
    golden = json.loads((ROOT / args.golden).read_text())
    ok = True
    for case in golden['cases']:
        got = model(case['inputs'])
        ref = case['output']
        lg_abs, _ = _diff(got['logits'], ref['logits'])
        _, act_rel = _diff(got['act_logits'], ref['act_logits'])
        import numpy as np
        ge, re_ = np.array(got['embeddings']), np.array(ref['embeddings'])
        cos_rows = (ge * re_).sum(-1) / (np.linalg.norm(ge, axis=-1) * np.linalg.norm(re_, axis=-1))
        cos = float(cos_rows.min())
        status = lg_abs <= 0.25 and act_rel <= 0.05 and cos >= 0.999
        ok &= status
        print(f"{'PASS' if status else 'FAIL'} {case['name']:16s} "
              f"logits.maxabs={lg_abs:.5f} act.rel={act_rel:.5f} emb.cos={cos:.6f}")
    print('全部通过' if ok else '存在不达标用例')
    return 0 if ok else 1


def cmd_bench(args):
    import numpy as np
    golden = json.loads((ROOT / 'models/english/golden.json').read_text())
    case = next(c for c in golden['cases'] if c['name'] == 'mixed')
    model = LayaMLX(args.model_dir, dtype=mx.float16 if args.fp16 else mx.float32)
    for _ in range(args.warmup):
        model(case['inputs'])
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        model(case['inputs'])
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    n, length = np.array(case['inputs']['input_ids']).shape
    print(f'batch={n} len={length} dtype={"fp16" if args.fp16 else "fp32"} '
          f'mean={times.mean():.1f}ms p50={np.percentile(times, 50):.1f}ms '
          f'min={times.min():.1f}ms （ONNX CPU 基线 ~520ms）')
    return 0


def cmd_serve(args):
    model_dir = resolve_model_dir(args.model_dir)
    model = LayaMLX(model_dir, dtype=mx.float16 if args.fp16 else mx.float32)
    emit = lambda obj: (sys.stdout.write(json.dumps(obj) + '\n'), sys.stdout.flush())
    emit({'ready': True, 'dtype': 'fp16' if args.fp16 else 'fp32', 'model_dir': str(model_dir)})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = None
        try:
            req = json.loads(line)
            emit({'id': req.get('id'), 'outputs': model(req['inputs'])})
        except Exception as error:
            emit({'id': req.get('id') if isinstance(req, dict) else None,
                  'error': f'{type(error).__name__}: {error}'})
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, help_ in [('validate', '对照 golden.json 验证数值'), ('bench', '延迟基准'), ('serve', 'Node sidecar 服务')]:
        p = sub.add_parser(name, help=help_)
        p.add_argument('--model-dir', default=None, help='含 model.safetensors 的目录（默认项目 .cache）')
        p.add_argument('--fp16', action='store_true')
    sub.choices['validate'].add_argument('--golden', default='models/english/golden.json')
    sub.choices['bench'].add_argument('--warmup', type=int, default=3)
    sub.choices['bench'].add_argument('--iters', type=int, default=10)
    args = ap.parse_args()
    return {'validate': cmd_validate, 'bench': cmd_bench, 'serve': cmd_serve}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
