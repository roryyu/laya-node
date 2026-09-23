#!/usr/bin/env python3
"""完整 Laya ONNX 导出；仅构建阶段使用，不是 Node.js 运行时依赖。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import onnx
import onnxruntime as ort
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast
from laya.agent import Agent
from laya.common import DecisionModel, QTYPES, build_sequence, collate_items, clamp_temperature

REVISION = '1c5edc17a7acd8701df6fc341c0d179f1c62c982'
INPUTS = ['input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype']
OUTPUTS = ['logits', 'act_logits', 'embeddings']


class ExportModel(nn.Module):
    """保留上游 DecisionModel 数学语义，额外暴露加入类型嵌入前的均值池化。"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        m = self.model
        h = m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        embeddings = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
        h = h + m.type_emb(qtype)[:, None, :]
        if m.head is not None:
            for layer in m.head.layers:
                # 原生 MultiheadAttention 的 legacy 导出会冻结序列长度；展开为等价基础算子。
                n, length, width = h.shape
                heads = layer.self_attn.num_heads
                dim = width // heads
                normalized = layer.norm1(h)
                qkv = F.linear(normalized, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
                q, k_attn, v = [part.reshape(n, length, heads, dim).transpose(1, 2) for part in qkv.chunk(3, -1)]
                scores = torch.matmul(q, k_attn.transpose(-2, -1)) * (layer.self_attn.head_dim ** -0.5)
                scores = scores.masked_fill(~attention_mask[:, None, None, :].bool(), float('-inf'))
                attended = torch.matmul(torch.softmax(scores, -1), v).transpose(1, 2).reshape(n, length, width)
                h = h + layer.self_attn.out_proj(attended)
                h = h + layer.linear2(layer.activation(layer.linear1(layer.norm2(h))))
        indices = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = m.scorer(torch.gather(h, 1, indices)).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits, -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        # 所有 batch 均将候选维度补齐至至少 2，单候选不触发静态 trace 分支。
        top2 = p.topk(2, -1).values
        features = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255], -1)
        action = m.act_head(torch.cat([h[:, 0].float(), features], -1))
        return logits, action, embeddings


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def tiny_agent(checkpoint):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
    from transformers import ModernBertConfig
    torch.manual_seed(1729)
    backend = Tokenizer(models.BPE(unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator([
        'choice score noul question level false true yes no statement holds',
        'Which team handles invoices payments refunds bugs billing support?',
        'I was charged twice please refund today. This is not spam.',
        '中文：发票重复扣款，请退款。 Français: être facturé. 日本語と絵文字🙂',
    ], trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                          special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']))
    backend.post_processor = processors.TemplateProcessing(
        single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, cls_token='[CLS]', sep_token='[SEP]',
                                mask_token='[MASK]', pad_token='[PAD]', unk_token='[UNK]')
    config = ModernBertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=3,
                             num_attention_heads=4, intermediate_size=48, max_position_embeddings=1024,
                             local_attention=8, global_attn_every_n_layers=3, pad_token_id=0,
                             reference_compile=False, attention_dropout=0, embedding_dropout=0,
                             mlp_dropout=0, local_rope_theta=160000 if checkpoint == 'multilingual' else 10000)
    encoder = AutoModel.from_config(config, attn_implementation='eager')
    model = DecisionModel(encoder, head_layers=2, n_act=2).eval().float()
    cfg = {'encoder': 'tiny-modernbert', 'head_layers': 2, 'max_len': 192, 'head_max_len': 128,
           'temperature': [1.6, 1.25, 1.98], 'temperature_by_options': {'choice:2': 1.9},
           'act_costs': {'escalate': 0.5}}
    return make_agent(model, tok, cfg)


def make_agent(model, tokenizer, cfg):
    agent = Agent.__new__(Agent)
    agent.model, agent.tok, agent.cfg = model, tokenizer, cfg
    agent.device, agent.dtype = torch.device('cpu'), torch.float32
    agent.temperature = [clamp_temperature(t) for t in cfg.get('temperature', [1, 1, 1])]
    agent.temperature_by_options = {k: clamp_temperature(v) for k, v in cfg.get('temperature_by_options', {}).items()}
    return agent


def real_agent(args):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    sub = '' if args.checkpoint == 'english' else args.checkpoint + '/'
    if args.source:
        source = Path(args.source).resolve()
    else:
        root = snapshot_download('convaiinnovations/laya', revision=args.revision,
                                 cache_dir=str(Path(args.cache_dir).resolve()),
                                 allow_patterns=[sub + p for p in ['rl_agent_config.json', 'model.safetensors', 'encoder/config.json', 'tokenizer/*']])
        source = Path(root) / sub
    cfg = json.loads((source / 'rl_agent_config.json').read_text())
    cfg.setdefault('temperature', [1, 1, 1])
    cfg.setdefault('max_len', 512)
    cfg.setdefault('head_max_len', 192)
    config = AutoConfig.from_pretrained(source / 'encoder', local_files_only=True)
    if config.model_type != 'modernbert':
        raise ValueError('只支持经验证的 ModernBERT/mmBERT 编码器')
    config.reference_compile = False
    encoder = AutoModel.from_config(config, attn_implementation='eager')
    model = DecisionModel(encoder, cfg['head_layers'], len(cfg.get('act_costs', {})) + 1)
    model.load_state_dict(load_file(source / 'model.safetensors'), strict=True)
    model.eval().float()
    # 不改写源 tokenizer_config；直接从 tokenizer.json 和特殊 token 配置构建。
    tcfg = json.loads((source / 'tokenizer/tokenizer_config.json').read_text())
    special = {k: v for k, v in tcfg.items() if k.endswith('_token')}
    tok = PreTrainedTokenizerFast(tokenizer_file=str(source / 'tokenizer/tokenizer.json'), **special)
    return make_agent(model, tok, cfg)


def batch_for(agent, state, questions, options):
    items = []
    for qdef in questions.values():
        q = agent._to_internal(qdef)
        ids, markers = build_sequence(agent.tok, state, q,
                                      options.get('maxLength', agent.cfg['max_len']),
                                      options.get('headMaxLength', agent.cfg['head_max_len']),
                                      truncate_left=options.get('truncateLeft', False))
        items.append({'ids': ids, 'markers': markers, 'qtype': QTYPES[q['t']]})
    b = collate_items([items], agent.tok.pad_token_id)
    if b['marker_pos'].shape[1] < 2:
        n = b['marker_pos'].shape[0]
        b['marker_pos'] = torch.cat([b['marker_pos'], torch.zeros((n, 1), dtype=torch.long)], 1)
        b['marker_mask'] = torch.cat([b['marker_mask'], torch.zeros((n, 1), dtype=torch.bool)], 1)
    return {k: b[k] for k in INPUTS}


def cases():
    choice = {'type': 'choice', 'instructions': 'Which team should handle this?',
              'criteria': {'billing': {'description': 'invoices and refunds'}, 'tech': 'bugs', 'zero': 0, 'no': False}}
    score = {'type': 'score', 'instructions': 'How urgent is this?', 'criteria': ['calm', 'soon', 'critical']}
    noul = {'type': 'noul', 'instructions': 'Is a refund requested?', 'criteria': {'true': ['refund'], 'false': {'desc': 'none'}}}
    single = {'type': 'choice', 'instructions': 'Pick one', 'criteria': ['only']}
    return [
        {'name': 'mixed', 'state': {'body': 'I was charged twice, please refund today.'},
         'questions': {'department': choice, 'urgency': score, 'refund': noul, 'single': single}},
        {'name': 'single', 'state': 'Short', 'questions': {'one': single}},
        {'name': 'unicode', 'state': {'body': '发票重复扣款，请退款。🙂 café', 'turns': ['你好', 'hello']}, 'questions': {'a': noul, 'b': choice}},
        {'name': 'truncated-right', 'state': 'refund later. ' * 80, 'questions': {'a': noul}, 'options': {'maxLength': 96, 'headMaxLength': 64}},
        {'name': 'truncated-left', 'state': 'refund later. ' * 80, 'questions': {'a': noul}, 'options': {'maxLength': 96, 'headMaxLength': 64, 'truncateLeft': True}},
        {'name': 'many-options', 'state': 'payments', 'questions': {'a': {'type': 'choice', 'instructions': 'Choose', 'criteria': [str(i) for i in range(12)]}}},
        {'name': 'numeric-label-order', 'state': 'Pick ten', 'questions': {'a': {'type': 'choice', 'instructions': 'Choose', 'criteria': ['10', '2', '0']}}},
    ]


def verify_and_golden(agent, model_path, output):
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    golden = []
    for case in cases():
        opts = case.get('options', {})
        b = batch_for(agent, case['state'], case['questions'], opts)
        with torch.no_grad():
            reference_logits, reference_action = agent.model(**b)
            h = agent.model.encoder(input_ids=b['input_ids'], attention_mask=b['attention_mask']).last_hidden_state
            mask = b['attention_mask'].unsqueeze(-1).to(h.dtype)
            reference_embedding = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
            reference = [reference_logits, reference_action, reference_embedding]
            actual = session.run(OUTPUTS, {k: v.numpy() for k, v in b.items()})
            for name, expected, got in zip(OUTPUTS, reference, actual):
                np.testing.assert_allclose(got, expected.numpy(), atol=1e-4, rtol=1e-3, err_msg=f'{case["name"]}/{name}')
            # 上游公开接口不接受截断参数；有自定义预算时仅比较原始张量。
            answers = agent.system_one(case['state'], case['questions']) if not opts else None
        golden.append({**case, 'inputs': {k: v.tolist() for k, v in b.items()},
                       'output': {name: value.tolist() for name, value in zip(OUTPUTS, reference)}, 'result': answers})
    embedding_texts = ['hello world', '中文退款🙂', '', 'A long refund request. 中文退款。 ' * 100]
    encoded = agent.tok(embedding_texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    with torch.no_grad():
        h = agent.model.encoder(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask).last_hidden_state
        mask = encoded.attention_mask.unsqueeze(-1).to(h.dtype)
        embeddings = ((h * mask).sum(1) / mask.sum(1).clamp_min(1)).tolist()
    write_json(output / 'golden.json', {'cases': golden, 'embedding': {
        'texts': embedding_texts, 'values': embeddings,
        'inputs': {k: encoded[k].tolist() for k in ['input_ids', 'attention_mask']}}})
    return len(golden)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=['english', 'multilingual', 'typed-decisions'], default='english')
    parser.add_argument('--source', help='已有原始 checkpoint 目录，直接包含 rl_agent_config.json')
    parser.add_argument('--output', required=True, help='全新的导出目录，禁止覆盖已有目录')
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--cache-dir', default=str(Path(__file__).resolve().parents[1] / '.cache/huggingface'))
    parser.add_argument('--tiny', action='store_true', help='离线生成小型随机模型，仅用于协议和数值验证')
    parser.add_argument('--verify-only', action='store_true', help='重验已有 ONNX，仅更新 golden 和 manifest，不覆盖图或 checkpoint')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.verify_only:
        existing = json.loads((output / 'config.json').read_text())['laya']
        if (existing['checkpoint'], existing['tiny'], existing['revision']) != (args.checkpoint, args.tiny, args.revision):
            parser.error('重验参数与现有制品 checkpoint、tiny、revision 不一致')
    elif output.exists():
        parser.error('输出目录已经存在，请使用新目录，避免覆盖已有制品')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.mha.set_fastpath_enabled(False)
    started = time.perf_counter()
    agent = tiny_agent(args.checkpoint) if args.tiny else real_agent(args)
    model_path = output / 'onnx/model.onnx'
    if args.verify_only:
        if args.tiny:
            agent.tok = PreTrainedTokenizerFast.from_pretrained(output, local_files_only=True)
        onnx.checker.check_model(str(model_path))
        count = verify_and_golden(agent, model_path, output)
        write_manifest(output, args, count, started)
        return
    wrapper = ExportModel(agent.model).eval()
    output.mkdir(parents=True)
    (output / 'onnx').mkdir()
    agent.tok.save_pretrained(output)
    tcfg_path = output / 'tokenizer_config.json'
    tcfg = json.loads(tcfg_path.read_text())
    tcfg['tokenizer_class'] = 'PreTrainedTokenizerFast'
    write_json(tcfg_path, tcfg)
    write_json(output / 'rl_agent_config.json', agent.cfg)
    config = agent.model.encoder.config
    metadata = {'format_version': 1, 'checkpoint': args.checkpoint, 'tiny': args.tiny,
                'source': args.source or 'convaiinnovations/laya', 'revision': args.revision,
                'hidden_size': config.hidden_size, 'max_position_embeddings': config.max_position_embeddings,
                'n_act': agent.model.act_head[-1].out_features, 'agent_config': agent.cfg}
    write_json(output / 'config.json', {'model_type': 'custom', 'laya': metadata})
    example = cases()[0]
    batch = batch_for(agent, example['state'], example['questions'], {})
    model_path = output / 'onnx/model.onnx'
    axes = {name: {0: 'batch', 1: 'sequence' if name in ['input_ids', 'attention_mask'] else 'options'} for name in INPUTS[:-1]}
    axes['qtype'] = {0: 'batch'}
    axes.update({'logits': {0: 'batch', 1: 'options'}, 'act_logits': {0: 'batch'}, 'embeddings': {0: 'batch'}})
    with torch.no_grad():
        torch.onnx.export(wrapper, tuple(batch[k] for k in INPUTS), str(model_path),
                          input_names=INPUTS, output_names=OUTPUTS, dynamic_axes=axes,
                          opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(model_path))
    count = verify_and_golden(agent, model_path, output)
    write_manifest(output, args, count, started)


def write_manifest(output, args, count, started):
    files = {}
    for p in sorted(output.rglob('*')):
        if p.is_file() and p.name != 'manifest.json':
            with p.open('rb') as f:
                files[str(p.relative_to(output))] = {'sha256': hashlib.file_digest(f, 'sha256').hexdigest(), 'bytes': p.stat().st_size}
    write_json(output / 'manifest.json', {'format_version': 1, 'source_revision': args.revision, 'files': files,
                                        'reference_laya_version': '0.3.5',
                                        'reference_commit': '573e5b62696ba441230cd6be71d593331b5d23af',
                                        'runtime': {'torch': torch.__version__, 'onnxruntime': ort.__version__},
                                        'verification_only': args.verify_only,
                                        'verification': {'cases': count, 'atol': 1e-4, 'rtol': 1e-3},
                                        'elapsed_seconds': time.perf_counter() - started})
    print(json.dumps({'output': str(output), 'tiny': args.tiny, 'verified_cases': count}, ensure_ascii=False))


if __name__ == '__main__':
    main()
