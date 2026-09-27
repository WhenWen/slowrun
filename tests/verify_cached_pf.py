"""CUDA diagnostic for shared CE/PF prefix graphs and compiled cached decoding.

Default: small model, full 2048-token context. --full-model uses the actual 1.44B
model. Neither mode runs the optimizer or measures validation quality. The PF
generator gate is forced open to exercise its complete gradient path.
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import time
import types

import torch
from torch._dynamo.utils import counters

from professor_forcing import ProfessorForcing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['fa2', 'fa3'], required=True)
    parser.add_argument('--rollout', type=int, default=32)
    parser.add_argument('--full-model', action='store_true')
    parser.add_argument('--activation-checkpointing', action='store_true')
    args = parser.parse_args()
    if not 4 <= args.rollout < 2048:
        parser.error('rollout must be between 4 and 2047')
    module = types.ModuleType('cached_pf_gpu_diagnostic')
    sys.modules[module.__name__] = module
    source = Path(__file__).resolve().parents[1].joinpath('train.py').read_text()
    old_argv = sys.argv
    try:
        sys.argv = ['train.py', '--attention-backend', args.backend]
        exec(compile(ast.parse(source.split('# Compute init\n')[0]),
                     'train.py', 'exec'), module.__dict__)
    finally:
        sys.argv = old_argv
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    overrides = {} if args.full_model else dict(n_layer=4, n_head=4, n_kv_head=4, n_embd=256)
    vocab = 50257 if args.full_model else 257
    config = module.GPTConfig(vocab_size=vocab, dropout=.05, stoch_depth=0,
                             use_iha=True, activation_checkpointing=args.activation_checkpointing,
                             **overrides)
    with torch.device('meta'):
        original = module.GPT(config)
    original.to_empty(device=device)
    original.init_weights()
    model = torch.compile(original, dynamic=False)
    decode = torch.compile(original, dynamic=True)
    pf = ProfessorForcing(config.n_embd, device, context=2048-args.rollout,
                          rollout=args.rollout, batch=1, generator_min_accuracy=-1)
    print(json.dumps({'phase': 'model_ready', 'full_model': args.full_model,
                      'parameters': sum(p.numel() for p in original.parameters()),
                      'context_length': 2048, 'rollout': args.rollout,
                      'forced_generator_gate': True}), flush=True)
    for iteration in range(2):
        tokens = torch.randint(vocab, (1, 2049), device=device)
        graph_counts = []

        def checked_decode(*inputs, **kwargs):
            result = decode(*inputs, **kwargs)
            graph_counts.append(int(counters['stats']['unique_graphs']))
            if len(graph_counts) > 3 and graph_counts[-1] != graph_counts[2]:
                raise AssertionError(f'Decoder recompiles as the KV cache grows: {graph_counts}')
            return result

        # Preserve the explicit mode check in amortized_loss.
        checked_decode.training = original.training
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ce, _, state = model(tokens[:, :-1], tokens[:, 1:], cache_prefix=2048-args.rollout,
                                 cache_batch=1)
            torch.cuda.synchronize()
            ce_seconds = time.perf_counter() - start
            pf_start = time.perf_counter()
            loss, metrics = pf.amortized_loss(model, checked_decode, state, .02)
            torch.cuda.synchronize()
            pf_seconds = time.perf_counter() - pf_start
        (ce + loss).backward()
        torch.cuda.synchronize()
        assert metrics['pf_g_enabled'] == 1
        assert all(p.grad is None for p in pf.discriminator.parameters())
        gradients = [p.grad for p in original.parameters()]
        assert all(g is not None for g in gradients)
        assert torch.stack([torch.isfinite(g).all() for g in gradients]).all().item()
        first_grad = original.transformer.h[0].attn.c_q.weight.grad.norm().item()
        assert first_grad > 0
        print(json.dumps({'iteration': iteration, 'status': 'PASS',
                          'seconds_including_compile': time.perf_counter() - start,
                          'ce_forward_seconds': ce_seconds, 'pf_forward_seconds': pf_seconds,
                          'decode_calls': len(graph_counts), 'decode_graph_counts': graph_counts,
                          'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                          'first_attention_gradient_norm': first_grad}), flush=True)
        # Validation references must not keep the previous ~5.4 GiB gradient
        # bank (or exported prefix buffers) alive during the next iteration.
        del gradients, state, ce, loss, metrics
        original.zero_grad(set_to_none=True)
    print('CACHED_PF_JOINT_BACKWARD_PASS', flush=True)


if __name__ == '__main__':
    main()
