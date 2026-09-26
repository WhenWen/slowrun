"""Full-model CUDA replay diagnostic, separate from the training smoke.

Force the generator gate open to test compiled BF16 gradients immediately. This
does not measure the natural gate, optimizer memory, convergence, or track timing.
Run on an otherwise idle GPU inside an authorized allocation.
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import time
import types

import torch

from professor_forcing import ProfessorForcing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['fa2', 'fa3'], required=True)
    parser.add_argument('--activation-checkpointing', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1].joinpath('train.py').read_text()
    module = types.ModuleType('slowrun_gpu_diagnostic')
    sys.modules[module.__name__] = module
    old_argv = sys.argv
    try:
        sys.argv = ['train.py', '--attention-backend', args.backend]
        exec(compile(ast.parse(source.split('# Compute init\n')[0]),
                     'train.py', 'exec'), module.__dict__)
    finally:
        sys.argv = old_argv
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    torch.manual_seed(42)
    config = module.GPTConfig(vocab_size=50257, dropout=.05,
                             use_iha=True, iha_mix_v=True,
                             activation_checkpointing=args.activation_checkpointing)
    with torch.device('meta'):
        original = module.GPT(config)
    original.to_empty(device=device)
    original.init_weights()
    model = torch.compile(original, dynamic=False)
    pf = ProfessorForcing(config.n_embd, device, batch=1,
                          generator_min_accuracy=-1)
    tokens = torch.randint(50257, (1, 80), device=device)
    print(json.dumps({'phase': 'model_ready',
                      'parameters': sum(p.numel() for p in original.parameters()),
                      'activation_checkpointing': args.activation_checkpointing,
                      'forced_generator_gate': True}), flush=True)
    for iteration in range(2):
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            metrics = pf.backward(model, tokens, .02)
        assert metrics['pf_g_enabled'] == 1
        assert model.training
        checks = []
        for name, param in original.transformer.h.named_parameters():
            assert param.grad is not None, name
            checks.append(torch.isfinite(param.grad).all())
        assert torch.stack(checks).all().item()
        gradient_norm = original.transformer.h[0].attn.c_q.weight.grad.norm().item()
        assert gradient_norm > 0
        assert original.lm_head.weight.grad is None
        assert all(p.grad is None for p in pf.discriminator.parameters())
        torch.cuda.synchronize()
        print(json.dumps({'iteration': iteration, 'status': 'PASS',
                          'seconds_including_compile': time.perf_counter() - start,
                          'first_attention_gradient_norm': gradient_norm,
                          'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                          **{k: v.item() if isinstance(v, torch.Tensor) else v
                             for k, v in metrics.items()}}), flush=True)
        original.zero_grad(set_to_none=True)
    print('COMPILED_PF_REPLAY_PASS', flush=True)


if __name__ == '__main__':
    main()
