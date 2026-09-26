"""GPU preflight: verify selected attention kernel's windows and backward math.

Run separately from unittest discovery inside an authorized GPU allocation.
"""
import argparse
import json

import torch
from torch.nn import functional as F
from kernels import get_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['fa2', 'fa3'], required=True)
    args = parser.parse_args()
    kernel = get_kernel('kernels-community/flash-attn2' if args.backend == 'fa2'
                        else 'kernels-community/flash-attn3', version=3 if args.backend == 'fa2' else 1)
    torch.manual_seed(42)
    report = []
    for length in (80, 2048):
        for window in ((-1, -1), (16, 0), (1024, 0)):
            inputs = [torch.randn(1, length, 2, 128, device='cuda', dtype=torch.bfloat16,
                                   requires_grad=True) for _ in range(3)]
            reference = [x.detach().float().requires_grad_() for x in inputs]
            position = torch.arange(length, device='cuda')
            mask = position[:, None] >= position[None, :]
            if window[0] >= 0:
                mask &= position[:, None] - position[None, :] <= window[0]
            expected = F.scaled_dot_product_attention(*(x.transpose(1, 2) for x in reference),
                                                       attn_mask=mask).transpose(1, 2)
            actual = kernel.flash_attn_func(*inputs, causal=True, window_size=window)
            torch.testing.assert_close(actual.float(), expected, atol=.02, rtol=.03)
            probe = torch.randn_like(expected)
            actual_grads = torch.autograd.grad((actual.float() * probe).sum(), inputs)
            expected_grads = torch.autograd.grad((expected * probe).sum(), reference)
            errors = []
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                assert torch.isfinite(actual_grad).all()
                relative_rms = ((actual_grad.float() - expected_grad).square().mean().sqrt()
                                / expected_grad.square().mean().sqrt().clamp_min(1e-12)).item()
                assert relative_rms < .03, relative_rms
                errors.append(relative_rms)
            report.append(dict(length=length, window=window, gradient_relative_rms=errors))
    print(json.dumps(dict(backend=args.backend, gpu=torch.cuda.get_device_name(),
                          status='PASS', cases=report), indent=2))


if __name__ == '__main__':
    main()
