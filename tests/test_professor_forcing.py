"""CPU correctness tests; the real eight-GPU run is validated separately."""
import ast
from pathlib import Path
import sys
import types
import unittest
import tempfile
from datetime import timedelta

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from professor_forcing import ProfessorForcing, sample_continuation


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 8)
        self.head = nn.Linear(8, 13)
        self.seen = []

    def forward(self, tokens, return_hidden=False, logit_positions=None):
        self.seen.append(tokens.detach().clone())
        hidden = self.embedding(tokens).cumsum(1)
        if return_hidden:
            return hidden
        if logit_positions is not None:
            hidden = hidden.index_select(1, logit_positions)
        return self.head(hidden)


def load_gpt_definitions():
    # train.py is an executable script. Load its definitions without its training
    # side effects; retain the actual model code rather than duplicating it.
    source = Path(__file__).resolve().parents[1].joinpath('train.py').read_text()
    prefix = source.split('# Compute init\n')[0]
    module = types.ModuleType('slowrun_test_defs')
    sys.modules[module.__name__] = module
    old_argv = sys.argv
    try:
        sys.argv = ['train.py', '--mtp-weight', '0']
        exec(compile(ast.parse(prefix), 'train.py', 'exec'), module.__dict__)
    finally:
        sys.argv = old_argv
    def attention(q, k, v, causal=False, window_size=(-1, -1)):
        t = q.size(1)
        i = torch.arange(t)
        mask = i[:, None] >= i[None, :]
        if window_size[0] >= 0:
            mask &= i[:, None] - i[None, :] <= window_size[0]
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                              v.transpose(1, 2), attn_mask=mask).transpose(1, 2)
    module.flash_attn.flash_attn_func = attention
    return module


def distributed_pf_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        # Deliberately distinct local data/model behavior must still produce one
        # identical discriminator update and a common gate decision on both ranks.
        torch.manual_seed(100 + rank)
        model = ToyLM()
        pf = ProfessorForcing(8, torch.device('cpu'), context=2, rollout=4,
                              generator_min_accuracy=-1)
        for _ in range(2):
            metrics = pf.backward(model, torch.randint(13, (4, 6)), .02)
            flat = torch.cat([p.detach().flatten() for p in pf.discriminator.parameters()])
            gathered = [torch.empty_like(flat) for _ in range(2)]
            dist.all_gather(gathered, flat)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
            accs = [torch.zeros_like(metrics['pf_accuracy']) for _ in range(2)]
            dist.all_gather(accs, metrics['pf_accuracy'])
            torch.testing.assert_close(accs[0], accs[1], rtol=0, atol=0)
            assert metrics['pf_g_enabled'] == 1
            assert model.embedding.weight.grad.norm() > 0
            model.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group()


class ProfessorForcingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_rollout_conditions_on_previous_samples(self):
        model = ToyLM().eval()
        prompt = torch.tensor([[2, 3], [4, 5]])
        result = sample_continuation(model, prompt, 4, torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(result[:, :2], prompt))
        self.assertEqual(result.shape, (2, 6))
        for t, seen in enumerate(model.seen):
            self.assertTrue(torch.equal(seen[:, :2+t], result[:, :2+t]))
        self.assertFalse(result.requires_grad)

    def test_two_rank_discriminator_and_gate_synchronization(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_pf_worker, args=(str(Path(directory) / 'rendezvous'),),
                     nprocs=2, join=True)

    def test_discriminator_cannot_backprop_into_generator_when_gated(self):
        model = ToyLM().train()
        pf = ProfessorForcing(8, torch.device('cpu'), context=2, rollout=4,
                              generator_min_accuracy=1.0)
        before = [p.clone() for p in pf.discriminator.parameters()]
        pf.backward(model, torch.randint(13, (4, 6)), 0.02)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, pf.discriminator.parameters())))
        self.assertTrue(model.training)

    def test_generator_gradient_and_rng_isolation(self):
        model = ToyLM().train()
        rng = torch.get_rng_state().clone()
        pf = ProfessorForcing(8, torch.device('cpu'), context=2, rollout=4,
                              generator_min_accuracy=-1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        tokens = torch.randint(13, (4, 6))
        rng = torch.get_rng_state().clone()
        result = pf.backward(model, tokens, 0.02)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(result['pf_g_enabled'], 1)
        self.assertGreater(model.embedding.weight.grad.norm().item(), 0)
        self.assertIsNone(model.head.weight.grad)  # discrete samples are detached
        self.assertTrue(all(p.grad is None for p in pf.discriminator.parameters()))
        self.assertTrue(model.training)

    def test_actual_gpt_causality_and_behavior_replay(self):
        m = load_gpt_definitions()
        cfg = m.GPTConfig(sequence_len=16, vocab_size=13, n_layer=4, n_head=2,
                          n_kv_head=2, n_embd=32, dropout=0, stoch_depth=0, use_iha=True)
        model = m.GPT(cfg)
        model.init_weights()
        model.eval()
        tokens = torch.randint(13, (2, 10))
        full = model(tokens)
        pos = torch.tensor([4])
        torch.testing.assert_close(model(tokens, logit_positions=pos), full[:, 4:5])
        torch.testing.assert_close(model(tokens[:, :5])[:, -1:], full[:, 4:5])
        changed = tokens.clone()
        changed[:, 5:] = 0
        torch.testing.assert_close(model(changed)[:, :5], full[:, :5])
        pf = ProfessorForcing(32, torch.device('cpu'), context=4, rollout=4,
                              generator_min_accuracy=-1)
        pf.backward(model, tokens, 0.02)
        self.assertGreater(model.transformer.h[0].attn.c_q.weight.grad.norm().item(), 0)
        self.assertFalse(model.training)

    def test_bf16_replay_does_not_reuse_detached_autocast_weights(self):
        m = load_gpt_definitions()
        cfg = m.GPTConfig(sequence_len=16, vocab_size=13, n_layer=4, n_head=2,
                          n_kv_head=2, n_embd=32, dropout=.05, stoch_depth=0, use_iha=True)
        model = m.GPT(cfg)
        model.init_weights()
        pf = ProfessorForcing(32, torch.device('cpu'), context=4, rollout=4,
                              generator_min_accuracy=-1)
        x, y = torch.randint(13, (4, 16)), torch.randint(13, (4, 16))
        for _ in range(2):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                pf.backward(model, x, .02)
                # All trunk matrices must receive the hidden-state gradient;
                # no-gradient sampling must not poison autocast's weight cache.
                for name, param in model.transformer.h.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all(), name)
                loss, _ = model(x, y)
            loss.backward()
            for name, param in model.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
            model.zero_grad(set_to_none=True)


if __name__ == '__main__':
    unittest.main()
