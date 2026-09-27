"""CPU correctness tests; the real eight-GPU run is validated separately."""
import ast
import copy
from dataclasses import replace
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


def load_gpt_definitions(mtp_weight=0):
    # train.py is an executable script. Load its definitions without its training
    # side effects; retain the actual model code rather than duplicating it.
    source = Path(__file__).resolve().parents[1].joinpath('train.py').read_text()
    prefix = source.split('# Compute init\n')[0]
    module = types.ModuleType('slowrun_test_defs')
    sys.modules[module.__name__] = module
    old_argv = sys.argv
    try:
        sys.argv = ['train.py', '--mtp-weight', str(mtp_weight)]
        exec(compile(ast.parse(prefix), 'train.py', 'exec'), module.__dict__)
    finally:
        sys.argv = old_argv
    def attention(q, k, v, causal=False, window_size=(-1, -1)):
        # FlashAttention aligns a shorter query to the RIGHT of the key cache.
        qi = torch.arange(k.size(1) - q.size(1), k.size(1))
        ki = torch.arange(k.size(1))
        mask = qi[:, None] >= ki[None, :]
        if window_size[0] >= 0:
            mask &= qi[:, None] - ki[None, :] <= window_size[0]
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

    def test_attached_prefix_reuses_ce_and_preserves_full_parameter_gradients(self):
        m = load_gpt_definitions(mtp_weight=.3)
        cfg = m.GPTConfig(sequence_len=16, vocab_size=23, n_layer=4, n_head=2,
                          n_kv_head=2, n_embd=32, dropout=0, stoch_depth=0,
                          use_iha=True, window_pattern='SL')
        for checkpointed in (False, True):
            for dupe in (False, True):
                with self.subTest(checkpointed=checkpointed, dupe=dupe):
                    reference = m.GPT(replace(cfg, activation_checkpointing=checkpointed))
                    reference.init_weights()
                    if dupe:
                        reference.set_dupe_layers(2, 3, 2)
                    shared = copy.deepcopy(reference)
                    x, y = torch.randint(23, (2, 16)), torch.randint(23, (2, 16))
                    prefix, pf_batch = 11, 1
                    fake = x[:pf_batch].clone()
                    fake[:, prefix:] = torch.randint(23, (pf_batch, 16-prefix))
                    probe = torch.randn(pf_batch, 16-prefix, cfg.n_embd)
                    ce_ref, _ = reference(x, y)
                    h_ref = reference(fake, return_hidden=True)[:, prefix:]
                    pf_embedding_grad_ref, = torch.autograd.grad(
                        (h_ref * probe).mean(), reference.transformer.wte.weight,
                        retain_graph=True)
                    (ce_ref + .02 * (h_ref * probe).mean()).backward()

                    ce_shared, _, state = shared(x, y, cache_prefix=prefix, cache_batch=pf_batch)
                    self.assertEqual(len(state['prefix_kv']), 6 if dupe else 4)
                    self.assertTrue(all(k.requires_grad and v.requires_grad
                                        for k, v in state['prefix_kv']))
                    self.assertFalse(state['real_hidden'].requires_grad)
                    self.assertFalse(state['next_logits'].requires_grad)
                    h_shared = shared(fake[:, prefix:], past_key_values=state['prefix_kv'],
                                      return_hidden=True)
                    pf_embedding_grad_shared, = torch.autograd.grad(
                        (h_shared * probe).mean(), shared.transformer.wte.weight,
                        retain_graph=True)
                    torch.testing.assert_close(pf_embedding_grad_shared, pf_embedding_grad_ref,
                                               rtol=3e-4, atol=2e-6)
                    torch.testing.assert_close(h_shared, h_ref, rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(state['real_hidden'],
                                               shared(x[:pf_batch], return_hidden=True)[:, prefix:])
                    torch.testing.assert_close(state['next_logits'], shared(x[:pf_batch])[:, prefix-1])
                    (ce_shared + .02 * (h_shared * probe).mean()).backward()
                    for (name, expected), (_, actual) in zip(reference.named_parameters(), shared.named_parameters()):
                        self.assertIsNotNone(expected.grad, name)
                        self.assertIsNotNone(actual.grad, name)
                        torch.testing.assert_close(actual.grad, expected.grad,
                                                   rtol=3e-4, atol=2e-6, msg=name)

    def test_incremental_cache_matches_full_context_through_sliding_windows_and_dupe(self):
        m = load_gpt_definitions()
        cfg = m.GPTConfig(sequence_len=16, vocab_size=23, n_layer=4, n_head=2,
                          n_kv_head=2, n_embd=32, dropout=0, stoch_depth=0,
                          use_iha=True, window_pattern='SL')
        model = m.GPT(cfg).eval()
        model.init_weights()
        model.set_dupe_layers(2, 3, 2)
        x = torch.randint(23, (2, 16))
        with torch.no_grad():
            full = model(x)
            logits, cache = model(x[:, :10], return_cache=True)
            torch.testing.assert_close(logits, full[:, :10])
            for position in range(10, 16):
                logits, cache = model(x[:, position:position+1], past_key_values=cache,
                                      return_cache=True)
                torch.testing.assert_close(logits, full[:, position:position+1],
                                           rtol=2e-5, atol=2e-6)
                self.assertEqual(len(cache), 6)
                self.assertTrue(all(k.size(1) == position+1 for k, _ in cache))

    def test_amortized_pf_reuses_ce_and_preserves_dropout_rng_and_closed_gate_gradients(self):
        m = load_gpt_definitions(mtp_weight=.3)
        cfg = m.GPTConfig(sequence_len=16, vocab_size=23, n_layer=4, n_head=2,
                          n_kv_head=2, n_embd=32, dropout=.1, stoch_depth=0,
                          use_iha=True, activation_checkpointing=True)
        reference = m.GPT(cfg).train()
        reference.init_weights()
        for gate in (1.0, -1.0):
            model = copy.deepcopy(reference)
            pf = ProfessorForcing(32, torch.device('cpu'), context=11, rollout=5,
                                  batch=1, generator_min_accuracy=gate)
            x, y = torch.randint(23, (2, 16)), torch.randint(23, (2, 16))
            reference.zero_grad(set_to_none=True)
            torch.manual_seed(812)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                baseline, _ = reference(x, y)
            (baseline / 8).backward()
            calls = []
            hook = model.register_forward_pre_hook(
                lambda module, inputs, kwargs: calls.append(inputs[0].size(1)), with_kwargs=True)
            torch.manual_seed(812)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                ce, _, state = model(x, y, cache_prefix=11, cache_batch=1)
                rng = torch.get_rng_state().clone()
                loss, metrics = pf.amortized_loss(model, model, state, .02)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(all(p.grad is None for p in model.parameters()))
            (ce / 8 + loss).backward()
            hook.remove()
            self.assertEqual(calls, [16, 1, 1, 1, 1, 5])
            self.assertTrue(model.training)
            self.assertEqual(metrics['pf_g_enabled'], float(gate < 0))
            self.assertTrue(all(p.grad is None for p in pf.discriminator.parameters()))
            differences = []
            for (name, expected), (_, actual) in zip(reference.named_parameters(), model.named_parameters()):
                self.assertTrue(torch.isfinite(actual.grad).all(), name)
                if gate == 1:
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0, msg=name)
                differences.append((actual.grad - expected.grad).abs().sum())
            if gate < 0:
                self.assertGreater(torch.stack(differences).sum().item(), 0)

    def test_checkpointing_preserves_mtp_dropout_gradients_and_pf_replay(self):
        m = load_gpt_definitions(mtp_weight=.3)
        cfg = m.GPTConfig(sequence_len=16, vocab_size=13, n_layer=4, n_head=2,
                          n_kv_head=2, n_embd=32, dropout=.1, stoch_depth=.2,
                          use_iha=True)
        reference = m.GPT(cfg)
        reference.init_weights()
        checkpointed = copy.deepcopy(reference)
        checkpointed.config = replace(cfg, activation_checkpointing=True)
        x, y = torch.randint(13, (2, 16)), torch.randint(13, (2, 16))
        for dupe in (False, True):
            if dupe:
                reference.set_dupe_layers(2, 3, 2)
                checkpointed.set_dupe_layers(2, 3, 2)
            outcomes = []
            for model in (reference, checkpointed):
                model.zero_grad(set_to_none=True)
                torch.manual_seed(177)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    loss, _ = model(x, y)
                loss.backward()
                outcomes.append((loss.detach(), torch.get_rng_state().clone()))
            torch.testing.assert_close(outcomes[0][0], outcomes[1][0])
            self.assertTrue(torch.equal(outcomes[0][1], outcomes[1][1]))
            for (name, expected), (_, actual) in zip(reference.named_parameters(),
                                                     checkpointed.named_parameters()):
                self.assertIsNotNone(expected.grad, name)
                self.assertIsNotNone(actual.grad, name)
                torch.testing.assert_close(actual.grad, expected.grad, msg=name)
        checkpointed.zero_grad(set_to_none=True)
        pf = ProfessorForcing(32, torch.device('cpu'), context=4, rollout=4,
                              generator_min_accuracy=-1)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            pf.backward(checkpointed, x, .02)
        for name, param in checkpointed.transformer.h.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)


if __name__ == '__main__':
    unittest.main()
