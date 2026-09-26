"""Sparse Professor Forcing (Lamb et al., arXiv:1610.09038) for a causal LM.

Sample discrete tokens without gradients, then replay them with gradients. This
preserves the paper's hidden-state gradient without differentiating token choices.
The discriminator sees equally long continuations in both domains, without the
shared prompt. No ground-truth next-token targets are applied to sampled contexts.
"""

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist


class BehaviorDiscriminator(nn.Module):
    def __init__(self, width, hidden=128):
        super().__init__()
        self.project = nn.Linear(width, hidden)
        self.temporal = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.output = nn.Linear(2 * hidden, 1)

    def forward(self, states):
        z = F.gelu(self.project(states.float())).transpose(1, 2)
        z = F.gelu(self.temporal(z))
        return self.output(torch.cat((z.mean(-1), z.amax(-1)), -1)).squeeze(-1)


@torch.no_grad()
def sample_continuation(model, prompt, length, generator=None):
    """Fixed-size causal canvas avoids a separate compiled graph per token."""
    b, p = prompt.shape
    tokens = prompt.new_zeros(b, p + length)
    tokens[:, :p] = prompt
    for offset in range(length):
        position = torch.tensor([p + offset - 1], device=tokens.device)
        logits = model(tokens, logit_positions=position)[:, 0]
        sampled = torch.multinomial(logits.float().softmax(-1), 1, generator=generator)
        tokens[:, p + offset] = sampled[:, 0]
    return tokens


def global_mean(tensor):
    value = tensor.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(value)
        value /= dist.get_world_size()
    return value


class ProfessorForcing:
    def __init__(self, width, device, *, context=64, rollout=16, batch=4,
                 hidden=128, lr=1e-4, generator_min_accuracy=0.75):
        if min(context, rollout, batch) < 1:
            raise ValueError("PF context, rollout, and batch must be positive")
        self.context, self.rollout, self.batch = context, rollout, batch
        self.generator_min_accuracy = generator_min_accuracy
        # Do not alter the base training RNG when initializing the discriminator.
        cuda_devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(1729)
            self.discriminator = BehaviorDiscriminator(width, hidden).to(device)
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=lr, weight_decay=0)
        self.generator = torch.Generator(device=device).manual_seed(2718 + (dist.get_rank() if dist.is_initialized() else 0))

    def backward(self, model, tokens, weight):
        """Accumulate weighted generator gradient; update synchronized discriminator.

        Called before ordinary microbatches, so the combined objective participates
        in the existing meta-gradient adaptation just like the CE/MTP objective.
        """
        # Sampling uses no_grad and D changes both weights and requires_grad.
        # Cached autocast weight casts would otherwise silently detach the replay
        # gradient or reuse D's pre-update weights within the enclosing context.
        device_type = tokens.device.type
        with torch.autocast(device_type, enabled=torch.is_autocast_enabled(device_type),
                            dtype=torch.get_autocast_dtype(device_type), cache_enabled=False):
            return self._backward(model, tokens, weight)

    def _backward(self, model, tokens, weight):
        p, r = self.context, self.rollout
        if tokens.size(1) < p + r:
            raise ValueError("PF context + rollout exceeds training sequence length")
        real_tokens = tokens[:self.batch, :p+r].contiguous()
        was_training = model.training
        model.eval()  # no dropout/stochastic-depth shortcut for domain classification
        try:
            fake_tokens = sample_continuation(model, real_tokens[:, :p], r, self.generator)
            with torch.no_grad():
                real = model(real_tokens, return_hidden=True)[:, p:]
                fake = model(fake_tokens, return_hidden=True)[:, p:]
            d = self.discriminator
            real_logits, fake_logits = d(real.detach()), d(fake.detach())
            d_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
            accuracy = global_mean(0.5 * ((real_logits > 0).float().mean() + (fake_logits < 0).float().mean()))
            self.optimizer.zero_grad(set_to_none=True)
            if accuracy.item() <= 0.99:
                d_loss.backward()
                if dist.is_initialized():
                    for param in d.parameters():
                        dist.all_reduce(param.grad)
                        param.grad.div_(dist.get_world_size())
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            g_loss = torch.zeros((), device=tokens.device)
            enabled = accuracy.item() > self.generator_min_accuracy
            if enabled:
                d.requires_grad_(False)
                try:
                    fake_states = model(fake_tokens, return_hidden=True)[:, p:]
                    g_loss = F.softplus(-d(fake_states)).mean()
                    (weight * g_loss).backward()
                finally:
                    d.requires_grad_(True)
            return {"pf_d_loss": global_mean(d_loss), "pf_g_loss": global_mean(g_loss),
                    "pf_accuracy": accuracy, "pf_g_enabled": float(enabled)}
        finally:
            model.train(was_training)
