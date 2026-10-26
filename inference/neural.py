"""Frozen-policy architecture and inference only; no optimizer or loss."""
import torch
from torch.distributions import Categorical


class NeuralProgramPolicy(torch.nn.Module):
    """Task-conditioned autoregressive GRU; no table of complete programs.

    Six instructions are generated with shared weights. Input features encode
    the task, scale and visible monitoring cues. Sampling and teacher forcing
    use exactly the same prefix-dependent mask supplied by the caller.
    """
    def __init__(self, feature_dim=12, hidden=32, steps=6, vocabulary=4):
        super().__init__()
        self.steps, self.vocabulary = steps, vocabulary
        self.encoder = torch.nn.Sequential(torch.nn.Linear(feature_dim, hidden),
                                           torch.nn.Tanh(), torch.nn.Linear(hidden, hidden),
                                           torch.nn.Tanh())
        self.tokens = torch.nn.Embedding(vocabulary + 1, 8)
        self.positions = torch.nn.Embedding(steps, 8)
        self.decoder = torch.nn.GRU(16, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, vocabulary)

    def forward(self, features, actions, mask_fn):
        batch = features.shape[0]
        start = torch.full((batch, 1), self.vocabulary, dtype=torch.long, device=features.device)
        previous = torch.cat([start, actions[:, :-1]], dim=1)
        positions = torch.arange(self.steps, device=features.device)[None].expand(batch, -1)
        inp = torch.cat([self.tokens(previous), self.positions(positions)], dim=-1)
        hidden, _ = self.decoder(inp, self.encoder(features)[None])
        logits = self.head(hidden)
        masks = torch.stack([mask_fn(features, actions[:, :t], t) for t in range(self.steps)], dim=1)
        if not masks.any(dim=-1).all():
            raise ValueError("No valid token for a program prefix")
        return logits.masked_fill(~masks, -torch.inf), masks

    @torch.no_grad()
    def generate(self, features, mask_fn, greedy=False):
        b = len(features)
        hidden = self.encoder(features)[None]
        previous = torch.full((b,), self.vocabulary, dtype=torch.long, device=features.device)
        actions, logps = [], []
        prefix = torch.empty((b, 0), dtype=torch.long, device=features.device)
        for t in range(self.steps):
            pos = torch.full((b,), t, dtype=torch.long, device=features.device)
            inp = torch.cat([self.tokens(previous), self.positions(pos)], dim=-1)[:, None]
            output, hidden = self.decoder(inp, hidden)
            mask = mask_fn(features, prefix, t)
            logits = self.head(output[:, 0]).masked_fill(~mask, -torch.inf)
            dist = Categorical(logits=logits)
            action = logits.argmax(-1) if greedy else dist.sample()
            actions.append(action)
            logps.append(dist.log_prob(action))
            prefix = torch.stack(actions, dim=1)
            previous = action
        return prefix, torch.stack(logps, dim=1)
