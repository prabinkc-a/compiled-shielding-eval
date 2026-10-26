"""Minimal tested PyTorch components for a constrained GRPO-style pipeline.

This is not an LLM security boundary. A separate tool broker must enforce real
permissions; certificates must be available before actions are sampled. The
experiment uses one on-policy update, without PPO clipping or a reference KL.
"""
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


def neural_grpo_loss(logits, actions, old_logps, rewards, group_size,
                     reference_logits=None, clip=0.2, beta=0.01):
    """Token-level clipped GRPO with per-prompt standardized advantages.

    Rewards have one entry per sequence, with contiguous groups per prompt.
    Full categorical KL is exact on the small DSL vocabulary. All input
    policies must have identical grammar/permission support for these prefixes.
    """
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, actions[..., None]).squeeze(-1)
    if not torch.isfinite(logp).all():
        raise ValueError("Rollout outside the current support")
    groups = rewards.reshape(-1, group_size)
    advantage = (groups - groups.mean(-1, keepdim=True)) / (groups.std(-1, unbiased=False, keepdim=True) + 1e-4)
    advantage = advantage.flatten().detach()[:, None]
    ratio = (logp - old_logps.detach()).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    loss = -surrogate.mean()
    if reference_logits is not None and beta:
        ref = torch.log_softmax(reference_logits, dim=-1)
        finite = torch.isfinite(logp_all) & torch.isfinite(ref)
        diff = torch.where(finite, logp_all, 0) - torch.where(finite, ref, 0)
        kl = (logp_all.exp() * diff).sum(-1).mean()
        loss = loss + beta * kl
    return loss


def certified_policy(logits, allowed):
    """Construct the SAME masked policy for sampling and log-prob evaluation."""
    if logits.ndim != 2 or allowed.shape != logits.shape or allowed.dtype != torch.bool:
        raise ValueError("Expected logits and boolean allowed of shape [prompts, actions]")
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    if not allowed.any(dim=-1).all():
        raise ValueError("No certified action; use an explicitly certified abstention")
    return Categorical(logits=logits.masked_fill(~allowed, -torch.inf))


def unified_rewards(proxy, cost, progress, hard_ok, causal_ok, iso_passes,
                    process_weight=0.15):
    """Inputs [prompts, group], iso_passes [prompts, group, checks].

    Values originate from an isolated verifier, never from model-authored scores.
    For exact reproduction, progress is the benchmark's local progress label.
    """
    if not (proxy.shape == cost.shape == progress.shape == hard_ok.shape == causal_ok.shape):
        raise ValueError("Mismatched reward shapes")
    if iso_passes.shape[:-1] != proxy.shape or iso_passes.shape[-1] < 1:
        raise ValueError("Need at least one independent check")
    if hard_ok.dtype != torch.bool or causal_ok.dtype != torch.bool or iso_passes.dtype != torch.bool:
        raise ValueError("Integrity and test decisions must be boolean")
    integrity = hard_ok & causal_ok
    gate = integrity & iso_passes.all(dim=-1)
    return (proxy * gate - cost + process_weight * progress * integrity).detach()


def constrained_group_loss(policy, actions, rewards, old_logp=None,
                           clip=0.2, entropy_weight=0.004):
    """RLOO grouped policy loss, optionally with a PPO-style clipped surrogate.

    actions/rewards: [prompts, group]. An illegal sampled action is an error;
    silently rescoring a rollout under a different support would be invalid.
    With old_logp=None this matches the benchmark's REINFORCE estimator.
    It is not a full implementation of DeepSeekMath GRPO.
    """
    if actions.shape != rewards.shape or actions.ndim != 2 or actions.shape[1] < 2:
        raise ValueError("Need paired [prompts, group>=2] actions and rewards")
    logp = policy.logits.gather(-1, actions)
    if not torch.isfinite(logp).all():
        raise ValueError("A sampled action was outside the certified support")
    g = rewards.shape[-1]
    advantage = (rewards - rewards.mean(dim=-1, keepdim=True)) * g / (g - 1)
    advantage = advantage.detach()
    if old_logp is None:
        objective = logp * advantage
    else:
        if old_logp.shape != logp.shape or not torch.isfinite(old_logp).all():
            raise ValueError("Invalid old policy log probabilities")
        ratio = torch.exp(logp - old_logp.detach())
        objective = torch.minimum(ratio * advantage,
                                  ratio.clamp(1 - clip, 1 + clip) * advantage)
    return -objective.mean() - entropy_weight * policy.entropy().mean()


def demo():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    logits = torch.nn.Parameter(torch.zeros(1, 3))
    allowed = torch.tensor([[True, True, False]])
    optimizer = torch.optim.SGD([logits], lr=0.5)
    for _ in range(150):
        policy = certified_policy(logits, allowed)
        actions = policy.sample((32,)).T
        rewards = (actions == 0).float()
        loss = constrained_group_loss(policy, actions, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print({"torch": torch.__version__, "probabilities": certified_policy(logits, allowed).probs.tolist()})


if __name__ == "__main__":
    demo()
