"""Algorithm F integration for TRL 1.16.0, text-only, single process.

The backend is the security boundary and must reject forbidden actions BEFORE
execution. No arbitrary generated Python is executed by this adapter. This
wrapper preserves TRL's rollout likelihoods; it does not silently inject a
generation-only token mask. A task-specific backend is intrinsically required.
"""
from collections import deque
from dataclasses import dataclass
import importlib.metadata
import math
import random
from typing import Protocol
import torch
from trl import GRPOTrainer


@dataclass
class Assessment:
    admissible: bool
    proxy: float
    cost: float
    complexity: float
    key: str


class VerificationBackend(Protocol):
    def quick(self, prompt: str, completion: str) -> Assessment:
        """Validate permissions before any execution, then obtain cheap feedback."""
    def audit(self, prompt: str, completion: str) -> bool:
        """Run independent adaptive challenge executions; update adversary state."""
    def challenge_prompt(self, prompt: str) -> str:
        """Return a fresh challenge for on-policy neural re-querying."""
    def invariant_prompt(self, prompt: str, completion: str) -> str | None:
        """Return a change under which this program's behavior should generalize."""


class AlgorithmFGRPOTrainer(GRPOTrainer):
    """Sparse adversarial rewards plus invariance and counterexample replay.

    Version-pinned because the prompt-group generation hook is protected API.
    Custom tasks supply four backend methods; DSLBackend is a runnable example.
    This does not claim to certify arbitrary Qwen/Llama output or support DDP.
    """
    def __init__(self, *args, backend: VerificationBackend, audit_floor=0.02,
                 penalty=3.0, invariant_weight=0.08, replay_weight=0.3,
                 f_seed=7, **kwargs):
        version = importlib.metadata.version("trl")
        if version != "1.16.0":
            raise RuntimeError(f"Validated against trl==1.16.0, found {version}; revalidate the hooks before use")
        if kwargs.get("reward_funcs") is not None:
            raise ValueError("Supply task scores through backend.quick; do not add a reward that bypasses integrity")
        if any(kwargs.get(k) is not None for k in ("tools", "environment_factory", "rollout_func")):
            raise ValueError("This adapter requires backend-owned execution; external tool paths are not brokered")
        if not 0 < audit_floor <= 1:
            raise ValueError("audit_floor must be positive and at most one")
        self.f_backend = backend
        self.f_rng = random.Random(f_seed)
        self.f_floor, self.f_penalty = audit_floor, penalty
        self.f_invariant_weight, self.f_replay_weight = invariant_weight, replay_weight
        self.f_failed = set()
        self.f_replay, self.f_recent = deque(maxlen=128), deque(maxlen=32)
        self.f_calls = 0
        self.f_metrics = {"quick_checks": 0, "audits": 0, "revoked": 0,
                          "blocked": 0, "cache_hits": 0, "challenged_groups": 0,
                          "invariance_updates": 0, "replay_updates": 0}
        kwargs["reward_funcs"] = self.algorithm_f_reward
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError("Single-process adapter: synchronize verifier/replay state before enabling distributed training")

    def _generate_and_score_completions(self, inputs):
        # Keep every GRPO group on ONE prompt after mutation.
        prepared = [{**row, "f_force_audit": False} for row in inputs]
        group = self.num_generations if self.model.training else self.num_generations_eval or self.num_generations
        if self.model.training and self.f_calls % 4 == 0 and len(prepared) >= group:
            rows = prepared[-group:]
            if not all(row["prompt"] == rows[0]["prompt"] for row in rows):
                raise RuntimeError("TRL prompt grouping changed; refusing to mix group advantages")
            mutated = self.f_backend.challenge_prompt(rows[0]["prompt"])
            for row in rows:
                row["prompt"], row["f_force_audit"] = mutated, True
            self.f_metrics["challenged_groups"] += 1
        self.f_calls += 1
        return super()._generate_and_score_completions(prepared)

    def algorithm_f_reward(self, prompts, completions, f_force_audit=None,
                           trainer_state=None, log_metric=None, **kwargs):
        if not all(isinstance(x, str) for x in prompts + completions):
            raise TypeError("This validated adapter accepts plain-text prompts and completions")
        if len(prompts) != len(completions):
            raise ValueError("Mismatched rollout batches")
        assessments = [self.f_backend.quick(p, c) for p, c in zip(prompts, completions)]
        provisional = [a.proxy - a.cost for a in assessments]
        mean = sum(provisional) / max(1, len(provisional))
        forced = f_force_audit if f_force_audit is not None else [False] * len(prompts)
        result = []
        for prompt, completion, a, base, force in zip(prompts, completions, assessments, provisional, forced):
            self.f_metrics["quick_checks"] += 1
            if not a.admissible:
                self.f_metrics["blocked"] += 1
                result.append(-self.f_penalty)
                continue
            self.f_recent.append((prompt, completion))
            q = 1.0 if force else min(1.0, self.f_floor + 0.12 * (base > mean + 0.05) + 0.08 * (a.complexity > 0))
            failed = a.key in self.f_failed
            if failed:
                self.f_metrics["cache_hits"] += 1
            elif self.f_rng.random() < q:
                self.f_metrics["audits"] += 1
                failed = not self.f_backend.audit(prompt, completion)
            if failed:
                self.f_failed.add(a.key)
                self.f_replay.append((prompt, completion))
                self.f_metrics["revoked"] += 1
                result.append(-self.f_penalty - a.cost)
            else:
                result.append(base - 0.04 * a.complexity)
        if log_metric is not None:
            for key, value in self.f_metrics.items():
                log_metric("algorithm_f/" + key, float(value))
        return result

    def _completion_logits(self, model, prompt, completion_ids):
        tokenizer = self.processing_class
        ids = tokenizer.encode(prompt, add_special_tokens=False)[-512:]
        if not ids:
            ids = [tokenizer.bos_token_id or tokenizer.eos_token_id]
        if ids[0] is None:
            raise ValueError("Need a nonempty prompt or BOS/EOS token")
        all_ids = torch.tensor([ids + completion_ids], dtype=torch.long, device=self.accelerator.device)
        output = model(input_ids=all_ids, attention_mask=torch.ones_like(all_ids), use_cache=False)
        start = len(ids) - 1
        return output.logits[0, start:start + len(completion_ids)].float()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        base_loss = loss
        if not model.training or self.state.global_step % 4:
            return loss
        if self.f_recent and self.f_invariant_weight:
            prompt, completion = self.f_rng.choice(list(self.f_recent))
            changed = self.f_backend.invariant_prompt(prompt, completion)
            ids = self.processing_class.encode(completion, add_special_tokens=False)[:32]
            if changed is not None and ids:
                a = torch.log_softmax(self._completion_logits(model, prompt, ids), -1)
                b = torch.log_softmax(self._completion_logits(model, changed, ids), -1)
                symmetric = 0.5 * ((a.exp() - b.exp()) * (a - b)).sum(-1).mean()
                loss = loss + self.f_invariant_weight * symmetric
                self.f_metrics["invariance_updates"] += 1
        if self.f_replay and self.f_replay_weight:
            prompt, completion = self.f_rng.choice(list(self.f_replay))
            ids = self.processing_class.encode(completion, add_special_tokens=False)
            # Never penalize a truncated prefix as if it were a complete bad
            # program. Generic long outputs need a task-specific failure span.
            if 0 < len(ids) <= 64:
                logits = self._completion_logits(model, prompt, ids)
                token_ids = torch.tensor(ids, device=logits.device)
                lp = torch.log_softmax(logits, -1).gather(-1, token_ids[:, None]).sum()
                probability = lp.exp().clamp(max=1 - 1e-6)
                loss = loss - self.f_replay_weight * torch.log1p(-probability)
                self.f_metrics["replay_updates"] += 1
        # TRL normalizes its own policy loss; normalize ONLY our auxiliary term.
        normalizer = getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps)
        return base_loss + (loss - base_loss) / normalizer

    def train(self, *args, **kwargs):
        resume = kwargs.get("resume_from_checkpoint", args[0] if args else None)
        if resume:
            raise ValueError("Resume requires persistent synchronized verifier state; this tested adapter supports fresh runs only")
        return super().train(*args, **kwargs)
