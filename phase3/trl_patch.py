"""Version-pinned TRL G adapter with support-consistent compiled grammar masks.

Single-process, plain text, beta=0, one fresh rollout per optimizer step. Supply
a task-specific PythonBackend and PythonGrammar. No claim of generic LLM safety.
"""
import importlib.metadata
import random
import torch
from trl import GRPOTrainer


class AlgorithmGGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, backend, grammar, audit_floor=.005, method="G", **kwargs):
        if importlib.metadata.version("trl") != "1.16.0":
            raise RuntimeError("Revalidate protected hooks for versions other than trl==1.16.0")
        if any(kwargs.get(k) is not None for k in ("reward_funcs", "rollout_func", "tools", "environment_factory")):
            raise ValueError("G owns generation and execution; bypass paths are unsupported")
        if method not in ("baseline", "G", "G_soft"):
            raise ValueError("Unknown method")
        if not 0 < audit_floor <= 1:
            raise ValueError("Invalid independent audit floor")
        self.g_backend, self.g_grammar, self.g_method = backend, grammar, method
        self.g_floor = audit_floor
        self.g_rng = random.Random(kwargs["args"].seed + 52)
        self.g_metrics = dict(sequences=0, audits=0, revoked=0, blocked=0, auxiliary_forwards=0)
        self.g_failed = set()
        kwargs["reward_funcs"] = self.compiled_reward
        kwargs["rollout_func"] = self.compiled_rollout
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1 or self.beta != 0 or self.args.num_iterations != 1 or self.args.gradient_accumulation_steps != 1:
            raise ValueError("Validated configuration: one process, beta=0, one iteration, accumulation=1")
        if self.args.steps_per_generation != 1:
            raise ValueError("New shield versions require a fresh rollout at each optimizer step")

    def compiled_rollout(self, prompts, trainer):
        return self.g_grammar.sample(self.model, prompts)

    def _generate_and_score_completions(self, inputs):
        # Previous rollout has completed its optimizer update. Never mutate
        # grammar support during the reward callback or current loss evaluation.
        self.g_backend.publish()
        self.g_grammar.refresh()
        return super()._generate_and_score_completions(inputs)

    def compiled_reward(self, prompts, completions, **kwargs):
        step = self.state.global_step
        assessments = [self.g_backend.quick(c, step) for c in completions]
        mean = sum(score for _, score in assessments) / len(assessments)
        values = []
        for code, (ok, score) in zip(completions, assessments):
            self.g_metrics["sequences"] += 1
            if not ok:
                self.g_metrics["blocked"] += 1
                values.append(-3.)
                continue
            if self.g_method == "baseline":
                values.append(score)
                continue
            # Saturation is measured against this finite grammar, not inferred
            # merely from silence. Until all known rule families are discovered,
            # retain exploratory audits. This can cost more than DSL G.
            saturated = len(self.g_backend.rules) >= 4
            q = self.g_floor + (.01 if saturated else .15) + .1 * (score > mean + .05)
            bad = code in self.g_failed
            if not bad and self.g_rng.random() < q:
                self.g_metrics["audits"] += 1
                bad = not self.g_backend.audit(code, step)
            if bad:
                self.g_failed.add(code)
                self.g_metrics["revoked"] += 1
            values.append(-3. if bad else score)
        return values

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
            logits_to_keep, batch_size=None, compute_entropy=False, compute_aux_loss=False, **kwargs):
        if compute_aux_loss:
            raise ValueError("MoE auxiliary loss is not supported by this reference adapter")
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = self.g_grammar.masked_logits(out.logits, input_ids, logits_to_keep)
        lp = torch.log_softmax(logits, -1)
        targets = input_ids[:, -logits_to_keep:]
        selected = lp.gather(-1, targets[..., None]).squeeze(-1)
        # Padding after EOS is excluded by TRL's completion mask; use zero to
        # avoid 0*(-inf) in downstream loss, never modify real-token logps.
        selected = torch.where(targets == self.processing_class.pad_token_id, 0., selected)
        if not torch.isfinite(selected).all():
            raise RuntimeError("Rollout/scoring shield support mismatch")
        entropy = -(lp.exp() * torch.where(torch.isfinite(lp), lp, 0.)).sum(-1) if compute_entropy else None
        return selected, entropy, None

    def train(self, *args, **kwargs):
        if kwargs.get("resume_from_checkpoint") or args:
            raise ValueError("Restore synchronized compiler/optimizer state before enabling resume")
        return super().train(**kwargs)
