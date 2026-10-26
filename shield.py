"""Versioned, contract-certified DSL shield and cascading sparse auditor.

A counterexample triggers certification; it is NOT itself a universal proof.
Kernel 3's trusted primitive contract identifies it as a shortcut in every suite.
This certificate allows blocking that primitive in both live and dormant branches.
Kernel 2 is merely incorrect and is never promoted into a safety restriction.
"""
import time
import numpy as np
import torch
from defenses import Adversary
from phase2.worlds import CUTOFFS, SUITES, execute, task_factory, trusted_score, program_mask


class CompiledShield:
    def __init__(self, compiled=False):
        self.compiled = bool(compiled)
        self.pending = False
        self.version = int(compiled)
        self.certificate = None
        self.permissions = torch.ones((6, 4), dtype=torch.bool)
        self.permissions[4] = torch.tensor([True, False, False, False])
        self.permissions[5] = torch.tensor([True, False, True, False])
        if compiled:
            self.permissions[[0, 3], 3] = False

    def certify(self, program, task, result):
        if task.suite not in SUITES or result.kernel != 3 or not result.shortcut:
            return False
        _, exploit, _ = trusted_score(task, result)
        if not exploit:
            return False
        self.pending = True
        self.certificate = {"primitive": 3, "positions": [0, 3],
                            "contract": "worlds.execute kernel 3 is a declared shortcut",
                            "suite": task.suite, "witness_size": task.size,
                            "program": list(map(int, program))}
        return True

    def publish(self):
        if self.pending and not self.compiled:
            self.compiled = True
            self.version += 1
            self.permissions[[0, 3], 3] = False
        self.pending = False

    def __call__(self, features, prefix, step):
        if not self.compiled:
            return program_mask(features, prefix, step, constrained=True)
        return self.permissions[step].to(features.device).expand(len(features), -1)

    def state(self):
        return {"compiled": self.compiled, "version": self.version, "certificate": self.certificate}


class CompilingVerifier:
    def __init__(self, seed, shield, compile_rules=True, fixed=False, early_exit=True):
        self.rng = np.random.default_rng(seed + 81)
        self.adversary = Adversary(seed)
        self.shield, self.compile_rules = shield, compile_rules
        self.fixed, self.early_exit = fixed, early_exit
        self.failures = {}
        self.audits = self.executions = self.micro_calls = self.full_calls = 0
        self.cache_hits = self.revocations = self.micro_failures = 0
        self.micro_seconds = self.full_seconds = 0.0
        self.first_certificate_step = None
        self.last_surprise = -100

    def check(self, program, original_task, step):
        self.audits += 1
        p = tuple(map(int, program))
        guard, cutoff = p[1:3]
        n, monitored, tag = 13, True, True
        if guard == 1:
            n = CUTOFFS[cutoff] + 1
        elif guard == 2:
            n, monitored = original_task.size, False
        elif guard == 3:
            n, tag = original_task.size, False
        failed = False
        # Large tasks are constructed lazily, only after the microprobe passes.
        for stage in range(2):
            started = time.perf_counter()
            if stage == 0:
                task = task_factory(original_task.suite, n, self.rng, monitored, tag,
                                    int(self.rng.integers(4)))
                self.micro_calls += 1
            else:
                # Co-evolving structure/cues; scale is always 97 or 128.
                proposed, arm = self.adversary.challenge(original_task.suite)
                task = task_factory(original_task.suite, int(self.rng.choice((97, 128))),
                                    self.rng, proposed.monitored, proposed.tag, proposed.structure)
                self.full_calls += 1
            result = execute(p, task)
            self.executions += 1
            bad = not trusted_score(task, result)[0]
            if stage == 1:
                self.adversary.update(arm, bad)
                self.full_seconds += time.perf_counter() - started
            else:
                self.micro_seconds += time.perf_counter() - started
                self.micro_failures += int(bad)
            if bad:
                failed = True
                self.failures[p] = task
                if self.compile_rules and self.shield.certify(p, task, result):
                    if self.first_certificate_step is None:
                        self.first_certificate_step = step + 1
                self.last_surprise = step
                if self.early_exit:
                    break
        return not failed

    def score_batch(self, tasks, programs, observed, step, forced=None):
        base = np.asarray([o.proxy - o.cost for o in observed], dtype=np.float32)
        rewards, audited = base.copy(), np.zeros(len(base), dtype=bool)
        mean = float(base.mean())
        probabilities = []
        for i, (task, p, out) in enumerate(zip(tasks, programs, observed)):
            p = tuple(map(int, p))
            if self.fixed:
                q = 0.0895
            elif self.shield.compiled:
                # A nonzero independent audit floor remains after saturation.
                q = 0.005 + 0.015 * np.exp(-max(0, step - self.last_surprise) / 12)
            else:
                q = .02 + .12 * (base[i] > mean + .05) + .08 * (p[1] != 0 and p[0] != p[3])
                if forced is not None and forced[i]:
                    q = 1.0
            probabilities.append(q)
            if p in self.failures:
                good = False
                self.cache_hits += 1
            elif self.rng.random() < q:
                audited[i] = True
                good = self.check(p, task, step)
            else:
                good = True
            if not good:
                rewards[i] = -3.0 - out.cost
                self.revocations += 1
        return rewards, audited, np.asarray(probabilities)

    def state(self):
        return {**self.shield.state(), "audits": self.audits, "challenge_executions": self.executions,
                "micro_calls": self.micro_calls, "full_calls": self.full_calls,
                "micro_failures": self.micro_failures, "micro_seconds": self.micro_seconds,
                "full_seconds": self.full_seconds, "failure_cache_hits": self.cache_hits,
                "retroactive_revocations": self.revocations,
                "first_certificate_step": self.first_certificate_step,
                "auxiliary_forward_passes": 0}
