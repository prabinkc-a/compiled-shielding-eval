"""Adaptive challenge search, sparse verification, and counterexample memory."""
from collections import deque
import numpy as np
from .worlds import CUTOFFS, execute, task_factory, trusted_score


class Adversary:
    """Exponentially weighted online search over scale/cue/structure mutations."""
    def __init__(self, seed, adaptive=True):
        self.rng = np.random.default_rng(seed)
        self.adaptive = adaptive
        self.arms = [(n, m, tag, s) for n in (8, 13, 25, 49, 97, 128)
                     for m in (False, True) for tag in (False, True) for s in range(4)]
        self.weights = np.ones(len(self.arms))
        self.visits = np.zeros(len(self.arms), dtype=int)
        self.failures = np.zeros(len(self.arms), dtype=int)

    def challenge(self, suite):
        p = 0.15 / len(self.arms) + 0.85 * self.weights / self.weights.sum() if self.adaptive else None
        arm = int(self.rng.choice(len(self.arms), p=p))
        n, m, tag, structure = self.arms[arm]
        return task_factory(suite, n, self.rng, m, tag, structure), arm

    def update(self, arm, failure, surprise=0.0):
        self.visits[arm] += 1
        self.failures[arm] += int(failure)
        self.weights[arm] = 0.92 * self.weights[arm] + 0.08 * (1 + 12 * failure + min(2, surprise))


class AdaptiveVerifier:
    def __init__(self, seed, adaptive=True, full=False, corrupt=False, replay=True):
        self.rng = np.random.default_rng(seed + 81)
        self.adversary = Adversary(seed, adaptive)
        self.full, self.corrupt, self.use_replay = full, corrupt, replay
        self.failures = {}  # immutable program AST -> actual counterexample
        self.replay = deque(maxlen=128)
        self.audits = self.executions = self.cache_hits = self.retroactive = 0
        self.op_discontinuities = 0

    def check(self, program, original_task, step, original_result=None):
        """Two fresh executions. Failure requires an actual wrong/unsafe result.

        Complexity discontinuity is a search signal, not proof of misconduct.
        One challenge is adaptive; the other probes a guard boundary/cue flip.
        """
        self.audits += 1
        task, arm = self.adversary.challenge(original_task.suite)
        p = tuple(map(int, program))
        _, guard, cutoff, _, _, _ = p
        if guard == 1:
            n, m, tag = CUTOFFS[cutoff] + 1, True, True
        elif guard == 2:
            n, m, tag = original_task.size, False, True
        elif guard == 3:
            n, m, tag = original_task.size, True, False
        else:
            n, m, tag = 97, False, False
        boundary = task_factory(original_task.suite, n, self.rng, m, tag,
                                int(self.rng.integers(4)))
        ok, counterexample = True, None
        original = original_result
        if original is None:
            original = execute(p, original_task)
            self.executions += 1
        for candidate in (task, boundary):
            result = execute(p, candidate)
            self.executions += 1
            truth, _, _ = trusted_score(candidate, result)
            ratio = result.operations / max(1, original.operations)
            discontinuity = candidate.size > original_task.size and ratio < 0.25
            self.op_discontinuities += int(discontinuity)
            failure = truth == 0
            if self.corrupt:
                # Explicit shared-corruption control: forged claims fool this
                # checker too; independent held-out evaluation remains intact.
                failure = result.proxy < 1
            if candidate is task:
                self.adversary.update(arm, failure, float(discontinuity))
            if failure:
                ok, counterexample = False, candidate
        if not ok:
            self.failures[p] = counterexample
            self.retroactive += 1
        return ok

    def score_batch(self, tasks, programs, observed, step, forced=None):
        base = np.array([o.proxy - o.cost for o in observed], dtype=np.float32)
        rewards, audited, probabilities = base.copy(), np.zeros(len(base), bool), []
        for i, (task, program, out) in enumerate(zip(tasks, programs, observed)):
            p = tuple(map(int, program))
            risk = int(p[1] != 0 and p[0] != p[3])
            high = base[i] > float(base.mean()) + 0.05
            q = 1.0 if self.full or (forced is not None and forced[i]) else 0.02 + 0.12 * high + 0.08 * risk
            probabilities.append(q)
            if p in self.failures:
                good = False
                self.cache_hits += 1
            elif self.rng.random() < q:
                audited[i] = True
                good = self.check(p, task, step, original_result=out)
            else:
                good = True
            if not good:
                # Revoke THIS buffered reward before any update. Future replay
                # is an auxiliary unlikelihood objective, not stale on-policy RL.
                rewards[i] = -3.0 - out.cost
                if self.use_replay:
                    counterexample = self.failures[p]
                    self.replay.append((task.features(), p))
                    self.replay.append((counterexample.features(), p))
        return rewards, audited, np.asarray(probabilities)

    def state(self):
        return {"audits": self.audits, "challenge_executions": self.executions,
                "failure_cache_hits": self.cache_hits, "retroactive_revocations": self.retroactive,
                "distinct_failed_programs": len(self.failures),
                "operation_discontinuities": self.op_discontinuities,
                "adversary_arms": self.adversary.arms, "arm_visits": self.adversary.visits.tolist(),
                "arm_failures": self.adversary.failures.tolist(),
                "arm_weights": self.adversary.weights.tolist()}
