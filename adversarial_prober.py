"""Trigger-blind input search; no source parsing or attack-catalog imports.

Search ranges are public, finite experimental choices. The prober receives only
correct/incorrect feedback and cannot infer coverage from an absence of errors.
"""
from itertools import product
import random


class InputProber:
    """Reweight scale, value-range and structure arms after observed failures."""
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.arms = list(product((8, 13, 25, 49, 97, 160), (20, 64, 256), range(3)))
        self.weights = [1.0] * len(self.arms)
        self.calls = 0

    def sample(self):
        weights = [.15 / len(self.arms) + .85 * w / sum(self.weights) for w in self.weights]
        index = self.rng.choices(range(len(self.arms)), weights)[0]
        n, amplitude, structure = self.arms[index]
        n = max(1, n + self.rng.randint(-3, 3))
        x = [self.rng.randint(-amplitude, amplitude) for _ in range(n)]
        if structure == 1:
            x.sort()
        elif structure == 2:
            x = [x[0]] * n
        self.calls += 1
        return index, x

    def observe(self, index, failed):
        self.weights[index] = .92 * self.weights[index] + .08 * (1 + 12 * bool(failed))

