"""Replay saved Phase 1 grader policies on a size sweep, without retraining.

This is a descriptive bridge. Program languages, training, and task observation
differ between phases; it is not a controlled tabular-versus-neural ablation.
"""
from pathlib import Path
import numpy as np
from environments import Grader, PLANS, lis_reference
from .run import csv_write


def main():
    root = Path(__file__).resolve().parent
    policies = np.load(root.parent / "results" / "policies.npz")
    rows = []
    for blind in (False, True):
        scenario = "blindspot" if blind else "main"
        for size in (8, 12, 20, 40, 80, 160):
            rng = np.random.default_rng(59800 + size)
            values = np.zeros((len(PLANS), 3))
            for _ in range(8):
                env = Grader(rng, blind=blind)
                env.hidden = [tuple(rng.integers(-20, 21, size=size)) for _ in range(8)]
                env.hidden_truth = [lis_reference(x) for x in env.hidden]
                for j, program in enumerate(PLANS):
                    result = env.run(program, verify=False)
                    values[j] += np.array([result.truth, result.exploit, result.functional]) / 8
            for seed in range(8):
                p = policies[f"{scenario}__grader__E_unified__{seed}"]
                truth, exploit, functional = p @ values
                rows.append({"phase": 1, "suite": "grader", "scenario": scenario, "method": "E",
                             "seed": seed, "size": size, "truth": float(truth),
                             "exploit": float(exploit), "functional": float(functional)})
            print(f"legacy {scenario}, n={size}", flush=True)
    csv_write(root / "results" / "legacy_generalization.csv", rows)


if __name__ == "__main__":
    main()
