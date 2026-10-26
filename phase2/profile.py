"""Serial, randomized-order warm-checkpoint cost calibration on this CPU.

Reuse the EXACT tested fit loop with a factory loading a final policy. The
reference is a frozen copy of that checkpoint, optimizer/auditor are fresh, and
40 further updates run on disposable clones. Release policies are not changed.
This is a workload measurement, not a new accuracy-selected training run.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch_integration import NeuralProgramPolicy
from . import train
from .run import csv_write
from .worlds import SUITES

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    jobs = [(suite, method, seed) for seed in range(3) for suite in SUITES for method in train.METHODS]
    order = np.random.default_rng(4118).permutation(len(jobs))
    rows = []
    original_factory = train.NeuralProgramPolicy
    for i in order:
        suite, method, seed = jobs[int(i)]
        path = ROOT / "results" / "runs" / f"{suite}__{method}__{seed}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        def factory():
            model = NeuralProgramPolicy()
            model.load_state_dict(payload["state_dict"])
            return model
        train.NeuralProgramPolicy = factory
        try:
            _, _, _, _, stats = train.fit(suite, method, seed + 800, updates=40,
                                           eval_count=1, checkpoints=False)
        finally:
            train.NeuralProgramPolicy = original_factory
        stats.pop("verifier_state", None)
        stats["seed"] = seed
        stats["profile_updates"] = 40
        rows.append(stats)
        print(f"profile {len(rows)}/{len(jobs)} {suite}/{method}/{seed}: "
              f"cpu {stats['cpu_train_seconds']:.3f}s", flush=True)
    csv_write(ROOT / "results" / "serial_profile.csv", rows)
    (ROOT / "results" / "profile_protocol.json").write_text(json.dumps({
        "order_seed": 4118, "models": len(rows), "updates_per_model": 40,
        "seeds": [0, 1, 2], "threads": 1,
        "policy": "frozen release checkpoint copied, then disposable calibration updates",
        "reference": "frozen calibration starting checkpoint", "verifier": "fresh state",
        "timing": "process CPU time; final evaluation excluded; serial execution",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
