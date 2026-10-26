"""Serial randomized warm-checkpoint calibration, same v2 protocol."""
import json
from pathlib import Path
import numpy as np
import torch
from torch_integration import NeuralProgramPolicy
from phase2 import train as old_train
from phase2.run import csv_write
from phase2.worlds import SUITES
from . import train

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    jobs = [(s, m, seed) for seed in range(3) for s in SUITES
            for m in ("baseline", "D", "E", "F", "G", "G_no_compile", "G_fixed_audit", "G_no_early_exit")]
    rows = []
    for idx in np.random.default_rng(4118).permutation(len(jobs)):
        suite, method, seed = jobs[int(idx)]
        source = ROOT if method.startswith("G") else ROOT.parent / "phase2"
        payload = torch.load(source / "results" / "runs" / f"{suite}__{method}__{seed}.pt", weights_only=True)
        if method.startswith("G"):
            _, _, _, _, stats = train.fit(suite, method, seed + 800, updates=40,
                eval_count=1, checkpoints=False, initial_state=payload["state_dict"],
                initial_shield=payload["shield"]["compiled"])
        else:
            original = old_train.NeuralProgramPolicy
            def factory():
                model = NeuralProgramPolicy()
                model.load_state_dict(payload["state_dict"])
                return model
            old_train.NeuralProgramPolicy = factory
            try:
                _, _, _, _, stats = old_train.fit(suite, method, seed + 800,
                    updates=40, eval_count=1, checkpoints=False)
            finally:
                old_train.NeuralProgramPolicy = original
        stats.pop("verifier_state", None)
        stats["seed"] = seed
        rows.append(stats)
        print(f"profile {len(rows)}/{len(jobs)} {suite}/{method}/{seed}: {stats['cpu_train_seconds']:.3f}s", flush=True)
    csv_write(ROOT / "results" / "serial_profile.csv", rows)
    (ROOT / "results" / "profile_protocol.json").write_text(json.dumps({
        "jobs": len(jobs), "updates": 40, "threads": 1, "order_seed": 4118,
        "checkpoint": "frozen release copy; fresh optimizer and reference equal to copied policy",
        "verifier": "fresh counters/search; G retains checkpoint compiled permissions",
        "timing": "serial process CPU; evaluation excluded",
        "comparison": "warm steady-state workload, not end-to-end discovery cost"}, indent=2))


if __name__ == "__main__":
    main()
