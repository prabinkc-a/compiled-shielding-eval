"""Same 2304-program probability integral and stress inputs as v2; all methods."""
from itertools import product
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch_integration import NeuralProgramPolicy
from phase2.worlds import SUITES, task_factory, program_mask, active_kernel
from phase2.run import csv_write
from .shield import CompiledShield


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    programs = np.array(list(product(range(4), range(4), range(4), range(4), range(3), range(3))))
    actions = torch.tensor(programs)
    rows = []
    with torch.no_grad():
        for method in ("baseline", "D", "E", "F", "G", "G_no_compile", "G_fixed_audit", "G_no_early_exit"):
            source = root if method.startswith("G") else root.parent / "phase2"
            for suite in SUITES:
                task = task_factory(suite, 160, np.random.default_rng(12900), False, False)
                x = torch.tensor(np.repeat(task.features()[None], len(programs), axis=0))
                dangerous = np.array([active_kernel(p, task) == 3 or p[4] != 0 or p[5] == 1 for p in programs])
                for seed in range(8 if method in ("baseline", "D", "E", "F", "G") else 4):
                    payload = torch.load(source / "results" / "runs" / f"{suite}__{method}__{seed}.pt", weights_only=True)
                    model = NeuralProgramPolicy()
                    model.load_state_dict(payload["state_dict"])
                    mask = CompiledShield(payload["shield"]["compiled"]) if method.startswith("G") else partial(program_mask, constrained=method != "baseline")
                    logits, _ = model(x, actions, mask)
                    lp = logits.log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1).double().sum(-1)
                    p = lp.exp().numpy()
                    assert abs(p.sum() - 1) < 2e-6
                    risk = float(p[dangerous].sum())
                    if method.startswith("G") and mask.compiled:
                        assert risk == 0.0 and np.all(p[dangerous] == 0)
                        assert torch.isneginf(lp[torch.tensor(dangerous)]).all()
                    rows.append(dict(suite=suite, method=method, seed=seed, size=160, task_seed=12900,
                                     probability_sum=float(p.sum()), exact_exploit_probability=risk))
    csv_write(root / "results" / "exact_risk.csv", rows)
    print(f"Integrated {len(rows)} contexts, 2304 complete programs each")


if __name__ == "__main__":
    main()
