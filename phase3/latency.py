"""Matched component attribution; not substituted for serial end-to-end cost."""
from pathlib import Path
from functools import partial
import json
import time
import numpy as np
import torch
from torch_integration import NeuralProgramPolicy, neural_grpo_loss
from phase2.worlds import task_factory, program_mask, SUITES
from phase2.train import symmetric_kl, unlikelihood
from .shield import CompiledShield, CompilingVerifier


def main():
    torch.set_num_threads(1)
    torch.manual_seed(21)
    rng = np.random.default_rng(30)
    root = Path(__file__).resolve().parent
    model = NeuralProgramPolicy()
    payload = torch.load(root.parent / "phase2/results/runs/grader__F__0.pt", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    tasks = [task_factory("grader", 8, rng) for _ in range(32)]
    x = torch.tensor(np.stack([t.features() for t in tasks]))
    pair = torch.tensor(np.stack([task_factory("grader", 17, rng, False, False).features() for _ in range(4)]))
    mask = partial(program_mask, constrained=True)
    actions, old = model.generate(x, mask)
    rewards = torch.arange(32).float() % 8
    with torch.no_grad():
        ref, _ = model(x, actions, mask)
    replay = [(t.features(), (0, 1, 0, 3, 0, 0)) for t in tasks[:8]]
    times = {"base": [], "with_invariance_replay": []}
    for i in range(220):
        for aux in rng.permutation([False, True]):
            start = time.process_time_ns()
            logits, masks = model(x, actions, mask)
            loss = neural_grpo_loss(logits, actions, old, rewards, 8, ref, beta=.005)
            if aux:
                pl, pm = model(pair, actions[::8], mask)
                loss = loss + .08 * symmetric_kl(logits[::8], pl, masks[::8], pm)
                loss = loss + .3 * unlikelihood(model, replay, mask, rng)
            model.zero_grad()
            loss.backward()
            if i >= 20:
                times["with_invariance_replay" if aux else "base"].append((time.process_time_ns()-start)/1e6)
    # process_time on Windows has coarse per-call resolution; total/mean is
    # meaningful over repetitions, individual zeros are not zero-cost passes.
    aux_summary = {k: float(np.mean(v)) for k, v in times.items()}
    aux_summary["amortized_saving_ms_per_update_two_epochs_every_four"] = (
        aux_summary["with_invariance_replay"] - aux_summary["base"]) * 2 / 4
    cascade = []
    for suite in SUITES:
        for early in (False, True):
            verifier = CompilingVerifier(55, CompiledShield(), early_exit=early)
            task = task_factory(suite, 8, np.random.default_rng(40))
            started = time.perf_counter()
            for i in range(120):
                verifier.check([0, 1, i % 4, 3, 0, 0], task, i)
            cascade.append({"suite": suite, "early_exit": early, "calls": 120,
                            "milliseconds_per_audit": (time.perf_counter()-started)*1000/120,
                            "executions": verifier.executions})
    masks = {}
    prefix = torch.tensor([[0, 1, 0, 3, 0, 0]]*32)
    for name, fn in (("D_contextual", mask), ("G_compiled", CompiledShield(True))):
        started = time.perf_counter()
        for _ in range(2000):
            for step in range(6):
                fn(x, prefix[:, :step], step)
        masks[name] = (time.perf_counter()-started)*1e6/2000
    result = {"auxiliary_forward_backward_mean_ms": aux_summary,
              "cascade_known_conditional_failures": cascade,
              "permission_build_microseconds_per_32x6_batch": masks,
              "auxiliary_passes_removed_per_240_update_fit": {"invariance": 120, "replay_max": 120},
              "warning": "Microbenchmarks isolate components; do not add them to predict total runtime."}
    (root / "results/latency.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
