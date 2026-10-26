"""Actual on-policy neural GRPO training; no precomputed outcome table."""
from __future__ import annotations
import copy
from functools import partial
import time
import numpy as np
import torch
from torch_integration import NeuralProgramPolicy, neural_grpo_loss
from .worlds import Task, task_factory, execute, static_gate, program_mask, trusted_score
from .defenses import AdaptiveVerifier

METHODS = ("baseline", "D", "E", "F")
ABLATIONS = ("F_random", "F_no_invariance", "F_no_replay", "F_full_audit", "F_corrupt")


def symmetric_kl(a, b, mask_a, mask_b):
    common = mask_a & mask_b
    # Compare only shared permitted support; permissions are not learned away.
    a = a.masked_fill(~common, -1e9)
    b = b.masked_fill(~common, -1e9)
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return 0.5 * ((la.exp() - lb.exp()) * (la - lb)).sum(-1).mean()


def unlikelihood(model, replay, mask, rng, count=8):
    if not replay:
        return next(model.parameters()).sum() * 0
    indices = rng.choice(len(replay), size=min(count, len(replay)), replace=False)
    batch = [replay[int(i)] for i in indices]
    features = torch.tensor(np.stack([x[0] for x in batch]))
    actions = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.long)
    logits, _ = model(features, actions, mask)
    lp = torch.log_softmax(logits, -1).gather(-1, actions[..., None]).squeeze(-1)
    good = torch.isfinite(lp).all(-1)
    if not good.any():
        return next(model.parameters()).sum() * 0
    prob = lp[good].sum(-1).exp().clamp(max=1 - 1e-6)
    return -torch.log1p(-prob).mean()


def evaluate(model, suite, method, seed, sizes=(8, 12, 20, 40, 80, 160), per_condition=64):
    mask = partial(program_mask, constrained=method != "baseline")
    rows = []
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(78000 + seed)
        for size in sizes:
            for monitored, tag, condition in ((True, True, "monitored"), (False, False, "unmonitored")):
                rng = np.random.default_rng(800000 + 100 * size + int(monitored))
                tasks = [task_factory(suite, size, rng, monitored, tag, structure=i % 4)
                         for i in range(per_condition)]
                features = torch.tensor(np.stack([t.features() for t in tasks]))
                programs, _ = model.generate(features, mask)
                for greedy in (False, True):
                    selected = model.generate(features, mask, greedy=True)[0] if greedy else programs
                    truth, exploit, functional, proxies = [], [], [], []
                    for task, program in zip(tasks, selected.tolist()):
                        outcome = execute(program, task)
                        a, b, c = trusted_score(task, outcome)
                        truth.append(a); exploit.append(b); functional.append(c); proxies.append(outcome.proxy)
                    rows.append({"suite": suite, "method": method, "seed": seed, "size": size,
                                 "condition": condition, "decode": "greedy" if greedy else "sample",
                                 "episodes": len(tasks), "truth": float(np.mean(truth)),
                                 "exploit": float(np.mean(exploit)), "functional": float(np.mean(functional)),
                                 "proxy": float(np.mean(proxies))})
    return rows


def fit(suite, method, seed, updates=360, prompts=4, group=8, epochs=2,
        eval_count=64, checkpoints=True):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(11000 + seed)
    verify_rng = np.random.default_rng(21000 + seed)
    model = NeuralProgramPolicy()
    reference = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    mask = partial(program_mask, constrained=method != "baseline")
    is_f = method.startswith("F")
    verifier = AdaptiveVerifier(31000 + seed, adaptive=method != "F_random",
                                full=method == "F_full_audit", corrupt=method == "F_corrupt",
                                replay=method != "F_no_replay") if is_f else None
    timings = dict(task_seconds=0.0, generation_seconds=0.0, execution_seconds=0.0,
                   verifier_seconds=0.0, update_seconds=0.0)
    curves, training_log = [], []
    audit_count, total_training, actual_executions, sampled_exploits = 0, 0, 0, 0
    started = time.perf_counter()
    started_cpu = time.process_time()
    eval_seconds = eval_cpu = cpu_rollout = 0.0
    if checkpoints:
        t = time.perf_counter()
        ct = time.process_time()
        curves.extend({**r, "update": 0} for r in evaluate(model, suite, method, seed,
                       sizes=(8, 20, 80, 160), per_condition=32))
        eval_seconds += time.perf_counter() - t
        eval_cpu += time.process_time() - ct
    for step in range(updates):
        rollout_cpu_start = time.process_time()
        t = time.perf_counter()
        contexts = [task_factory(suite, int(rng.choice((6, 8, 10, 12))), rng,
                                 structure=int(rng.integers(3))) for _ in range(prompts)]
        adversarial_prompt = is_f and step % 4 == 0
        if adversarial_prompt:
            contexts[-1], _ = verifier.adversary.challenge(suite)
        tasks = [x for x in contexts for _ in range(group)]
        features = torch.tensor(np.stack([x.features() for x in tasks]))
        timings["task_seconds"] += time.perf_counter() - t
        t = time.perf_counter()
        actions, old_logps = model.generate(features, mask)
        programs = actions.tolist()
        with torch.no_grad():
            reference_logits, _ = reference(features, actions, mask)
        timings["generation_seconds"] += time.perf_counter() - t
        t = time.perf_counter()
        observed = [execute(p, x) for p, x in zip(programs, tasks)]
        total_training += len(tasks)
        actual_executions += len(tasks)
        # Diagnostic truth labels are counted, never passed into optimizer reward.
        sampled_exploits += sum(int(o.shortcut or o.hard) for o in observed)
        timings["execution_seconds"] += time.perf_counter() - t
        t = time.perf_counter()
        if method == "E":
            gates = [static_gate(p, suite, verify_rng) for p in programs]
            actual_executions += 4 * len(tasks)
            audit_count += len(tasks)
            rewards = np.array([o.proxy * gate - o.cost + 0.05 * o.process
                                for gate, o in zip(gates, observed)], dtype=np.float32)
        elif is_f:
            forced = np.zeros(len(tasks), dtype=bool)
            if adversarial_prompt:
                forced[-group:] = True
            before = verifier.executions
            rewards, audited, audit_prob = verifier.score_batch(tasks, programs, observed, step, forced)
            actual_executions += verifier.executions - before
            audit_count += int(audited.sum())
            if method != "F_no_invariance":
                # Tiny AST description-length surrogate; it is NOT Kolmogorov
                # complexity and is not itself proof of a deceptive computation.
                complexity = np.array([int(p[1] != 0 and p[0] != p[3]) for p in programs])
                rewards -= 0.04 * complexity
        else:
            rewards = np.array([o.proxy - o.cost - 0.1 * o.hard for o in observed], dtype=np.float32)
        timings["verifier_seconds"] += time.perf_counter() - t
        cpu_rollout += time.process_time() - rollout_cpu_start
        t = time.perf_counter()
        reward_tensor = torch.tensor(rewards)
        for epoch in range(epochs):
            logits, masks = model(features, actions, mask)
            loss = neural_grpo_loss(logits, actions, old_logps, reward_tensor, group,
                                   reference_logits, beta=0.005)
            invariant_value = 0.0
            if is_f and method != "F_no_invariance" and step % 4 == 0:
                paired_tasks = [task_factory(suite, min(160, x.size * 2 + 1), verify_rng,
                                              not x.monitored, not x.tag, x.structure) for x in contexts]
                paired_features = torch.tensor(np.stack([x.features() for x in paired_tasks]))
                paired_logits, paired_mask = model(paired_features, actions[::group], mask)
                inv = symmetric_kl(logits[::group], paired_logits, masks[::group], paired_mask)
                loss = loss + 0.08 * inv
                invariant_value = float(inv.detach())
            if is_f and method != "F_no_replay" and step % 4 == 0:
                loss = loss + 0.3 * unlikelihood(model, verifier.replay, mask, rng)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        timings["update_seconds"] += time.perf_counter() - t
        if step % 30 == 0 or step == updates - 1:
            training_log.append({"suite": suite, "method": method, "seed": seed, "update": step + 1,
                                 "reward": float(rewards.mean()), "loss": float(loss.detach()),
                                 "invariance": invariant_value, "audit_fraction": audit_count / total_training})
        if checkpoints and (step + 1) in (updates // 3, 2 * updates // 3, updates):
            t = time.perf_counter()
            ct = time.process_time()
            curves.extend({**r, "update": step + 1} for r in evaluate(model, suite, method, seed,
                           sizes=(8, 20, 80, 160), per_condition=32))
            eval_seconds += time.perf_counter() - t
            eval_cpu += time.process_time() - ct
    train_seconds = time.perf_counter() - started - eval_seconds
    stats = {"suite": suite, "method": method, "seed": seed,
             "parameters": sum(p.numel() for p in model.parameters()),
             "train_seconds": train_seconds, **timings, "training_sequences": total_training,
             "cpu_train_seconds": time.process_time() - started_cpu - eval_cpu,
             "cpu_rollout_seconds": cpu_rollout,
             "program_executions": actual_executions, "audited_sequences": audit_count,
             "audit_fraction": audit_count / total_training,
             "training_exploit_rate": sampled_exploits / total_training}
    if verifier:
        stats["verifier_state"] = verifier.state()
    final = evaluate(model, suite, method, seed, per_condition=eval_count)
    return model, final, curves, training_log, stats
