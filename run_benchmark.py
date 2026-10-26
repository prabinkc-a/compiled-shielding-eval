"""Run reproducible finite-policy experiments and emit raw, auditable evidence."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
from algorithms import MAIN, ABLATIONS, TrainSignals, supervision, train
from environments import ENVIRONMENTS, FIELDS, PLANS, build_bank

ROOT = Path(__file__).resolve().parent


def signals_from(bank):
    return TrainSignals(**{f: bank[:, :, FIELDS.index(f)] for f in TrainSignals.__annotations__})


def evaluate(bank, p):
    means = bank.mean(axis=0)
    out = dict(zip(FIELDS, map(float, p @ means)))
    out["gap"] = out["proxy"] - out["truth"]
    return out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def bootstrap_ci(values, seed=7041, repeats=10000):
    x = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(len(x), size=(repeats, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "low": float(np.quantile(b, 0.025)),
            "high": float(np.quantile(b, 0.975)), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0}


def summarize(rows):
    """Bootstrap paired seed blocks AFTER equal-weight averaging of suites."""
    groups = {}
    for r in rows:
        key = (r["scenario"], r["split"], r["method"])
        groups.setdefault(key, {}).setdefault(r["seed"], []).append(r)
    summary = []
    metrics = ("proxy", "truth", "exploit", "gap", "functional", "seconds",
               "base_work", "verify_work", "certificate_work")
    for (scenario, split, method), by_seed in groups.items():
        record = {"scenario": scenario, "split": split, "method": method}
        for metric in metrics:
            values = [np.mean([r[metric] for r in block]) for block in by_seed.values()]
            record[metric] = bootstrap_ci(values)
        summary.append(record)
    return summary


def profile_online(banks, policies, seeds, count=160, repeats=5):
    """Actually replay uncached training rollouts, excluding hidden test scoring.

    Same number of rollouts per method. Randomized method order, five repeats.
    Times include environment simulation and requested verifier, but exclude
    task construction, policy updates, certificate construction and hidden tests.
    """
    rows = []
    for suite_id, suite in enumerate(ENVIRONMENTS):
        rng = np.random.default_rng(88000 + suite_id)
        envs = [ENVIRONMENTS[suite](rng) for _ in range(12)]
        schedules = {}
        for method in MAIN:
            p = np.mean([policies[("main", suite, method, s)] for s in seeds], axis=0)
            rr = np.random.default_rng(17003)
            schedules[method] = rr.choice(len(PLANS), size=count, p=p)
        # Warm up code paths outside measured intervals.
        for e in envs:
            e.run(PLANS[0], verify=True, evaluate=False)
        for repeat in range(repeats):
            for method in rng.permutation(MAIN):
                use_gate = method in ("B_nyaya", "E_unified")
                start = time.perf_counter_ns()
                for i, action in enumerate(schedules[method]):
                    envs[i % len(envs)].run(PLANS[action], verify=use_gate, evaluate=False)
                ns = (time.perf_counter_ns() - start) / count
                rows.append({"suite": suite, "method": str(method), "repeat": repeat,
                             "episodes": count, "microseconds_per_rollout": ns / 1000})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--updates", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--train-tasks", type=int, default=32)
    ap.add_argument("--test-tasks", type=int, default=96)
    ap.add_argument("--output", default="results")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.updates, args.train_tasks, args.test_tasks = 3, 200, 8, 12
    dest = ROOT / args.output
    dest.mkdir(exist_ok=True, parents=True)
    start = time.perf_counter()
    seeds = list(range(args.seeds))
    metadata = {"config": vars(args), "python": sys.version, "platform": platform.platform(),
                "processor": platform.processor(), "numpy": np.__version__,
                "cpu_count": os.cpu_count(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "split_seeds": {}, "bank_seconds": {}, "source_sha256": {}}
    for p in sorted(ROOT.glob("*.py")):
        metadata["source_sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    # Main task distributions plus a separate common-mode/adaptive attacker scenario.
    banks = {}
    for scenario in ("main", "blindspot"):
        for j, suite in enumerate(ENVIRONMENTS):
            for z, split in enumerate(("train", "id", "ood")):
                count = args.train_tasks if split == "train" else args.test_tasks
                seed = 100000 + j * 1000 + z * 100
                path = dest / f"bank_{scenario}_{suite}_{split}.npz"
                t = time.perf_counter()
                bank = build_bank(suite, count, seed, ood=split == "ood", blind=scenario == "blindspot")
                banks[(scenario, suite, split)] = bank
                np.savez_compressed(path, values=bank, fields=FIELDS, plans=PLANS)
                key = f"{scenario}/{suite}/{split}"
                metadata["split_seeds"][key] = seed
                metadata["bank_seconds"][key] = time.perf_counter() - t
                print(f"bank {key}: {bank.shape}, {metadata['bank_seconds'][key]:.2f}s", flush=True)
    rows, curves, policies = [], [], {}
    experiments = [("main", m, {}) for m in MAIN + ABLATIONS]
    experiments += [("blindspot", m, {}) for m in MAIN]
    experiments += [("main", "E_gate_miss_25", {"gate_miss": 0.25}),
                    ("main", "E_gate_reject_10", {"gate_false_reject": 0.10})]
    for scenario, label, options in experiments:
        method = "E_unified" if label.startswith("E_gate_") and label != "E_gate_only" else label
        for suite in ENVIRONMENTS:
            bank = banks[(scenario, suite, "train")]
            sig = signals_from(bank)
            train_method = method
            if method == "oracle_outcome":
                # Explicit diagnostic upper-bound control; never mixed into main algorithms.
                sig.proxy = bank[:, :, FIELDS.index("truth")].copy()
                train_method = "baseline"
            for seed in seeds:
                result = train(sig, train_method, seed=seed, updates=args.updates,
                               batch_size=args.batch, lr=args.lr, **options)
                p = result.pop("p")
                curve = result.pop("curve")
                policies[(scenario, suite, label, seed)] = p
                for step, reward, proxy in curve:
                    curves.append({"scenario": scenario, "suite": suite, "method": label,
                                   "seed": seed, "update": step, "training_reward": reward,
                                   "training_proxy": proxy})
                # Train evaluation is diagnostic only; no checkpoint selection.
                for split in ("train", "id", "ood"):
                    row = {"scenario": scenario, "suite": suite, "method": label,
                           "seed": seed, "split": split, **result,
                           **evaluate(banks[(scenario, suite, split)], p)}
                    rows.append(row)
            print(f"trained {scenario}/{suite}/{label}: {len(seeds)} seeds", flush=True)
    write_csv(dest / "per_seed_metrics.csv", rows)
    write_csv(dest / "learning_curves.csv", curves)
    np.savez_compressed(dest / "policies.npz", **{"__".join(map(str, k)): v for k, v in policies.items()})
    summary = summarize(rows)
    (dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    online = profile_online(banks, policies, seeds, count=80 if args.quick else 240)
    write_csv(dest / "online_profile.csv", online)
    # Direct paired differences, not subtraction of separate confidence intervals.
    comparisons = []
    for competitor in MAIN[:-1]:
        for metric in ("truth", "exploit"):
            diffs = []
            for seed in seeds:
                def mean(m):
                    return np.mean([r[metric] for r in rows if r["scenario"] == "main"
                                    and r["split"] == "id" and r["method"] == m and r["seed"] == seed])
                diffs.append(mean("E_unified") - mean(competitor))
            comparisons.append({"comparison": f"E_unified - {competitor}", "metric": metric,
                                **bootstrap_ci(diffs)})
    (dest / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    metadata["elapsed_seconds"] = time.perf_counter() - start
    metadata["model_count"] = len(policies)
    metadata["sampled_training_episodes"] = len(policies) * args.updates * args.batch
    metadata["bank_episodes_executed"] = sum(b.shape[0] * b.shape[1] for b in banks.values())
    (dest / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done: {len(policies)} policies, {metadata['elapsed_seconds']:.1f}s -> {dest}", flush=True)


if __name__ == "__main__":
    main()
