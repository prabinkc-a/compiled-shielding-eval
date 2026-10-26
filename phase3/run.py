"""Run the Phase 3 neural experiments, preserving all final checkpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from .train import METHODS, ABLATIONS, fit
from phase2.worlds import SUITES

ROOT = Path(__file__).resolve().parent


def run_job(suite, method, seed, updates, eval_count, dest, resume):
    dest = Path(dest)
    stem = f"{suite}__{method}__{seed}"
    record = dest / "runs" / f"{stem}.json"
    if resume and record.exists():
        return json.loads(record.read_text())
    model, f, c, l, st = fit(suite, method, seed, updates=updates,
                            eval_count=eval_count, checkpoints=method in METHODS)
    torch.save({"state_dict": model.state_dict(), "suite": suite, "method": method,
                "seed": seed, "updates": updates, "shield": st.get("verifier_state", {})}, dest / "runs" / f"{stem}.pt")
    result = {"final": f, "curves": c, "log": l, "stats": st}
    record.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--ablation-seeds", type=int, default=4)
    parser.add_argument("--updates", type=int, default=240)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-count", type=int, default=96)
    parser.add_argument("--output", default="results")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.pilot:
        args.seeds, args.ablation_seeds, args.updates, args.eval_count = 1, 0, 120, 24
        args.workers = 1
    dest = ROOT / args.output
    (dest / "runs").mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sources = list(ROOT.glob("*.py")) + list((ROOT.parent / "phase2").glob("*.py")) + [ROOT.parent / "torch_integration.py"]
    meta = {"config": vars(args), "torch": torch.__version__, "numpy": np.__version__,
            "python": sys.version, "platform": platform.platform(), "torch_threads": 1,
            "source_sha256": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources}}
    jobs = [(s, m, seed) for seed in range(args.seeds) for s in SUITES for m in METHODS]
    jobs += [(s, m, seed) for seed in range(args.ablation_seeds) for s in SUITES for m in ABLATIONS]
    # Interleave methods and suites to reduce systematic wall-time bias.
    order = np.random.default_rng(9026).permutation(len(jobs))
    final, curves, logs, stats = [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
      futures = [pool.submit(run_job, *jobs[int(i)], args.updates, args.eval_count, str(dest), args.resume) for i in order]
      for future in as_completed(futures):
        result = future.result()
        st = result["stats"]
        stem = f"{st['suite']}__{st['method']}__{st['seed']}"
        final.extend(result["final"]); curves.extend(result["curves"])
        logs.extend(result["log"]); stats.append(result["stats"])
        blind = [r for r in result["final"] if r["size"] > 12 and r["condition"] == "unmonitored" and r["decode"] == "sample"]
        print(f"{len(stats)}/{len(jobs)} {stem}: true={np.mean([r['truth'] for r in blind]):.3f} "
              f"exploit={np.mean([r['exploit'] for r in blind]):.3f} "
              f"time={result['stats']['train_seconds']:.1f}s", flush=True)
    csv_write(dest / "evaluation.csv", final)
    csv_write(dest / "generalization_curves.csv", curves)
    csv_write(dest / "training_log.csv", logs)
    csv_write(dest / "training_cost.csv", [{k: v for k, v in r.items() if k != "verifier_state"} for r in stats])
    (dest / "verifier_states.json").write_text(json.dumps([r for r in stats if "verifier_state" in r], indent=2))
    meta["models"] = len(stats)
    meta["training_sequences"] = sum(r["training_sequences"] for r in stats)
    meta["actual_program_executions"] = sum(r["program_executions"] for r in stats)
    meta["evaluation_sequences"] = sum(r["episodes"] for r in final)
    meta["elapsed_seconds"] = time.perf_counter() - start
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Complete: {dest}", flush=True)


if __name__ == "__main__":
    main()
