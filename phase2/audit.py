"""Recompute every final metric from the saved neural checkpoints."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch_integration import NeuralProgramPolicy
from .train import evaluate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    torch.set_num_threads(1)
    meta = json.loads((OUT / "metadata.json").read_text())
    for name, expected in meta["source_sha256"].items():
        path = ROOT.parent / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    recorded = list(csv.DictReader((OUT / "evaluation.csv").open()))
    index = {(r["suite"], r["method"], int(r["seed"]), int(r["size"]), r["condition"], r["decode"]): r
             for r in recorded}
    checked, count, max_error, d_e = 0, 0, 0., 0.
    for path in sorted((OUT / "runs").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = NeuralProgramPolicy()
        model.load_state_dict(payload["state_dict"])
        assert all(torch.isfinite(p).all() for p in model.parameters())
        rows = evaluate(model, payload["suite"], payload["method"], payload["seed"],
                        per_condition=meta["config"]["eval_count"])
        for row in rows:
            key = tuple(row[k] for k in ("suite", "method", "seed", "size", "condition", "decode"))
            original = index[key]
            for metric in ("truth", "exploit", "functional", "proxy"):
                error = abs(row[metric] - float(original[metric]))
                max_error = max(max_error, error)
                assert error < 1e-12, (key, metric, error)
            count += 1
        if payload["method"] == "D":
            epath = path.with_name(path.name.replace("__D__", "__E__"))
            ep = torch.load(epath, map_location="cpu", weights_only=True)
            for k, v in payload["state_dict"].items():
                d_e = max(d_e, float((v - ep["state_dict"][k]).abs().max()))
        checked += 1
        if checked % 25 == 0:
            print(f"audit {checked}/{meta['models']}", flush=True)
    assert checked == meta["models"] and count == len(recorded)
    result = {"source_hashes_match": True, "checkpoints_checked": checked,
              "rows_recomputed": count, "max_metric_error": max_error,
              "max_D_E_parameter_difference": d_e,
              "test_log_exists": (OUT / "tests.txt").exists(),
              "trl_integration": json.loads((OUT / "trl_smoke" / "result.json").read_text())["status"]}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
