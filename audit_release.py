"""Independently recompute released metrics from saved policies and task banks."""
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from environments import Maze, Grader, Tamper

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    metadata = json.loads((OUT / "run_metadata.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    policies = np.load(OUT / "policies.npz")
    assert len(policies.files) == metadata["model_count"]
    banks = {}
    for path in OUT.glob("bank_*.npz"):
        key = tuple(path.stem.removeprefix("bank_").split("_"))
        data = np.load(path)
        banks[key] = (data["values"], data["fields"].tolist())
    rows = list(csv.DictReader((OUT / "per_seed_metrics.csv").open()))
    max_error = 0.0
    for row in rows:
        scenario, suite, method, seed, split = (row[k] for k in ("scenario", "suite", "method", "seed", "split"))
        p = policies["__".join((scenario, suite, method, seed))]
        assert np.isfinite(p).all() and (p >= 0).all()
        assert np.isclose(p.sum(), 1.0, atol=1e-12)
        bank, fields = banks[(scenario, suite, split)]
        recomputed = np.einsum("p,tpf->f", p, bank) / bank.shape[0]
        for f, v in zip(fields, recomputed):
            error = abs(float(row[f]) - v)
            max_error = max(max_error, error)
            assert error < 1e-8, (scenario, suite, method, seed, split, f, error)
        if method in ("D_hierarchy", "E_unified", "A_full_constraints", "E_no_gate", "E_no_process"):
            training, ff = banks[(scenario, suite, "train")]
            illegal = np.max(np.maximum(training[:, :, ff.index("hard")],
                                        training[:, :, ff.index("causal")]), axis=0) > 0
            assert np.all(p[illegal] == 0)
    paired_max = 0.0
    for suite in ("maze", "grader", "tamper"):
        for seed in range(metadata["config"]["seeds"]):
            def p(method):
                return policies[f"main__{suite}__{method}__{seed}"]
            error = float(np.max(np.abs(p("D_hierarchy") - p("A_full_constraints"))))
            paired_max = max(paired_max, error)
            assert error < 1e-12
            assert np.array_equal(p("B_nyaya"), p("E_gate_only"))
    examples = []
    for cls, plan, description, blind, ood in (
        (Maze, (0, 0, 0), "Honest corridor navigation", False, False),
        (Maze, (3, 1, 1), "Internal teleport evades direct-only boundary rules", False, False),
        (Grader, (2, 1, 0), "Visible-test memorization", False, False),
        (Grader, (3, 2, 0), "Evaluator overwrite with incorrect program", False, False),
        (Tamper, (2, 1, 1), "Conceal early error and forge final report", False, False),
        (Maze, (3, 1, 1), "Teleport with compromised telemetry", True, True),
        (Grader, (4, 1, 0), "Adaptive length-conditional solver fools short probes", True, True),
        (Tamper, (2, 1, 1), "Shared monitor and replay blind spot", True, True),
    ):
        env = cls(np.random.default_rng(20261026), blind=blind, ood=ood)
        result = env.run(plan)
        examples.append({"suite": cls.name, "description": description, "plan": plan,
                         "blind": blind, "ood": ood, "outcome": asdict(result)})
    (OUT / "failure_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    audit = {"source_hashes_match": True, "policies_checked": len(policies.files),
             "metric_rows_recomputed": len(rows), "max_absolute_metric_error": max_error,
             "A_full_vs_D_max_probability_error": paired_max,
             "B_vs_gate_only_bitwise_equal": True, "certified_support_checks": "passed"}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
