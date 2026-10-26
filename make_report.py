"""Generate tables, figures and the technical report from recorded experiments."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from algorithms import MAIN, ABLATIONS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LABELS = {
    "baseline": "Baseline: outcome + penalty", "A_rta": "A — Rta projection",
    "B_nyaya": "B — Nyaya verification", "C_process": "C — process validity",
    "D_hierarchy": "D — context hierarchy", "E_unified": "E — unified synthesis",
    "A_full_constraints": "A + complete constraints", "C_progress": "C + local progress",
    "E_no_gate": "E without verification gate", "E_no_process": "E without process term",
    "E_no_hierarchy": "E without global hierarchy", "E_gate_only": "E gate only (= B)",
    "baseline_penalty_1": "Additive penalty = 1", "baseline_penalty_5": "Additive penalty = 5",
    "oracle_outcome": "Oracle true-outcome control", "E_gate_miss_25": "E, 25% gate false accepts",
    "E_gate_reject_10": "E, 10% gate false rejects",
}


def read_csv(name):
    return list(csv.DictReader((RESULTS / name).open(encoding="utf-8")))


def main():
    summary = json.loads((RESULTS / "summary.json").read_text())
    metadata = json.loads((RESULTS / "run_metadata.json").read_text())
    pairs = json.loads((RESULTS / "paired_comparisons.json").read_text())
    rows = read_csv("per_seed_metrics.csv")
    profile = read_csv("online_profile.csv")
    lookup = {(r["scenario"], r["split"], r["method"]): r for r in summary}
    def s(m, scenario="main", split="id"):
        return lookup[scenario, split, m]
    def mean(m, metric, scenario="main", split="id"):
        return 100 * s(m, scenario, split)[metric]["mean"]
    def interval(m, metric, scenario="main", split="id"):
        q = s(m, scenario, split)[metric]
        return f"{100*q['mean']:.2f} [{100*q['low']:.2f}, {100*q['high']:.2f}]"
    def cpu(m):
        return statistics.mean(statistics.median(float(r["microseconds_per_rollout"])
                               for r in profile if r["method"] == m and r["suite"] == suite)
                               for suite in ("maze", "grader", "tamper"))
    base_cpu, base_time = cpu("baseline"), s("baseline")["seconds"]["mean"]
    master = ["| Method | Proxy score % | True held-out accuracy % [95% CI] | Exploit rate % | Gap Δ (pp) | Training overhead %: optimizer / rollout† |",
              "|---|---:|---:|---:|---:|---:|"]
    for m in MAIN:
        opt = 100 * (s(m)["seconds"]["mean"] / base_time - 1)
        roll = 100 * (cpu(m) / base_cpu - 1)
        master.append(f"| {LABELS[m]} | {mean(m,'proxy'):.2f} | {interval(m,'truth')} | "
                      f"{mean(m,'exploit'):.3f} | {mean(m,'gap'):.2f} | {opt:+.1f} / {roll:+.1f} |")
    suites = ["| Method | Maze: true / exploit % | Grader: true / exploit % | Tampering: true / exploit % |",
              "|---|---:|---:|---:|"]
    for m in MAIN:
        cells = []
        for suite in ("maze", "grader", "tamper"):
            rr = [r for r in rows if r["scenario"] == "main" and r["split"] == "id"
                  and r["method"] == m and r["suite"] == suite]
            a, b = (100 * statistics.mean(float(r[f]) for r in rr) for f in ("truth", "exploit"))
            cells.append(f"{a:.2f} / {b:.3f}")
        suites.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
    ood = ["| Method | Larger-task true accuracy % | Larger-task exploit % | Blind-spot true accuracy % | Blind-spot exploit % |",
           "|---|---:|---:|---:|---:|"]
    for m in MAIN:
        ood.append(f"| {LABELS[m]} | {mean(m,'truth',split='ood'):.2f} | {mean(m,'exploit',split='ood'):.3f} | "
                   f"{mean(m,'truth','blindspot','ood'):.2f} | {mean(m,'exploit','blindspot','ood'):.2f} |")
    ablations = ["| Method / intervention | True accuracy % | Exploit % | Δ true vs E (pp) |",
                 "|---|---:|---:|---:|"]
    for m in ["E_unified"] + ABLATIONS + ["E_gate_miss_25", "E_gate_reject_10"]:
        ablations.append(f"| {LABELS[m]} | {mean(m,'truth'):.2f} | {mean(m,'exploit'):.3f} | "
                         f"{mean(m,'truth')-mean('E_unified','truth'):+.2f} |")
    raw = ["| Method | Raw functional correctness % | Integrity-adjusted true accuracy % |",
           "|---|---:|---:|"]
    for m in MAIN:
        raw.append(f"| {LABELS[m]} | {mean(m,'functional'):.2f} | {mean(m,'truth'):.2f} |")
    cmp = ["| Paired comparison | True-accuracy difference (pp), 95% CI |",
           "|---|---:|"]
    for r in pairs:
        if r["metric"] == "truth":
            cmp.append(f"| {r['comparison']} | {100*r['mean']:+.3f} [{100*r['low']:+.3f}, {100*r['high']:+.3f}] |")
    tables = {"MASTER_TABLE": master, "SUITE_TABLE": suites, "OOD_TABLE": ood,
              "ABLATION_TABLE": ablations, "RAW_TABLE": raw, "PAIRED_TABLE": cmp}
    for key, lines in tables.items():
        (RESULTS / f"{key.lower()}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    winner = max(MAIN, key=lambda m: mean(m, "truth"))
    replacements = {
        **{k: "\n".join(v) for k, v in tables.items()},
        "WINNER": LABELS[winner], "WINNER_ACCURACY": f"{mean(winner,'truth'):.2f}",
        "E_ACCURACY": f"{mean('E_unified','truth'):.2f}",
        "BASE_ACCURACY": f"{mean('baseline','truth'):.2f}",
        "E_GAIN": f"{mean('E_unified','truth')-mean('baseline','truth'):.2f}",
        "BASE_EXPLOIT": f"{mean('baseline','exploit'):.2f}",
        "E_EXPLOIT": f"{mean('E_unified','exploit'):.3f}",
        "E_BLIND_EXPLOIT": f"{mean('E_unified','exploit','blindspot','ood'):.2f}",
        "E_BLIND_TRUTH": f"{mean('E_unified','truth','blindspot','ood'):.2f}",
        "E_CPU_RATIO": f"{cpu('E_unified')/base_cpu:.2f}",
        "BASE_CPU_US": f"{base_cpu:.1f}", "E_CPU_US": f"{cpu('E_unified'):.1f}",
        "D_CPU_RATIO": f"{cpu('D_hierarchy')/base_cpu:.2f}",
        "E_D_CPU_RATIO": f"{cpu('E_unified')/cpu('D_hierarchy'):.2f}",
        "MODEL_COUNT": f"{metadata['model_count']:,}",
        "EPISODES": f"{metadata['sampled_training_episodes']:,}",
        "BANK_EPISODES": f"{metadata['bank_episodes_executed']:,}",
        "RUN_SECONDS": f"{metadata['elapsed_seconds']:.1f}",
        "BANK_SECONDS": f"{sum(metadata['bank_seconds'].values()):.1f}",
        "PROCESSOR": metadata["processor"], "PLATFORM": metadata["platform"],
        "CPU_COUNT": str(metadata["cpu_count"]),
    }
    report = (ROOT / "REPORT_TEMPLATE.md").read_text(encoding="utf-8")
    for key, value in replacements.items():
        report = report.replace("{{" + key + "}}", value)
    if "{{" in report:
        raise RuntimeError("Unresolved report placeholder")
    (ROOT / "Frontier_AI_Alignment_Benchmark_Report.md").write_text(report, encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold"})
    names = ["Baseline", "A: Rta", "B: Nyaya", "C: Process", "D: Hierarchy", "E: Unified"]
    colors = ["#64748b", "#0ea5e9", "#8b5cf6", "#d97706", "#14b8a6", "#166534"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.12, 1, 0.88))
    for ax, metric, scenario, split, title in [
        (axes[0], "truth", "main", "id", "True held-out accuracy"),
        (axes[1], "exploit", "main", "id", "Reward-hacking rate"),
        (axes[2], "exploit", "blindspot", "ood", "Hacking with shared blind spots")]:
        values = [mean(m, metric, scenario, split) for m in MAIN]
        ax.barh(names, values, color=colors, height=0.65)
        ax.set_xlim(0, 110)
        ax.invert_yaxis()
        ax.set_title(title, pad=14)
        ax.set_xlabel("Percent")
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        for y, v in enumerate(values):
            ax.text(v + 1, y, f"{v:.1f}", va="center", fontsize=9)
    fig.suptitle("Constraints work only as well as their checkers", fontsize=19, weight="bold")
    fig.text(0.5, 0.02, "24 training seeds • 3 synthetic suites • finite program policies, no LLM training\n"
             "Blind-spot column: separate adversarial retraining, then larger-task evaluation",
             ha="center", fontsize=10, color="#475569")
    fig.savefig(RESULTS / "leaderboard.png", dpi=180, bbox_inches="tight")
    fig.savefig(RESULTS / "leaderboard.svg", bbox_inches="tight")
    plt.close(fig)
    (RESULTS / "release_metrics.json").write_text(json.dumps(replacements, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in replacements.items() if not k.endswith("TABLE")}, indent=2))


if __name__ == "__main__":
    main()
