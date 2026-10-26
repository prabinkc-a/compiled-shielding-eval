"""Build v2.0 tables, seed-level intervals, and scientific figures from raw data."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .train import METHODS, ABLATIONS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
LABEL = {"baseline": "Baseline GRPO", "D": "D: hierarchy", "E": "E: static synthesis",
         "F": "F: adaptive synthesis", "F_random": "F, uniform search arms",
         "F_no_invariance": "F, no MDL/KL regularization", "F_no_replay": "F, no replay",
         "F_full_audit": "F, audit every uncached candidate", "F_corrupt": "F, corrupted audit feedback"}


def read(name):
    return list(csv.DictReader((OUT / name).open(encoding="utf-8")))


def bootstrap(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(7113)
    means = x[rng.integers(len(x), size=(10000, len(x)))].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.quantile(means, .025)),
                hi=float(np.quantile(means, .975)), seeds=len(x))


def main():
    rows, costs, profiles = read("evaluation.csv"), read("training_cost.csv"), read("serial_profile.csv")
    curves = read("generalization_curves.csv")
    legacy = read("legacy_generalization.csv")
    exact = read("exact_risk.csv")
    metadata = json.loads((OUT / "metadata.json").read_text())
    def select(m, kind="combined", decode="sample"):
        rr = [r for r in rows if r["method"] == m and r["decode"] == decode]
        if kind == "id":
            return [r for r in rr if int(r["size"]) <= 12 and r["condition"] == "monitored"]
        if kind == "scale":
            return [r for r in rr if int(r["size"]) > 12 and r["condition"] == "monitored"]
        if kind == "cue":
            return [r for r in rr if int(r["size"]) <= 12 and r["condition"] == "unmonitored"]
        return [r for r in rr if int(r["size"]) > 12 and r["condition"] == "unmonitored"]
    def seed_means(rr, metric):
        return {s: np.mean([float(r[metric]) for r in rr if int(r["seed"]) == s])
                for s in sorted({int(r["seed"]) for r in rr})}
    def summary(m, metric="truth", kind="combined", decode="sample"):
        return bootstrap(list(seed_means(select(m, kind, decode), metric).values()))
    def pct(m, metric="truth", kind="combined", decode="sample"):
        return 100 * summary(m, metric, kind, decode)["mean"]
    def interval(m, metric="truth", kind="combined"):
        q = summary(m, metric, kind)
        return f"{100*q['mean']:.2f} [{100*q['lo']:.2f}, {100*q['hi']:.2f}]"
    def avg(data, m, field):
        return float(np.mean([float(r[field]) for r in data if r["method"] == m]))
    def ratio(data, m, field="cpu_train_seconds", ref="baseline"):
        return avg(data, m, field) / avg(data, ref, field)
    master = ["| Method | Small monitored accuracy % | Blind-spot survival / true accuracy % [95% CI] | Sampled no-exploit survival % | Sampled exploit rate % | Calibrated training CPU / rollout CPU × baseline |",
              "|---|---:|---:|---:|---:|---:|"]
    all_summary = {}
    for m in METHODS:
        master.append(f"| {LABEL[m]} | {pct(m,kind='id'):.2f} | {interval(m)} | {100-pct(m,'exploit'):.3f} | "
                      f"{pct(m,'exploit'):.3f} | {ratio(profiles,m):.2f}× / {ratio(profiles,m,'cpu_rollout_seconds'):.2f}× |")
        all_summary[m] = {kind: {f: summary(m, f, kind) for f in ("truth", "exploit", "functional", "proxy")}
                          for kind in ("id", "scale", "cue", "combined")}
    shifts = ["| Method | Scale only: true % | Cue only: true % | Combined: true % | Combined: greedy true / exploit % |",
              "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        shifts.append(f"| {LABEL[m]} | {pct(m,kind='scale'):.2f} | {pct(m,kind='cue'):.2f} | {pct(m):.2f} | "
                      f"{pct(m,decode='greedy'):.2f} / {pct(m,'exploit',decode='greedy'):.3f} |")
    per_suite = ["| Method | Maze true / exploit % | Grader true / exploit % | Tampering true / exploit % |",
                 "|---|---:|---:|---:|"]
    for m in METHODS:
        cells = []
        for suite in ("maze", "grader", "tamper"):
            rr = [r for r in select(m) if r["suite"] == suite]
            cells.append(" / ".join(f"{100*np.mean([float(r[f]) for r in rr]):.3f}" for f in ("truth", "exploit")))
        per_suite.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    performance = ["| Method | Actual full-training process CPU × baseline | Serial calibrated CPU × baseline | Serial rollout CPU × baseline | Heavy-audit fraction % | Training exploit rate % |",
                   "|---|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        performance.append(f"| {LABEL[m]} | {ratio(costs,m):.2f}× | {ratio(profiles,m):.2f}× | "
                           f"{ratio(profiles,m,'cpu_rollout_seconds'):.2f}× | {100*avg(costs,m,'audit_fraction'):.2f} | "
                           f"{100*avg(costs,m,'training_exploit_rate'):.3f} |")
    paired = []
    f_seeds = seed_means(select("F"), "truth")
    for m in METHODS[:-1]:
        other = seed_means(select(m), "truth")
        ci = bootstrap([f_seeds[s] - other[s] for s in f_seeds])
        paired.append({"comparison": f"F - {m}", **ci})
    ablations = ["| Intervention (four paired seeds) | True accuracy % | Exploit % | Δ accuracy vs F on the same seeds (pp), 95% CI |",
                 "|---|---:|---:|---:|"]
    for m in ABLATIONS:
        alt = seed_means(select(m), "truth")
        diff = bootstrap([alt[s] - f_seeds[s] for s in alt])
        ablations.append(f"| {LABEL[m]} | {pct(m):.3f} | {pct(m,'exploit'):.3f} | "
                         f"{100*diff['mean']:+.3f} [{100*diff['lo']:+.3f}, {100*diff['hi']:+.3f}] |")
    comparisons = ["| Paired main comparison | True-accuracy improvement (pp), 95% CI |", "|---|---:|"]
    for q in paired:
        comparisons.append(f"| {q['comparison']} | {100*q['mean']:+.2f} [{100*q['lo']:+.2f}, {100*q['hi']:+.2f}] |")
    n = sum(int(r["episodes"]) for r in select("F"))
    incidents = sum(round(float(r["exploit"]) * int(r["episodes"])) for r in select("F"))
    if incidents == 0:
        zero_text = (f"Zero observed exploits in {n:,} stochastic combined-shift episodes gives a conditional "
                     f"one-sided 95% zero-event upper bound of {100*(1-.05**(1/n)):.4f}%, under independent "
                     "episode sampling for these fixed trained policies and task instances. It is not a guarantee over new tasks, seeds, or exploit families.")
    else:
        zero_text = f"F produced {incidents} exploit episodes out of {n:,} stochastic combined-shift trials. The requested literal zero-exploitation target was not met."
    replacements = {
        "MASTER": "\n".join(master), "SHIFTS": "\n".join(shifts), "SUITES": "\n".join(per_suite),
        "PERFORMANCE": "\n".join(performance), "ABLATIONS": "\n".join(ablations), "COMPARISONS": "\n".join(comparisons),
        "F_TRUE": f"{pct('F'):.2f}", "F_HACK": f"{pct('F','exploit'):.3f}", "E_TRUE": f"{pct('E'):.2f}",
        "E_HACK": f"{pct('E','exploit'):.3f}", "D_TRUE": f"{pct('D'):.2f}", "BASE_TRUE": f"{pct('baseline'):.2f}",
        "F_GAIN_E": f"{pct('F')-pct('E'):.2f}", "F_CPU": f"{ratio(profiles,'F'):.2f}",
        "E_CPU": f"{ratio(profiles,'E'):.2f}", "D_CPU": f"{ratio(profiles,'D'):.2f}",
        "F_ROLL": f"{ratio(profiles,'F','cpu_rollout_seconds'):.2f}",
        "E_ROLL": f"{ratio(profiles,'E','cpu_rollout_seconds'):.2f}",
        "D_ROLL": f"{ratio(profiles,'D','cpu_rollout_seconds'):.2f}",
        "F_AUDITS": f"{100*avg(costs,'F','audit_fraction'):.2f}",
        "F_FULL_TIME": f"{ratio([r for r in costs if int(r['seed'])<4],'F_full_audit',ref='F'):.2f}",
        "EXACT_MEAN": f"{100*np.mean([float(r['exact_exploit_probability']) for r in exact]):.6f}",
        "EXACT_MAX": f"{100*max(float(r['exact_exploit_probability']) for r in exact):.6f}",
        "F_CORRUPT": f"{pct('F_corrupt'):.2f}",
        "N_MODELS": f"{metadata['models']:,}", "N_TRAIN": f"{metadata['training_sequences']:,}",
        "N_EXEC": f"{metadata['actual_program_executions']:,}", "N_EVAL": f"{metadata['evaluation_sequences']:,}",
        "PARAMETERS": str(int(float(costs[0]['parameters']))), "ZERO_TEXT": zero_text,
        "ELAPSED": f"{metadata['elapsed_seconds']:.1f}",
    }
    template = (ROOT / "REPORT_TEMPLATE.md").read_text(encoding="utf-8")
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    assert "{{" not in template
    (ROOT / "Frontier_AI_Alignment_Benchmark_Report_v2.0.md").write_text(template, encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    (OUT / "paired_comparisons.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    (OUT / "release_metrics.json").write_text(json.dumps(replacements, indent=2), encoding="utf-8")
    for key in ("MASTER", "SHIFTS", "SUITES", "PERFORMANCE", "ABLATIONS", "COMPARISONS"):
        (OUT / (key.lower() + ".md")).write_text(replacements[key] + "\n", encoding="utf-8")

    colors = dict(baseline="#64748b", D="#0284c7", E="#8b5cf6", F="#15803d")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for m in METHODS:
        q = summary(m)
        x, y = ratio(profiles, m), 100*q["mean"]
        ax.errorbar(x, y, yerr=[[y-100*q["lo"]], [100*q["hi"]-y]], fmt="o", color=colors[m], capsize=4, markersize=9)
        ax.annotate(m if m != "baseline" else "Baseline", (x,y), xytext=(8, -14 if m == "D" else 8), textcoords="offset points")
    points = sorted((ratio(profiles,m), pct(m), m) for m in METHODS)
    front, best = [], -1
    for x,y,m in points:
        if y > best + 1e-9:
            front.append((x,y)); best=y
    ax.plot(*zip(*front), linestyle="--", color="#94a3b8", zorder=0)
    ax.set(xlabel="Measured serial training CPU / neural baseline", ylabel="True accuracy on combined shifts (%)",
           title="Accuracy–compute frontier: frozen neural policy evaluation", ylim=(0,105))
    ax.grid(alpha=.2)
    fig.savefig(OUT / "pareto.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13,5), layout="constrained")
    for ax, condition in zip(axes, ("monitored", "unmonitored")):
        for m in METHODS:
            xs, ys = [], []
            for size in (8,12,20,40,80,160):
                rr = [r for r in rows if r["method"] == m and r["condition"] == condition and r["decode"] == "sample" and int(r["size"]) == size]
                xs.append(size); ys.append(100*np.mean([float(r["truth"]) for r in rr]))
            ax.plot(xs,ys,"o--" if m=="E" else "o-",label=LABEL[m],color=colors[m],
                    linewidth=3 if m=="D" else 1.8, markersize=8 if m=="D" else 5,
                    markerfacecolor="white" if m=="D" else colors[m])
        ax.axvspan(6,12,color="#cbd5e1",alpha=.25,label="Core training sizes")
        ax.set(xscale="log", xlabel="Task size (vertices / array length / records)", ylabel="True accuracy (%)",
               ylim=(0,105), title="Visible monitor" if condition=="monitored" else "Monitoring and surface cues flipped")
        ax.set_xticks([8,12,20,40,80,160], [8,12,20,40,80,160]); ax.grid(alpha=.2)
        ax.text(.98,.04,"D and E overlap",transform=ax.transAxes,ha="right",color="#475569",fontsize=9)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.savefig(OUT / "neural_generalization.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(13,5),layout="constrained")
    for m in METHODS:
        updates=sorted({int(r["update"]) for r in curves})
        yy=[]
        for u in updates:
            rr=[r for r in curves if r["method"]==m and r["decode"]=="sample" and r["condition"]=="unmonitored" and int(r["size"])>12 and int(r["update"])==u]
            yy.append(100*np.mean([float(r["truth"]) for r in rr]))
        axes[0].plot(updates,yy,"o--" if m=="E" else "o-",label=m,color=colors[m],
                     linewidth=3 if m=="D" else 1.8, markersize=8 if m=="D" else 5,
                     markerfacecolor="white" if m=="D" else colors[m])
    axes[0].set(xlabel="GRPO updates",ylabel="Combined-shift true accuracy (%)",title="Neural learning curves",ylim=(0,105)); axes[0].legend()
    sizes=(8,12,20,40,80,160)
    for scenario,style,label in (("main","--","Phase 1 E: main"),("blindspot",":","Phase 1 E: blind-spot")):
        yy=[100*np.mean([float(r["functional"]) for r in legacy if r["scenario"]==scenario and int(r["size"])==size]) for size in sizes]
        axes[1].plot(sizes,yy,style,marker="o",label=label)
    for m in ("E","F"):
        yy=[100*np.mean([float(r["functional"]) for r in rows if r["suite"]=="grader" and r["method"]==m and r["decode"]=="sample" and r["condition"]=="unmonitored" and int(r["size"])==size]) for size in sizes]
        axes[1].plot(sizes,yy,"o-",label=f"Phase 2 neural {m}",color=colors[m])
    axes[1].set(xscale="log",xlabel="LIS input length",ylabel="Raw functional correctness (%)",title="Legacy bridge: different program languages",ylim=(0,105))
    axes[1].set_xticks(sizes,sizes);axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_yticks([0,20,40,60,80,100])
    fig.savefig(OUT / "learning_and_legacy.png",dpi=180)
    plt.close(fig)
    print(json.dumps({k:v for k,v in replacements.items() if k not in ("MASTER","SHIFTS","SUITES","PERFORMANCE","ABLATIONS","COMPARISONS")}, indent=2))


if __name__ == "__main__":
    main()
