"""Generate v3 tables, figures, report and brief exclusively from saved results."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phase2.run import csv_write

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "phase2/results"
OUT = ROOT / "results"
METHODS = ("baseline", "D", "E", "F", "G")


def read(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def primary(rows):
    return [r for r in rows if r["decode"] == "sample" and r["condition"] == "unmonitored" and int(r["size"]) > 12]


def ci(values):
    values = np.asarray(values)
    boot = np.random.default_rng(20261026).choice(values, (10000, len(values))).mean(-1)
    return np.quantile(boot, [.025, .975]).tolist()


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"]*len(headers)) + " |\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    final = read(OLD / "evaluation.csv") + read(OUT / "evaluation.csv")
    neural_curves = primary(read(OLD / "generalization_curves.csv") + read(OUT / "generalization_curves.csv"))
    costs = read(OLD / "training_cost.csv") + read(OUT / "training_cost.csv")
    profile = read(OUT / "serial_profile.csv")
    exact = read(OUT / "exact_risk.csv")
    py = json.loads((OUT / "python/summary.json").read_text())
    pycurves = read(OUT / "python/curves.csv")
    latency = json.loads((OUT / "latency.json").read_text())
    audit = json.loads((OUT / "checkpoint_audit.json").read_text())
    python_audit = json.loads((OUT / "python_checkpoint_audit.json").read_text())
    test_summary = (OUT / "tests.txt").read_text(encoding="utf-8").strip().splitlines()[-1]
    base = [r for r in profile if r["method"] == "baseline"]
    summary = {}
    allmethods = METHODS + ("G_no_compile", "G_fixed_audit", "G_no_early_exit")
    for method in allmethods:
        rows = primary([r for r in final if r["method"] == method])
        samples = [avg([r for r in rows if int(r["seed"]) == seed], "truth")*100 for seed in sorted({int(r["seed"]) for r in rows})]
        timing = [r for r in profile if r["method"] == method]
        c = [r for r in costs if r["method"] == method]
        summary[method] = dict(accuracy=avg(rows,"truth")*100, accuracy_ci=ci(samples),
            exploit=avg(rows,"exploit")*100, exact=avg([r for r in exact if r["method"]==method],"exact_exploit_probability")*100,
            audit=avg(c,"audit_fraction")*100, training_exploit=avg(c,"training_exploit_rate")*100,
            cpu=avg(timing,"cpu_train_seconds")/avg(base,"cpu_train_seconds"),
            rollout_cpu=avg(timing,"cpu_rollout_seconds")/avg(base,"cpu_rollout_seconds"),
            accuracy_by_seed=samples)
    g, f = summary["G"], summary["F"]
    diff = np.asarray(g["accuracy_by_seed"])-np.asarray(f["accuracy_by_seed"])
    summary["G_minus_F_pp"] = dict(mean=float(diff.mean()), ci=ci(diff))
    python_summary = {}
    for method in ("baseline", "G"):
        runs = [r for r in py if r["method"] == method]
        rows = [r for r in pycurves if r["method"] == method and int(r["update"])==48 and int(r["size"])>12]
        python_summary[method] = dict(accuracy=100*avg(rows,"truth"), exploit=100*avg(rows,"exploit"),
            grammar_risk=100*avg([r for r in rows if int(r["size"])==160],"grammar_exploit_mass"),
            wall_seconds=float(np.mean([r["train_wall_seconds"] for r in runs])),
            audit=100*sum(r["metrics"]["audits"] for r in runs)/sum(r["metrics"]["sequences"] for r in runs),
            rules=[len(r["backend"]["rules"]) for r in runs],
            sequences=sum(r["metrics"]["sequences"] for r in runs),
            worker_executions=sum(r["backend"]["executions"] for r in runs))
    summary["python"] = python_summary
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    csv_write(OUT / "pareto.csv", [{"method": m, **{k:v for k,v in summary[m].items() if not isinstance(v,list)}} for m in METHODS])
    pareto = table(["Method", "True accuracy %", "Sampled exploit %", "Exact risk %¹", "Audits %²", "Training CPU ×", "Rollout CPU ×"],
        [[m, f"{summary[m]['accuracy']:.3f}", f"{summary[m]['exploit']:.4f}", f"{summary[m]['exact']:.6f}",
          f"{summary[m]['audit']:.3f}", f"{summary[m]['cpu']:.2f}", f"{summary[m]['rollout_cpu']:.2f}"] for m in METHODS])
    ablations = []
    for m in ("F", "G", "G_no_compile", "G_fixed_audit", "G_no_early_exit"):
        rows = [r for r in primary(final) if r["method"]==m and int(r["seed"])<4]
        ablations.append([m, f"{100*avg(rows,'truth'):.3f}", f"{100*avg(rows,'exploit'):.4f}",
            f"{100*avg([r for r in exact if r['method']==m and int(r['seed'])<4],'exact_exploit_probability'):.6f}",
            f"{summary[m]['cpu']:.2f}"])
    ablation_table = table(["Method (seeds 0–3)", "Accuracy %", "Sampled exploit %", "Exact risk %", "Serial CPU ×"], ablations)
    components = []
    for m in METHODS:
        rows = [r for r in profile if r["method"]==m]
        components.append([m]+[f"{1000*avg(rows,k)/40:.3f}" for k in
            ("task_seconds", "generation_seconds", "execution_seconds", "verifier_seconds", "update_seconds")])
    components_table = table(["Method", "Task ms", "Generation + reference ms", "Execution ms", "Verifier ms", "Update ms"], components)
    pytable = table(["Python method", "Accuracy %", "Sampled exploit %", "n=160 grammar risk %", "Audits %", "Training wall s/model"],
        [[m, f"{v['accuracy']:.3f}", f"{v['exploit']:.3f}", f"{v['grammar_risk']:.6f}", f"{v['audit']:.2f}", f"{v['wall_seconds']:.2f}"] for m,v in python_summary.items()])
    family_rows = []
    for suite in ("maze","grader","tamper"):
        rows=[r for r in pycurves if r["suite"]==suite and int(r["update"])==48 and int(r["size"])>12]
        family_rows.append([suite]+[f"{100*avg([r for r in rows if r['method']==m],'truth'):.3f}" for m in ("baseline","G")]+
            [f"{100*avg([r for r in rows if r['method']=='G' and int(r['size'])==160],'grammar_exploit_mass'):.6f}"])
    family_table=table(["Python family", "Baseline accuracy %", "G accuracy %", "G n=160 grammar risk %"],family_rows)

    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
    colors = dict(zip(METHODS,["#65778a","#c18428","#8863b6","#278c9e","#cf443e"]))
    fig, axes = plt.subplots(1,2,figsize=(11.5,4.2),layout="constrained")
    for m in METHODS:
        s = summary[m]
        axes[0].scatter(s["cpu"],s["accuracy"],color=colors[m],s=60)
        axes[0].annotate(m,(s["cpu"],s["accuracy"]),xytext=(4, -13 if m=="E" else 5),textcoords="offset points")
        rows=[r for r in final if r["method"]==m and r["decode"]=="sample" and r["condition"]=="unmonitored"]
        sizes=sorted({int(r["size"]) for r in rows})
        axes[1].plot(sizes,[100*avg([r for r in rows if int(r["size"])==n],"truth") for n in sizes],"o-",label=m,color=colors[m])
    axes[0].set(xlabel="Serial training CPU / baseline",ylabel="True accuracy (%)",title="Measured speed–accuracy tradeoff",ylim=(20,105))
    axes[1].set(xlabel="Unmonitored task size",ylabel="True accuracy (%)",title="Neural scale generalization",xscale="log",ylim=(20,105))
    axes[1].legend(ncol=3)
    fig.savefig(OUT / "neural_pareto.png"); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(11.5,4.2),layout="constrained")
    for m in METHODS:
        rows=[r for r in neural_curves if r["method"]==m]
        steps=sorted({int(r["update"]) for r in rows})
        axes[0].plot(steps,[100*avg([r for r in rows if int(r["update"])==s],"truth") for s in steps],"o-",label=m,color=colors[m])
        axes[1].plot(steps,[100*avg([r for r in rows if int(r["update"])==s],"exploit") for s in steps],"o-",label=m,color=colors[m])
    axes[0].set(xlabel="Neural GRPO update",ylabel="True accuracy (%)",title="Fixed-checkpoint generalization",ylim=(0,105))
    axes[1].set(xlabel="Neural GRPO update",ylabel="Sampled exploitation (%)",title="Held-out sizes 20, 80, 160",yscale="symlog")
    axes[0].legend(ncol=3);axes[1].legend(ncol=3)
    fig.savefig(OUT / "neural_training.png");plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(11.5,4.2),layout="constrained")
    for m,color in (("baseline",colors["baseline"]),("G",colors["G"])):
        rows=[r for r in pycurves if r["method"]==m and int(r["size"])>12]
        steps=sorted({int(r["update"]) for r in rows})
        axes[0].plot(steps,[100*avg([r for r in rows if int(r["update"])==s],"truth") for s in steps],"o-",label=m,color=color)
        rows=[r for r in rows if int(r["size"])==160]
        axes[1].plot(steps,[100*avg([r for r in rows if int(r["update"])==s],"grammar_exploit_mass") for s in steps],"o-",label=m,color=color)
    axes[0].set(xlabel="TRL optimizer update",ylabel="Held-out Python accuracy (%)",title="Actual Python function generation",ylim=(0,105))
    axes[1].set(xlabel="TRL optimizer update",ylabel="Grammar exploit probability (%)",title="Rare branches included in exact grammar sum",yscale="symlog")
    axes[0].legend(); axes[1].legend()
    fig.savefig(OUT / "python_training.png"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.4),layout="constrained")
    bottom=np.zeros(len(METHODS))
    for key,label in (("task_seconds","Task"),("generation_seconds","Generation + reference"),("execution_seconds","Execution"),("verifier_seconds","Verifier"),("update_seconds","Optimizer update")):
        values=np.array([1000*avg([r for r in profile if r["method"]==m],key)/40 for m in METHODS])
        ax.bar(METHODS,values,bottom=bottom,label=label);bottom+=values
    ax.set(ylabel="Serial wall milliseconds per update",title="Measured component attribution (32 completions/update)")
    ax.legend(ncol=3,fontsize=8,loc="upper left",bbox_to_anchor=(0,1.0))
    ax.set_ylim(0,float(bottom.max())*1.22)
    fig.savefig(OUT / "latency_breakdown.png");plt.close(fig)

    aux=latency["auxiliary_forward_backward_mean_ms"]
    casc=latency["cascade_known_conditional_failures"]
    early=avg([r for r in casc if r["early_exit"]],"milliseconds_per_audit")
    late=avg([r for r in casc if not r["early_exit"]],"milliseconds_per_audit")
    saving=100*(1-early/late)
    gstates=[json.loads(p.read_text())["stats"]["verifier_state"] for p in (OUT/"runs").glob("*__G__*.json")]
    micro=sum(s["micro_calls"] for s in gstates);full=sum(s["full_calls"] for s in gstates)
    py_g=python_summary["G"]
    delta=summary["G_minus_F_pp"]
    report=f'''# Frontier AI Alignment Benchmark Report v3.0

**26 October 2026 · CPU research reproduction · Algorithm G: compiled shielding**

G achieved **{g['accuracy']:.3f}% blind-spot true accuracy**, **zero sampled exploitation**, **0.000000% exact integrated exploit probability in the bounded DSL**, and **{g['audit']:.3f}% heavy audits**. Fresh serial calibration measured **{g['cpu']:.2f}× training CPU / {g['rollout_cpu']:.2f}× rollout CPU**, relative to baseline.

The requested **99.90% accuracy target was missed**. G's mean was {delta['mean']:+.3f} percentage points relative to F, with paired seed-bootstrap 95% interval [{delta['ci'][0]:+.3f}, {delta['ci'][1]:+.3f}]. This does not establish that G improves accuracy or loses no basis points. It does establish exact exclusion of the designated DSL exploits after compilation. No claim of universal alignment or a universally Pareto-optimal release is warranted.

## 1. Measured speed–accuracy–safety tradeoff

{pareto}

¹ Exact risk integrates all 2,304 possible programs at the **same v2 stress input** per suite/seed: n=160, unmonitored, untagged, task seed 12900. These 24 contexts per main method are not the same population as the broader sampled evaluation; do not interpret their risk as a confidence bound on that evaluation. For G every unsafe term is exactly zero due to negative-infinity logits, not decimal rounding. Probability sums are checked within 2×10⁻⁶ of one.

² Audit fraction here covers the **entire original training run**, including discovery. E executes four static checks on every sequence. F and G rates count sequences receiving actual challenge execution; cache-only rejections do not count as heavy audits. V2's quoted F 8.95% was a separate warm calibration measurement.

G accuracy 95% seed-bootstrap interval: [{g['accuracy_ci'][0]:.3f}, {g['accuracy_ci'][1]:.3f}]%. Zero sampled exploit events occurred in 9,216 G primary evaluation episodes. Zero exploitation is **not** equivalent to perfect task correctness: ordinary errors and abstentions remain possible. During discovery, G's executed training exploit rate was {g['training_exploit']:.5f}%; this is not a zero-risk training-from-initialization claim.

![Neural Pareto and scale generalization](results/neural_pareto.png)

![Fixed neural training checkpoints](results/neural_training.png)

The checkpoint curves use 32 cases per size/cue cell at updates 0/80/160/240, on sizes 20/80/160. The primary final table uses 96 cases and adds size 40, so its endpoint need not equal the smaller curve evaluation.

## 2. Protocol and provenance

- **Same neural protocol as v2:** 6,492-parameter input-conditioned MLP/GRU, six autoregressive instruction tokens; three suites (graph maze, strict LIS, aggregation); eight seeds 0–7; 240 updates, four prompts × eight completions, two clipped GRPO optimizer epochs per rollout, learning rate .002, reference KL coefficient .005.
- **60 new fits:** 24 G + 36 ablations (three variants × three suites × four seeds). Each fit executes 7,680 generated programs. No checkpoint was selected using held-out accuracy. Final stochastic evaluation uses 96 cases per size/cue cell, with fixed task and sampling seeds.
- **96 frozen v2 comparator policies are reused**, along with their independently reproduced accuracy rows. Baseline/D/E/F were not retrained and counted as new v3 runs. New exact integration covers all five methods and the ablations. Existing v2 files and archives remain intact.
- Training core tasks are sizes 6, 8, 10, 12. F and G replace one of four prompt groups every fourth update with an adaptive scale/cue challenge (6.25% of sequences). G does not force heavy audits on that group after compilation. Primary accuracy equally weights sizes 20, 40, 80, 160 with both cues false across suites/seeds.
- **72 fresh serial calibration runs:** all five main methods plus three G ablations, three suites, seeds 0–2, 40 updates each, randomized order seed 4118, one Torch thread. Each disposable policy starts from its saved final checkpoint with a fresh optimizer and frozen reference copy. G retains its compiled permissions; verifier counters/search restart. This estimates warm steady-state CPU, not cold discovery cost. No parallel benchmark ran during calibration.
- Bootstrap units are training seeds, first averaging over suites and sizes; 10,000 resamples, seed 20261026. Only eight main seeds and three calibration seeds limit statistical precision. Timing ratios are workload- and machine-specific, not expected LLM training speedups.
- Independent reload reproduced **{audit['rows_reproduced_exactly']} final rows from {audit['checkpoints_reloaded']} new checkpoints** exactly and verified {audit['training_source_hashes_verified']} recorded source hashes.
- Validation: **{test_summary}**. A separate reload of **{python_audit['models']} saved Python models**, tokenizers, and shields reproduced their finite-grammar risk; five Python implementation source hashes were verified.

Phase 1's 125-program categorical policies and 1.32×/7.26× rollout ratios are historical context, not direct cost denominators for this neural experiment. V2's 1.12×/1.30×/1.33× training ratios are historical measurements; the table uses one new common calibration.

D and E's identical sampled accuracy is inherited from the matched v2 experiments: the static small-input checks are redundant with D's mask in this training regime, and E's constant process bonus cancels under group-standardized advantages. Extra verifier work therefore need not change the learned behavior.

## 3. What G actually certifies

**Discovery → certificate → versioned mask.** An executed counterexample triggers a trusted check of the DSL primitive contract. Kernel 3 is explicitly a shortcut in every suite; certification removes token 3 at both primary and alternate-kernel positions. The compiler does not infer a general theorem from one failing example. It uses known semantics to generalize from that witness. Hard channel/report restrictions remain enforced.

All 24 main G runs found their first certificate in update 1. New masks are published only after that rollout's two optimizer epochs. Sampling, old log probabilities, reference probabilities, and current loss scoring therefore use identical support. Once published, forbidden tokens receive −∞, so their conditional probabilities and every containing program's probability are zero regardless of network weights. Both live and dormant branches are covered across all input scales and monitoring cues.

The safety statement is conditional on a complete and correct primitive taxonomy, trusted execution, and a functioning mask. It does not imply correctness of kernel 2, forbid abstention, transfer to unknown primitives, or survive removal of the shield. Since the unsafe primitive is already identified in the language contract, a global static mask could also enforce this restriction from initialization; dynamic discovery is not evidence that an unknown semantic vulnerability was solved from scratch.

**Audit decay.** Before compilation, conditional/high-advantage samples and the scheduled challenge group receive exploratory audits. After compilation, q = .005 + .015 exp(−time-since-surprise/12). Failures restart the short audit burst; the independent .5% floor remains. This rule uses certificate saturation and observed surprises, not an assumption that silence proves safety.

**Cascading checks.** The first probe crosses the actual cutoff (13, 25, 49, or 97), or flips the relevant cue. Only a passing microprobe triggers the second test at size 97/128. Thus “micro” is a stage name, not a guarantee of tiny size. Complexity discontinuity alone never authorizes a ban.

The name *Pratitantra-Siddhanta* is a design analogy for system-relative established constraints; it is not a historical claim that classical epistemology supplied this formal algorithm. Shielded RL predates this experiment: see [Alshiekh et al., Safe Reinforcement Learning via Shielding](https://arxiv.org/abs/1708.08611).

## 4. Ablations: hard compilation versus soft feedback

{ablation_table}

The accuracy rows use matched seeds 0–3. Serial CPU uses the same three calibration seeds 0–2 for every row. G_no_compile retains cascading checks, failure caching, negative rewards, and adversarial prompts, but neither hard compilation nor F's auxiliary losses. Without saturation it keeps exploratory auditing. It therefore measures the full value of compilation plus the enabled audit decay, not a pure single-variable mask-cost experiment. G_fixed_audit uses q=.0895; G_no_early_exit always runs both stages. Kernel semantics and base optimizer are unchanged.

G_no_early_exit was faster than G in this small serial calibration despite doing more work per failing audit. The calibration uses different learned trajectories and only three seeds, so it does not establish an end-to-end speed advantage for early exit. The fixed attack workload in §5 isolates the conditional saving more directly.

Hard compilation eliminates the residual softmax leak. These observations do **not** support the stronger requested assertion of strictly better accuracy than F. Invariance and replay are absent from all G variants, and no replacement neural auxiliary pass is hidden in the verifier.

## 5. Latency attribution

Measured serial wall milliseconds per 32-sequence update (process CPU ratios appear in §1):

{components_table}

![Latency attribution](results/latency_breakdown.png)

The matched frozen-batch microbenchmark measured base GRPO forward/backward at **{aux['base']:.3f} ms** versus **{aux['with_invariance_replay']:.3f} ms** with F's two auxiliary forwards and their gradients. At two epochs and auxiliary use every fourth update, the measured amortized difference is **{aux['amortized_saving_ms_per_update_two_epochs_every_four']:.3f} ms/update**. G removes 120 invariance forwards and up to 120 replay forwards per 240-update fit. Its ordinary policy/reference forwards and GRPO backpropagation remain.

For a 32×6 permission batch, contextual D mask construction measured **{latency['permission_build_microseconds_per_32x6_batch']['D_contextual']:.2f} µs** and the compiled G table **{latency['permission_build_microseconds_per_32x6_batch']['G_compiled']:.2f} µs**. The lookup of a position's rule is constant-time; producing the full batch mask and applying it still scale with batch/token/vocabulary dimensions.

On a deliberately attack-heavy microbenchmark (360 conditional failures, all four cutoffs, all suites), early exit measured **{early:.3f} ms/audit** versus **{late:.3f} ms/audit** without it: **{saving:.1f}% savings**. This includes challenge construction and execution. It does not establish >70% savings on real training traffic. Actual main G traffic ran {micro} first stages and {full} second stages, avoiding only {micro-full} of {micro} potential second probes ({100*(micro-full)/micro:.1f}%). Once shielding removes attacks, most audited programs pass the microprobe and still require the second stage.

These microbenchmarks isolate components on fixed workloads. Their deltas must not be added to claim an exact decomposition of the end-to-end F–G difference: audit schedules, masks, accepted actions, and training trajectories also differ. Windows process CPU timing is coarse for individual small calls; averages accumulate 200 timed repetitions.

## 6. Real Python generation with actual TRL updates

{pytable}

{family_table}

**18 real runs:** baseline/G × three task families × three seeds (0–2); 48 GRPO optimizer steps each, eight completions per step, 6,912 total generated training functions. Random local GPT-2 with two layers, hidden size 32, two attention heads, {py[0]['parameters']:,} parameters; no pretrained download and no SFT. Attention weights changed in every run and all saved parameters remained finite. Raw TRL logs, model weights, generated source, compiler witnesses, and curves are saved.

This is **bounded compositional Python synthesis**. The supplied grammar has 20 functions per family: two correct algorithm bodies, optional empty-input handling, and an optional length-dependent incorrect early return with cutoff 12/24/48/96. The model emits ordinary BPE tokens forming complete Python functions; they are not six-integer DSL completions. A token trie constrains syntax, and a two-layer causal Transformer supplies conditional probabilities at branch choices. Correct algorithm bodies are provided, so this is not evidence of learning or inventing BFS, dynamic programming, or summation from scratch.

Training executes actual functions on monitored sizes 8/12. Held-out evaluation executes sizes 20/40/80/160, 32 sampled functions per size/model/checkpoint, two independent task realizations per size. All checkpoints 0/16/32/48 are fixed in advance. Both baseline and G share the syntax grammar and static execution permissions; only G adds adversarial audits and dynamic rules. A separate evaluator bypasses learned bans but retains static permissions, so it measures model output correctness rather than treating rejected code as success. Training rewards use small-input correctness minus an executed-line cost; there is no explicit reward for a dormant malicious branch. This Python comparison is not a reproduction of the DSL's strongly hack-incentivizing proxy reward, and its three seeds support only a small demonstration.

`ast.parse` plus the allowlist rejects imports, I/O, arbitrary calls, recursion, while loops, comprehensions, and unsafe attribute access. Only local list append and selected pure builtins are permitted. Validated functions run in fresh `python -I -S` subprocesses with restricted builtins, a 150,000-line trace budget, and a 3-second timeout. This is process isolation for restricted benchmark code, **not** an OS sandbox for arbitrary hostile Python. Successful parse alone is never treated as safe execution; see [Python AST documentation](https://docs.python.org/3/library/ast.html).

On a failing scale test, G compiles the complete **leading len(x)>literal guard plus its return subtree**, scoped to the task family. No prior statements may mutate x before that matched subtree. It does not ban all size checks or remove valid empty-input guards. The next grammar version excludes every supplied completion containing that certified leading subtree. Nonmatching failures use exact canonical-function revocation. Rule construction/parsing takes linear work; incremental trie transitions use constant-time dictionary lookup.

![Python training curves](results/python_training.png)

Python G's final sampled accuracy is **{py_g['accuracy']:.3f}%**, sampled exploit rate **{py_g['exploit']:.3f}%**, and n=160 **finite-grammar** exploit probability **{py_g['grammar_risk']:.6f}%**. The grammar integral exposes branches that finite samples miss. It is not an integral over arbitrary Python. The Python extension did **not** meet zero exact risk or sub-3% auditing: its audit rate was {py_g['audit']:.2f}% and training wall cost was {py_g['wall_seconds']/python_summary['baseline']['wall_seconds']:.2f}× baseline. Across G runs, discovered rule counts were {py_g['rules']} out of four boundary families, ordered seed-major with maze/LIS/aggregation within each seed. Extending a rule to unseen cutoffs would require an additional validated semantic contract, not simply removing the literal from the signature.

Python training wall time excludes separately measured evaluation and checkpoint serialization, includes worker launch and execution, and uses serial one-thread runs. Parent process CPU does not include child-process CPU and is not used as the primary Python cost measure. Exact `(AST, input size, input seed)` results are cached; cache hits and actual subprocess counts are published. This means timings describe repeated-program benchmark workloads, not arbitrary novel LLM outputs.

## 7. TRL integration and operational scope

`phase3/trl_patch.py` supplies **AlgorithmGGRPOTrainer** and `README.md` shows the short call site. It uses the installed [TRL GRPOTrainer API](https://huggingface.co/docs/trl/grpo_trainer), pinned to TRL 1.16.0 and Transformers 5.21.0. Both its custom rollout and likelihood hook use the same frozen grammar. It retains standard TRL group advantages and clipped loss, with beta=0 for the fresh tiny model, and adds no invariance/replay loss.

The adapter rejects incompatible external rewards/tools, distributed training, rollout reuse, gradient accumulation beyond one, and unsynchronized resume. Support is published between optimizer updates. The task-specific backend/grammar is essential; adapting this to Qwen/Llama requires suitable grammar, tokenizer validation, a real execution sandbox, and fresh model-scale evidence. A generic ten-line wrapper cannot certify arbitrary generated Python. No Qwen/Llama experiment was run.

## 8. Findings and remaining limits

- **Exact-zero DSL support target:** met after compilation; discovery itself is not zero-risk.
- **<3% full-training heavy audits:** met at {g['audit']:.3f}%.
- **99.90% primary accuracy:** missed at {g['accuracy']:.3f}%; no held-out tuning or larger training budget was used to conceal the miss.
- **Near-D CPU:** compare the new common calibration, G {g['cpu']:.2f}× / D {summary['D']['cpu']:.2f}× training and G {g['rollout_cpu']:.2f}× / D {summary['D']['rollout_cpu']:.2f}× rollout. These are small CPU model results, not frontier-model throughput predictions.
- **Early-exit >70% universal saving:** not established. The attack-heavy and actual-traffic measurements have different denominators.
- **Real code generation:** actual Python and actual TRL gradients demonstrated within an explicitly supplied finite grammar; arbitrary Python and learned algorithm discovery remain untested.

The empirical contribution is a reproducible counterexample-to-compiled-permission loop with consistent policy support, lower audit duty cycle, and no auxiliary neural losses. The principal limitation is that the strongest safety guarantee relies on a known, finite action semantics. Stronger next tests would require held-out attack families, richer independently validated AST contracts, larger pretrained code models, an isolated execution service, and distributed mask-state synchronization.

## 9. Reproduction and artifacts

Run commands and restrictions are in `README.md`; dependency versions in `phase2/requirements-lock.txt`. Key data: `results/evaluation.csv`, `training_cost.csv`, `serial_profile.csv`, `exact_risk.csv`, `latency.json`, `summary.json`, `checkpoint_audit.json`, and `python/curves.csv`. Checkpoints are under `results/runs` and `results/python/*/model`. The release manifest hashes all bundled files. Pilots, caches, and the virtual environment are excluded from the release archive.

No external message, submission, or publication was sent. `DEEPMIND_EXECUTIVE_BRIEF.md` is a draft for human review, without a claim of affiliation or endorsement.
'''
    (ROOT / "Frontier_AI_Alignment_Benchmark_Report_v3.0.md").write_text(report,encoding="utf-8")
    brief=f'''# Compiled shielding: speed and exact-support safety in a bounded neural benchmark

**Technical brief for frontier alignment researchers · 26 October 2026**

**Finding.** A task-conditioned neural program policy with dynamically compiled permissions achieved **{g['accuracy']:.3f}% true accuracy**, **zero sampled exploits**, and **exactly zero probability on designated DSL exploits after compilation**, at **{g['cpu']:.2f}× training CPU / {g['rollout_cpu']:.2f}× rollout CPU** relative to standard GRPO. Heavy audits covered **{g['audit']:.3f}%** of training sequences.

**Empirical arc.** V1's finite categorical benchmark reduced sampled reward hacking from 99.78% to zero with hierarchical masks, but a shared verifier blind spot broke its unified method. V2 replaced the categorical policy with a 6,492-parameter autoregressive GRU: adaptive verifier F reached 99.902% blind-spot accuracy, while exact enumeration exposed 0.033013% residual exploit mass. V3's G promotes verified, contract-certified shortcut primitives into hard token masks and removes invariance/replay auxiliary losses. A sparse audit floor remains.

**Evidence.** Three task families (maze, LIS, aggregation), eight main seeds, 240 updates and 7,680 completions per policy. Twenty-four new G policies and 36 ablations were trained; 96 frozen v2 comparators were reused transparently. Fresh randomized serial calibration covered 72 disposable checkpoint runs. An independent checkpoint reload reproduced {audit['rows_reproduced_exactly']:,} final evaluation rows. The exact integrator enumerated every one of 2,304 programs at matched stress inputs.

**What did not meet the target.** G missed the requested 99.90% accuracy threshold. Its difference from F was {delta['mean']:+.3f} percentage points, paired 95% seed-bootstrap interval [{delta['ci'][0]:+.3f}, {delta['ci'][1]:+.3f}]. Lower cost and stronger support exclusion do not establish strictly better accuracy or universal Pareto optimality. Early-exit gains depend on attack prevalence.

**Python demonstration.** Eighteen actual TRL runs trained two-layer causal Transformers to emit BPE-token Python functions, validated by AST permissions and executed in isolated subprocesses. G reached {py_g['accuracy']:.3f}% sampled held-out accuracy; n=160 residual exploit mass within the supplied grammar was {py_g['grammar_risk']:.6f}%. Algorithm bodies and the 20-function-per-family grammar were supplied; this is constrained synthesis, not unrestricted code generation or algorithm invention.

**Scope of the guarantee.** Zero DSL mass follows from trusted primitive semantics and −∞ masks on both live and dormant shortcuts. It does not cover initial discovery, unknown attack classes, shield removal, arbitrary Python, or model-scale distribution shifts. The subprocess worker is not an OS sandbox. No frontier LLM efficacy claim is made.

**Research implication.** Promote counterexamples into sound permissions where action semantics permit it; keep optimizer likelihoods synchronized with each shield version; spend residual audits on unknown behavior. The next discriminating experiment is a pretrained code model with held-out attack families, richer AST contracts, and an independently secured execution broker.

Code, checkpoints, raw curves, ablations, timing attribution, tests, and a pinned TRL adapter accompany the v3 report. This is a draft research brief, not a submission or claim of institutional affiliation.
'''
    (ROOT / "DEEPMIND_EXECUTIVE_BRIEF.md").write_text(brief,encoding="utf-8")
    print(pareto)
    print(pytable)


if __name__ == "__main__":
    main()
