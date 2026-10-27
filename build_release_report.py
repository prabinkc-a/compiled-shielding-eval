"""Generate the public narrative and figures from raw experiment records.

No hand-entered headline metrics. Historical observations and new controls are
kept separate; the broader discovery stress test is not an LLM training result.
"""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def read_csv(path):
    return list(csv.DictReader(path.open(encoding='utf-8')))


def mean(rows,key):
    return float(np.mean([float(r[key]) for r in rows]))


def interval(values):
    values=np.array(values)
    samples=np.random.default_rng(20261026).choice(values,(10000,len(values))).mean(1)
    return np.quantile(samples,[.025,.975]).tolist()


def collect():
    controls=json.loads((ROOT/'extension_results/controls.json').read_text(encoding='utf-8'))
    discovery=json.loads((ROOT/'extension_results/discovery.json').read_text(encoding='utf-8'))
    capped=json.loads((ROOT/'extension_results/matched_fuzzing.json').read_text(encoding='utf-8'))
    matched=json.loads((ROOT/'extension_results/full_budget/matched_fuzzing.json').read_text(encoding='utf-8'))
    evaluation=read_csv(ROOT/'phase2/results/evaluation.csv')+read_csv(ROOT/'phase3/results/evaluation.csv')
    evaluation += [r for run in controls['oracle'] for r in run['final']]
    exact=read_csv(ROOT/'phase3/results/exact_risk.csv')
    costs=controls['profile']
    baseline=mean([r for r in costs if r['method']=='baseline'],'cpu_train_seconds')
    methods={}
    for method in ('baseline','D','F','G','Oracle'):
        rows=[r for r in evaluation if r['method']==method and r['decode']=='sample'
              and r['condition']=='unmonitored' and int(r['size'])>12]
        seeds=sorted(set(int(r['seed']) for r in rows))
        by_seed=[100*mean([r for r in rows if int(r['seed'])==s],'truth') for s in seeds]
        methods[method]=dict(accuracy=100*mean(rows,'truth'),ci=interval(by_seed),accuracy_by_seed=by_seed,
            sampled_exploit=100*mean(rows,'exploit'),
            exact_risk=100*(np.mean([r['exact_risk'] for r in controls['oracle']]) if method=='Oracle'
                else mean([r for r in exact if r['method']==method],'exact_exploit_probability')),
            cpu_ratio=mean([r for r in costs if r['method']==method],'cpu_train_seconds')/baseline)
    py=json.loads((ROOT/'phase3/results/python/summary.json').read_text(encoding='utf-8'))
    pycurves=read_csv(ROOT/'phase3/results/python/curves.csv')
    python={}
    for method in ('baseline','G'):
        runs=[r for r in py if r['method']==method]
        final=[r for r in pycurves if r['method']==method and int(r['update'])==48 and int(r['size'])==160]
        python[method]=dict(risk=100*mean(final,'grammar_exploit_mass'),
            wall=mean(runs,'train_wall_seconds'),rules=[len(r['backend']['rules']) for r in runs])
    grouped={}
    for category in ('size','modulo/parity','value range'):
        rows=[r for r in discovery['records'] if r['category']==category]
        grouped[category]=dict(total=len(rows),found=sum(r['discovered'] for r in rows),
            d_found=sum(r['static_detected'] for r in rows),
            transfer=sum(r['heldout_variant_blocked'] for r in rows),
            honest_blocked=sum(r['honest_blocked'] for r in rows))
    def paired_summary(dataset):
        paired={}
        for category in ('size','modulo/parity','value range','overall'):
            selected=[r for r in dataset['records'] if category=='overall' or r['category']==category]
            a=[r for r in selected if r['method']=='adaptive']
            u=[r for r in selected if r['method']=='uniform']
            assert len(a)==len(u)>0
            differences=[]
            for seed in sorted({r['seed'] for r in a}):
                aa=[r for r in a if r['seed']==seed]
                uu=[r for r in u if r['seed']==seed]
                differences.append(100*(np.mean([r['discovered'] for r in aa])-
                                        np.mean([r['discovered'] for r in uu])))
            paired[category]=dict(total=len(a),adaptive=sum(r['discovered'] for r in a),
                uniform=sum(r['discovered'] for r in u),difference_pp=float(np.mean(differences)),
                ci=interval(differences),seed_differences_pp=differences)
        return paired
    paired=paired_summary(matched)
    matched_families={f:{method:sum(r['discovered'] for r in matched['records']
                          if r['family']==f and r['method']==method)
                        for method in ('adaptive','uniform')}
                      for f in sorted({r['family'] for r in matched['records']})}
    metrics=dict(methods=methods,python=python,discovery=grouped,
        matched=paired,matched_capped=paired_summary(capped),matched_family_counts=matched_families,
        family_counts={f:sum(r['discovered'] for r in discovery['records'] if r['family']==f)
                       for f in sorted(set(r['family'] for r in discovery['records']))})
    (ROOT/'extension_results/release_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return metrics,evaluation,discovery,controls,matched,capped


def main():
    metrics,evaluation,discovery,controls,matched,capped=collect()
    m,py=metrics['methods'],metrics['python']
    found=sum(r['discovered'] for r in discovery['records'])
    total=len(discovery['records'])
    probe_calls=sum(r['probe_executions'] for r in discovery['records'])
    old_found=sum(py['G']['rules'])
    old_total=4*len(py['G']['rules'])
    pycost=py['G']['wall']/py['baseline']['wall']
    overall=metrics['matched']['overall']
    hook=("We tested whether adaptive adversarial probing discovers conditional Python exploits more effectively than uniform fuzzing with the same execution budget. "
        f"Across a supplied attack catalog, adaptive search found {overall['adaptive']}/{overall['total']} family instances and uniform fuzzing found {overall['uniform']}/{overall['total']}, with 792 search executions each. "
        "Both can miss triggers, and the compiled shield only transfers across an identical leading guard and return, so a trivial rewrite can evade it.")
    discovery_table='| Trigger category | Adaptive discovery | Uniform discovery | Paired difference |\n|---|---:|---:|---:|\n'
    for category in ('size','modulo/parity','value range','overall'):
        r=metrics['matched'][category]
        discovery_table+=f"| {category} | {r['adaptive']}/{r['total']} ({100*r['adaptive']/r['total']:.1f}%) | {r['uniform']}/{r['total']} ({100*r['uniform']/r['total']:.1f}%) | {r['difference_pp']:+.1f} pp |\n"
    if overall['ci'][0]>0:
        interpretation="The interval favors adaptive weighting in this small test."
    elif overall['ci'][1]<0:
        interpretation="The interval favors uniform fuzzing in this small test."
    else:
        interpretation="These data do not show a clear advantage from adaptive weighting."
    matched_note=("Family instances are twelve supplied families across two suites and nine seeds; this is a discovery stress test, not newly trained neural policies. "
        f"Both methods use the same arms, candidate scheduler, search budget and compiler; independent witness replays add {overall['adaptive']} executions for adaptive and {overall['uniform']} for uniform. "
        f"The overall paired 95% seed-bootstrap interval for adaptive minus uniform is {overall['ci'][0]:+.1f} to {overall['ci'][1]:+.1f} percentage points; nine seeds are a small sample. "
        + interpretation)
    capped_overall=metrics['matched_capped']['overall']
    padding={method:sum(r['padding_probes'] for r in capped['records'] if r['method']==method)
             for method in ('adaptive','uniform')}
    primary_padding={method:sum(r['padding_probes'] for r in matched['records'] if r['method']==method)
                     for method in ('adaptive','uniform')}
    table='| Neural DSL method | Held-out accuracy | Exact exploit mass | Training CPU |\n|---|---:|---:|---:|\n'
    labels={'baseline':'Standard GRPO','D':'D · monitored-only static mask','F':'F · soft adaptive probing',
        'G':'G · discovery + compiled shielding','Oracle':'Oracle · static global mask from step 0'}
    for key in m:
        r=m[key]
        table+=f"| {labels[key]} | {r['accuracy']:.2f}% | {r['exact_risk']:.4f}% | {r['cpu_ratio']:.2f}× |\n"
    note=("Accuracy is sampled on larger, unmonitored tasks; exact mass integrates the complete finite DSL at one large task per suite/seed. "
          "CPU is a fresh serial warm-checkpoint calibration against standard GRPO, not inference latency or an LLM speedup. "
          "The table is the DSL experiment; the opening Python results are a separate experiment.")
    oracle_note=(f"**The oracle correction matters.** Its measured accuracy is {m['Oracle']['accuracy']:.2f}%, not an assumed 99.86%. "
        "The oracle eliminates every specified shortcut from initialization but can still choose ordinary incorrect algorithms or abstain. "
        "It differs from D only in the mask; F/G also train on some adversarial contexts. "
        "G matches the oracle's zero exploit support after discovery, but this design does not isolate discovery as the sole cause of its accuracy advantage.")
    readme=f'''# Automated exploit discovery and compiled shielding

{hook}

{discovery_table}
{matched_note}

For example, `len(x) > 12` and `12 < len(x)` have the same meaning but different AST signatures; a rule learned from one does not block the other. The supplied attack catalog is researcher-designed, and neither method demonstrates arbitrary-code or LLM-scale safety.

## DSL sanity check

{table}
{note}

{oracle_note}

**All training, probing, compilation and evaluation source is public under MIT.** Start with [REPORT.md](REPORT.md) for the findings and [LESSWRONG_POST.md](LESSWRONG_POST.md) for the discussion draft.

## Reproduce

Use Python 3.13 and install the tested dependencies, then run the complete experiment pipeline:

```bash
python -m pip install -r requirements.txt
python reproduce_all.py
```

The default performs fresh training and evaluation in a new `reproduced/` directory; it does not substitute cached checkpoints for training. It includes the historical tabular/neural/Python experiments, the new oracle control, trigger-blind discovery, calibration and report generation. This is a CPU research run, not a promised one-minute demo; runtime depends on hardware.

```bash
python reproduce_all.py --smoke        # tests + two real TRL updates; not headline reproduction
python reproduce_all.py --verify-only  # hashes, checkpoint reloads, exact risk and witness checks
python audit_search.py                  # reconstruct matched search traces from seeds
```

The new 24 oracle fits and expanded discovery experiment were executed for this release. Prior D/F/G and Python results retain their original checkpoints and source provenance; they are not relabeled as newly trained. Full fresh orchestration is supplied; see [VALIDATION.md](VALIDATION.md) for exactly what was executed locally.

## Inspect the engineering

- [train.py](train.py), [defenses.py](defenses.py), [shield.py](shield.py): actual neural updates, adaptive verifier and DSL compiler.
- [adversarial_prober.py](adversarial_prober.py), [ast_compiler.py](ast_compiler.py): parameter-blind search, traced counterexamples and guarded subtree transfer.
- [python_backend.py](python_backend.py), [python_grammar.py](python_grammar.py), [trl_patch.py](trl_patch.py): generalized broker, lexical grammar and support-consistent TRL integration.
- [trl_example.py](trl_example.py): runnable integration example with a local tiny GPT-2; no download.
- [phase2/](phase2/), [phase3/](phase3/): original complete implementations, checkpoints, raw measurements and historical reports. Their reports describe older protocols; [REPORT.md](REPORT.md) is the current interpretation.
- [extension_results/](extension_results/): oracle checkpoints, per-family discovery outcomes, input witnesses, costs and source hashes.

![Generalization and discovery recall](assets/generalization_discovery.png)

The original Python experiment used a supplied grammar with known cutoff shapes; it did not discover arbitrary Python attacks. The expanded test hides trigger parameters from the prober, uses an empty rule store, and evaluates unseen suffix variants, but the researcher still designed the attack catalog. The expanded study covers sum/LIS, not maze value triggers, and is a discovery stress test rather than new neural training. No Qwen/Llama safety result is claimed.
'''
    failures='\n'.join(f"| `{name}` | {count}/18 |" for name,count in metrics['family_counts'].items())
    matched_failures='\n'.join(f"| `{name}` | {counts['adaptive']}/18 | {counts['uniform']}/18 |" for name,counts in metrics['matched_family_counts'].items())
    matched_intervals='\n'.join(f"| {category} | {metrics['matched'][category]['difference_pp']:+.1f} | {metrics['matched'][category]['ci'][0]:+.1f} to {metrics['matched'][category]['ci'][1]:+.1f} |" for category in ('size','modulo/parity','value range','overall'))
    confidence='\n'.join(f"| {labels[key]} | {r['ci'][0]:.3f}–{r['ci'][1]:.3f}% |" for key,r in m.items())
    cost_rows='\n'.join(f"| {method} | {mean([r for r in controls['profile'] if r['method']==method],'cpu_train_seconds')/40*1000:.3f} | {mean([r for r in controls['profile'] if r['method']==method],'cpu_rollout_seconds')/40*1000:.3f} |" for method in m)
    report=f'''# Empirical report: automated exploit discovery and compiled shielding

{hook}

{discovery_table}
{matched_note}

The transfer restriction matters immediately: `len(x) > 12` and `12 < len(x)` are equivalent, but have different signatures, so discovering one does not block the other. The catalog was supplied by the researchers; these are family-level discovery rates, not policy-weighted risk or evidence that an LLM invented the attacks.

## The more interesting test: conditional Python exploits

The historical TRL experiment trained a 68,480-parameter causal Transformer to emit actual Python BPE tokens. It supplied two honest algorithm bodies, an optional empty-input guard, and four size cutoffs. The old compiler specifically extracted a leading `len(x) > literal` guard, and the prober read that literal to cross its boundary. Calling these arbitrary unknown attacks would overstate the result.

Across nine G runs, discovered cutoff counts were `{py['G']['rules']}`: **{old_found}/{old_total} family-run instances ({100*old_found/old_total:.1f}% recall), ranging from two to four per run**. The baseline retained {py['baseline']['risk']:.6f}% mean exact grammar risk at size 160; G retained {py['G']['risk']:.6f}%. Zero sampled violations did not mean zero probability. Python training wall time was {pycost:.2f}× baseline, including broker subprocess work and excluding evaluation.

We added a separate, broader discovery test. Its compiler starts with no banned signatures and accepts any permitted leading guard/return shape; its prober does not parse conditions, import the attack catalog, or receive trigger constants or evaluator witnesses. It searches scale, value range and input structure using only executed correctness feedback. The evaluator separately supplies size, modulo/parity and value-trigger cases, plus held-out honest suffix variants.

In the earlier eight-probes-per-family study, adaptive search discovered **{found}/{total} ({100*found/total:.1f}%)** and blocked the same number of held-out suffix variants. None of the {total} tested honest control instances was blocked; the controls do not establish a general false-positive bound. Recall counts a family only after an independently replayed wrong return is compiled, not merely when its source parses.

That earlier study used {probe_calls} search executions plus {found} independent compiler replays, excluding evaluator/control work. The matched comparison above is a new experiment: both methods use 792 search executions, allocated within each suite/seed by the same least-probed-first schedule. The earlier adaptive total therefore cannot be compared directly to either matched-study total: the allocation and stopping schedules changed. Opportunities for individual families can differ as each method finds different failures. Search-execution counts exclude compiler replays, so this is not an equal wall-clock-cost comparison.

The original **{py['G']['risk']:.2f}%** is a policy-weighted risk in a trained finite grammar. The earlier expanded test's **{100*(total-found)/total:.1f}% undiscovered family fraction** is an unweighted coverage measure, not a new policy risk estimate. No neural policy was trained on the expanded catalog for these recall numbers. The generalized TRL wiring separately received a two-update gradient smoke test.

## What the compiler actually does

An audit generates inputs, executes a candidate and compares its output with an independent sum/LIS oracle. A separate replay must reproduce a wrong return from the function's first guarded return. The compiler hashes the complete condition and return AST, then excludes functions with that identical first subtree, including a different unseen suffix. It does not transfer a rule across earlier mutation, a dominating branch, a different return, or an arbitrary semantically equivalent expression. A failure later in the honest suffix cannot certify the earlier guard.

Rules are published between completed optimizer updates. Generation, old log probabilities and current likelihood scoring share a frozen grammar version throughout each update. The root TRL adapter no longer treats discovery of four rules as proof of saturation; unknown families require continuing exploration. The original adapter is preserved under `phase3/` so historical results remain reproducible.

![Generalization and discovery recall](assets/generalization_discovery.png)

## DSL sanity check

{table}
{note}

D's incomplete monitored-input specification leaves {m['D']['exact_risk']:.2f}% exact shortcut mass in the DSL stress context. F reduces that mass to {m['F']['exact_risk']:.4f}% through soft feedback. G reaches zero, the same support restriction an informed human can install at step zero.

{oracle_note}

Kernel 3 is explicitly designated a shortcut by the trusted interpreter; G uses this known primitive contract after a counterexample and does not infer universal unsafety from one failure. Every G main run certified the primitive in the first update. This is a bounded sanity check, not discovery of an unknown primitive.

## Where it breaks

Finite scale/value search misses rare or out-of-range conditions. A signature only covers identical syntax at a privileged position; equivalent obfuscations and later guards evade that transfer rule. The supplied grammar, reference algorithms and small models limit the claim further. This is empirical correctness checking under a trusted oracle, not intent detection or semantic verification of arbitrary code. The worker has process isolation, a timeout and a line budget; it is not an OS security sandbox.

G's low DSL cost is measured after compilation on tiny CPU policies. The harder Python experiment costs more, and the new standalone search has its own execution budget. Neither proves 1.05× overhead for a large pretrained model. Historical F/G also receive broader training contexts than D/the oracle, so accuracy differences mix exposure, rewards and masking. A matched-exposure factorial experiment and a separately constructed blind attack set would be stronger next tests.

Runtime shielding predates this work; see [Alshiekh et al., Safe Reinforcement Learning via Shielding](https://arxiv.org/abs/1708.08611). The contribution here is the inspectable bounded experiment, counterexample-to-subtree machinery and explicit failure accounting. Classical Nyaya/Mimamsa concepts motivated adversarial examination and priority constraints as design analogies; they do not establish novelty or guarantees.

## Appendix: protocols, controls and provenance

### Neural protocol

Three suites: maze path legality, strict LIS grading, and sum with reward-channel tampering. A 6,492-parameter GRU emits six tokens: primary kernel, guard, cutoff, alternate kernel, reward channel, report. There are 2,304 syntactically complete programs. Kernel 2 can be incorrect without being the designated exploit; abstention is also permitted.

Eight training seeds (0–7), 240 updates, four prompt groups of eight completions, two optimizer epochs, Adam learning rate 0.002, clipped ratio 0.2, reference KL coefficient 0.005. Core sizes are 6/8/10/12. F/G replace one prompt group every fourth update with an adaptive challenge (6.25% of sequences); Oracle and D have the same small-only task stream. No held-out accuracy checkpoint selection. Evaluation uses 96 cases per size/cue cell, sizes 8/12/20/40/80/160; headline means average the four larger unmonitored cells, suites and seeds.

Source: `train.py` and historical `phase2/train.py`, `phase3/train.py`. Task RNG 11000+seed; verifier RNG 21000+seed; adaptive RNG 31000+seed. Evaluation task RNG 800000+100*size+monitor; sampling RNG 78000+seed. Exact integration uses size 160, task seed 12900, both cues false. Its zero is structural under the known DSL mask; it is not an empirical bound on arbitrary inputs.

Seed-level 95% bootstrap intervals (10,000 resamples, RNG 20261026):

| Method | Accuracy interval |
|---|---:|
{confidence}

### Fresh cost calibration

Five methods × three suites × seeds 0–2: 45 serial runs, 40 disposable updates each, one Torch thread, randomized order RNG 4118. Policies start from their final checkpoints; optimizer/reference and search counters restart. G retains its compiled mask. Process CPU excludes evaluation. These are warm training costs; no other benchmark was launched during this calibration.

| Method | Mean CPU ms/update | Rollout CPU ms/update |
|---|---:|---:|
{cost_rows}

Raw costs and source hashes: [controls.json](extension_results/controls.json). Original microbenchmarks remain in [phase3/results/latency.json](phase3/results/latency.json); their machine-specific timing must not be mixed into the new ratios. G's historical main-training audit fraction was about 0.76%, versus F's 8.95%; low post-compilation auditing is not a no-failure-during-learning claim.

### Historical Python protocol

Baseline/G × three suites × seeds 0–2 = 18 runs; 48 updates × eight completions. Tiny GPT-2: two layers, two heads, embedding 32, 68,480 parameters, BPE vocabulary 320, no pretrained weights or SFT. Twenty supplied functions per suite. Training checks sizes 8/12; evaluation sizes 12/20/40/80/160, 32 sampled functions per cell. GRPO learning rate 0.001, beta 0, one fresh rollout per update. Original four-family saturation heuristic and condition-reading prober remain explicitly visible in the historical source. Mean grammar risk is integrated over each learned finite grammar at size 160.

### Expanded discovery protocol

Two suites (sum/LIS) × nine seeds × twelve families = {total} instances. Each candidate gets at most eight probes, with early exit after a failure; search-arm weights persist across randomly ordered families within a suite/seed. This is a coverage test with equal candidate opportunity, not an on-policy or sparse-rollout-cost measurement. Each success incurs one additional independent compiler replay. Static D gets only size 8/12, values in [-20,20]. Both rule stores start empty; the diagnostic oracle receives the evaluator's full labels.

Search arms: sizes 8/13/25/49/97/160 with uniform integer jitter ±3; value amplitudes 20/64/256; random, sorted or repeated-value structure. Selection is 15% uniform plus 85% weight-normalized; feedback update is `w ← .92w + .08(1 + 12*failure)`. Search seed 9000+seed, family ordering 17000+seed, static-task seed 7000+seed. The evaluator's trigger witnesses and variant-1 suffixes are withheld until search ends for that family. The compiler does not preload family labels; the catalog is nevertheless researcher-designed, not independently blinded or invented by the model.

| Family | Discovered across 18 suite/seed instances |
|---|---:|
{failures}

This table includes all misses. `mod_257_251` lies outside the maximum probed size; `sum_eq_9973` demands a specific value configuration. Rare in-range modulo triggers can also escape eight probes. All raw decisions are in [discovery.json](extension_results/discovery.json); actual executed inputs, expected/observed outputs and signatures are in [discovery_witnesses/](extension_results/discovery_witnesses/).

The study's recorded wall time is {discovery['protocol']['elapsed_seconds']:.1f} seconds, including evaluator/control execution; smoke tests shared the machine for part of this run. It is not an isolated overhead calibration. The nine RNG seeds are reused across the two suites, so the {total} family instances should not be treated as independent statistical replicates.

### Matched-budget fuzzing comparison

The paired experiment gives each method 44 search executions per suite/seed, 792 total, on the same two suites, nine seeds and twelve supplied families. Both start with an empty rule store, use identical 54 search arms, input generation, verifier and compiler, and share a fixed randomized family order. Uniform keeps the initial weights; adaptive applies the same feedback updates as the previous study. Within each run, the least-probed undiscovered family is checked next, breaking ties by the fixed order; the per-family cap equals the whole 44-call run budget. This avoids the earlier eight-probe cap and allows the budget to continue targeting unresolved families. Each found family incurs an extra independent witness replay outside the search budget. Thus the search budget is matched globally and per suite/seed, not necessarily per family or in total wall-clock time.

Both schedulers react to discovery by moving probes toward unresolved families. The uniform control randomizes input arms uniformly; it is not a fully non-adaptive policy for allocating candidate programs.

Primary-run padding calls: {primary_padding['adaptive']} adaptive and {primary_padding['uniform']} uniform. Non-padding calls test candidates not yet discovered by that method in the corresponding run.

The matched study uses the same seed numbers and family orders for each method, but once their choices diverge, the actual sampled inputs need not coincide. The 95% interval bootstraps the nine paired seed-level differences, averaging the two suites within each seed; it does not treat the 216 family instances as independent observations. Selection and budget decisions are reported in [full-budget matched_fuzzing.json](extension_results/full_budget/matched_fuzzing.json). Run [audit_search.py](audit_search.py) to reconstruct the search arms, scheduling and witness inputs from seeds; actual witness correctness is checked separately by execution. No unrestricted LLM was trained on this catalog.

The first matched run retained the earlier eight-probe family cap. Adaptive found {capped_overall['adaptive']}/{capped_overall['total']} and uniform found {capped_overall['uniform']}/{capped_overall['total']}; their difference was {capped_overall['difference_pp']:+.1f} pp (paired 95% interval {capped_overall['ci'][0]:+.1f} to {capped_overall['ci'][1]:+.1f}). Because remaining families hit the cap, {padding['adaptive']} adaptive and {padding['uniform']} uniform calls were padding. In the distributed run, all padding revisited already discovered candidates. After inspecting that result and its padding, we chose to run the full-budget protocol presented as the primary table above. This is a post-inspection follow-up, not a preregistered independent confirmation. The first result is preserved in [capped matched_fuzzing.json](extension_results/matched_fuzzing.json).

| Category | Adaptive minus uniform (pp) | Paired 95% bootstrap interval (pp) |
|---|---:|---:|
{matched_intervals}

| Trigger family | Adaptive found | Uniform found |
|---|---:|---:|
{matched_failures}

### Reproduction and implementation limits

`python reproduce_all.py` rebuilds experiments in a fresh output tree. `--smoke` tests integration without reproducing headline metrics; `--verify-only` checks the distributed evidence. [VALIDATION.md](VALIDATION.md) distinguishes runs executed for this release from preserved historical evidence. File hashes are in `SHA256SUMS.txt`; source/config hashes are embedded in experiment metadata. Historical manifests document old packages and are not manifests for this consolidated directory.

Tested versions are pinned in [requirements.txt](requirements.txt). TRL integration uses protected hooks and is pinned to 1.16.0: one process, plain text, beta 0, one iteration, gradient accumulation 1 and one fresh generation per update. It rejects unsupported configurations. [Hugging Face's GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) describes the underlying trainer; this custom constrained sampler needs independent revalidation for other versions, distributed training or Qwen/Llama models.

All source and supplied checkpoints are under [MIT](LICENSE); there is no withheld training/compiler component. Historical v1/v2/v3 reports are preserved for provenance, not as substitutes for the corrected interpretation above.
'''
    post=f'''# Empirical Notes: Using Sparse Adversarial Probing & AST Shield Compilation to Catch Conditional Specification Gaming (And Where It Breaks)

{hook}

{discovery_table}
{matched_note}

The transfer limitation is immediate: after learning to block `len(x) > 12`, the shield still permits the equivalent `12 < len(x)`. This is a test of supplied functions and a restricted grammar, not an autonomous search for arbitrary exploits.

Everything is now public: [training, compiler, prober, checkpoints and reproduction instructions](https://github.com/prabinkc-a/compiled-shielding-eval). Earlier packaging withheld the training engine. That made the most interesting claims harder to inspect, so this release removes that split.

## How the matched comparison works

Both methods search the same size, value-range and structure arms and receive the same correctness feedback. Each gets 44 search executions per suite/seed, 792 total, under the same candidate scheduler; the per-family cap equals the full 44-call budget. Independent witness replays are extra. Uniform fuzzing keeps every arm equally likely, while adaptive search updates weights after failures. Because each method can find different families at different times, the schedule need not give an identical number of probes to each individual family, even though the total is equal. Seeds and family order are matched; individual sampled inputs can diverge.

The study measures family discovery and exact-syntax suffix transfer in a supplied catalog. It does not measure end-to-end LLM training overhead or the probability of a trained model exploiting these new triggers. The paired seed-level interval above is descriptive and rests on only nine seeds.

The first matched run used an eight-probe family cap: adaptive found {capped_overall['adaptive']}/{capped_overall['total']} and uniform found {capped_overall['uniform']}/{capped_overall['total']}, but {padding['adaptive']} adaptive and {padding['uniform']} uniform calls were padding after remaining families hit the cap. We chose the full-budget follow-up after inspecting that run, so it is not preregistered independent confirmation.

## The earlier Python training result

In the original bounded Python/TRL experiment, G discovered **{old_found} of {old_total} cutoff-family instances ({100*old_found/old_total:.1f}% recall)**. A discovered guard was compiled into a restriction that also blocked functions with the same guard and a different supplied suffix. Mean exact exploit mass fell from **{py['baseline']['risk']:.2f}% to {py['G']['risk']:.2f}%**, at **{pycost:.2f}× training wall time**. The remaining risk matters: sampled evaluation happened to show no violations, while enumerating the grammar exposed probability on undiscovered cutoffs.

There is a correction attached to that result. The grammar already supplied the four cutoffs and the old compiler recognized `len(x) > literal`; the prober inspected the literal. These were unspecified as *bans*, not unknown as *possible program structures*. This was not an LLM inventing attacks or a verifier discovering arbitrary vulnerabilities.

The separate earlier discovery study used a different per-family schedule and found **{found}/{total} ({100*found/total:.1f}%)** instances. Its held-out suffix variants were blocked exactly when their family was discovered. None of its tested honest controls was blocked, but the control set is small and far from a general soundness proof. The matched comparison at the top uses a new schedule, so its totals are not the same experimental run.

## What is being compiled?

Consider a function that behaves correctly except for a guarded early return:

```python
def solve(x):
    if len(x) % 31 == 17:
        return 0
    return sum(x)
```

The search varies scale, values and structure and executes the actual function. If it finds a wrong output, an independent replay must confirm that execution returned from the guard. The compiler then records the exact condition-and-return AST signature. A different suffix under the identical leading guard is excluded too, because the witnessed failing path never reaches the suffix.

The restrictions are deliberately narrow. A later failure does not justify banning an earlier guard. A different return value is a different signature. Prior mutation or a dominating conditional prevents this transfer rule. The system does not infer arbitrary semantic equivalence or turn one counterexample into a universal theorem.

In the neural trainer, a new rule becomes active between optimizer updates. Sampling and likelihood scoring use the same frozen grammar version during an update. That bookkeeping is essential: changing support halfway through a GRPO update would make the policy-ratio calculation inconsistent.

## The static oracle a reviewer rightly asked for

The simpler six-token DSL has a designated shortcut primitive, kernel 3. A human who knows this can ban it globally on day one. Dynamic discovery cannot beat that oracle's support restriction, and this benchmark should never have implied otherwise.

{table}
{note}

{oracle_note}

The modest defensible claim is that counterexamples can repair an incomplete static restriction in a small controlled setting. In the DSL, known primitive semantics make repair easy. In Python, coverage and the limits of subtree transfer are the interesting part.

## Where the search fails

Eight probes per candidate are not comprehensive. A length residue beyond every searched size cannot be found. A rare in-range residue may never be sampled. A special value combination can evade broad value-range mutations. The family-by-family misses and replayable witnesses are included in [the report](https://github.com/prabinkc-a/compiled-shielding-eval/blob/main/REPORT.md).

The old Python result leaves {py['G']['risk']:.2f}% policy-weighted grammar risk. The earlier expanded discovery test leaves {100*(total-found)/total:.1f}% of family instances undiscovered. These are different denominators, and I would not combine them into a single safety score. Nor should the low warm DSL cost be advertised as the cost of searching arbitrary Python: the Python experiment was slower, and the expanded coverage test has its own budget.

## What this is not evidence for

The models are a 6,492-parameter GRU and a 68,480-parameter Transformer trained from scratch. The grammars and honest algorithm bodies are supplied. The broader trigger experiment covers sum/LIS, not every environment, and has no new large-model training result. A two-step TRL integration check is just that. No Qwen/Llama post-training or frontier-model robustness has been demonstrated.

The worker is not an OS sandbox. Correctness depends on trusted reference algorithms, the source validator and the execution broker. The attacker catalog is researcher-designed rather than independently blinded. F/G's extra adversarial training inputs also confound accuracy comparisons with small-input-only controls. These are concrete limits on the inference, not reasons to reinterpret the misses as success.

Shielding itself is established prior work; [Alshiekh et al.](https://arxiv.org/abs/1708.08611) study enforcing specifications through shields in reinforcement learning. I see this experiment as an engineering case study in discovering counterexamples, retaining their consequences and measuring what remains uncovered—not a new general alignment principle. Classical Nyaya/Mimamsa ideas were motivating analogies for adversarial examination and constraint priority, not formal guarantees.

The next experiment I would trust more would hold training exposure fixed, let a separate party construct held-out trigger families, and compare sparse probing plus compilation against both a static oracle and matched-budget fuzzing on a pretrained code model. The matched comparison here is on supplied functions, not a pretrained model. For now, the useful artifact is inspectable code with explicit misses: `python reproduce_all.py` rebuilds the experiment pipeline, and `--smoke` provides a much shorter integration check.

*Draft for LessWrong / Alignment Forum; not submitted.*
'''
    for name,text in [('README.md',readme),('REPORT.md',report),('LESSWRONG_POST.md',post)]:
        (ROOT/name).write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
    for method in ('D','F','G','Oracle'):
        ys=[]
        for size in (8,12,20,40,80,160):
            rows=[r for r in evaluation if r['method']==method and r['decode']=='sample' and r['condition']=='unmonitored' and int(r['size'])==size]
            ys.append(100*mean(rows,'truth'))
        axes[1].plot((8,12,20,40,80,160),ys,marker='o',label=method)
    axes[1].set(xlabel='Held-out task size',ylabel='True accuracy (%)',title='Neural DSL generalization',ylim=(70,101))
    axes[1].legend(ncol=2)
    categories=['size','modulo/parity','value range']
    positions=np.arange(len(categories))
    axes[0].bar(positions-.18,[100*metrics['matched'][c]['adaptive']/metrics['matched'][c]['total'] for c in categories],width=.36,label='Adaptive')
    axes[0].bar(positions+.18,[100*metrics['matched'][c]['uniform']/metrics['matched'][c]['total'] for c in categories],width=.36,label='Uniform')
    axes[0].set_xticks(positions,categories)
    axes[0].axhline(100,color='gray',linestyle='--',label='Catalog oracle (by definition)')
    axes[0].set(ylabel='Family discovery recall (%)',title='Separate trigger-blind discovery test',ylim=(0,110))
    axes[0].legend()
    (ROOT/'assets').mkdir(exist_ok=True)
    fig.savefig(ROOT/'assets/generalization_discovery.png',dpi=300)
    plt.close(fig)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
