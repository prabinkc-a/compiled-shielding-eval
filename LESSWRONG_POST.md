# Empirical Notes: Using Sparse Adversarial Probing & AST Shield Compilation to Catch Conditional Specification Gaming (And Where It Breaks)

Can a system discover an omitted condition in its tests and turn the resulting counterexample into a restriction that survives further training? I tested a small version of that question using neural policies that generate bounded programs. The useful result is partial: witnessed conditional exploits can be removed cheaply once their structure is known, but finite probing leaves real holes.

Everything is now public: [training, compiler, prober, checkpoints and reproduction instructions](https://github.com/prabinkc-a/compiled-shielding-eval). Earlier packaging withheld the training engine. That made the most interesting claims harder to inspect, so this release removes that split.

## The result I would lead with

In the original bounded Python/TRL experiment, G discovered **32 of 36 cutoff-family instances (88.9% recall)**. A discovered guard was compiled into a restriction that also blocked functions with the same guard and a different supplied suffix. Mean exact exploit mass fell from **2.70% to 0.44%**, at **1.34× training wall time**. The remaining risk matters: sampled evaluation happened to show no violations, while enumerating the grammar exposed probability on undiscovered cutoffs.

There is a correction attached to that result. The grammar already supplied the four cutoffs and the old compiler recognized `len(x) > literal`; the prober inspected the literal. These were unspecified as *bans*, not unknown as *possible program structures*. This was not an LLM inventing attacks or a verifier discovering arbitrary vulnerabilities.

I therefore added a stricter, separate discovery study. The new prober receives correctness feedback but cannot read trigger parameters or evaluator witnesses. Its rule store starts empty; its compiler handles any permitted first guard/return shape rather than one length comparison. Across sum and LIS functions with size, modulo/parity and value triggers, it found **162/216 (75.0%)** family instances. That weaker recall is part of the result, not an inconvenient number to hide.

| Trigger category | Family instances | D discovery recall | G discovery recall | Held-out suffixes blocked |
|---|---:|---:|---:|---:|
| size | 72 | 0.0% | 97.2% (70/72) | 70/72 |
| modulo/parity | 72 | 0.0% | 55.6% (40/72) | 40/72 |
| value range | 72 | 0.0% | 72.2% (52/72) | 52/72 |

Those are unweighted discovery-recall figures from a supplied attack catalog. They are **not** measurements of a newly trained model's exploit probability. Held-out suffix variants were blocked exactly when their family was discovered. None of the tested honest controls was blocked, but the control set is small and far from a general soundness proof.

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

| Neural DSL method | Held-out accuracy | Exact exploit mass | Training CPU |
|---|---:|---:|---:|
| Standard GRPO | 30.15% | 69.7919% | 1.00× |
| D · monitored-only static mask | 90.31% | 11.6023% | 1.04× |
| F · soft adaptive probing | 99.90% | 0.0330% | 1.38× |
| G · discovery + compiled shielding | 99.86% | 0.0000% | 0.99× |
| Oracle · static global mask from step 0 | 98.84% | 0.0000% | 1.02× |

Accuracy is sampled on larger, unmonitored tasks; exact mass integrates the complete finite DSL at one large task per suite/seed. CPU is a fresh serial warm-checkpoint calibration against standard GRPO, not inference latency or an LLM speedup. The table is the DSL experiment; the opening Python results are a separate experiment.

**The oracle correction matters.** Its measured accuracy is 98.84%, not an assumed 99.86%. The oracle eliminates every specified shortcut from initialization but can still choose ordinary incorrect algorithms or abstain. It differs from D only in the mask; F/G also train on some adversarial contexts. G matches the oracle's zero exploit support after discovery, but this design does not isolate discovery as the sole cause of its accuracy advantage.

The modest defensible claim is that counterexamples can repair an incomplete static restriction in a small controlled setting. In the DSL, known primitive semantics make repair easy. In Python, coverage and the limits of subtree transfer are the interesting part.

## Where the search fails

Eight probes per candidate are not comprehensive. A length residue beyond every searched size cannot be found. A rare in-range residue may never be sampled. A special value combination can evade broad value-range mutations. The family-by-family misses and replayable witnesses are included in [the report](https://github.com/prabinkc-a/compiled-shielding-eval/blob/main/REPORT.md).

The old Python result leaves 0.44% policy-weighted grammar risk. The broader test leaves 25.0% of family instances undiscovered. These are different denominators, and I would not combine them into a single safety score. Nor should the low warm DSL cost be advertised as the cost of searching arbitrary Python: the Python experiment was slower, and the expanded coverage test has its own budget.

## What this is not evidence for

The models are a 6,492-parameter GRU and a 68,480-parameter Transformer trained from scratch. The grammars and honest algorithm bodies are supplied. The broader trigger experiment covers sum/LIS, not every environment, and has no new large-model training result. A two-step TRL integration check is just that. No Qwen/Llama post-training or frontier-model robustness has been demonstrated.

The worker is not an OS sandbox. Correctness depends on trusted reference algorithms, the source validator and the execution broker. The attacker catalog is researcher-designed rather than independently blinded. F/G's extra adversarial training inputs also confound accuracy comparisons with small-input-only controls. These are concrete limits on the inference, not reasons to reinterpret the misses as success.

Shielding itself is established prior work; [Alshiekh et al.](https://arxiv.org/abs/1708.08611) study enforcing specifications through shields in reinforcement learning. I see this experiment as an engineering case study in discovering counterexamples, retaining their consequences and measuring what remains uncovered—not a new general alignment principle. Classical Nyaya/Mimamsa ideas were motivating analogies for adversarial examination and constraint priority, not formal guarantees.

The next experiment I would trust more would hold training exposure fixed, let a separate party construct held-out trigger families, and compare sparse probing plus compilation against both a static oracle and matched-budget fuzzing on a pretrained code model. For now, the useful artifact is inspectable code with explicit misses: `python reproduce_all.py` rebuilds the experiment pipeline, and `--smoke` provides a much shorter integration check.

*Draft for LessWrong / Alignment Forum; not submitted.*
