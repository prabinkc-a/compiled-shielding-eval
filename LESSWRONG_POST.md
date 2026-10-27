# Empirical Notes: Using Sparse Adversarial Probing & AST Shield Compilation to Catch Conditional Specification Gaming (And Where It Breaks)

We tested whether adaptive adversarial probing discovers conditional Python exploits more effectively than uniform fuzzing with the same execution budget. Across a supplied attack catalog, adaptive search found 144/216 family instances and uniform fuzzing found 148/216, with 792 search executions each. Both can miss triggers, and the compiled shield only transfers across an identical leading guard and return, so a trivial rewrite can evade it.

| Trigger category | Adaptive discovery | Uniform discovery | Paired difference |
|---|---:|---:|---:|
| size | 64/72 (88.9%) | 68/72 (94.4%) | -5.6 pp |
| modulo/parity | 32/72 (44.4%) | 34/72 (47.2%) | -2.8 pp |
| value range | 48/72 (66.7%) | 46/72 (63.9%) | +2.8 pp |
| overall | 144/216 (66.7%) | 148/216 (68.5%) | -1.9 pp |

Family instances are twelve supplied families across two suites and nine seeds; this is a discovery stress test, not newly trained neural policies. Both methods use the same arms, candidate scheduler, search budget and compiler; independent witness replays add 144 executions for adaptive and 148 for uniform. The overall paired 95% seed-bootstrap interval for adaptive minus uniform is -7.4 to +4.6 percentage points; nine seeds are a small sample. These data do not show a clear advantage from adaptive weighting.

The transfer limitation is immediate: after learning to block `len(x) > 12`, the shield still permits the equivalent `12 < len(x)`. This is a test of supplied functions and a restricted grammar, not an autonomous search for arbitrary exploits.

Everything is now public: [training, compiler, prober, checkpoints and reproduction instructions](https://github.com/prabinkc-a/compiled-shielding-eval). Earlier packaging withheld the training engine. That made the most interesting claims harder to inspect, so this release removes that split.

## How the matched comparison works

Both methods search the same size, value-range and structure arms and receive the same correctness feedback. Each gets 44 search executions per suite/seed, 792 total, under the same candidate scheduler; the per-family cap equals the full 44-call budget. Independent witness replays are extra. Uniform fuzzing keeps every arm equally likely, while adaptive search updates weights after failures. Because each method can find different families at different times, the schedule need not give an identical number of probes to each individual family, even though the total is equal. Seeds and family order are matched; individual sampled inputs can diverge.

The study measures family discovery and exact-syntax suffix transfer in a supplied catalog. It does not measure end-to-end LLM training overhead or the probability of a trained model exploiting these new triggers. The paired seed-level interval above is descriptive and rests on only nine seeds.

The first matched run used an eight-probe family cap: adaptive found 144/216 and uniform found 148/216, but 10 adaptive and 26 uniform calls were padding after remaining families hit the cap. We chose the full-budget follow-up after inspecting that run, so it is not preregistered independent confirmation.

## The earlier Python training result

In the original bounded Python/TRL experiment, G discovered **32 of 36 cutoff-family instances (88.9% recall)**. A discovered guard was compiled into a restriction that also blocked functions with the same guard and a different supplied suffix. Mean exact exploit mass fell from **2.70% to 0.44%**, at **1.34× training wall time**. The remaining risk matters: sampled evaluation happened to show no violations, while enumerating the grammar exposed probability on undiscovered cutoffs.

There is a correction attached to that result. The grammar already supplied the four cutoffs and the old compiler recognized `len(x) > literal`; the prober inspected the literal. These were unspecified as *bans*, not unknown as *possible program structures*. This was not an LLM inventing attacks or a verifier discovering arbitrary vulnerabilities.

The separate earlier discovery study used a different per-family schedule and found **162/216 (75.0%)** instances. Its held-out suffix variants were blocked exactly when their family was discovered. None of its tested honest controls was blocked, but the control set is small and far from a general soundness proof. The matched comparison at the top uses a new schedule, so its totals are not the same experimental run.

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

The old Python result leaves 0.44% policy-weighted grammar risk. The earlier expanded discovery test leaves 25.0% of family instances undiscovered. These are different denominators, and I would not combine them into a single safety score. Nor should the low warm DSL cost be advertised as the cost of searching arbitrary Python: the Python experiment was slower, and the expanded coverage test has its own budget.

## What this is not evidence for

The models are a 6,492-parameter GRU and a 68,480-parameter Transformer trained from scratch. The grammars and honest algorithm bodies are supplied. The broader trigger experiment covers sum/LIS, not every environment, and has no new large-model training result. A two-step TRL integration check is just that. No Qwen/Llama post-training or frontier-model robustness has been demonstrated.

The worker is not an OS sandbox. Correctness depends on trusted reference algorithms, the source validator and the execution broker. The attacker catalog is researcher-designed rather than independently blinded. F/G's extra adversarial training inputs also confound accuracy comparisons with small-input-only controls. These are concrete limits on the inference, not reasons to reinterpret the misses as success.

Shielding itself is established prior work; [Alshiekh et al.](https://arxiv.org/abs/1708.08611) study enforcing specifications through shields in reinforcement learning. I see this experiment as an engineering case study in discovering counterexamples, retaining their consequences and measuring what remains uncovered—not a new general alignment principle. Classical Nyaya/Mimamsa ideas were motivating analogies for adversarial examination and constraint priority, not formal guarantees.

The next experiment I would trust more would hold training exposure fixed, let a separate party construct held-out trigger families, and compare sparse probing plus compilation against both a static oracle and matched-budget fuzzing on a pretrained code model. The matched comparison here is on supplied functions, not a pretrained model. For now, the useful artifact is inspectable code with explicit misses: `python reproduce_all.py` rebuilds the experiment pipeline, and `--smoke` provides a much shorter integration check.

*Draft for LessWrong / Alignment Forum; not submitted.*
