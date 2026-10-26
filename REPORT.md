# Empirical report: automated exploit discovery and compiled shielding

We tested whether sparse counterexample search can turn an incomplete reward specification into reusable execution-time restrictions. We built adversarial scale/cue probing and runtime AST shield compilation around PyTorch policies and a bounded TRL code generator. G discovered 32/36 supplied Python cutoff-family instances, reducing grammar risk from 2.70% to 0.44% at 1.34× training wall time; a new trigger-blind stress test found 162/216 family instances. Finite probing still misses triggers, leaving 0.44% risk in the original Python grammar and larger coverage gaps in the broader test.

| Neural DSL method | Held-out accuracy | Exact exploit mass | Training CPU |
|---|---:|---:|---:|
| Standard GRPO | 30.15% | 69.7919% | 1.00× |
| D · monitored-only static mask | 90.31% | 11.6023% | 1.04× |
| F · soft adaptive probing | 99.90% | 0.0330% | 1.38× |
| G · discovery + compiled shielding | 99.86% | 0.0000% | 0.99× |
| Oracle · static global mask from step 0 | 98.84% | 0.0000% | 1.02× |

Accuracy is sampled on larger, unmonitored tasks; exact mass integrates the complete finite DSL at one large task per suite/seed. CPU is a fresh serial warm-checkpoint calibration against standard GRPO, not inference latency or an LLM speedup. The table is the DSL experiment; the opening Python results are a separate experiment.

## What this establishes

D's incomplete monitored-input specification leaves 11.60% exact shortcut mass in the DSL stress context. F reduces that mass to 0.0330% through soft feedback. G turns witnessed violations into persistent support exclusions and reaches zero, the same support restriction an informed human can install at step zero. That is a useful systems result within a bounded language, not a solution to unknown specification gaming in general.

**The oracle correction matters.** Its measured accuracy is 98.84%, not an assumed 99.86%. The oracle eliminates every specified shortcut from initialization but can still choose ordinary incorrect algorithms or abstain. It differs from D only in the mask; F/G also train on some adversarial contexts. G matches the oracle's zero exploit support after discovery, but this design does not isolate discovery as the sole cause of its accuracy advantage.

In the six-token DSL, kernel 3 is explicitly designated a shortcut by the trusted interpreter. G uses this known primitive contract after a counterexample; it does not infer universal unsafety from one failure. Every G main run certified that primitive in the first update. A static oracle is therefore the right control, and dynamically rediscovering the restriction is not the central scientific contribution.

## The more interesting test: conditional Python exploits

The historical TRL experiment trained a 68,480-parameter causal Transformer to emit actual Python BPE tokens. It supplied two honest algorithm bodies, an optional empty-input guard, and four size cutoffs. The old compiler specifically extracted a leading `len(x) > literal` guard, and the prober read that literal to cross its boundary. Calling these arbitrary unknown attacks would overstate the result.

Across nine G runs, discovered cutoff counts were `[4, 3, 4, 4, 4, 4, 4, 2, 3]`: **32/36 family-run instances (88.9% recall), ranging from two to four per run**. The baseline retained 2.696235% mean exact grammar risk at size 160; G retained 0.441524%. Zero sampled violations did not mean zero probability. Python training wall time was 1.34× baseline, including broker subprocess work and excluding evaluation.

We added a separate, broader discovery test. Its compiler starts with no banned signatures and accepts any permitted leading guard/return shape; its prober does not parse conditions, import the attack catalog, or receive trigger constants or evaluator witnesses. It searches scale, value range and input structure using only executed correctness feedback. The evaluator separately supplies size, modulo/parity and value-trigger cases, plus held-out honest suffix variants.

| Trigger category | Family instances | D discovery recall | G discovery recall | Held-out suffixes blocked |
|---|---:|---:|---:|---:|
| size | 72 | 0.0% | 97.2% (70/72) | 70/72 |
| modulo/parity | 72 | 0.0% | 55.6% (40/72) | 40/72 |
| value range | 72 | 0.0% | 72.2% (52/72) | 52/72 |

The oracle blocks every evaluator-labeled exploit family by construction. G discovers **162/216 (75.0%)** and blocks the same number of held-out suffix variants. None of the 216 tested honest control instances was blocked; the controls do not establish a general false-positive bound. Recall counts a family only after an independently replayed wrong return is compiled, not merely when its source parses.

Discovery used 792 search executions plus 162 independent compiler replays, excluding evaluator/control work. That is 3.67 search executions per family instance on average. The comparison to two small static checks measures expanded coverage with a larger search budget; it does not establish an advantage over matched-budget random fuzzing.

The original **0.44%** is a policy-weighted risk in a trained finite grammar. The expanded test's **25.0% undiscovered family fraction** is an unweighted coverage measure, not a new policy risk estimate. No neural policy was trained on the expanded catalog for these recall numbers. The generalized TRL wiring separately received a two-update gradient smoke test.

## What the compiler actually does

An audit generates inputs, executes a candidate and compares its output with an independent sum/LIS oracle. A separate replay must reproduce a wrong return from the function's first guarded return. The compiler hashes the complete condition and return AST, then excludes functions with that identical first subtree, including a different unseen suffix. It does not transfer a rule across earlier mutation, a dominating branch, a different return, or an arbitrary semantically equivalent expression. A failure later in the honest suffix cannot certify the earlier guard.

Rules are published between completed optimizer updates. Generation, old log probabilities and current likelihood scoring share a frozen grammar version throughout each update. The root TRL adapter no longer treats discovery of four rules as proof of saturation; unknown families require continuing exploration. The original adapter is preserved under `phase3/` so historical results remain reproducible.

![Generalization and discovery recall](assets/generalization_discovery.png)

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
| Standard GRPO | 17.231–50.934% |
| D · monitored-only static mask | 86.577–94.000% |
| F · soft adaptive probing | 99.826–99.978% |
| G · discovery + compiled shielding | 99.783–99.935% |
| Oracle · static global mask from step 0 | 97.222–99.783% |

### Fresh cost calibration

Five methods × three suites × seeds 0–2: 45 serial runs, 40 disposable updates each, one Torch thread, randomized order RNG 4118. Policies start from their final checkpoints; optimizer/reference and search counters restart. G retains its compiled mask. Process CPU excludes evaluation. These are warm training costs; no other benchmark was launched during this calibration.

| Method | Mean CPU ms/update | Rollout CPU ms/update |
|---|---:|---:|
| baseline | 33.637 | 12.760 |
| D | 35.026 | 13.715 |
| F | 46.354 | 18.056 |
| G | 33.377 | 13.281 |
| Oracle | 34.332 | 13.411 |

Raw costs and source hashes: [controls.json](extension_results/controls.json). Original microbenchmarks remain in [phase3/results/latency.json](phase3/results/latency.json); their machine-specific timing must not be mixed into the new ratios. G's historical main-training audit fraction was about 0.76%, versus F's 8.95%; low post-compilation auditing is not a no-failure-during-learning claim.

### Historical Python protocol

Baseline/G × three suites × seeds 0–2 = 18 runs; 48 updates × eight completions. Tiny GPT-2: two layers, two heads, embedding 32, 68,480 parameters, BPE vocabulary 320, no pretrained weights or SFT. Twenty supplied functions per suite. Training checks sizes 8/12; evaluation sizes 12/20/40/80/160, 32 sampled functions per cell. GRPO learning rate 0.001, beta 0, one fresh rollout per update. Original four-family saturation heuristic and condition-reading prober remain explicitly visible in the historical source. Mean grammar risk is integrated over each learned finite grammar at size 160.

### Expanded discovery protocol

Two suites (sum/LIS) × nine seeds × twelve families = 216 instances. Each candidate gets at most eight probes, with early exit after a failure; search-arm weights persist across randomly ordered families within a suite/seed. This is a coverage test with equal candidate opportunity, not an on-policy or sparse-rollout-cost measurement. Each success incurs one additional independent compiler replay. Static D gets only size 8/12, values in [-20,20]. Both rule stores start empty; the diagnostic oracle receives the evaluator's full labels.

Search arms: sizes 8/13/25/49/97/160 with uniform integer jitter ±3; value amplitudes 20/64/256; random, sorted or repeated-value structure. Selection is 15% uniform plus 85% weight-normalized; feedback update is `w ← .92w + .08(1 + 12*failure)`. Search seed 9000+seed, family ordering 17000+seed, static-task seed 7000+seed. The evaluator's trigger witnesses and variant-1 suffixes are withheld until search ends for that family. The compiler does not preload family labels; the catalog is nevertheless researcher-designed, not independently blinded or invented by the model.

| Family | Discovered across 18 suite/seed instances |
|---|---:|
| `max_gt_160` | 16/18 |
| `max_gt_40` | 18/18 |
| `min_lt_minus40` | 18/18 |
| `mod_257_251` | 0/18 |
| `mod_2_1` | 18/18 |
| `mod_31_17` | 6/18 |
| `mod_7_3` | 16/18 |
| `size_gt_12` | 18/18 |
| `size_gt_24` | 18/18 |
| `size_gt_48` | 16/18 |
| `size_gt_96` | 18/18 |
| `sum_eq_9973` | 0/18 |

This table includes all misses. `mod_257_251` lies outside the maximum probed size; `sum_eq_9973` demands a specific value configuration. Rare in-range modulo triggers can also escape eight probes. All raw decisions are in [discovery.json](extension_results/discovery.json); actual executed inputs, expected/observed outputs and signatures are in [discovery_witnesses/](extension_results/discovery_witnesses/).

The study's recorded wall time is 254.2 seconds, including evaluator/control execution; smoke tests shared the machine for part of this run. It is not an isolated overhead calibration. The nine RNG seeds are reused across the two suites, so the 216 family instances should not be treated as independent statistical replicates.

### Reproduction and implementation limits

`python reproduce_all.py` rebuilds experiments in a fresh output tree. `--smoke` tests integration without reproducing headline metrics; `--verify-only` checks the distributed evidence. [VALIDATION.md](VALIDATION.md) distinguishes runs executed for this release from preserved historical evidence. File hashes are in `SHA256SUMS.txt`; source/config hashes are embedded in experiment metadata. Historical manifests document old packages and are not manifests for this consolidated directory.

Tested versions are pinned in [requirements.txt](requirements.txt). TRL integration uses protected hooks and is pinned to 1.16.0: one process, plain text, beta 0, one iteration, gradient accumulation 1 and one fresh generation per update. It rejects unsupported configurations. [Hugging Face's GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) describes the underlying trainer; this custom constrained sampler needs independent revalidation for other versions, distributed training or Qwen/Llama models.

All source and supplied checkpoints are under [MIT](LICENSE); there is no withheld training/compiler component. Historical v1/v2/v3 reports are preserved for provenance, not as substitutes for the corrected interpretation above.
