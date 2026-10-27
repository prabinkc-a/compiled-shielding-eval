# Automated exploit discovery and compiled shielding

We tested whether adaptive adversarial probing discovers conditional Python exploits more effectively than uniform fuzzing with the same execution budget. Across a supplied attack catalog, adaptive search found 144/216 family instances and uniform fuzzing found 148/216, with 792 search executions each. Both can miss triggers, and the compiled shield only transfers across an identical leading guard and return, so a trivial rewrite can evade it.

| Trigger category | Adaptive discovery | Uniform discovery | Paired difference |
|---|---:|---:|---:|
| size | 64/72 (88.9%) | 68/72 (94.4%) | -5.6 pp |
| modulo/parity | 32/72 (44.4%) | 34/72 (47.2%) | -2.8 pp |
| value range | 48/72 (66.7%) | 46/72 (63.9%) | +2.8 pp |
| overall | 144/216 (66.7%) | 148/216 (68.5%) | -1.9 pp |

Family instances are twelve supplied families across two suites and nine seeds; this is a discovery stress test, not newly trained neural policies. Both methods use the same arms, candidate scheduler, search budget and compiler; independent witness replays add 144 executions for adaptive and 148 for uniform. The overall paired 95% seed-bootstrap interval for adaptive minus uniform is -7.4 to +4.6 percentage points; nine seeds are a small sample. These data do not show a clear advantage from adaptive weighting.

For example, `len(x) > 12` and `12 < len(x)` have the same meaning but different AST signatures; a rule learned from one does not block the other. The supplied attack catalog is researcher-designed, and neither method demonstrates arbitrary-code or LLM-scale safety.

## DSL sanity check

| Neural DSL method | Held-out accuracy | Exact exploit mass | Training CPU |
|---|---:|---:|---:|
| Standard GRPO | 30.15% | 69.7919% | 1.00× |
| D · monitored-only static mask | 90.31% | 11.6023% | 1.04× |
| F · soft adaptive probing | 99.90% | 0.0330% | 1.38× |
| G · discovery + compiled shielding | 99.86% | 0.0000% | 0.99× |
| Oracle · static global mask from step 0 | 98.84% | 0.0000% | 1.02× |

Accuracy is sampled on larger, unmonitored tasks; exact mass integrates the complete finite DSL at one large task per suite/seed. CPU is a fresh serial warm-checkpoint calibration against standard GRPO, not inference latency or an LLM speedup. The table is the DSL experiment; the opening Python results are a separate experiment.

**The oracle correction matters.** Its measured accuracy is 98.84%, not an assumed 99.86%. The oracle eliminates every specified shortcut from initialization but can still choose ordinary incorrect algorithms or abstain. It differs from D only in the mask; F/G also train on some adversarial contexts. G matches the oracle's zero exploit support after discovery, but this design does not isolate discovery as the sole cause of its accuracy advantage.

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
