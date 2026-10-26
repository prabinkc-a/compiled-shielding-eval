# Fully open-source research release

This supersedes the evaluation-only packaging. All training code, adversarial search rules, AST compilation, TRL adapters, checkpoints and raw results are now distributed together under MIT. Historical reports remain in their original phase folders; the corrected current account is `REPORT.md`.

New evidence:

- 24 fresh Oracle Static Mask fits: 98.84% held-out accuracy and zero exact DSL exploit mass. The proposed 99.86% oracle figure was not assumed; it is G's historical result. The oracle shares D's small-only training contexts, while F/G receive some adversarial contexts.
- 45 serial calibration runs: D 1.04×, F 1.38×, G 0.99× and Oracle 1.02× baseline training CPU. These are warm CPU measurements, not universal speedups.
- Expanded trigger-blind Python discovery: 162/216 family instances found and compiled (75.0% recall); size 70/72, modulo/parity 40/72, value-range 52/72. All 162 discovered signatures block held-out suffix variants. The report includes every missed family.
- The original Python/TRL result remains 32/36 supplied cutoff-family instances discovered, with exact grammar risk reduced from 2.696235% to 0.441524%. It is explicitly distinguished from unknown arbitrary-code discovery and from the new unweighted recall study.
- 52 tests pass, plus two real updates through the generalized TRL adapter. The complete fresh-run pipeline is `python reproduce_all.py`; `--smoke` and `--verify-only` have distinct meanings documented in `VALIDATION.md`.
- Evidence verification passed: 39 training-source hashes, 144 neural checkpoint reloads reproducing 3,456 evaluation cells, 18 Python checkpoint risk integrals and 162 independently replayed discovery witnesses.

`README.md` and `REPORT.md` open with a four-sentence summary and one comparison table. `LESSWRONG_POST.md` is a draft only; it has not been submitted to a forum.

Checkpoints moved from the old inference-only layout into their full experimental contexts under `phase2/results`, `phase3/results` and `extension_results/oracle`. The previous `verify_leaderboard.py` entry point remains as a compatibility alias for the expanded evidence verifier.
