# Release validation

This release distinguishes new experiments from preserved historical evidence.

- **New oracle experiment:** 24 fresh fits (three suites, eight seeds, 240 updates each), saved weights, complete evaluation rows and exact support integration. The oracle receives D's small-only task stream; there are no audits or verifier rewards.
- **New cost experiment:** 45 serial warm-checkpoint runs, five methods, three suites, three seeds and 40 updates per run. No other benchmark was launched during calibration.
- **New discovery study:** 216 family instances across two suites, nine seeds and twelve trigger families, with a maximum of eight search probes per instance; independent witness replay, unseen suffix transfer and honest-control checks. This is not expanded neural training.
- **Tests:** `python reproduce_all.py --smoke` passed all 52 tests and performed two real updates through the generalized TRL adapter. Weights changed and remained finite; four audits found one revoked subtree. The smoke result is preserved in `extension_results/trl_smoke.json`.
- **Historical evidence:** complete v1/v2/v3 sources and results are preserved. Prior D/F/G and Python headline results were not retrained and relabeled as new experiments in this release.

`python reproduce_all.py` supplies the complete fresh pipeline in a new directory, including historical experiments, new controls, reports and verification. The entire historical retraining pipeline was not rerun during release preparation; its constituent original experiments and the new stages have execution evidence. `--smoke` does not establish full-run metric reproducibility.

`python reproduce_all.py --verify-only` passed: 39 recorded training-source hashes matched, all 144 main neural checkpoints reproduced 3,456 sampled/greedy evaluation cells and exact finite-DSL risk, all 18 Python policies reproduced grammar risk, and all 162 new discovery witnesses replayed successfully. This check took 75.5 seconds on this machine. The machine-readable result is in `extension_results/verification_summary.json`; it is distinct from the smoke test and from full retraining.

The finalized manifest is checked again after packaging. `python verify_release.py --hash-only` checks file integrity without claiming to recompute scientific results.

Timing is hardware-dependent. The package makes no one-minute full-reproduction claim, no independently blinded attack-set claim and no pretrained-LLM safety claim.
