# Compiled shielding: speed and exact-support safety in a bounded neural benchmark

**Technical brief for frontier alignment researchers · 26 October 2026**

**Finding.** A task-conditioned neural program policy with dynamically compiled permissions achieved **99.859% true accuracy**, **zero sampled exploits**, and **exactly zero probability on designated DSL exploits after compilation**, at **1.00× training CPU / 0.99× rollout CPU** relative to standard GRPO. Heavy audits covered **0.760%** of training sequences.

**Empirical arc.** V1's finite categorical benchmark reduced sampled reward hacking from 99.78% to zero with hierarchical masks, but a shared verifier blind spot broke its unified method. V2 replaced the categorical policy with a 6,492-parameter autoregressive GRU: adaptive verifier F reached 99.902% blind-spot accuracy, while exact enumeration exposed 0.033013% residual exploit mass. V3's G promotes verified, contract-certified shortcut primitives into hard token masks and removes invariance/replay auxiliary losses. A sparse audit floor remains.

**Evidence.** Three task families (maze, LIS, aggregation), eight main seeds, 240 updates and 7,680 completions per policy. Twenty-four new G policies and 36 ablations were trained; 96 frozen v2 comparators were reused transparently. Fresh randomized serial calibration covered 72 disposable checkpoint runs. An independent checkpoint reload reproduced 1,440 final evaluation rows. The exact integrator enumerated every one of 2,304 programs at matched stress inputs.

**What did not meet the target.** G missed the requested 99.90% accuracy threshold. Its difference from F was -0.043 percentage points, paired 95% seed-bootstrap interval [-0.087, +0.000]. Lower cost and stronger support exclusion do not establish strictly better accuracy or universal Pareto optimality. Early-exit gains depend on attack prevalence.

**Python demonstration.** Eighteen actual TRL runs trained two-layer causal Transformers to emit BPE-token Python functions, validated by AST permissions and executed in isolated subprocesses. G reached 100.000% sampled held-out accuracy; n=160 residual exploit mass within the supplied grammar was 0.441524%. Algorithm bodies and the 20-function-per-family grammar were supplied; this is constrained synthesis, not unrestricted code generation or algorithm invention.

**Scope of the guarantee.** Zero DSL mass follows from trusted primitive semantics and −∞ masks on both live and dormant shortcuts. It does not cover initial discovery, unknown attack classes, shield removal, arbitrary Python, or model-scale distribution shifts. The subprocess worker is not an OS sandbox. No frontier LLM efficacy claim is made.

**Research implication.** Promote counterexamples into sound permissions where action semantics permit it; keep optimizer likelihoods synchronized with each shield version; spend residual audits on unknown behavior. The next discriminating experiment is a pretrained code model with held-out attack families, richer AST contracts, and an independently secured execution broker.

Code, checkpoints, raw curves, ablations, timing attribution, tests, and a pinned TRL adapter accompany the v3 report. This is a draft research brief, not a submission or claim of institutional affiliation.
