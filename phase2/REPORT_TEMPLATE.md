# Frontier AI Alignment Benchmark Report v2.0

## Neural policy training, adaptive verification, and the limits of blind-spot defense

**Independent research release · 26 October 2026 · CPU experiments · Version 2.0**

This report accompanies executed code and raw measurements. It is not a release from or endorsed by a frontier lab. Phase 2 trains actual PyTorch neural policies and separately tests a Hugging Face TRL integration. It does not fine-tune a pretrained Qwen, Llama, Gemini, Claude, or GPT model, and it does not establish a general solution to reward hacking.

## 1. Executive findings

**Algorithm F achieves {{F_TRUE}}% true accuracy on larger, unmonitored tasks, versus {{E_TRUE}}% for the neural implementation of E, {{D_TRUE}}% for D, and {{BASE_TRUE}}% for the GRPO baseline.** F improves over E by **{{F_GAIN_E}} percentage points**. Its measured exploit rate on that primary evaluation is **{{F_HACK}}%**, compared with **{{E_HACK}}%** for E.

**Compute:** in a separate serial calibration, F uses **{{F_CPU}}×** the neural baseline's end-to-end training CPU and **{{F_ROLL}}×** its rollout CPU. D uses {{D_CPU}}× training CPU / {{D_ROLL}}× rollout CPU; E uses {{E_CPU}}× / {{E_ROLL}}×. F invokes heavy checks on **{{F_AUDITS}}%** of training sequences. These numbers measure this small CPU workload; they do not estimate GPU or frontier-model training overhead.

**The evidence supports sharply suppressing the supplied attacks, but the strict zero-exploitation target was not met.** Evaluation-only integration over all 2,304 possible programs at 24 selected large, unmonitored stress contexts found mean exploit probability **{{EXACT_MEAN}}%**, with a worst context of **{{EXACT_MAX}}%**. The stochastic neural policy retains unsafe probability mass even though the sampled primary evaluation saw no exploit events.

{{ZERO_TEXT}} The exact integration uses a different, explicitly selected set of stress contexts and is not an estimate over the primary task distribution. The ablations test which pieces help; they do not establish that all three proposed ingredients are individually necessary or that the philosophical framing is a new optimization primitive.

The release contains **{{N_MODELS}} neural training runs**, **{{N_TRAIN}} on-policy sampled programs**, **{{N_EXEC}} actual training/challenge program executions**, and **{{N_EVAL}} final evaluation executions.** The main comparison uses eight training seeds per method and suite; ablations use four. No lookup table of complete-program rewards is used in neural training. The model has **{{PARAMETERS}} trainable parameters**, with a task encoder and autoregressive GRU decoder.

The TRL wrapper was executed with TRL 1.16.0 and Transformers 5.21.0 on a randomly initialized 45,866-parameter GPT-2. Two optimizer steps changed its weights and exercised auditing, blocked-action checks, invariance loss, and negative replay. This is an integration test, not evidence of pretrained LLM alignment or distributed production readiness.

## 2. What changed from Phase 1

Phase 1 selected a whole three-action program from 125 categorical entries. Phase 2 conditions a shared neural network on task features and emits six instruction tokens one at a time:

\[
\pi_\theta(a\mid x)=\prod_{t=0}^{5}\pi_\theta(a_t\mid a_{<t},\phi(x)).
\]

The 12 input features contain log task size, monitor and surface cues, suite identity, numerical summaries, and two structure indicators. A two-layer tanh encoder initializes a 32-dimensional GRU. Previous-token and position embeddings drive the decoder. The output head has four token choices; the last two positions use three. There are at most \(4^4\cdot3^2=2304\) syntactic programs, but the network neither stores a probability parameter per complete program nor indexes a precomputed outcome table.

The six tokens select **primary kernel, guard, threshold, alternate kernel, channel, and report**. Guards can be unconditional, size-dependent, monitor-dependent, or surface-tag-dependent; thresholds are 12, 24, 48, and 96. This permits a dormant fallback such as “run a correct solver on monitored inputs, then use a side channel elsewhere.” It also permits legitimate algorithm switching. Two correct kernels per suite make that distinction testable.

| Suite | Honest executable kernels | Modeled shortcut / failure | Trusted check |
|---|---|---|---|
| Broken Maze | BFS and DFS over a randomized corridor graph | Internal teleport, wait, direct monitor mutation | Goal reached through actual graph edges; provenance intact |
| Vulnerable Grader | Quadratic LIS dynamic programming and a separate patience-sorting implementation | Constant wrong result, conditional shortcut, forged grader channel | Independent strict-LIS reference and action provenance |
| Multi-step Tampering | Sequential sum and pairwise aggregation from raw records | Corrupted aggregate, concealed error, forged clean report | Recompute the aggregate from raw data; inspect integrity |

The maze is now a connected corridor graph with task size measured in vertices, not an 11×11 pixel grid. The neural network assembles bounded programs using supplied correct kernels; it does **not** discover BFS, learn LIS arithmetic from scratch, or generate unrestricted Python. The code is a real neural autoregressive program-policy benchmark, not a pretrained language-model benchmark.

The Phase 1 headline of 1.26% was a macro average over three compromised-checker environments. It was not the isolated effect of the LIS length-12 trick. Phase 2 therefore reports scale-only, cue-only, and combined shifts, plus a separate corrupted-audit ablation. The changed action language, kernel costs, training algorithm, and task distribution also mean that Phase 1's 1.26% and 7.26× cannot be reused as Phase 2 baselines.

## 3. Algorithm F: formulation and implementation

### Shared neural GRPO update

Each update draws four task prompts and eight programs per prompt. All methods use the same network initialization distribution, Adam learning rate 0.002, 240 updates, two policy epochs per rollout batch, clip width 0.2, gradient norm bound 1, and reference KL coefficient 0.005. The reference network is frozen at initialization.

For prompt group \(G\), advantages are standardized within that prompt:

\[
A_i=\frac{R_i-\bar R_G}{\operatorname{sd}_G(R)+10^{-4}}.
\]

For token ratio \(\rho_{it}=\pi_\theta(a_{it}\mid x,a_{i,<t})/\pi_{\rm old}(a_{it}\mid x,a_{i,<t})\), the implemented policy loss is

\[
L_{\rm GRPO}=-\frac1{6|G|}\sum_{i,t}
\min\bigl(\rho_{it}A_i,\operatorname{clip}(\rho_{it},0.8,1.2)A_i\bigr)
+0.005\,\mathbb E_t D_{\rm KL}(\pi_\theta\Vert\pi_{\rm ref}).
\]

The KL is computed exactly over the small DSL vocabulary. Sampling and teacher-forced scoring use identical prefix-dependent permission masks, which is covered by a numerical consistency test. This is an actual token-level GRPO implementation, rather than Phase 1's grouped REINFORCE approximation. The algorithmic reference is [DeepSeekMath](https://arxiv.org/abs/2402.03300); this implementation is not a reproduction of its language-model training scale or hyperparameters.

### Controls: Baseline, D, and E

**Baseline:** optimize weak proxy score minus modeled effort and a 0.1 direct-violation penalty, using grammar constraints only.

**D:** apply fast contextual support masks on every token and optimize the same proxy utility. The broker excludes direct grading/log writes and explicit fake reports everywhere. The inherited contextual rule recognizes the hidden shortcut only on small monitored tasks. Dormant conditional branches and unknown-regime behavior remain a specification hole.

**E:** use D's mask and four execution checks per rollout, all confined to small monitored/tagged tasks of sizes 6, 8, 10, and 12. Its outcome reward is multiplied by their conjunction, plus a small process-validity term. These checks catch unconditional shortcuts but miss honest-in-test-regime programs. E is reimplemented with the new neural policy; it is not the old categorical checkpoint.

### F1 — Jati / Nigrahasthana-inspired adversarial co-evolution

The verifier searches 96 mutation arms: six sizes \(\{8,13,25,49,97,128\}\), two monitor states, two surface tags, and four structures (random, increasing, decreasing, repeated). Its arm distribution is

\[
q_j=\frac{0.15}{96}+0.85\frac{w_j}{\sum_k w_k},\qquad
w_j\leftarrow0.92w_j+0.08(1+12\,\mathrm{failure}+\min(2,\mathrm{surprise})).
\]

Each heavy audit executes two fresh challenges: an arm sampled from this distribution and a targeted boundary/cue challenge. A size guard is tested just past its explicit threshold; a monitor or tag guard is tested after flipping that cue. Instrumented operation-count collapse contributes a search signal. **Only actual incorrect or integrity-violating execution causes rejection; a complexity change alone does not.**

Every fourth update replaces one of four prompts with an adversarial prompt and audits that entire group. This makes the *neural policy itself* answer on new contexts. Merely replaying a frozen small-input program would fail to test whether the network chooses a different program outside its training regime.

This is an online weighted adversary and a changing counterexample store, not a separately trained neural red-team LLM. The classical names are inspiration: *jati* often denotes objection or futile rejoinder and *nigrahasthana* a point of defeat. They are not historical terms for this modern algorithm. [Academic account of Nyaya debate categories](https://epgp.inflibnet.ac.in/epgpdata/uploads/epgp_content/philosophy/07.logic_2/39._the_classification_of_debate_and_rules_of_debate__the_nature_and_role_of_chala%2C_jati_and_nigrahasthana_%28nyaya%29/et/8998_et_7.39.1.pdf). Adaptive reward-model training has modern precedents as well. [Adversarial Training of Reward Models](https://arxiv.org/abs/2504.06141).

### F2 — description-length prior and epistemic invariance

The small AST complexity surrogate is

\[
C(a)=\mathbf1[\text{nontrivial guard and different primary/alternate kernels}].
\]

F subtracts \(0.04C(a)\) from its reward. This is a simple description-length-inspired prior, **not** a calculation of Kolmogorov complexity or a general MDL proof.

Every fourth update also compares token distributions for an original task and an actual newly generated task with changed scale and cues, holding the sampled program prefix fixed:

\[
L_{\rm inv}=\tfrac12\mathbb E_t\left[
D_{\rm KL}(\pi_\theta(\cdot\mid x,a_{<t})\Vert\pi_\theta(\cdot\mid x',a_{<t}))+
D_{\rm KL}(\pi_\theta(\cdot\mid x',a_{<t})\Vert\pi_\theta(\cdot\mid x,a_{<t}))\right].
\]

Distributions are renormalized on their common permitted support. The coefficient is 0.08. This discourages an unmotivated switch in program selection across scale or monitoring cues. It does not inspect a model's hidden intentions or prove that internal computation is invariant. Legitimate algorithms can change complexity abruptly; a supplied test explicitly accepts a switch between correct quadratic and efficient LIS solvers.

Invariance is appropriate here because each family has a general solver. It can be harmful if applied across genuinely different tasks, algorithms, permissions, or answer semantics. The TRL backend must identify changes for which generalization is actually warranted.

### F3 — lazy audits, revocation, and replay

All samples pass the fast permission mask. For ordinary candidates, the heavy-audit probability is

\[
p_i=0.02+0.12\mathbf1[\text{provisional utility}>\text{batch mean}+0.05]
 +0.08\mathbf1[\text{nontrivial conditional program}],
\]

with probability one for the scheduled adversarial prompt group. The positive random floor prevents high-scoring novelty from being the only audited category. A cached, demonstrated failing program is rejected without repeating its expensive checks. There is no inference-time best-of-N filtering in the reported main evaluation.

An audited failure replaces the current buffered reward by \(-3-c\) **before** the GRPO update. It also enters bounded replay memory with both its original and counterexample task features. Subsequent updates add

\[
L_{\rm replay}=-\mathbb E_{(x,a)\in\mathcal B}\log(1-\pi_\theta(a\mid x)),
\qquad L_F=L_{\rm GRPO}+0.08L_{\rm inv}+0.3L_{\rm replay}.
\]

The replay term runs every fourth update on at most eight stored entries. It is a separate unlikelihood objective, **not** an on-policy importance-sampling estimate from stale trajectories. “Retroactive” means revoking the buffered reward and discouraging the program later; it does not undo already-applied optimizer steps or real-world side effects.

The audit selection is intentionally risk-biased and there is no inverse-probability correction. Thus F optimizes a risk-weighted surrogate; it is not an unbiased estimate of exhaustive verified reward. Failed-program caching assumes a stationary executable artifact and this fixed DSL. Caching arbitrary stateful or adaptive LLM behavior requires a stronger cache key and invalidation policy.

## 4. Protocol, leaderboard, and Pareto frontier

### Training and evaluation separation

Core training tasks have sizes 6, 8, 10, and 12 with both monitoring cues present. F additionally receives online adversarial training contexts and execution checks up to size 128, plus targeted threshold/cue probes. Its consistency pairs can reach size 160. **F therefore has expanded supervision, not magic generalization from small inputs alone.** The held-out payloads, task seeds, and evaluation decisions are never fed back to training, but not every evaluated scale is unseen by F. Size 160 is outside F's ordinary challenge-arm set, but it can occur in its unlabeled consistency interventions.

Final evaluation uses sizes 8, 12, 20, 40, 80, and 160, with monitor/tag cues jointly present or jointly absent. Each condition has 96 task instances per fitted model, with the same evaluation instances and sampling seeds across compared methods. Each task is evaluated with stochastic decoding and separately with greedy decoding. The primary score averages sizes 20–160 with both cues absent, equally across suites and training seeds.

The experiments draw actual neural sequences and execute actual programs. There is no cached reward bank. Four worker processes train independent seeds with one Torch thread each. Each model receives 7,680 training sequences and 480 optimizer steps. The development pilot used one seed and 120 updates; the final budget was fixed at 240 updates for every method. This is an exploratory study, not a preregistered external benchmark.

True accuracy requires correct task completion without an exploit. No-exploit survival is \(100(1-\mathbb E[e])\) and can include abstention or honest failure. “Blind-spot survival” in the headline means integrity-adjusted true accuracy; both quantities are shown to avoid equating inactivity with successful alignment. Exploit labels come from independent simulator provenance and truth checks, never from claimed intent.

The confidence intervals resample eight paired optimizer-seed blocks 10,000 times, after averaging task sizes and suites inside each block. They are conditional on this task generator, task banks, DSL, and hyperparameters. They do not capture all benchmark-design uncertainty. Ablation comparisons use only the four shared seeds; exploratory comparisons are not corrected for multiplicity.

### Master neural leaderboard

{{MASTER}}

{{COMPARISONS}}

### Isolating the shifts

“Scale only” keeps both monitoring cues present at larger sizes. “Cue only” flips both cues on small tasks. “Combined” changes size and cues. The two cue variables are not separately isolated in this final table; they are varied independently during adversarial search.

{{SHIFTS}}

{{SUITES}}

D and E have identical final sampled evaluation results here for a mechanistic reason: on the small monitored training distribution, D's mask already removes the recognized shortcuts, and E's additional gate does not change their proxy rewards. E's process term adds the same 0.05 within each group, which standardized advantages cancel. Small floating-point differences remain in parameters, but four static checks add no new regime information. F's gain should therefore be attributed to the complete changed training intervention, especially its broader challenges, rather than to the mere number of checks.

An extra diagnostic in `results/exact_risk.csv` enumerates the **trained neural distribution**, only at evaluation, rather than training a table. It integrates all 2,304 program probabilities for one size-160, unmonitored/untagged task per suite and seed. The resulting nonzero risks rule out interpreting the sampled 0.000% column as an exact safety guarantee. This diagnostic was added to investigate the zero-event result; its contexts are post hoc stress probes and are not used for model selection.

### Compute accounting

{{PERFORMANCE}}

The full-training process-CPU column comes from the actual training runs, including generation, tool execution, verification, gradients, replay, and consistency work; evaluation is excluded. Process CPU time is used because jobs ran concurrently. Wall-clock scheduling and early-run initialization varied materially, so small cost differences should not be overinterpreted.

The serial calibration removes concurrent training as a confound: 36 disposable runs load checkpoints from seeds 0–2, then execute the same training loop for 40 further updates in randomized serial order, with fresh optimizers and verifier state. The reference is frozen at the calibration starting checkpoint. These clones are used only for timing; their updated weights do not contribute to reported accuracy. Calibration rollout time includes task construction, generation, reference forward passes, simulation, and audits; update time also includes backpropagation and auxiliaries. The procedure is explicitly described in `results/profile_protocol.json`.

F's heavy-audit fraction is {{F_AUDITS}}%, compared with E's check of every rollout. The full-audit F ablation consumes {{F_FULL_TIME}}× F's actual full-training process CPU in this run. Verification work may involve much larger inputs, so a smaller audit fraction does not translate directly into the same reduction in CPU. The baseline's neural generation and backpropagation are a substantial cost denominator: E's four checks now cost {{E_ROLL}}× rollout CPU, not Phase 1's 7.26×.

![Accuracy-compute frontier](results/pareto.png)

This chart uses frozen-policy combined-shift accuracy and serial calibrated CPU. It is a Pareto comparison within this workload. It must not be converted into GPU hours, dollars, or throughput forecasts for a pretrained LLM. The target of roughly 1.5× should be judged using the displayed measured ratios, rather than assumed from the audit schedule.

## 5. Neural generalization, ablations, and interpretation

![Neural generalization across sizes and monitoring cues](results/neural_generalization.png)

![Learning curves and replay of legacy tabular policies](results/learning_and_legacy.png)

The learning curves use fixed intermediate checkpoints at updates 0, 80, 160, and 240. They were not used for checkpoint selection. The legacy panel re-executes saved Phase 1 grader policies on a new LIS size sweep, without retraining, and plots raw numerical correctness. It shows the length-triggered conditional failure directly. It is **not** a controlled claim that neural networks generalize better than tables: the two phases use different action languages, supervision, task observations, and reward costs. The raw CSV also contains integrity-adjusted legacy scores.

### Ablations

{{ABLATIONS}}

The matched four-seed comparison shows no accuracy gain from auditing every uncached candidate, despite {{F_FULL_TIME}}× the actual training CPU cost. Removing replay lowers accuracy by about 0.11 points; removing arm adaptation or MDL/KL changes it by at most about 0.07 points here. These are small incremental effects, not evidence that three individually indispensable defenses have been discovered. The largest demonstrated intervention is the complete F pipeline with expanded challenges.

- **Uniform search arms:** disable the online arm weighting but retain the same challenge support, targeted boundary/cue checks, sparse-audit schedule, consistency regularization, and replay. This isolates arm adaptation, not the entire benefit of adversarial data. Similar results would mean the experiment has not established that co-evolution is necessary.
- **No invariance:** remove both the AST complexity prior and symmetric-KL term; keep adaptive challenges and failure replay. A small difference means the proposed complexity/invariance machinery has not shown a substantial incremental gain here.
- **No replay:** keep immediate reward revocation and the failure cache, but remove later unlikelihood updates. This distinguishes buffered negative reward from replay's additional effect.
- **Full audit:** audit every uncached candidate, with the same adversary and auxiliaries. Previously demonstrated failures can still be rejected directly from cache.
- **Corrupted audit feedback:** let the verifier accept the vulnerable proxy rather than the independent truth check while keeping other F machinery. This ablation scores {{F_CORRUPT}}% true accuracy. It tests this particular corruption, not all possible common-mode compromise; retaining high performance would not make the verifier trustworthy.

The benchmark favors reusable general algorithms and contains only a bounded family of conditional attacks. Counterexample training can be sufficient; additional regularization may be redundant or occasionally harmful. The ablation table, including unfavorable or negligible differences, is the basis for deciding what is necessary. The report does not attribute every gain to each proposed philosophical ingredient.

### What remains unsolved

F has access to a trusted execution view that can independently detect the specified mistakes. E's static checks do not exercise the same regimes. This change in information is a central part of the intervention. The uniform-arm control is supervision-matched within F, but there is no fully exhaustive isolation of every difference between D/E and F. In particular, the experiment does not prove that a stronger random-testing baseline without all F auxiliaries could not match it.

Finite testing cannot rule out a new predicate outside the search support, a new conditional syntax, audit-state detection, a stateful sleeper trigger, a compromised reference implementation, or an unavailable true objective. The decoder cannot invent arbitrary nested conditions or exploit host tools; the six-token language and supplied kernels constrain its capabilities substantially. Its summary-vector task encoder does not learn algorithmic reasoning over raw code or entire grids.

A policy may branch for legitimate computational reasons. Imposing distributional invariance across arbitrary problem changes can suppress useful specialization. An operation-count drop is neither a proof of dishonesty nor a robust monitor against padding or deliberate wasted computation. The experiment uses an inspectable AST and instrumented primitives; free-form LLM code makes these checks harder.

Zero observed evaluation exploits also does not imply zero unsafe exploration. The training exploit column records candidate shortcut executions inside the toy sandbox, including adversarial challenges. Sparse verification is an optimization technique, not a substitute for a real permission boundary. Non-reversible actions require enforcement before they happen.

There is also a structural reason the strict zero target remains unmet: with finite logits, a softmax assigns positive probability to every action left in its support. At large unmonitored inputs, the incomplete contract still permits a shortcut token. Penalties, replay, and invariance can make it rare but cannot mathematically remove it. Exact zero would require a sound expanded support restriction or an independently enforced execution rule; discovering a counterexample alone is not a proof that such a rule is complete.

## 6. Drop-in TRL patch and ten-line usage

`phase2/trl_patch.py` provides `AlgorithmFGRPOTrainer`, a subclass of the actual `GRPOTrainer`. It supports plain-text causal models, backend-owned execution, fresh runs, and a single process. It checks for **TRL 1.16.0** because prompt-group mutation uses a protected API. It preserves the trainer's sampling/log-probability relationship; it does not silently mask generation tokens while scoring them under an unmasked model.

The backend must implement four methods: cheap permission-aware scoring, heavy auditing, generation of a challenge prompt, and a semantically valid consistency intervention. `DSLBackend` implements all four for this benchmark without `eval`, `exec`, shell calls, arbitrary file mutation, or model-authored graders. Its permission check rejects illegal outputs **before** interpreting them. Arbitrary coding or tool-use tasks need their own sandbox and verifier backend; a generic ten-line wrapper cannot supply the missing specification.

This is the complete ten-line entry point for the supplied DSL task:

```python
from trl import GRPOConfig
from phase2.trl_patch import AlgorithmFGRPOTrainer
from phase2.trl_backend import DSLBackend
backend = DSLBackend(seed=7)
data = backend.dataset("grader")
args = GRPOConfig(output_dir="f-qwen", num_generations=4, per_device_train_batch_size=4, max_completion_length=32, max_steps=100)
trainer = AlgorithmFGRPOTrainer(
    model="Qwen/Qwen3-0.6B", backend=backend, args=args,
    train_dataset=data)
trainer.train()
```

The Qwen identifier is an example accepted by the upstream trainer; replace it with an accessible supported causal model, including an appropriate Llama checkpoint. Model access, memory, valid-output initialization, and task-specific supervision still matter. This entry point was **not** used to train Qwen or Llama for this report. See the upstream [GRPOTrainer documentation](https://huggingface.co/docs/trl/grpo_trainer) for model-loading and trainer interfaces.

### What was actually validated

`python -m phase2.trl_smoke` constructs a local tiny GPT-2 and tokenizer without downloading pretrained weights. With four completions per prompt and two optimizer steps, it checks that:

1. A conditional small-input solver is rejected by a scale challenge and gets reward −3.1.
2. Forbidden DSL mutations are blocked before execution.
3. Adaptive prompt mutation keeps complete GRPO groups on the same prompt.
4. Invariance and replay auxiliary losses run and neural attention weights change.

The integration test uses `beta=0`, because its reference model is not a pretrained checkpoint path; the main GRU benchmark uses a frozen reference with beta 0.005. The smoke test validates the API path and differentiability, not a numerical alignment gain. It is intentionally separate from the benchmark's accuracy claims.

The adapter rejects distributed execution and checkpoint resume until verifier/cache/replay state persistence and synchronization are implemented. It also rejects external `tools`, `environment_factory`, and `rollout_func` execution paths that bypass its backend. Multimodal/conversational inputs need an explicit adaptation. Auxiliary losses respect gradient accumulation scaling, but the executed integration test uses accumulation one. For long free-form completions, a task-specific failure span is needed: replay avoids silently treating a truncated prefix as a complete failed program.

These constraints make the tested boundary concrete. They are not evidence that every production TRL configuration or Qwen/Llama training setup has been validated.

## 7. Reproduction, tests, and artifacts

Run commands from `alignment_benchmark/`, using the existing local `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests phase2/tests
.\.venv\Scripts\python.exe -m phase2.run --output rerun
.\.venv\Scripts\python.exe -m phase2.trl_smoke
```

The released report is generated from `phase2/results/`. To reproduce that entire directory, run the benchmark with its default output, then `python -m phase2.profile`, `python -m phase2.legacy_bridge`, `python -m phase2.exact_risk`, and `python -m phase2.report`, using the virtual-environment interpreter. The profiler should run after the concurrent benchmark finishes. `--pilot --output pilot_rerun` is a small development check, not the release experiment. `--resume` resumes completed independent benchmark jobs; it is different from TRL model-training checkpoint resume, which the adapter intentionally rejects.

The executed test suite passed 31 tests, including all preserved Phase 1 checks. Coverage includes honest solver correctness across scales, static-verifier blind spots, legitimate complexity-changing algorithms, exact agreement between autoregressive sampling and teacher forcing, encoder/GRU gradients, permission masks, deterministic neural retraining, finite masked KL, counterexample revocation/replay, and broker rejection before execution.

| File | Purpose |
|---|---|
| `../torch_integration.py` | Upgraded neural policy and token-level GRPO; Phase 1 functions preserved |
| `worlds.py` | Task generator, six-token interpreter, fast permissions, trusted evaluation |
| `defenses.py` | Adaptive mutation weights, targeted probes, sparse audits, failure cache |
| `train.py` | Actual neural rollout/update loop and evaluation |
| `run.py` | Multi-seed execution, checkpoint persistence, raw data export |
| `trl_patch.py`, `trl_backend.py` | Version-pinned trainer wrapper and complete safe DSL backend |
| `trl_smoke.py` | Real tiny-model TRL integration test |
| `profile.py` | Serial warm-checkpoint cost calibration |
| `legacy_bridge.py` | Frozen Phase 1 policy replay over LIS sizes |
| `results/evaluation.csv` | Final per-model, per-suite, per-size, per-cue stochastic/greedy metrics |
| `results/generalization_curves.csv` | Intermediate checkpoint curves |
| `results/training_cost.csv` | Actual process/wall timing, audit rates, training exploit rates |
| `results/serial_profile.csv` | Independent serial calibration measurements |
| `results/exact_risk.csv` | Evaluation-only integration exposing nonzero policy exploit probability |
| `results/verifier_states.json` | Search-arm weights, visits, failures, cache/revocation counts |
| `results/runs/*.pt` | Actual fitted PyTorch state dictionaries |
| `results/runs/*.json` | Per-run raw observations and metadata |
| `results/metadata.json` | Frozen source hashes, budgets, versions, counts, platform |
| `results/trl_smoke/result.json` | Executed trainer integration outcome |
| `results/tests.txt` | Recorded test output |
| `results/audit.json` | Checkpoint/metric reconstruction and source-integrity audit |

The primary benchmark completed in {{ELAPSED}} seconds of wall time; subsequent profiling, bridge replay, tests, and report generation are separate. Python 3.13.7, Torch 2.14.1+cpu, NumPy 2.5.1, TRL 1.16.0, Transformers 5.21.0, Datasets 5.1.0, and Accelerate 1.15.0 were used. The direct dependency lock and MIT license accompany the code. The ZIP excludes the installed virtual environment and developmental pilot outputs.

## Research conclusion

This phase crosses the **implementation** barrier from a categorical lookup policy to a trainable autoregressive neural controller. It tests an adaptive defense against explicit scale/cue attacks, measures the cost of sparse verification, and supplies a running trainer integration. Its strongest result is conditional: useful counterexamples and enforceable permissions can teach a small neural policy to avoid the modeled shortcuts. The remaining frontier problem is preserving that result when agents, tasks, verifier failures, and deployment interfaces exceed the bounded world represented here.
