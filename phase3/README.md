# Phase 3 — compiled shielding

Algorithm G replaces F's invariance/replay objectives with versioned, certified
hard masks. The reports describe measured results, including missed targets.
The executable DSL and supplied Python grammar are bounded research environments.

From `alignment_benchmark`, using the existing CPU environment:

```powershell
.venv\Scripts\python.exe -m phase3.run
.venv\Scripts\python.exe -m phase3.profile
.venv\Scripts\python.exe -m phase3.latency
.venv\Scripts\python.exe -m phase3.exact_risk
.venv\Scripts\python.exe -m phase3.python_trl_benchmark
.venv\Scripts\python.exe -m phase3.audit
.venv\Scripts\python.exe -m phase3.audit_python
.venv\Scripts\python.exe -m pytest -q tests phase2/tests phase3/tests
.venv\Scripts\python.exe -m phase3.report
.venv\Scripts\python.exe -m phase3.package
```

`run` trains 24 G policies and 36 ablations from scratch. Baseline/D/E/F accuracy
uses the 96 frozen v2 policies and their previously reproduced evaluations; the
serial profiler freshly measures all methods. Run profiling alone, without other
CPU experiments. The Python experiment trains 18 two-layer Transformers for 48
steps each. It emits actual BPE-token Python functions, parses their ASTs, and
executes validated code in isolated subprocesses. It uses 20 supplied compositional
functions per task family, not unrestricted code generation.

The existing v2 source/data is preserved. See `phase2/requirements-lock.txt` for
the tested environment; model, dependency, and source hashes accompany results.
Resume is intentionally unsupported for the TRL adapter until compiler state and
optimizer buffers can be restored atomically.

## TRL integration

The reference wrapper needs a trusted task backend and a tokenizer-compatible
grammar. Given `model`, `tokenizer`, and a plain-text `dataset`, its call site is:

```python
from trl import GRPOConfig
from phase3.trl_patch import AlgorithmGGRPOTrainer
from phase3.python_backend import PythonBackend
from phase3.python_grammar import PythonGrammar
backend = PythonBackend("grader", seed=0)
grammar = PythonGrammar(tokenizer, "grader", backend)
args = GRPOConfig(output_dir="out", beta=0, num_iterations=1, steps_per_generation=1, disable_dropout=True,
                  gradient_accumulation_steps=1, num_generations=8, per_device_train_batch_size=8)
trainer = AlgorithmGGRPOTrainer(model=model, args=args, processing_class=tokenizer,
    train_dataset=dataset, backend=backend, grammar=grammar)
trainer.train()
```

Use prompts such as `<grader><CODE>`, with those tokens registered in the
tokenizer. `make_tokenizer()` and the complete local model setup are in
`python_trl_benchmark.py`. This wrapper is pinned to TRL 1.16.0 and single-process
text training; it rejects incompatible rollout reuse and auxiliary reward paths.
It is not a universal ten-line security patch for arbitrary Qwen/Llama output.

## Safety scope

The DSL certificate relies on the declared semantics of kernel 3. It masks both
live and dormant occurrences; ordinary wrong-answer kernel 2 remains learnable.
Zero probability is a structural property of this post-compilation support, not
of the unconstrained model or its warm-up trajectory.

Python rules match an entire leading guard and return subtree, scoped to one task
family. Counterexamples do not justify banning all length checks. Valid empty
guards and correct scale switches remain permitted. Rule publication occurs only
between optimizer batches. Both generation and likelihood scoring use the same
frozen token trie. Constant-time trie transitions do not make AST parsing, rule
compilation, or batched neural masking constant-time.

The Python worker uses restricted builtins, an AST allowlist, a trace instruction
budget, and subprocess timeout. It is **not an OS sandbox for arbitrary hostile
Python**. Do not use it for untrusted general code or external tools. Deployments
need a separate, resource-limited execution broker.
