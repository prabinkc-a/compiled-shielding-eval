# Phase 2: neural policies and Algorithm F

Read `Frontier_AI_Alignment_Benchmark_Report_v2.0.md` for the measured results, limitations, and full mathematical specification.

All commands run from the parent `alignment_benchmark` directory:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests phase2/tests
.\.venv\Scripts\python.exe -m phase2.run --output rerun
.\.venv\Scripts\python.exe -m phase2.trl_smoke
```

The default benchmark runs 156 actual neural training experiments: eight seeds per main method/suite and four seeds per ablation/suite. Four CPU worker processes each use one Torch thread. `--pilot --output pilot_rerun` runs a smaller development experiment. The release data are in `phase2/results`; choosing another output directory preserves them.

For the full report pipeline, run the main benchmark with its default output, then run `-m phase2.profile`, `-m phase2.legacy_bridge`, `-m phase2.exact_risk`, `-m phase2.report`, and `-m phase2.audit`. Run the serial profiler only after parallel training completes. The legacy bridge requires the included Phase 1 `results/policies.npz` and `environments.py`.

The neural policy lives in the upgraded parent `torch_integration.py`; original Phase 1 helpers remain compatible. It is a GRU controller over a bounded six-token program language with supplied solver primitives. It is not an LLM trained to discover algorithms or synthesize arbitrary code.

`AlgorithmFGRPOTrainer` is a tested TRL 1.16.0 integration. `DSLBackend` is fully implemented. Other task families need their own permission enforcement and independent verifier. The tested adapter supports fresh, text-only, single-process runs; unsupported resume/distributed/tool execution paths fail explicitly. Qwen/Llama alignment performance was not measured.

The benchmark source and adapter are MIT-licensed. No external repository has been published by this task.
