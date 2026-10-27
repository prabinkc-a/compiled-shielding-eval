# Matched-budget fuzzing research update

A paired control gives adaptive search and uniform fuzzing 792 search executions each on the same two suites, nine seeds, twelve supplied trigger families, input arms, candidate scheduler, verifier and compiler. Adaptive search discovered 144/216 family instances; uniform fuzzing discovered 148/216. The paired difference is -1.9 percentage points (95% seed-bootstrap interval -7.4 to +4.6): no clear advantage from adaptive weighting.

Both methods used every search call on an unresolved family. Independent witness replays add 144 and 148 executions respectively, and family-specific search allocations can differ. The primary full-budget protocol was chosen after inspecting an eight-probe-per-family matched run, whose 144/216 versus 148/216 result and padding counts are preserved as a sensitivity check. The earlier 162/216 adaptive result used yet another scheduling protocol and is reported separately. No neural policy was trained on the new comparison catalog.

README, REPORT, and the draft LessWrong post now lead with the trigger-category comparison and immediately flag that a trivial equivalent guard rewrite evades an exact-signature shield. All source, raw runs, candidate allocation traces, witnessed inputs, and reproduction scripts are included under MIT.

The underlying DSL oracle and prior trained-model findings remain available, with their original limits. A complete fresh reproduction is `python reproduce_all.py`; `--smoke` and `--verify-only` have distinct meanings. The LessWrong post remains an unpublished draft.
