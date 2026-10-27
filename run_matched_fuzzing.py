"""Paired adaptive-versus-uniform discovery with equal search-execution budgets.

Each suite/seed gets 44 search executions, for 792 per method. The two methods
share candidate order, arms, input generator, feedback oracle, eight-probe cap,
compiler, and scheduling; only weight updates differ. Each round visits every
not-yet-discovered candidate once, so early successes free budget for the rest.
Any exhausted catalog is padded with audits of already discovered candidates.
Independent witness replays are recorded separately from the search budget.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from adversarial_prober import InputProber
from ast_compiler import SubtreeCompiler, run_inputs
from discovery_catalog import families, program

ROOT = Path(__file__).resolve().parent


class UniformProber(InputProber):
    """Exactly the same input generator, with equal arm weights throughout."""
    def observe(self, index, failed):
        pass


def run_one(suite, seed, method, budget=44, family_cap=8):
    search = (InputProber if method == 'adaptive' else UniformProber)(9000 + seed)
    compiler = SubtreeCompiler(suite)
    catalog = families()
    random.Random(17000 + seed).shuffle(catalog)
    records = {f['name']: dict(suite=suite, seed=seed, method=method,
        family=f['name'], category=f['category'], discovered=False, probes=0,
        heldout_variant_blocked=False, padding_probes=0) for f in catalog}
    trace = []
    for call in range(budget):
        candidates = [f for f in catalog if not records[f['name']]['discovered']
                      and records[f['name']]['probes'] < family_cap]
        padding = not candidates
        if padding:
            candidates = catalog
        # Least-probed candidate first, with fixed randomized catalog order as
        # the tie break; this avoids hidden adaptation based on trigger labels.
        family = min(candidates, key=lambda f: records[f['name']]['probes'])
        record = records[family['name']]
        arm, x = search.sample()
        source = program(suite, family)
        result = run_inputs(source, suite, [x])[0]
        search.observe(arm, not result['correct'])
        record['probes'] += 1
        record['padding_probes'] += int(padding)
        trace.append(dict(call=call, family=family['name'], arm=arm,
                          correct=result['correct'], padding=padding))
        if not result['correct'] and not record['discovered']:
            assert compiler.propose(source, x), (suite, seed, method, family['name'])
            compiler.publish()
            record['discovered'] = True
            record['heldout_variant_blocked'] = not compiler.admissible(program(suite, family, 1))
            assert record['heldout_variant_blocked']
    return dict(records=list(records.values()), trace=trace,
                witnesses=compiler.witnesses, rules=sorted(compiler.rules))


def run(output, seeds=9, budget=44, family_cap=8):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    jobs = [(suite, seed, method) for suite in ('tamper', 'grader')
            for seed in range(seeds) for method in ('adaptive', 'uniform')]
    # Interleaving order is fixed independently of observed outcomes.
    random.Random(21907).shuffle(jobs)
    runs = []
    for suite, seed, method in jobs:
        result = run_one(suite, seed, method, budget, family_cap)
        runs.append(result)
        print(f"{len(runs)}/{len(jobs)} {suite}/{seed}/{method}: "
              f"{sum(r['discovered'] for r in result['records'])}/12", flush=True)
    result = dict(records=[r for run in runs for r in run['records']],
        runs=runs, protocol=dict(seeds=seeds, suites=['tamper', 'grader'],
            methods=['adaptive', 'uniform'], budget_per_suite_seed=budget,
            total_search_executions_per_method=budget*seeds*2,
            family_cap=family_cap, job_order_seed=21907,
            input_seed='9000 + seed', catalog_order_seed='17000 + seed',
            scheduling='least-probed undiscovered eligible family; fixed catalog order breaks ties',
            padding='after all families discovered or capped, audit least-probed catalog entries',
            excluded_from_search_budget='one independent compiler replay per discovery',
            comparison='same total and per-suite-seed search budget; individual family opportunities may differ',
            elapsed_seconds=time.perf_counter()-start,
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__),ROOT/'adversarial_prober.py',ROOT/'ast_compiler.py',
                 ROOT/'discovery_catalog.py',ROOT/'discovery_worker.py']}))
    (output/'matched_fuzzing.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'extension_results')
    parser.add_argument('--seeds', type=int, default=9)
    parser.add_argument('--budget', type=int, default=44)
    args = parser.parse_args()
    run(args.output, args.seeds, args.budget)
